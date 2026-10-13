"""Cut mimicgen square_d0 FAILURE rollouts to an approach -> outcome clip (step 1 of a LoRA fine-tune).

usage: python robometer/finetune/segment_square.py --src <failure rollout dataset: local dir or HF id> --out <dir>

Rule:
  start           frame 0 -- the approach is KEPT; only the tail is cut
  end_rewarded    first frame with reward > 0
  end_unrewarded  argmin over frames at or after the grasp of |nut - ref|
  min_segment_frames 20

In square_d0 the peg is FIXED -- the nut position at first reward across the success split has std
[3.3, 4.2, 9.9] mm -- so the endpoint is |nut - ref| in world coordinates, with ref the seated nut pose
measured from that split. (The threading rule was the relative form |(needle - tripod) - ref|.)
square_d0 ONLY: square_d1 moves the peg, and no segmenter for square_d1 or threading_d0 is part of this
repository -- their segmented LoRAs are assets.

observation.state layout (23):
  [0:3] eef_pos  [3:7] eef_quat  [7:9] gripper_qpos  [9:12] nut_pos  [12:16] nut_quat
  [16:19] nut_to_eef_pos  [19:23] nut_to_eef_quat
Output: a LeRobot dataset of the cut clips (the input of make_rbm_from_seg.py). Successes are not segmented.
"""
import os, json, subprocess, shutil, argparse
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="failure rollout dataset (local dir or HF id)")
ap.add_argument("--out", required=True)
ap.add_argument("--near-mm", type=float, default=80.0)
ap.add_argument("--sustain", type=int, default=40)
ap.add_argument("--min-frames", type=int, default=20)
ap.add_argument("--ref", default="0.2288,0.1013,0.8872")
ap.add_argument("--keep-start", type=int, default=1,
                help="1 = keep the full approach and cut only the tail (start=0). The grasp is still "
                     "detected, but only to constrain WHERE the outcome frame is searched for.")
A = ap.parse_args()
REF = np.array([float(x) for x in A.ref.split(",")])

from huggingface_hub import snapshot_download
src = Path(A.src) if Path(A.src).is_dir() else Path(snapshot_download(A.src, repo_type="dataset"))
info = json.loads((src/"meta/info.json").read_text()); FPS = info["fps"]
CAMS = [k for k in info["features"] if k.startswith("observation.images.")]
dt = pq.read_table(next((src/"data").rglob("*.parquet")))
d = dt.to_pydict()
st = np.stack([np.asarray(x) for x in d["observation.state"]])
epi = np.asarray(d["episode_index"]); fr = np.asarray(d["frame_index"])
rw = np.asarray(d["reward"], dtype=float).ravel()
ept = pq.read_table(next((src/"meta/episodes").rglob("*.parquet"))).to_pydict()
vloc = {int(e): {c: (int(ept[f"videos/{c}/chunk_index"][i]), int(ept[f"videos/{c}/file_index"][i]),
                     float(ept[f"videos/{c}/from_timestamp"][i])) for c in CAMS}
        for i, e in enumerate(ept["episode_index"])}

segs = []
for e in np.unique(epi):
    m = epi == e; idx = np.flatnonzero(m)
    n2e = np.linalg.norm(st[m, 16:19], axis=1); nut = st[m, 9:12]; r = rw[m]; L = m.sum()
    fsep = st[m, 7] - st[m, 8]


    zb = np.median(nut[20:40, 2]); lifted = (nut[:, 2] - zb) > 0.030; lifted[:20] = False


    att = np.flatnonzero((n2e < A.near_mm/1000.0) & (fsep < 0.03))
    attempt = int(att[0]) if len(att) else -1
    near = n2e < A.near_mm/1000.0
    w = np.convolve(near.astype(float), np.ones(A.sustain)/A.sustain, "same")
    cand = np.flatnonzero(near & (w > 0.9))
    g = int(cand[0]) if len(cand) else -1
    if (r > 0).any():
        term = int(np.argmax(r > 0)); rewarded = True
    elif lifted.any():
        dd = np.linalg.norm(nut - REF, axis=1)
        if g >= 0: dd[:g] = np.inf
        term = int(np.argmin(dd)); rewarded = False
    elif attempt >= 0:


        w = np.arange(L) >= attempt
        dd = np.where(w, n2e, np.inf)
        term = int(np.argmin(dd)); rewarded = False
    else:

        term = int(L - 1); rewarded = False
    resid = float(np.linalg.norm(nut[term] - REF)*1000)
    start = 0 if A.keep_start else (g if g >= 0 else -1)
    if start < 0:
        segs.append(dict(source_episode_index=int(e), source_length=int(L), grasp_frame=g,
                         start_frame=-1, terminal_frame=-1, length=0, rewarded=False,
                         endpoint_residual_mm=float("nan"), kept=False, reason="no_grasp"))
        continue
    ln = term - start + 1
    if (r > 0).any():        kind = "rewarded"
    elif lifted.any():        kind = "lifted_then_failed"
    elif attempt >= 0:        kind = "failed_grasp"
    else:                     kind = "never_reached_nut"
    segs.append(dict(source_episode_index=int(e), source_length=int(L), grasp_frame=g,
                     grasp_attempt_frame=attempt, end_kind=kind,
                     start_frame=int(start), terminal_frame=term, length=int(ln),
                     rewarded=bool(rewarded), endpoint_residual_mm=resid,
                     kept=bool(ln >= A.min_frames),
                     reason="" if ln >= A.min_frames else "too_short"))
kept = [s for s in segs if s["kept"]]
print(f"  {len(segs)} episodes -> kept {len(kept)}, dropped {len(segs)-len(kept)}", flush=True)
print(f"  grasp frame  median {np.median([s['grasp_frame'] for s in kept]):.0f}", flush=True)
print(f"  clip length  median {np.median([s['length'] for s in kept]):.0f} "
      f"of {np.median([s['source_length'] for s in kept]):.0f}", flush=True)
print(f"  endpoint residual mm: median {np.median([s['endpoint_residual_mm'] for s in kept]):.1f} "
      f"p90 {np.percentile([s['endpoint_residual_mm'] for s in kept],90):.1f}", flush=True)


dst = Path(A.out); shutil.rmtree(dst, ignore_errors=True); (dst/"meta").mkdir(parents=True)
import imageio.v2 as imageio
def writer(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
        "-s","224x224","-r",str(FPS),"-i","-","-c:v","libsvtav1","-crf","30","-g","2","-preset","12",
        "-pix_fmt","yuv420p",str(p)], stdin=subprocess.PIPE)

rows_out, ep_rows, nwritten = [], [], {c: 0 for c in CAMS}
procs = {c: writer(dst/f"videos/{c}/chunk-000/file-000.mp4") for c in CAMS}
readers = {}
new_ep = 0; gidx = 0
for s in kept:
    e = s["source_episode_index"]; g, term = s["start_frame"], s["terminal_frame"]
    m = epi == e; rows = np.flatnonzero(m)[g:term+1]
    vt = {}
    for c in CAMS:
        ci, fi, t0 = vloc[e][c]
        key = (c, ci, fi)
        if key not in readers:
            readers[key] = imageio.get_reader(str(src/f"videos/{c}/chunk-{ci:03d}/file-{fi:03d}.mp4"))
        rd = readers[key]; base = int(round(t0*FPS))
        frm = nwritten[c]/FPS
        for k in range(g, term+1):
            procs[c].stdin.write(np.asarray(rd.get_data(base+k)).astype(np.uint8).tobytes())
        nwritten[c] += (term-g+1)
        vt[c] = (0, 0, frm, nwritten[c]/FPS)
    er = dict(episode_index=new_ep, length=int(term-g+1),
              **{f"videos/{c}/chunk_index": vt[c][0] for c in CAMS},
              **{f"videos/{c}/file_index": vt[c][1] for c in CAMS},
              **{f"videos/{c}/from_timestamp": vt[c][2] for c in CAMS},
              **{f"videos/{c}/to_timestamp": vt[c][3] for c in CAMS},
              dataset_from_index=gidx, dataset_to_index=gidx+int(term-g+1))
    ep_rows.append(er); rows_out.append(rows); gidx += int(term-g+1); new_ep += 1
    if new_ep % 20 == 0: print(f"    {new_ep}/{len(kept)}", flush=True)
for c in CAMS: procs[c].stdin.close(); procs[c].wait()
for rd in readers.values(): rd.close()

sel = np.concatenate(rows_out)
newd = {k: [v[i] for i in sel] for k, v in d.items()}
newd["episode_index"] = sum([[i]*len(r) for i, r in enumerate(rows_out)], [])
newd["frame_index"]   = sum([list(range(len(r))) for r in rows_out], [])
newd["index"]         = list(range(len(sel)))
newd["timestamp"]     = [float(f)/FPS for f in newd["frame_index"]]
(dst/"data/chunk-000").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pydict(newd), dst/"data/chunk-000/file-000.parquet")
(dst/"meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pydict({k: [r[k] for r in ep_rows] for k in ep_rows[0]}),
               dst/"meta/episodes/chunk-000/file-000.parquet")
info["total_episodes"] = len(kept); info["total_frames"] = int(len(sel))
info["splits"] = {"train": f"0:{len(kept)}"}
(dst/"meta/info.json").write_text(json.dumps(info, indent=4))
for f in ("meta/tasks.parquet","meta/stats.json"):
    if (src/f).exists(): shutil.copy2(src/f, dst/f)
(dst/"meta/segment_info.json").write_text(json.dumps(dict(
    source_repo_id=A.src,
    segment=dict(start="frame 0 -- the approach is kept in full; only the tail is cut",
                 grasp_detection=f"|eef - nut| < {A.near_mm:.0f} mm sustained over the next "
                       f"{A.sustain} frames (>90% under threshold); used ONLY to constrain the "
                       f"outcome search, not to cut the clip",
                 end_rewarded="first frame with reward > 0",
                 end_lifted_then_failed="argmin over frames at/after the grasp of |nut - ref|",
                 end_failed_grasp="argmin of |eef - nut| at/after the grasp ATTEMPT -- the moment "
                                  "the grasp failed. A grasp attempt is fingers closing within "
                                  f"{A.near_mm:.0f} mm of the nut, with NO sustain requirement, "
                                  "because a failed grasp does not stay near the nut.",
                 end_never_reached_nut="last frame -- nothing to cut at, episode kept whole",
                 endpoint_reference_mm=[float(x*1000) for x in REF],
                 endpoint_reference_source="mimicgen_square_d0_224x224_mtdit_flow_40k_success",
                 min_segment_frames=A.min_frames),
    episodes=[{**s, "episode_index": i} for i, s in enumerate(kept)] +
             [s for s in segs if not s["kept"]]), indent=2))
print(f"  SEG-OK {dst}  ({len(kept)} eps, {len(sel)} frames)", flush=True)
