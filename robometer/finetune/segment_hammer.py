"""Cut mimicgen hammer_cleanup FAILURE rollouts at the outcome, keeping the approach (step 1 of a LoRA fine-tune).

usage: python robometer/finetune/segment_hammer.py --src <failure rollout dataset: local dir or HF id> --out <dir>

hammer_cleanup is a THREE-stage task -- open the drawer, place the hammer, close the drawer -- so a
single endpoint rule does not fit. The cut depends on how far the attempt got:

  start                 frame 0 -- the approach is kept in full; only the tail is cut
  end_rewarded          first frame with reward > 0
  end_placed            argmin over frames at/after the hammer lift of |(hammer - drawer) - ref|
  end_failed_grasp      drawer opened but the hammer never left the table: argmin of |eef - hammer|
                        at/after the drawer opening -- the moment the grasp failed
  end_never_opened      last frame; the attempt never began, keep the episode whole
  min_segment_frames    20

ref = mean (hammer - drawer) at first reward over the success split. The RELATIVE form is required:
d1 randomises the drawer, so the hammer's world position at success has std [59.9, 38.7, 15.4] mm
while (hammer - drawer) has std [25.0, 24.2, 15.4] mm.

observation.state (37) = [0:9] robot | [9:23] hammer | [23:37] drawer, each object being
pos(3) + quat(4) + to_eef pos(3) + quat(4).
sim_state[17] is the drawer joint: 0 = closed, -0.134 = fully open.
Output: a LeRobot dataset of the cut clips (the input of make_rbm_from_seg.py). Successes are not segmented.
"""
import os, json, subprocess, shutil, argparse
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default="-0.0188,-0.0484,0.0676")
ap.add_argument("--near-mm", type=float, default=80.0)
ap.add_argument("--open-thresh", type=float, default=-0.05)
ap.add_argument("--lift-mm", type=float, default=30.0)
ap.add_argument("--min-frames", type=int, default=20)
A = ap.parse_args()
REF = np.array([float(x) for x in A.ref.split(",")])

from huggingface_hub import snapshot_download
src = Path(A.src) if Path(A.src).is_dir() else Path(snapshot_download(A.src, repo_type="dataset"))
info = json.loads((src/"meta/info.json").read_text()); FPS = info["fps"]
CAMS = [k for k in info["features"] if k.startswith("observation.images.")]
dfiles = sorted((src/"data").rglob("*.parquet"))
dt = pa.concat_tables([pq.read_table(f) for f in dfiles]); d = dt.to_pydict()
st = np.stack([np.asarray(x) for x in d["observation.state"]])
ss = np.stack([np.asarray(x) for x in d["observation.sim_state"]])
epi = np.asarray(d["episode_index"]); rw = np.asarray(d["reward"], dtype=float).ravel()
ept = pq.read_table(next((src/"meta/episodes").rglob("*.parquet"))).to_pydict()
vloc = {int(e): {c: (int(ept[f"videos/{c}/chunk_index"][i]), int(ept[f"videos/{c}/file_index"][i]),
                     float(ept[f"videos/{c}/from_timestamp"][i])) for c in CAMS}
        for i, e in enumerate(ept["episode_index"])}

segs = []
for e in np.unique(epi):
    m = epi == e; L = int(m.sum())
    dj = ss[m][:, 17]; ham = st[m][:, 9:12]; dr = st[m][:, 23:26]
    h2e = np.linalg.norm(st[m][:, 16:19], axis=1); r = rw[m]
    rel = np.linalg.norm((ham - dr) - REF, axis=1)
    zb = np.median(ham[20:40, 2]); lift = ham[:, 2] - zb; lift[:20] = 0
    opened = dj.min() < A.open_thresh
    open_f = int(np.argmax(dj < A.open_thresh)) if opened else -1
    lifted = lift > A.lift_mm/1000.0
    if (r > 0).any():
        term = int(np.argmax(r > 0)); kind = "rewarded"
    elif lifted.any():
        g = int(np.argmax(lifted)); dd = rel.copy(); dd[:g] = np.inf
        term = int(np.argmin(dd)); kind = "placed_or_attempted"
    elif opened:
        dd = h2e.copy(); dd[:open_f] = np.inf
        term = int(np.argmin(dd)); kind = "failed_grasp"
    else:
        term = L - 1; kind = "never_opened_drawer"
    ln = term + 1
    segs.append(dict(source_episode_index=int(e), source_length=L, start_frame=0,
                     terminal_frame=int(term), length=int(ln), end_kind=kind,
                     rewarded=bool((r > 0).any()), drawer_opened=bool(opened),
                     open_frame=open_f, hammer_lifted=bool(lifted.any()),
                     max_lift_mm=float(lift.max()*1000), max_drawer_open=float(dj.min()),
                     final_drawer=float(dj[-1]),
                     endpoint_residual_mm=float(rel[term]*1000),
                     kept=bool(ln >= A.min_frames), reason="" if ln >= A.min_frames else "too_short"))
kept = [s for s in segs if s["kept"]]
import collections
print(f"  {len(segs)} episodes -> kept {len(kept)}", flush=True)
for k, v in collections.Counter(s["end_kind"] for s in kept).most_common():
    sub = [s for s in kept if s["end_kind"] == k]
    print(f"    {k:<22}{v:>4}  median len {np.median([s['length'] for s in sub]):>5.0f}"
          f" of {np.median([s['source_length'] for s in sub]):.0f}"
          f"  resid {np.median([s['endpoint_residual_mm'] for s in sub]):>6.0f}mm", flush=True)

dst = Path(A.out); shutil.rmtree(dst, ignore_errors=True); (dst/"meta").mkdir(parents=True)
import imageio.v2 as imageio
def writer(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
        "-s","224x224","-r",str(FPS),"-i","-","-c:v","libsvtav1","-crf","30","-g","2","-preset","12",
        "-pix_fmt","yuv420p",str(p)], stdin=subprocess.PIPE)
rows_out, ep_rows, nw = [], [], {c: 0 for c in CAMS}
procs = {c: writer(dst/f"videos/{c}/chunk-000/file-000.mp4") for c in CAMS}
readers = {}; gidx = 0
for i_new, s in enumerate(kept):
    e = s["source_episode_index"]; term = s["terminal_frame"]
    rows = np.flatnonzero(epi == e)[:term+1]
    vt = {}
    for c in CAMS:
        ci, fi, t0 = vloc[e][c]; key = (c, ci, fi)
        if key not in readers:
            readers[key] = imageio.get_reader(str(src/f"videos/{c}/chunk-{ci:03d}/file-{fi:03d}.mp4"))
        rd = readers[key]; base = int(round(t0*FPS)); frm = nw[c]/FPS
        for k in range(0, term+1):
            procs[c].stdin.write(np.asarray(rd.get_data(base+k)).astype(np.uint8).tobytes())
        nw[c] += term+1; vt[c] = (0, 0, frm, nw[c]/FPS)
    ep_rows.append(dict(episode_index=i_new, length=int(term+1),
        **{f"videos/{c}/chunk_index": 0 for c in CAMS},
        **{f"videos/{c}/file_index": 0 for c in CAMS},
        **{f"videos/{c}/from_timestamp": vt[c][2] for c in CAMS},
        **{f"videos/{c}/to_timestamp": vt[c][3] for c in CAMS},
        dataset_from_index=gidx, dataset_to_index=gidx+term+1))
    rows_out.append(rows); gidx += term+1
    if (i_new+1) % 20 == 0: print(f"    {i_new+1}/{len(kept)}", flush=True)
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
(dst/"meta/segment_info.json").write_text(json.dumps(dict(source_repo_id=A.src, segment=dict(
    start="frame 0 -- the approach is kept in full; only the tail is cut",
    end_rewarded="first frame with reward > 0",
    end_placed_or_attempted="argmin over frames at/after the hammer lift of |(hammer - drawer) - ref|",
    end_failed_grasp="drawer opened but hammer never lifted: argmin of |eef - hammer| at/after the "
                     "drawer opening -- the moment the grasp failed",
    end_never_opened_drawer="last frame -- the attempt never began, episode kept whole",
    endpoint_reference_m=REF.tolist(),
    endpoint_reference_note="RELATIVE (hammer - drawer). d1 randomises the drawer, so the hammer's "
                            "WORLD position at success has std [59.9,38.7,15.4] mm while the "
                            "relative form has [25.0,24.2,15.4] mm.",
    endpoint_reference_source="mimicgen_hammer_cleanup_d1_224x224_mtdit_flow_35k_success",
    drawer_joint="sim_state[17]: 0 = closed, -0.134 = fully open",
    open_threshold=A.open_thresh, lift_threshold_mm=A.lift_mm,
    min_segment_frames=A.min_frames), episodes=segs), indent=2))
print(f"  SEG-OK {dst}  ({len(kept)} eps, {len(sel)} frames)", flush=True)
