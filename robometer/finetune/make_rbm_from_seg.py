"""Convert LeRobot datasets (segmented failures, successes, demos) into RBM format for RoboMeter fine-tuning (step 2).

usage: python robometer/finetune/make_rbm_from_seg.py --out <dir> --specs <dataset>,<dataset>,... \\
           --task "<training sentence>" --name-prefix mimicgen_<task> --lang-json <json with {task, lang_vector}>

Writes <out>/{train,test}/<name-prefix>_{train,test}_seg_rfm/: a parquet of {id, task, lang_vector, data_source,
frames, is_robot, quality_label, partial_success} plus one mp4 per trajectory (64 frames, linspace-subsampled, 224x224,
agentview). A dataset whose path contains "failure" is labelled failure, every other one successful. --test-frac 0.2
with seed 0: the released LoRAs were trained on 160 trajectories (80 failures + 80 successes, the successes half prior
rollouts and half expert demos) and evaluated on 40.
`lang_vector` is the 384-d all-MiniLM-L6-v2 embedding of the task sentence, identical for every row: pass it with
--lang-json, or lift it from an existing RBM dataset of the SAME sentence with --lang-ref.
Next steps (robometer/finetune/NOTES.md): push both sets to the Hub, register the success cutoff, train_lora.sh.
"""
import os, json, argparse, subprocess
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n-frames", type=int, default=64)
ap.add_argument("--test-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--specs", default="")
ap.add_argument("--task", required=True, help="the task sentence (robometer/teachers.sh TEACHER_TEXT)")
ap.add_argument("--lang-ref", default="",
                help="RBM dataset to lift lang_vector from -- MUST embed the same task string")
ap.add_argument("--lang-json", default="",
                help="instead of lifting: a json with {task, lang_vector}. Verified to reproduce the "
                     "existing datasets exactly with all-MiniLM-L6-v2 (cos 1.000000).")
ap.add_argument("--name-prefix", required=True, help="dataset name prefix, e.g. mimicgen_square_d0")
ap.add_argument("--id-prefix", default="",
                help="prefix of the per-row id (cosmetic; default: --name-prefix without 'mimicgen_')")
ap.add_argument("--max-per-spec", default="",
                help="comma list of caps aligned with --specs; blank/0 = use all")
ap.add_argument("--cam", default="observation.images.agentview",
                help="video feature to convert")
A = ap.parse_args()
A.id_prefix = A.id_prefix or A.name_prefix.removeprefix("mimicgen_")
assert A.lang_json or A.lang_ref, "pass --lang-json or --lang-ref (the embedding of the task sentence)"
TASK = None
CAM  = A.cam

from huggingface_hub import hf_hub_download, snapshot_download
if A.lang_json:
    _j = json.loads(Path(A.lang_json).read_text())
    LANG = [float(x) for x in _j["lang_vector"]]; TASK = A.task or _j["task"]
    assert len(LANG) == 384 and _j["task"] == TASK, "lang_json task must match --task"
    print(f"task: {TASK!r} (lang_vector from {A.lang_json})", flush=True)
else:
  from huggingface_hub import HfApi
  _rf = [f for f in (s_.rfilename for s_ in HfApi().dataset_info(A.lang_ref).siblings)
         if f.endswith(".parquet")][0]
  ref = pq.read_table(hf_hub_download(A.lang_ref, _rf, repo_type="dataset")).to_pydict()
  LANG = [float(x) for x in ref["lang_vector"][0]]
  TASK = A.task if A.task else ref["task"][0]
  assert len(LANG) == 384


  assert ref["task"][0] == TASK, f"lang_ref task {ref['task'][0]!r} != --task {TASK!r}"
  print(f"task: {TASK!r}", flush=True)
print(f"lang_vector lifted from reference: {len(LANG)}-d", flush=True)

SPECS = [s for s in A.specs.split(",") if s]
_caps = [c.strip() for c in A.max_per_spec.split(",")] if A.max_per_spec else []
CAPS = {sp: (int(_caps[i]) if i < len(_caps) and _caps[i] and int(_caps[i]) > 0 else 0)
        for i, sp in enumerate(SPECS)}
rows_by_split = {"train": [], "test": []}
rng = np.random.default_rng(A.seed)
out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
counters = {"train": 0, "test": 0}

import pyarrow.parquet as _pq
for spec in SPECS:
    label = "failure" if "failure" in spec else "successful"
    ps    = 0.0 if label == "failure" else 1.0


    local = Path(spec) if Path(spec).is_dir() else Path(snapshot_download(spec, repo_type="dataset"))
    info  = json.loads((local/"meta/info.json").read_text())
    ep_t  = _pq.read_table(sorted((local/"meta/episodes").rglob("*.parquet"))[0]).to_pydict()
    n_ep  = info["total_episodes"]

    vids  = {int(e): (int(ep_t[f"videos/{CAM}/chunk_index"][i]), int(ep_t[f"videos/{CAM}/file_index"][i]),
                      float(ep_t[f"videos/{CAM}/from_timestamp"][i]), float(ep_t[f"videos/{CAM}/to_timestamp"][i]))
             for i, e in enumerate(ep_t["episode_index"])}
    order = rng.permutation(n_ep)
    if CAPS.get(spec):
        order = order[:CAPS[spec]]
        n_ep_used = len(order)
    else:
        n_ep_used = n_ep
    n_test = int(round(n_ep_used * A.test_frac))
    split_of = {int(e): ("test" if k < n_test else "train") for k, e in enumerate(order)}
    print(f"{spec}: {n_ep} eps, using {n_ep_used} -> {n_test} test / {n_ep_used-n_test} train  [{label}]",
          flush=True)

    for e in sorted(split_of):
        sp = split_of[e]; ci, fi, t0, t1 = vids[e]
        src = local/f"videos/{CAM}/chunk-{ci:03d}/file-{fi:03d}.mp4"
        idx = counters[sp]; counters[sp] += 1
        name = f"{A.name_prefix}_{sp}_seg_rfm"
        rel  = f"{name}/batch_{idx//1000:04d}/trajectory_{idx:04d}.mp4"
        dst  = out/sp/rel; dst.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(["ffmpeg","-y","-loglevel","error","-ss",f"{t0}","-to",f"{t1}","-i",str(src),
                        "-vf",f"scale=224:224,fps={A.n_frames}/{max(t1-t0,1e-3)}",
                        "-frames:v",str(A.n_frames),"-c:v","libx264","-pix_fmt","yuv420p",str(dst)],
                       check=True)
        rows_by_split[sp].append(dict(
            id=f"{A.id_prefix}_{label}_{Path(spec).name.split('_')[-2]}_{e}", task=TASK, lang_vector=LANG,
            data_source=name, frames=rel, is_robot=True,
            quality_label=label, partial_success=ps))

schema = pa.schema([("id",pa.string()),("task",pa.string()),
                    ("lang_vector",pa.list_(pa.float32())),("data_source",pa.string()),
                    ("frames",pa.string()),("is_robot",pa.bool_()),
                    ("quality_label",pa.string()),("partial_success",pa.float32())])
for sp, rows in rows_by_split.items():
    if not rows: continue
    name = f"{A.name_prefix}_{sp}_seg_rfm"
    d = {k: [r[k] for r in rows] for k in rows[0]}
    p = out/sp/name/f"{'train' if sp=='train' else 'test'}-00000-of-00001.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(d, schema=schema), p)
    import collections
    print(f"  {sp}: {len(rows)} trajectories -> {p}", flush=True)
    print(f"      labels {dict(collections.Counter(d['quality_label']))}", flush=True)
print("RBM-OK", out, flush=True)
