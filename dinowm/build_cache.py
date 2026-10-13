"""Decode every frame of a task's prior rollouts (D_off) ONCE into the uint8 frame cache read by common_ws.

usage: WS_TASK=<task> python dinowm/build_cache.py [--task <task>] [--flow <rollout dataset>]

Reads  $DATA_ROOT/<flow>_success and <flow>_failure (LeRobot v3: data/*.parquet + videos/observation.images.<cam>/*.mp4;
       <flow> defaults to the task's campaign rollout set, common_ws.ROLLOUTS)
Writes $CACHE_ROOT/wm_frames/<task>_frames_u8.npy   (N, 2, 224, 224, 3) uint8, cameras (agentview, eye_in_hand),
                                                    rows = success then failure, sorted by (episode, frame); 17-24 GB
       $CACHE_ROOT/wm_frames/<task>_meta.npz        per row: split, ep, fr, T, state, action, reward, success
       $CACHE_ROOT/wm_frames/<task>_cache_checks.json

The frames are decoded with imageio/ffmpeg and checked twice before the cache is accepted: 24 random rows are read back
through common_ws.FrameReader, and 8 held-out rows are compared with LeRobotDataset's own decode (must be bit-identical,
max |diff| 0). The file is written as <task>_frames_u8.partial and renamed at the end, so an interrupted build leaves no
cache behind. This is the input of the world-model training, the RoboMeter scoring and the V fit.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import common_ws as C

DS = C.DATA
BUF = 256


def load_table(name):
    """parquet rows sorted by (episode, frame) == the global row order the videos are packed in."""
    import pyarrow.dataset as pads
    t = (pads.dataset(DS / name / "data", format="parquet")
         .to_table(columns=["index", "episode_index", "frame_index", "observation.state", "action",
                            "reward", "success"])
         .sort_by([("episode_index", "ascending"), ("frame_index", "ascending")]))
    return dict(
        index=np.asarray(t["index"].combine_chunks(), np.int64),
        ep=np.asarray(t["episode_index"].combine_chunks(), np.int64),
        fr=np.asarray(t["frame_index"].combine_chunks(), np.int64),
        state=np.asarray(t["observation.state"].combine_chunks().to_pylist(), np.float32),
        action=np.asarray(t["action"].combine_chunks().to_pylist(), np.float32),
        reward=np.asarray(t["reward"].combine_chunks().to_pylist(), np.float32),
        success=np.asarray(t["success"].combine_chunks().to_pylist(), bool),
    )


def frame_stream(name, cam):
    """Every frame of one camera in global row order."""
    import imageio.v2 as imageio
    files = sorted((DS / name / "videos" / f"observation.images.{cam}").rglob("*.mp4"))
    assert files, (name, cam)
    for path in files:
        rdr = imageio.get_reader(path, "ffmpeg")
        try:
            yield from rdr
        finally:
            rdr.close()


def npy_header(n):
    import io
    b = io.BytesIO()
    np.lib.format.write_array_header_1_0(b, {"descr": "|u1", "fortran_order": False, "shape": (n, *C.SHAPE1)})
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, choices=C.TASKS, help="must equal WS_TASK when given")
    ap.add_argument("--flow", default="", help="rollout dataset name without the _success/_failure suffix "
                                               "(default: the task's campaign rollout set)")
    args = ap.parse_args()
    task = C._need_task(args.task)
    flow = args.flow or C.ROLLOUTS[task]
    if f"mimicgen_{task}_" not in flow:
        raise SystemExit(f"--flow {flow} is not a {task} rollout set (WS_TASK={task!r} names the cache it would fill)")
    splits = {"success": f"{flow}_success", "failure": f"{flow}_failure"}
    missing = [str(DS / name) for name in splits.values() if not (DS / name / "data").is_dir()]
    if missing:
        raise SystemExit(f"rollout dataset(s) not found: {missing} (DATA_ROOT={DS})")
    print(f"task {task}: {flow}_{{success,failure}} -> {C.FRAMES}", flush=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tabs = {s: load_table(name) for s, name in splits.items()}
    n_tot = sum(len(t["ep"]) for t in tabs.values())
    hdr = npy_header(n_tot)
    print(f"{n_tot} rows, header {len(hdr)} B, file {(len(hdr) + n_tot * C.FRAME_BYTES) / 1e9:.2f} GB", flush=True)
    cols = {k: [] for k in ("split", "ep", "fr", "T", "state", "action", "reward", "success")}
    keep = {}
    rng = np.random.default_rng(0)
    probe_rows = set(rng.choice(n_tot, 24, replace=False).tolist())
    tmp = C.FRAMES.with_suffix(".partial")
    written = 0
    with open(tmp, "wb") as fh:
        fh.write(hdr)
        for split, name in splits.items():
            T = tabs[split]
            n = len(T["ep"])
            assert np.array_equal(T["index"], np.arange(T["index"][0], T["index"][0] + n)), "parquet index not contiguous"
            Tlen = np.zeros(n, np.int64)
            for e in np.unique(T["ep"]):
                m = T["ep"] == e
                assert np.array_equal(T["fr"][m], np.arange(m.sum())), "episode frames not contiguous"
                Tlen[m] = m.sum()
            buf, done = [], 0
            for frames in zip(*(frame_stream(name, c) for c in C.CAMS), strict=True):
                im = np.stack(frames)
                assert im.shape == C.SHAPE1 and im.dtype == np.uint8, im.shape
                if written + done in probe_rows:
                    keep[written + done] = im.copy()
                buf.append(im); done += 1
                if len(buf) == BUF:
                    fh.write(np.stack(buf).tobytes()); buf = []
            if buf:
                fh.write(np.stack(buf).tobytes())
            assert done == n, f"{split}: decoded {done} frames, parquet has {n}"
            written += n
            for c, v in (("split", np.full(n, split)), ("ep", T["ep"]), ("fr", T["fr"]), ("T", Tlen),
                         ("state", T["state"]), ("action", T["action"]), ("reward", T["reward"]),
                         ("success", T["success"])):
                cols[c].append(v)
            print(f"  {split}: {n} frames written [{time.time() - t0:.0f}s]", flush=True)
        fh.flush(); os.fsync(fh.fileno())
    assert tmp.stat().st_size == len(hdr) + n_tot * C.FRAME_BYTES
    os.replace(tmp, C.FRAMES)
    F = {k: np.concatenate(v) for k, v in cols.items()}
    F["split"] = F["split"].astype("U7")
    np.savez(C.META, header_bytes=np.int64(len(hdr)), frame_bytes=np.int64(C.FRAME_BYTES), **F)

    rd = C.FrameReader(C.FRAMES)
    for r, im in keep.items():
        assert np.array_equal(rd.read([r])[0], im), f"read-back mismatch at row {r}"
    checks = {"n_rows": int(n_tot), "readback_rows": len(keep)}


    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    par = []
    for split, name in splits.items():
        dsl = LeRobotDataset(repo_id=f"local/{name}", root=DS / name)
        sel = np.flatnonzero((F["split"] == split) & (F["ep"] >= C.TRAIN_EPISODES))
        for g in rng.choice(sel, 4, replace=False):
            local = int(g - np.flatnonzero(F["split"] == split)[0])
            it = dsl[local]
            assert int(it["episode_index"]) == F["ep"][g] and int(it["frame_index"]) == F["fr"][g]
            ref = np.stack([(it[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                            for c in C.CAMS])
            d = np.abs(rd.read([int(g)])[0].astype(int) - ref.astype(int))
            par.append(dict(row=int(g), split=split, ep=int(F["ep"][g]), fr=int(F["fr"][g]),
                            absdiff_mean=float(d.mean()), absdiff_max=int(d.max())))
            print(f"  parity {split}/{F['ep'][g]}/{F['fr'][g]}: |d| mean {d.mean():.4f} max {d.max()}", flush=True)
    rd.close()
    assert max(p["absdiff_max"] for p in par) == 0, "cache decode differs from LeRobotDataset decode"
    checks["lerobot_parity"] = par
    (C.CACHE / f"{C.WS_TASK}_cache_checks.json").write_text(json.dumps(checks, indent=1))
    print(f"wrote {C.FRAMES} ({C.FRAMES.stat().st_size / 1e9:.2f} GB) + {C.META}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
