"""Score a task's prior rollouts with a RoboMeter teacher: the targets of the offline V fit (value/fit_v_mqattn.py).

usage (main environment; the teacher must be serving --spool, see robometer/serve.sh):
  WS_TASK=<task> python robometer/score_rollouts.py --spool $SPOOL_ROOT/<teacher> \\
      --out $CKPT_ROOT/rm_scores/<task>/predictions.parquet                              first scoring of a task
  WS_TASK=<task> python robometer/score_rollouts.py --spool $SPOOL_ROOT/<other teacher> \\
      --rows $CKPT_ROOT/rm_scores/<task>/predictions.parquet \\
      --out  $CKPT_ROOT/rm_scores/<task>_seg/predictions.parquet                         the SAME rows, another teacher

ROWS. 64 frames per rollout episode, evenly spaced from the first to the last frame (np.linspace, both endpoints),
for all 100 success + 100 failure episodes of D_off: 12,800 rows (default_rows). --rows takes the row set of an
existing table instead.
PROTOCOL. Each episode's rows are sent as ONE clip (agentview camera, oldest frame first, uint8 224x224) -- exactly
what the online distiller sends -- and the reply is progress (and success probability) per frame. Frames come from the
frame cache (dinowm/build_cache.py), i.e. the frames the V fitter pairs with these targets.
OUTPUT. predictions.parquet with columns split, episode_index, frame_index, pred_progress, pred_success. The ROW ORDER
is part of the recipe, because the V fitter samples rows by position: a first scoring writes success episodes first
(the order default_rows lists them); a rescoring with --rows writes the groups sorted, failure episodes first. That
is the order of the tables behind the released V directories (pv4 and MC-return: first scoring; pv4_seg: rescoring).
With --rows the correlation between the new and the input progress is printed: 1.0000 when the server runs the LoRA
that produced the input (a protocol check), lower when the teacher changed (square_d1 segmented vs not: 0.88).

--limit-episodes N scores only the first N episodes per split (a smoke test).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_DINOWM = Path(__file__).resolve().parents[1] / "dinowm"
if str(_DINOWM) not in sys.path[:1]:
    sys.path.insert(0, str(_DINOWM))           # this repository's common_ws, whatever PYTHONPATH says

import common_ws as C  # noqa: E402

K = 64
AGENTVIEW = C.CAMS.index("agentview")
KEYS = ["split", "episode_index", "frame_index"]


def default_rows(M) -> pd.DataFrame:
    """The scored frames of a first scoring: per episode K frames from first to last, success episodes first."""
    rows = []
    for split in ("success", "failure"):
        m = M["split"] == split
        for ep in np.unique(M["ep"][m]):
            fr = np.sort(M["fr"][m & (M["ep"] == ep)])
            pick = fr[np.linspace(0, len(fr) - 1, min(K, len(fr))).round().astype(int)]
            rows += [(split, int(ep), int(f)) for f in pick]
    return pd.DataFrame(rows, columns=KEYS)


def episode_groups(df: pd.DataFrame, rescoring: bool) -> list:
    """(split, episode) groups in the order they are scored and written (see ROW ORDER in the module docstring)."""
    return list(df.groupby(["split", "episode_index"], sort=rescoring))


def score(spool: Path, clip: np.ndarray, rid: str, timeout: float) -> np.ndarray:
    tmp = spool / "req" / f"{rid}.tmp"
    with open(tmp, "wb") as fh:                                           # np.save would append .npy to "*.tmp"
        np.save(fh, clip)
    tmp.rename(spool / "req" / f"{rid}.npy")
    res, err, ts = spool / "res" / f"{rid}.npy", spool / "res" / f"{rid}.err", time.time()
    while True:
        if res.exists():
            o = np.load(res); res.unlink(missing_ok=True); return o
        if err.exists():
            raise RuntimeError(f"server error on {rid}: {err.read_text()[-800:]}")
        if time.time() - ts > timeout:
            raise TimeoutError(f"no reply for {rid} in {timeout}s (is robometer_server.py running on {spool}?)")
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default=None, choices=C.TASKS, help="must equal WS_TASK when given")
    ap.add_argument("--spool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=Path, default=None, help="re-score the rows of this predictions.parquet")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit-episodes", type=int, default=0, help="per split; 0 = all")
    a = ap.parse_args()
    C._need_task(a.task)
    if not (a.spool / "READY").exists():
        raise SystemExit(f"{a.spool} has no READY file: start the teacher first (robometer/serve.sh)")

    M = C.load_meta()
    df = pd.read_parquet(a.rows) if a.rows else default_rows(M)
    key = {(str(s), int(e), int(f)): i for i, (s, e, f) in enumerate(zip(M["split"], M["ep"], M["fr"]))}
    rd = C.FrameReader()
    (a.spool / "req").mkdir(parents=True, exist_ok=True)
    (a.spool / "res").mkdir(parents=True, exist_ok=True)
    groups = episode_groups(df, rescoring=a.rows is not None)
    if a.limit_episodes:
        seen = {}
        groups = [(k, g) for k, g in groups if seen.setdefault(k[0], 0) < a.limit_episodes
                  and not seen.__setitem__(k[0], seen[k[0]] + 1)]
    out_rows, t0 = [], time.time()
    for n_ep, ((split, ep), g) in enumerate(groups, 1):
        g = g.sort_values("frame_index")
        rows = np.array([key[(str(split), int(ep), int(f))] for f in g.frame_index])
        clip = np.ascontiguousarray(rd.read(rows)[:, AGENTVIEW])            # (T, 224, 224, 3) uint8
        o = score(a.spool, clip, f"score_{os.getpid()}_{split}_{int(ep):04d}", a.timeout)
        o = np.asarray(o, np.float64).reshape(-1, len(g))
        succ = o[1] if o.shape[0] > 1 else np.full(len(g), np.nan)
        out_rows += [(str(split), int(ep), int(f), float(p), float(s)) for f, p, s in zip(g.frame_index, o[0], succ)]
        if n_ep % 20 == 0:
            print(f"[score] {n_ep}/{len(groups)} episodes [{time.time() - t0:.0f}s]", flush=True)
    rd.close()
    new = pd.DataFrame(out_rows, columns=KEYS + ["pred_progress", "pred_success"])
    if a.rows is not None and "pred_progress" in df.columns:
        m = new.merge(df, on=KEYS, suffixes=("", "_old"))
        print(f"[score] corr(new, input) progress {np.corrcoef(m.pred_progress, m.pred_progress_old)[0, 1]:.4f}; "
              f"max |new - input| {float((m.pred_progress - m.pred_progress_old).abs().max()):.4f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(a.out)
    print(f"[score] wrote {a.out}: {len(new)} rows; mean progress "
          + ", ".join(f"{s} {new[new.split == s].pred_progress.mean():.3f}" for s in ("success", "failure")), flush=True)


if __name__ == "__main__":
    main()
