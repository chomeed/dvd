"""CPU test of the data rules of the offline pipeline (dinowm/common_ws.py, robometer/score_rollouts.py,
value/fit_v_mqattn.py): the two-frame context windows, the scored rows and their order, the MC-return target, and the
state scaler written into a V directory.

Synthetic checks always run. With the released assets under the scripts/env.sh roots the same rules are checked against
them: the frame-cache metadata ($CACHE_ROOT/wm_frames), the score tables ($CKPT_ROOT/rm_scores) and the V directories
($CKPT_ROOT/value) -- otherwise those tests print SKIP.
Run: python tests/pipeline/test_offline_data.py
"""
import importlib
import json
import os

import _pipeline_util as U

U.use_repo("dinowm", "robometer", "value")
os.environ.setdefault("WS_TASK", "square_d0")       # read once at import; the functions under test take M explicitly

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common_ws as C  # noqa: E402
import score_rollouts as S  # noqa: E402

KEYS = ["split", "episode_index", "frame_index"]


def _meta(lengths=(35, 8, 20, 120), n_success=2):
    """A tiny cache: episodes of the given lengths, the first n_success in the success split (reward on their last
    frame), the rest failures; rows = success then failure, sorted by (episode, frame), as build_cache writes them."""
    rows = []
    for i, n in enumerate(lengths):
        split = "success" if i < n_success else "failure"
        ep = i if split == "success" else i - n_success
        rows += [(split, ep, f, n, float(split == "success" and f == n - 1)) for f in range(n)]
    split, ep, fr, T, rew = zip(*rows)
    rng = np.random.default_rng(0)
    return dict(split=np.array(split).astype("U7"), ep=np.array(ep), fr=np.array(fr), T=np.array(T),
                reward=np.array(rew, np.float32), action=rng.standard_normal((len(rows), 7)).astype(np.float32),
                state=rng.standard_normal((len(rows), 12)).astype(np.float32))


def _check_windows(M, label):
    n_win = {}
    for which in ("train", "heldout"):
        W = C.windows(M, which)
        cur, prev, nxt = W["cur"], W["prev"], W["nxt"]
        fr = M["fr"][cur]
        assert np.all((fr == 0) | (fr >= C.H)), "a window starts at a frame with no real t-8 (frames 1..7)"
        assert np.array_equal(prev, np.where(fr >= C.H, cur - C.H, cur)), "context is not [t-8, t] / [0, 0]"
        assert np.array_equal(nxt, cur + C.H)
        for other in (prev, nxt):
            assert np.all(M["ep"][other] == M["ep"][cur]) and np.all(M["split"][other] == M["split"][cur])
        assert np.all(M["fr"][nxt] == fr + C.H) and np.all(M["fr"][prev] == np.where(fr >= C.H, fr - C.H, 0))
        want_ep = M["ep"][cur] < C.TRAIN_EPISODES if which == "train" else M["ep"][cur] >= C.TRAIN_EPISODES
        assert np.all(want_ep)
        # completeness: every admissible frame of every episode on this side of the split is a window
        side = (M["ep"] < C.TRAIN_EPISODES) if which == "train" else (M["ep"] >= C.TRAIN_EPISODES)
        admissible = side & ((M["fr"] == 0) | (M["fr"] >= C.H)) & (M["fr"] + C.H <= M["T"] - 1)
        assert np.array_equal(np.flatnonzero(admissible), cur), "windows() dropped or added frames"
        if len(cur):
            a = C.chunk(M, cur[:5])
            assert a.shape == (min(5, len(cur)), C.H * 7)
            assert np.array_equal(a[0], M["action"][cur[0]:cur[0] + C.H].reshape(-1)), "chunk is not actions t..t+7"
        n_win[which] = len(cur)
    print(f"{label}: {n_win['train']} train + {n_win['heldout']} held-out windows (context [t-8, t] -> t+8; t = 0 uses "
          f"frame 0 twice; frames 1..7 dropped; train = episodes < {C.TRAIN_EPISODES})")
    return n_win


def test_windows_synthetic():
    assert (C.H, C.TRAIN_EPISODES, C.CAMS, C.SHAPE1) == (8, 80, ("agentview", "eye_in_hand"), (2, 224, 224, 3))
    M = _meta()
    n = _check_windows(M, "synthetic")
    # per episode of length L: frame 0 (if L > 8) plus frames 8 .. L-9
    assert n["train"] == sum((1 if L > 8 else 0) + max(0, L - 16) for L in (35, 8, 20, 120)) and n["heldout"] == 0
    M2 = dict(M, ep=M["ep"] + 79)                     # move the episodes across the train / held-out boundary
    n2 = C.windows(M2, "train")["cur"].size, C.windows(M2, "heldout")["cur"].size
    assert n2[0] > 0 and n2[1] > 0 and sum(n2) == n["train"]


def test_scored_rows_synthetic():
    M = _meta()
    df = S.default_rows(M)
    assert list(df.columns) == KEYS and not df.duplicated(KEYS).any()
    for (split, ep), g in df.groupby(["split", "episode_index"]):
        length = int(M["T"][(M["split"] == split) & (M["ep"] == ep)][0])
        fr = g.frame_index.to_numpy()
        assert len(fr) == min(S.K, length) and fr[0] == 0 and fr[-1] == length - 1 and np.all(np.diff(fr) > 0)
    assert list(df.split.unique()) == ["success", "failure"], "a first scoring lists success episodes first"
    first = [k for k, _ in S.episode_groups(df, rescoring=False)]
    again = [k for k, _ in S.episode_groups(df, rescoring=True)]
    assert first == [("success", 0), ("success", 1), ("failure", 0), ("failure", 1)], first
    assert again == sorted(first) and again[0][0] == "failure", "a rescoring writes the groups sorted (failure first)"
    print(f"scored rows: {len(df)} = 64 per episode (short episodes: all frames), first and last frame included; "
          f"first scoring order {first[0]}..{first[-1]}, rescoring order {again[0]}..{again[-1]}")


def test_mc_targets_synthetic():
    os.environ["WS_TASK"] = C.WS_TASK
    F = importlib.import_module("fit_v_mqattn")
    M = _meta()
    df = S.default_rows(M)
    y = F.mc_targets(M, df)
    assert y.dtype == np.float32 and F.GAMMA == 0.99
    for (split, ep, f), v in zip(df.itertuples(index=False), y):
        if split == "failure":
            assert v == 0.0
        else:
            t_r = int(M["T"][(M["split"] == split) & (M["ep"] == ep)][0]) - 1
            assert abs(v - 0.99 ** (t_r - f)) < 1e-6, (split, ep, f, v)
    assert set(F.PREFIX) == {"pred_progress", "mc"} and F.PREFIX["mc"] == "v_dino_mc_return"
    print(f"MC target: gamma^(t_r - t) with gamma 0.99 on success rows (max {y.max():.3f} at the rewarded frame, min "
          f"{y[df.split == 'success'].min():.3f}), exactly 0 on the {int((df.split == 'failure').sum())} failure rows")


def _real_meta(task):
    p = U.root("CACHE_ROOT") / "wm_frames" / f"{task}_meta.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def test_windows_released_caches():
    done = 0
    for task in U.TASKS:
        M = _real_meta(task)
        if M is None:
            continue
        n_ep = {s: len(np.unique(M["ep"][M["split"] == s])) for s in ("success", "failure")}
        assert n_ep == {"success": 100, "failure": 100}, (task, n_ep)
        first_fail = int(np.flatnonzero(M["split"] == "failure")[0])
        assert np.all(M["split"][:first_fail] == "success") and np.all(M["split"][first_fail:] == "failure")
        n = _check_windows(M, f"{task} ({len(M['ep'])} frames)")
        assert n["train"] > 3 * n["heldout"] > 0
        done += 1
    if not done:
        U.skip("no frame-cache metadata below $CACHE_ROOT/wm_frames")


def test_cache_metadata_matches_datasets():
    """build_cache.load_table on the rollout datasets reproduces the metadata of the released frame cache: the same
    rows in the same order (success then failure, by episode and frame) with the same states, actions and rewards."""
    B = importlib.import_module("build_cache")
    done = 0
    for task in U.TASKS:
        M, flow = _real_meta(task), C.ROLLOUTS[task]
        if M is None or not all((C.DATA / f"{flow}_{s}" / "data").is_dir() for s in ("success", "failure")):
            continue
        tabs = [B.load_table(f"{flow}_{s}") for s in ("success", "failure")]
        assert sum(len(t["ep"]) for t in tabs) == len(M["ep"]), task
        for k_meta, k_tab in (("ep", "ep"), ("fr", "fr"), ("state", "state"), ("action", "action"), ("reward", "reward")):
            assert np.array_equal(M[k_meta], np.concatenate([t[k_tab] for t in tabs])), (task, k_meta)
        print(f"{task}: {flow}_{{success,failure}} -> {len(M['ep'])} cache rows, metadata identical; state width "
              f"{M['state'].shape[1]}, {int((M['reward'] > 0).sum())} rewarded frames")
        done += 1
        if done == 2:                    # two tasks are enough for a unit test (each reads ~60k parquet rows)
            break
    if not done:
        U.skip("no frame-cache metadata with its rollout datasets ($CACHE_ROOT/wm_frames, $DATA_ROOT)")


def test_score_tables_released():
    """The released score tables hold exactly default_rows, in the order score_rollouts writes them."""
    done = 0
    tables = [(t, t, False) for t in U.TASKS] + [("square_d1", "square_d1_seg", True), ("threading_d0", "threading_d0_seg", True)]
    for task, name, rescoring in tables:
        p, M = U.root("CKPT_ROOT") / "rm_scores" / name / "predictions.parquet", _real_meta(task)
        if M is None or not p.exists():
            continue
        df = pd.read_parquet(p)
        want = S.default_rows(M)
        if rescoring:
            want = pd.concat([g.sort_values("frame_index") for _, g in S.episode_groups(want, rescoring=True)])
        got = df[KEYS].astype({"episode_index": int, "frame_index": int})
        assert len(df) == 12800 and np.array_equal(got.to_numpy(), want.to_numpy()), f"{name}: row set or order differs"
        prog = df.pred_progress.to_numpy()
        print(f"rm_scores/{name}: 12800 rows in {'rescoring' if rescoring else 'first-scoring'} order; progress mean "
              f"success {prog[df.split == 'success'].mean():.3f} / failure {prog[df.split == 'failure'].mean():.3f}")
        done += 1
    a, b = (U.root("CKPT_ROOT") / "rm_scores" / n / "predictions.parquet" for n in ("threading_d0", "threading_d0_seg"))
    if a.exists() and b.exists():       # the same LoRA scored the same rows twice: identical targets, different order
        m = pd.read_parquet(a).merge(pd.read_parquet(b), on=KEYS, suffixes=("", "_seg"))
        d = float((m.pred_progress - m.pred_progress_seg).abs().max())
        print(f"threading_d0 vs threading_d0_seg: max |progress difference| {d}")
        assert len(m) == 12800 and d == 0.0
    a, b = (U.root("CKPT_ROOT") / "rm_scores" / n / "predictions.parquet" for n in ("square_d1", "square_d1_seg"))
    if a.exists() and b.exists():       # two different teachers
        m = pd.read_parquet(a).merge(pd.read_parquet(b), on=KEYS, suffixes=("", "_seg"))
        c = float(np.corrcoef(m.pred_progress, m.pred_progress_seg)[0, 1])
        print(f"square_d1 vs square_d1_seg: corr {c:.3f}, max |difference| {float((m.pred_progress - m.pred_progress_seg).abs().max()):.2f}")
        assert 0.5 < c < 0.99
    if not done:
        U.skip("no score table below $CKPT_ROOT/rm_scores with its frame-cache metadata")


def test_state_scaler_matches_released_v_dirs():
    """fit_v_mqattn writes state_scaler.json from the demo dataset's statistics; that must equal what the released V
    directories carry (the RL code asserts the V directory's scaler against its own)."""
    from online_rl_mimicgen.common import TASKS, StateScaler
    done = 0
    for d in sorted((U.root("CKPT_ROOT") / "value").glob("v_dino_*_mqattn_pv4*")):
        task = json.loads((d / "config.json").read_text())["task"]
        root = TASKS[task]["dataset_root"]
        if not (root / "meta/stats.json").exists():
            continue
        sc, ref = StateScaler(root), json.loads((d / "state_scaler.json").read_text())
        mine = {"q01": [float(x) for x in sc.low], "q99": [float(x) for x in sc.high]}
        assert mine == {"q01": ref["q01"], "q99": ref["q99"]}, f"{d.name}: recomputed scaler differs"
        done += 1
    if not done:
        U.skip("no released V directory with its demo dataset")
    print(f"state_scaler.json recomputed from the demo datasets equals the stored one in {done} V directories")


if __name__ == "__main__":
    U.main(globals())
