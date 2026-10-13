"""Offline pretraining of V: multi-query attention readout ("pv4") + 3-seed MLP ensemble on the prior's rollouts.

usage (value/fit_v.sh wraps the three recipes with the per-task values):
  WS_TASK=<task> python value/fit_v_mqattn.py --task <task> --tag <base wm tag> \\
      --wm $CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_<tag>/wm_dinov2s14_<tag>.pt \\
      --scores $CKPT_ROOT/rm_scores/<task>/predictions.parquet --per-view --queries 4 --out-dim 768 --suffix mqattn_pv4

THE THREE RECIPES OF THE CAMPAIGN (same rows, readout, optimiser and seeds; they differ in the target only)
  pv4      --suffix mqattn_pv4, target RoboMeter pred_progress of the task's first score table ("dpinit").
           -> v_dino_robometer_pred_progress_<task>_<tag>_mqattn_pv4: the V of every DICE-RL DVD cell and of the DSRL
           cells on square_d0 and hammer_cleanup_d1.
  pv4_seg  --suffix mqattn_pv4_seg --scores $CKPT_ROOT/rm_scores/<task>_seg/predictions.parquet, the same rows scored
           by the task's segmented LoRA. -> ..._mqattn_pv4_seg: the V of the DSRL cells on square_d1 and threading_d0.
  mc       --target mc --suffix mqattn_pv4: the Monte-Carlo success return gamma^(t_r - t) per primitive step
           (gamma 0.99, t_r = the first rewarded frame) on success episodes and 0 on failure episodes ("mcinit"); the
           score table only supplies the row set. -> v_dino_mc_return_<task>_<tag>_mqattn_pv4: DSRL arm oursmc.
<tag> is the task's BASE world model (ph_conv_150k square_d0 / square_d1, ph_conv_168k threading_d0, ph_conv_217k
hammer_cleanup_d1): every V was fitted through the base model even where the RL run loads the continuation.
Held-out Spearman of the released pv4 fits against progress: 0.92 square_d0, 0.89 square_d1, 0.94 threading_d0,
0.92 hammer_cleanup_d1. The fit is not bit-reproducible on GPU (the two threading fits on identical labels differ in
weights and agree in held-out metrics); the released directories are the reference. About 8 minutes, 10 GB of GPU.

Rows: the RoboMeter-scored frames of the task's prior rollouts (64 per episode, robometer/score_rollouts.py), train
episodes 0-79, held out 80-99; rows are sampled by position, so the table's row order is part of the recipe. Each
scored frame t gives two rows with the SAME target:
  actual     frozen DINOv2-S patches of frame t, raw state_t[:9]
  predicted  the world model's prediction of frame t from [t-16, t-8], proprio t-8 and actions t-8..t-1, with its
             predicted proprio (frames t < 8 have no predicted row)
Input = [MultiQueryAttnPool(patches) | proprio 9]. --per-view --queries 4 --out-dim 768: 4 queries per camera, each a
softmax over that camera's 196 patches of <patch, q>/sqrt(384), output the weighted mean of the raw patch vectors;
the 8 x 384 concatenation goes through one Linear to 768 (+9 = 777 = V's input).
  1) readout + VNet (256,256,256, LayerNorm + Mish) fitted jointly (seed 0), Adam 1e-3 cosine 20k, batch 512 (half
     actual, half predicted), MSE on the target
  2) readout frozen; VNet seeds 0-2 refit on the frozen pooled inputs, same recipe
No TD offline. Writes $CKPT_ROOT/value/<prefix>_<task>_<tag>_<suffix>/ = the --value-model of the RL runs:
  config.json        state_dim, input_dim 777, hidden, seeds, dino_pool mqattn + provenance and held-out metrics
  state_scaler.json  the trainer's StateScaler (q01 / q99 of the demo dataset); the RL code asserts it matches
  mqattn_pool.pt     the readout; its shape is read from the blob, not from flags; must reload bit-identically (checked)
  weights/v_td_seed{0,1,2}.pt     ("v_td" is a variant label of the loader, not TD training)
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ.get("MQATTN_OMP", "2")   # read by libgomp at torch import

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "dinowm"), str(_REPO / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)                 # this repository's modules, whatever PYTHONPATH says

import common_ws as C  # noqa: E402
import online_rl_mimicgen  # noqa: E402

assert _REPO / "rl" in Path(online_rl_mimicgen.__file__).resolve().parents, (
    f"online_rl_mimicgen resolved to {online_rl_mimicgen.__file__}, not to {_REPO / 'rl'}: another copy was imported first")

from online_rl_mimicgen import dino_dvd  # noqa: E402
from online_rl_mimicgen.common import TASKS, StateScaler  # noqa: E402
from online_rl_mimicgen.dino_wm import load_dino_wm  # noqa: E402
from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool  # noqa: E402
from online_rl_mimicgen.value_model import VNet  # noqa: E402

H, TRAIN_EPISODES, DIM, STEPS = 8, 80, 384, 20000
GAMMA = 0.99
HIDDEN = (256, 256, 256)
PREFIX = {"pred_progress": "v_dino_robometer_pred_progress", "mc": "v_dino_mc_return"}


def mc_targets(M, df):
    """--target mc: the Monte-Carlo SUCCESS return gamma^(t_r - t) for the scored frames of success episodes
    (t_r = the episode's first rewarded frame), 0 for every frame of a failure episode."""
    first = {}
    for s_, e in {(str(a), int(b)) for a, b in zip(df.split, df.episode_index)}:
        m = np.where((M["split"] == s_) & (M["ep"] == e))[0]
        hit = m[M["reward"][m] > 0]
        first[(s_, e)] = int(M["fr"][hit[0]]) if len(hit) else None
    out = np.zeros(len(df), np.float32)
    for i, (s_, e, f) in enumerate(zip(df.split, df.episode_index, df.frame_index)):
        tr = first[(str(s_), int(e))]
        if tr is not None:
            out[i] = GAMMA ** max(tr - int(f), 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--wm", required=True, help="the task's base DINO-WM (.pt)")
    ap.add_argument("--arch", default="", help="depth,heads,mlp of the WM predictor (default 6,6,2048)")
    ap.add_argument("--tag", required=True, help="WM tag used in the output name, e.g. ph_conv_150k")
    ap.add_argument("--scores", required=True, help="RoboMeter predictions.parquet (robometer/score_rollouts.py): the "
                                                    "rows, and with --target pred_progress the targets")
    ap.add_argument("--teacher", default="", help="free-text name of the RoboMeter model, recorded in config.json")
    ap.add_argument("--target", choices=tuple(PREFIX), default="pred_progress",
                    help="pred_progress = RoboMeter's per-frame progress; mc = the MC success return gamma^(t_r - t), 0 "
                         "on failures, on the SAME rows, readout and recipe (output prefix v_dino_mc_return)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--queries", type=int, default=1, help="K (per camera with --per-view)")
    ap.add_argument("--per-view", action="store_true", help="each camera gets its own K queries over its 196 patches")
    ap.add_argument("--out-dim", type=int, default=0, help="0 = no projection; else one flat Linear to this width")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--suffix", required=True, help="output dir suffix, e.g. mqattn_pv4")
    ap.add_argument("--out-root", default=str(C.CKPT / "value"),
                    help="parent of the V directory [$CKPT_ROOT/value; without CKPT_ROOT, <repo>/assets/checkpoints/value]")
    args = ap.parse_args()
    C._need_task(args.task)
    dev = torch.device("cuda")
    t0 = time.time()
    scores_path = Path(args.scores)
    df = pd.read_parquet(scores_path)
    print(f"scores: {scores_path} ({len(df)} rows)", flush=True)
    M = C.load_meta()
    key = {(str(s), int(e), int(f)): i for i, (s, e, f) in enumerate(zip(M["split"], M["ep"], M["fr"]))}
    rows_t = np.array([key[(s, int(e), int(f))] for s, e, f in zip(df.split, df.episode_index, df.frame_index)])
    y = df.pred_progress.to_numpy(np.float32)
    mc_all = mc_targets(M, df)                                    # also the held-out ranking metric of every fit
    if args.target == "mc":
        prog, y, _sp = y, mc_all, df.split.to_numpy()
        print(f"target MC return: success rows mean {y[_sp == 'success'].mean():.3f} (min {y[_sp == 'success'].min():.3f}), "
              f"failure rows {y[_sp == 'failure'].mean():.3f}; corr(progress, mc) {np.corrcoef(prog, y)[0, 1]:.3f}", flush=True)
    ep = df.episode_index.to_numpy()
    fr = df.frame_index.to_numpy()
    ST = M["state"][:, :9].astype(np.float32)
    kw = {}
    if args.arch:
        d_, h_, m_ = (int(x) for x in args.arch.split(","))
        kw = dict(depth=d_, heads=h_, mlp_dim=m_, heads_explicit=True)
    wm = load_dino_wm(args.wm, device=dev, **kw)
    A = dino_dvd.Bf16Adapter(wm, action_dim=7, bf16=True)
    rd = C.FrameReader()
    n = len(df)
    Za = torch.empty((n, 2, 196, DIM), dtype=torch.float16, device=dev)
    with torch.no_grad():
        for s0 in range(0, n, 128):
            Za[s0:s0 + 128] = dino_dvd.encode_frames(A, torch.as_tensor(rd.read(rows_t[s0:s0 + 128]), device=dev)).half()
    Pa = torch.as_tensor(ST[rows_t], device=dev)
    ok = fr >= H
    idx = np.where(ok)[0]
    cur = rows_t - H
    prev = np.where(fr - H >= H, rows_t - 2 * H, cur)
    Zp = torch.empty((len(idx), 2, 196, DIM), dtype=torch.float16, device=dev)
    Pp = torch.empty((len(idx), 9), dtype=torch.float32, device=dev)
    with torch.no_grad():
        for s0 in range(0, len(idx), 64):
            j = idx[s0:s0 + 64]
            both = rd.read(np.stack([prev[j], cur[j]], 1).reshape(-1)).reshape(len(j), 2, 2, 224, 224, 3)
            dino_dvd.set_context(A, dino_dvd.encode_frames(A, torch.as_tensor(both[:, 0], device=dev)),
                                 dino_dvd.encode_frames(A, torch.as_tensor(both[:, 1], device=dev)))
            act = torch.as_tensor(M["action"][cur[j][:, None] + np.arange(H)[None]].reshape(len(j), -1), device=dev, dtype=torch.float32)
            Zp[s0:s0 + len(j)] = A(torch.as_tensor(ST[cur[j]], device=dev), act).half()
            Pp[s0:s0 + len(j)] = A.last_state_pred[:, :9].float()
    rd.close()
    print(f"actual {n} + predicted {len(idx)} patch grids on GPU [{time.time() - t0:.0f}s]", flush=True)
    yt = torch.as_tensor(y, device=dev)
    tr_a = torch.as_tensor(np.where(ep < TRAIN_EPISODES)[0], device=dev)
    tr_p = torch.as_tensor(np.where(ep[idx] < TRAIN_EPISODES)[0], device=dev)
    ho_a = np.where(ep >= TRAIN_EPISODES)[0]
    ho_p = np.where(ep[idx] >= TRAIN_EPISODES)[0]
    y_p = yt[torch.as_tensor(idx, device=dev)]
    split = df.split.to_numpy()

    def batches(g):
        """half actual, half predicted rows per batch."""
        ia = tr_a[torch.randint(0, len(tr_a), (256,), device=dev, generator=g)]
        ip = tr_p[torch.randint(0, len(tr_p), (256,), device=dev, generator=g)]
        return (torch.cat([Za[ia], Zp[ip]]).float(), torch.cat([Pa[ia], Pp[ip]]), torch.cat([yt[ia], y_p[ip]]))

    # ---- stage 1: readout + VNet jointly (seed 0)
    torch.manual_seed(0)
    pool = MultiQueryAttnPool(dim=DIM, queries=args.queries, per_view=args.per_view, out_dim=(args.out_dim or None),
                              n_views=2, query_norm=False, share_queries=False, patches=196).to(dev)
    pdim = pool.pooled_dim
    print(f"stage 1 pool=mqattn queries={args.queries} per_view={args.per_view} concat_dim={pool.concat_dim} "
          f"pooled_dim={pdim} v_input={pdim + 9} params={sum(q.numel() for q in pool.parameters())}", flush=True)
    net0 = VNet(pdim + 9, HIDDEN).to(dev)
    opt = torch.optim.Adam(list(pool.parameters()) + list(net0.parameters()), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(args.steps):
        z, p, yy = batches(g)
        loss = F.mse_loss(net0(torch.cat([pool(z, "mqattn"), p], 1)).squeeze(-1), yy)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    pool.eval().requires_grad_(False)
    print(f"stage 1 (readout + V jointly): final loss {float(loss.detach()):.5f} [{time.time() - t0:.0f}s]", flush=True)

    # ---- stage 2: frozen readout -> pooled inputs -> VNet seeds 0..seeds-1
    with torch.no_grad():
        Xa = torch.cat([torch.cat([pool(Za[s:s + 512].float(), "mqattn"), Pa[s:s + 512]], 1) for s in range(0, n, 512)])
        Xp = torch.cat([torch.cat([pool(Zp[s:s + 512].float(), "mqattn"), Pp[s:s + 512]], 1) for s in range(0, len(idx), 512)])
    assert Xa.shape[1] == pdim + 9, (Xa.shape, pdim)
    out = Path(args.out_root) / f"{PREFIX[args.target]}_{args.task}_{args.tag}_{args.suffix}"
    (out / "weights").mkdir(parents=True, exist_ok=True)
    metrics = {}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        net = VNet(Xa.shape[1], HIDDEN).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        g = torch.Generator(device=dev).manual_seed(seed)
        for step in range(args.steps):
            ia = tr_a[torch.randint(0, len(tr_a), (256,), device=dev, generator=g)]
            ip = tr_p[torch.randint(0, len(tr_p), (256,), device=dev, generator=g)]
            loss = F.mse_loss(net(torch.cat([Xa[ia], Xp[ip]])).squeeze(-1), torch.cat([yt[ia], y_p[ip]]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            va = net(Xa[torch.as_tensor(ho_a, device=dev)]).squeeze(-1).cpu().numpy()
            vp = net(Xp[torch.as_tensor(ho_p, device=dev)]).squeeze(-1).cpu().numpy()
            va_m = net(Xa[torch.as_tensor(idx[ho_p], device=dev)]).squeeze(-1).cpu().numpy()
        ya, yp = y[ho_a], y[idx][ho_p]
        sa, sp = split[ho_a], split[idx][ho_p]
        # succ_minus_fail: success/failure separation of V (a progress-only V can rank progress well and still not
        # separate outcomes); spearman_mc: ranking against the true success return whatever the fit target was
        metrics[f"seed{seed}"] = dict(
            actual=dict(rmse=float(np.sqrt(((va - ya) ** 2).mean())), spearman=float(spearmanr(va, ya).statistic),
                        succ_minus_fail=float(va[sa == "success"].mean() - va[sa == "failure"].mean()),
                        spearman_mc=float(spearmanr(va, mc_all[ho_a]).statistic),
                        mean_v=float(va.mean()), mean_mc=float(mc_all[ho_a].mean())),
            predicted=dict(rmse=float(np.sqrt(((vp - yp) ** 2).mean())), spearman=float(spearmanr(vp, yp).statistic),
                           succ_minus_fail=float(vp[sp == "success"].mean() - vp[sp == "failure"].mean())),
            pearson_v_actual_vs_predicted=float(np.corrcoef(va_m, vp)[0, 1]),
            mean_abs_v_actual_minus_predicted=float(np.abs(va_m - vp).mean()))
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "weights" / f"v_td_seed{seed}.pt")
        print(f"  seed {seed}: {json.dumps(metrics[f'seed{seed}'])} [{time.time() - t0:.0f}s]", flush=True)

    # the readout is saved once, beside V, and must reload to bit-identical pooled outputs
    torch.save(pool.blob(fit=f"value/fit_v_mqattn.py stage 1 (seed 0), {args.steps} steps; queries={args.queries} "
                             f"per_view={args.per_view} out_dim={args.out_dim or None}"), out / "mqattn_pool.pt")
    back = MultiQueryAttnPool.from_blob(torch.load(out / "mqattn_pool.pt", map_location="cpu", weights_only=False),
                                        device=dev).eval()
    with torch.no_grad():
        ref, got = pool(Za[:256].float(), "mqattn"), back(Za[:256].float(), "mqattn")
    assert torch.equal(ref, got), f"mqattn_pool.pt does not round-trip (max |d| {float((ref - got).abs().max())})"

    # V directory metadata: the trainer's StateScaler of the demo set (value_model asserts it) + the net config
    sc = StateScaler(TASKS[args.task]["dataset_root"])
    (out / "state_scaler.json").write_text(json.dumps({"q01": [float(x) for x in sc.low], "q99": [float(x) for x in sc.high]}))
    cfg = dict(state_dim=int(sc.low.shape[0]), input_dim=int(Xa.shape[1]), hidden=list(HIDDEN), layernorm=True,
               activation="mish", gamma=GAMMA, task=args.task,
               estimator=("robometer_distill_on_dino:pred_progress" if args.target == "pred_progress" else
                          "mc_success_return_on_dino (gamma^(t_r - t) per primitive step, t_r = first rewarded frame; "
                          "0 on failures)"),
               target=args.target, world_model=Path(args.wm).name, robometer=args.teacher,
               scores="/".join(scores_path.parts[-2:]),
               train_episodes=f"0-{TRAIN_EPISODES - 1}", eval_episodes=f"{TRAIN_EPISODES}-99", seeds=args.seeds,
               fit_steps=args.steps, metrics=metrics, dino_pool="mqattn", mqattn_queries=args.queries,
               mqattn_per_view=args.per_view, mqattn_out_dim=(args.out_dim or None), mqattn_share_queries=False,
               mqattn_query_norm=False, mqattn_concat_dim=pool.concat_dim,
               mqattn_params=int(sum(q.numel() for q in pool.parameters())),
               recipe="readout+VNet jointly (seed 0), then frozen readout and VNet 256x3 seeds 0-2; Adam 1e-3 cosine "
                      f"{args.steps}, batch 512 (half actual, half predicted), MSE on "
                      + ("pred_progress" if args.target == "pred_progress" else "the MC success return (--target mc)"))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
