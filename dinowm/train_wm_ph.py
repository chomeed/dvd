"""Train the DINO world model f ("ph_conv": proprio head + conv action encoder) on a task's prior rollouts.

usage (dinowm/train_wm.sh wraps both stages with the per-task values below):
  WS_TASK=<task> python dinowm/train_wm_ph.py --act conv --steps <S> --resume                                  base
  WS_TASK=<task> python dinowm/train_wm_ph.py --act conv --steps <S> --init <base .pt> --lr 1e-4 --seed <k> \\
                                              --tag-suffix _cont<S/1000>k --snapshot-steps 50000,100000 --resume
                                                                                                         continuation

MODEL. Frozen DINOv2-S/14 at 196 px (14 x 14 patches, 384-d) on two cameras; a 6-layer / 6-head / MLP-2048 transformer
predictor (14.4 M parameters) maps the features of the two context frames [t-8, t], the proprio s_t[:9] and the 8 x 7
action chunk to the ABSOLUTE features of frame t+8. --act conv encodes the chunk with a Conv1d stack over its 8 steps
into one token. The proprio head reads the state token and predicts the RESIDUAL (s_{t+8} - s_t)[:9] divided by
state_delta_scale, the per-dimension std of that delta on the train windows (a buffer saved in the checkpoint).
LOSS. MSE(features) + state_w * MSE(normalised proprio residual), state_w 1.
OPTIMISATION. Batch 32, AdamW weight decay 0.01, 500 warm-up steps then cosine to 0.1 x the peak LR, bf16 autocast,
paired DrQ shift (pad 4), train windows of rollout episodes 0-79 (common_ws.windows), model init seed 0.

THE PRODUCTION WORLD MODELS (S = an equal number of passes over each task's windows):
  task               S        base tag       continuation tag          --seed of the continuation
  square_d0          150000   ph_conv_150k   ph_conv_150k_cont150k     1
  square_d1          150000   ph_conv_150k   ph_conv_150k_cont150k     0   (yes, 0: the same batch stream as its base)
  threading_d0       168000   ph_conv_168k   ph_conv_168k_cont168k     1
  hammer_cleanup_d1  217000   ph_conv_217k   ph_conv_217k_cont217k     1
  base          from scratch, peak LR 5e-4, data seed 0, S steps (square_d1 also passed --snapshot-steps 44000,80000)
  continuation  --init the base model, a FRESH AdamW and schedule with peak LR 1e-4, S more steps
Which one is used where: every V was fitted through the BASE model; online, DICE-RL loads the continuation for
square_d0, square_d1 and threading_d0 and the BASE model for hammer_cleanup_d1, DSRL loads the continuation for all
four. Held-out clean feature skill at the end of training, to check a retrain against: square_d0 0.467 / 0.453
(base / continuation), square_d1 0.379 / 0.364, threading_d0 0.548 / 0.539, hammer_cleanup_d1 0.491 / 0.479. A
continuation's step-0 validation must equal its base run's final validation. Training is not bit-reproducible
(GPU, compiled predictor); the released checkpoints are the reference. One base run takes 5-10 h on one GPU.

OUTPUTS, all named by the tag ph_<act>[_<S/1000>k][<tag suffix>]:
  $CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_<tag>/wm_dinov2s14_<tag>.pt    predictor state dict = --dino-wm-ckpt of
                                                    the RL runs (round-tripped through dino_wm.load_dino_wm before
                                                    it is accepted); .json beside it = architecture, recipe, final_val
  .../resume_<tag>.pt                               optimiser + curves, rewritten every --save-every steps (--resume)
  $RUNS_ROOT/wm_curves/curves_<tag>_<task>.json     train / validation curves
Validation every --val-every steps on 1024 fixed held-out windows, clean and with a fixed paired shift: feature skill
(1 - MSE / MSE of copying frame t), pooled skill, SHUFFLED-ACTION skill (how much of the skill is the action's),
proprio residual skill, end-effector error in mm against copying.

CONTINUATION FLAGS (all opt-in; without them the json keeps the base-run key set). --init CKPT loads the predictor
strictly and KEEPS its state_delta_scale; --lr PEAK; --seed N changes the batch-row / shift-fraction stream only (model
init and the validation subset stay fixed); --tag-suffix S is appended to every output name and is required with
--init; --snapshot-steps writes extra drop-in models <tag>_at<k>k. Guards of --init: its path must not name another
task, its head count must be on record (the weights cannot tell it), and its state_delta_scale must equal the scale
derived from THIS task's data within 0.1 %, which catches another task's model whatever the file is called.

torch.compile of the predictor needs an executable cache directory: where /tmp is mounted noexec, point
TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR at a normal disk (train_wm.sh does).
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path[:1]:
    sys.path.insert(0, str(_RL))               # this repository's package, whatever PYTHONPATH says

import common_ws as C  # noqa: E402
import online_rl_mimicgen  # noqa: E402

assert _RL in Path(online_rl_mimicgen.__file__).resolve().parents, (
    f"online_rl_mimicgen resolved to {online_rl_mimicgen.__file__}, not to {_RL}: another copy was imported first")

from online_rl_mimicgen.dino_wm import DinoDynamics, load_dino_wm  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
B, LR, WD, WARM, VAL_N, SEED = 32, 5e-4, 0.01, 500, 1024, 0
BB = "dinov2s14"
H_CTX = 2


class Prefetch:
    """Background thread that produces batch `step` (frame reads) ahead of the training loop, in step order."""

    def __init__(self, produce, start, stop, depth=4):
        self.q = queue.Queue(depth)
        self.t = threading.Thread(target=self._run, args=(produce, start, stop), daemon=True)
        self.t.start()

    def _run(self, produce, start, stop):
        for s in range(start, stop):
            self.q.put((s, produce(s)))
        self.q.put((None, None))

    def __iter__(self):
        while True:
            s, b = self.q.get()
            if s is None:
                return
            yield s, b


def trained_steps(m):
    """optimiser steps behind a WM json: a continuation's total > a snapshot's step > a file's own step > schedule length"""
    if "total_steps_incl_init" in m:
        return m["total_steps_incl_init"]
    return next((m[k] for k in ("snapshot_step", "step", "steps") if m.get(k) is not None), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default=None, choices=C.TASKS, help="must equal WS_TASK when given")
    ap.add_argument("--act", choices=("linear", "conv"), required=True, help="action encoder (the released WMs: conv)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--state-w", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="no deliverable or resume file written")
    ap.add_argument("--init", default=None, help="predictor checkpoint to continue from")
    ap.add_argument("--lr", type=float, default=LR, help="peak LR (500 warm-up, cosine to 0.1x)")
    ap.add_argument("--seed", type=int, default=SEED, help="batch-row / shift-fraction stream (model init + val subset fixed)")
    ap.add_argument("--tag-suffix", default="", help="appended to the tag (every output name)")
    ap.add_argument("--snapshot-steps", default="", help="comma list of steps: extra deliverables named <tag>_at<k>k")
    args = ap.parse_args()
    if args.init:
        args.init = os.path.abspath(args.init)
    task = C._need_task(args.task)
    NEW_DEFAULTS = dict(init=None, lr=LR, seed=SEED, tag_suffix="", snapshot_steps="")
    cont = any(getattr(args, k) != v for k, v in NEW_DEFAULTS.items())
    args_json = {k: v for k, v in vars(args).items() if k != "task" and (cont or k not in NEW_DEFAULTS)}
    snaps = sorted({int(x) for x in args.snapshot_steps.split(",") if x.strip()})
    assert all(0 < k < args.steps for k in snaps), f"--snapshot-steps {snaps} must lie inside (0, {args.steps})"
    assert not args.init or args.tag_suffix, "--init needs --tag-suffix (never overwrite the init run's outputs)"
    spec0 = C.SPECS[BB]
    depth, heads, mlp = C.DEPTH, spec0["heads"], spec0["mlp"]
    tag = (f"ph_{args.act}" + ("" if args.state_w == 1.0 else f"_sw{args.state_w:g}")
           + ("" if args.steps == 12000 else f"_{args.steps // 1000}k") + args.tag_suffix)

    def deliverable_path(t):
        return C.wm_path(task, t)

    def snap_tag(k):
        return f"{tag}_at{k // 1000}k" if k % 1000 == 0 else f"{tag}_at{k}"

    resume_p = deliverable_path(tag).with_name(f"resume_{tag}.pt")
    curves_p = C.OUT / f"curves_{tag}_{task}.json"

    init_meta, init_sd = {}, None
    if args.init:
        m = re.search(r"mimicgen_(.+?)_224x224_dinowm", str(args.init))
        assert m is None or m.group(1) == task, f"--init {args.init} is not this task's WM (WS_TASK={task!r})"
        outs = [deliverable_path(t) for t in [tag] + [snap_tag(k) for k in snaps]] + [resume_p]
        assert Path(args.init).resolve() not in {p.resolve() for p in outs}, f"--init {args.init} is one of this run's outputs"
        init_sd = torch.load(args.init, map_location="cpu", weights_only=False)
        rec = {}
        if isinstance(init_sd, dict) and "state_dict" in init_sd:
            rec = {k: v for k, v in init_sd.items() if k in ("depth", "heads", "mlp_dim", "history", "action_encoder", "step")}
            init_sd = init_sd["state_dict"]
        jp = Path(args.init).with_suffix(".json")
        init_meta = {**rec, **(json.loads(jp.read_text()) if jp.exists() else {})}
        want = dict(dino_wm_arch=f"{depth},{heads},{mlp}", depth=depth, heads=heads, mlp_dim=mlp, action_encoder=args.act,
                    history=H_CTX, trunk=spec0["hf"], predictor="transformer")
        bad = {k: (init_meta[k], w) for k, w in want.items() if k in init_meta and init_meta[k] != w}
        assert not bad, f"--init architecture differs from this run's (init, this run): {bad}"

        assert "heads" in init_meta or "dino_wm_arch" in init_meta, (
            f"--init {args.init}: no head-count record (json dino_wm_arch/heads next to it)")
    t0 = time.time()
    C.OUT.mkdir(parents=True, exist_ok=True)
    M = C.load_meta()
    Wt, Wv = C.windows(M, "train"), C.windows(M, "heldout")
    assert not set(np.unique(Wt["uid"])) & set(np.unique(Wv["uid"]))
    ST = M["state"][:, :9].astype(np.float32)
    scale = np.maximum((ST[Wt["nxt"]] - ST[Wt["cur"]]).std(0), 1e-4).astype(np.float32)
    rd = C.FrameReader()
    bb = C.Backbone(BB, dev)
    s = bb.spec
    torch.manual_seed(SEED)
    wm = DinoDynamics(bb.model, None, 2, 7, 9, history=H_CTX, depth=depth, heads=heads, mlp_dim=mlp,
                      dino_size=s["px"], image_size=224, state_head=True, action_encoder=args.act).to(dev)
    assert wm.has_state_head and wm.n_patches == bb.n_patches and wm.dim == bb.dim
    if not args.init:
        wm.state_delta_scale.copy_(torch.as_tensor(scale, device=dev))
    else:
        assert not [k for k in init_sd if k.startswith("dino.")], "--init carries trunk weights; the trunk stays the frozen HF one"
        mi, un = wm.load_state_dict(init_sd, strict=False)
        assert not [k for k in mi if not k.startswith("dino.")] and not un, (mi, un)
        ck_scale = wm.state_delta_scale.detach().cpu().numpy().astype(np.float32)

        rel = float(np.max(np.abs(ck_scale - scale) / scale))
        assert rel < 1e-3, (f"--init state_delta_scale is {rel:.1%} off this task's data (WS_TASK={task!r}): another task's WM? "
                            f"init {np.round(ck_scale, 5).tolist()} data {np.round(scale, 5).tolist()}")
        scale = ck_scale
        del init_sd
    wm.predictor.compile()
    params = C.predictor_params(wm)
    n_par = sum(p.numel() for p in params)
    n_act = sum(p.numel() for n, p in wm.named_parameters() if n.startswith(("act_conv.", "act_tok.")))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, C.lr_lambda(args.steps, WARM, 0.1))
    print(f"[{tag}] trunk {s['hf']} {bb.px}px grid {bb.side}x{bb.side}; predictor depth {depth} heads {heads} mlp {mlp}: "
          f"{n_par/1e6:.2f}M (action encoder {args.act}: {n_act/1e6:.3f}M; proprio head "
          f"{sum(p.numel() for p in wm.state_head.parameters())/1e6:.3f}M); state_w {args.state_w}; "
          f"delta scale {np.round(scale, 5).tolist()}; train windows {len(Wt['cur'])} held-out {len(Wv['cur'])}", flush=True)
    if cont:
        print(f"[{tag}] continuation: init {args.init} (its steps {trained_steps(init_meta) if args.init else 0}); "
              f"peak lr {args.lr:g}; data seed {args.seed}; snapshots at {snaps}", flush=True)

    rng = np.random.default_rng(args.seed)
    IDX = rng.integers(0, len(Wt["cur"]), size=(args.steps, B))
    U = rng.random((args.steps, B, 2, 2), dtype=np.float32)

    def produce(step):
        j = IDX[step]
        rows = np.stack([Wt["prev"][j], Wt["cur"][j], Wt["nxt"][j]], 1)
        return dict(frames=rd.read(rows.reshape(-1)).reshape(B, H_CTX + 1, 2, 224, 224, 3),
                    state=ST[Wt["cur"][j]], nstate=ST[Wt["nxt"][j]], action=C.chunk(M, Wt["cur"][j]),
                    off=C.frac_to_off(U[step], bb.pad))

    vj = np.sort(np.random.default_rng(12345).choice(len(Wv["cur"]), VAL_N, replace=False))
    vcur = Wv["cur"][vj]
    VU = np.random.default_rng(777).random((VAL_N, 2, 2), dtype=np.float32)
    v_state = torch.as_tensor(ST[vcur], device=dev)
    v_nstate = torch.as_tensor(ST[Wv["nxt"][vj]], device=dev)
    v_act = torch.as_tensor(C.chunk(M, vcur), device=dev)
    v_act_shuf = v_act[torch.as_tensor(np.random.default_rng(999).permutation(VAL_N), device=dev)]
    VZ = {"clean": [], "shift": []}
    for s0 in range(0, VAL_N, 64):
        jj = vj[s0:s0 + 64]
        rows = np.stack([Wv["prev"][jj], Wv["cur"][jj], Wv["nxt"][jj]], 1)
        fr = torch.as_tensor(rd.read(rows.reshape(-1)).reshape(len(jj), H_CTX + 1, 2, 224, 224, 3), device=dev)
        off = torch.as_tensor(C.frac_to_off(VU[s0:s0 + 64], bb.pad), device=dev)[:, None].expand(-1, H_CTX + 1, -1, -1)
        VZ["clean"].append(C.encode_u8(bb, fr, None).half())
        VZ["shift"].append(C.encode_u8(bb, fr, off).half())
    VZ = {k: torch.cat(v) for k, v in VZ.items()}
    print(f"val subset {VAL_N} windows encoded [{time.time()-t0:.0f}s]", flush=True)

    def fwd(z_ctx, act, st):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = wm.predict_from_features(z_ctx, act.reshape(len(st), C.H, 7), st)
        return p.float(), wm.last_state_delta, wm.last_state_pred

    @torch.no_grad()
    def evaluate():
        wm.eval()
        out = {}
        for cond, Z in VZ.items():
            acc = dict(num=0.0, den=0.0, pnum=0.0, pden=0.0, shnum=0.0, snum=0.0, sden=0.0, sshnum=0.0, mm=0.0, mmc=0.0)
            for s0 in range(0, VAL_N, 128):
                sl = slice(s0, s0 + 128)
                z = Z[sl].float()
                st, ns = v_state[sl], v_nstate[sl]
                tgt, cp = z[:, H_CTX], z[:, H_CTX - 1]
                pred, d, sp = fwd(z[:, :H_CTX], v_act[sl], st)
                acc["num"] += float(F.mse_loss(pred, tgt, reduction="sum"))
                acc["den"] += float(F.mse_loss(cp, tgt, reduction="sum"))
                acc["pnum"] += float(F.mse_loss(C.pool_mean(pred), C.pool_mean(tgt), reduction="sum"))
                acc["pden"] += float(F.mse_loss(C.pool_mean(cp), C.pool_mean(tgt), reduction="sum"))
                tgt_n = (ns - st) / wm.state_delta_scale
                acc["snum"] += float(((d - tgt_n) ** 2).sum())
                acc["sden"] += float((tgt_n ** 2).sum())
                acc["mm"] += float((sp[:, :3] - ns[:, :3]).norm(dim=-1).sum()) * 1000
                acc["mmc"] += float((st[:, :3] - ns[:, :3]).norm(dim=-1).sum()) * 1000
                pred_sh, d_sh, _ = fwd(z[:, :H_CTX], v_act_shuf[sl], st)
                acc["shnum"] += float(F.mse_loss(pred_sh, tgt, reduction="sum"))
                acc["sshnum"] += float(((d_sh - tgt_n) ** 2).sum())
            out[cond] = dict(mse=acc["num"] / Z[:, H_CTX].numel(), skill=1 - acc["num"] / acc["den"],
                             pooled_skill=1 - acc["pnum"] / acc["pden"],
                             skill_shuffled_action=1 - acc["shnum"] / acc["den"],
                             state_skill=1 - acc["snum"] / acc["sden"],
                             state_skill_shuffled_action=1 - acc["sshnum"] / acc["sden"],
                             eef_mm=acc["mm"] / VAL_N, eef_copy_mm=acc["mmc"] / VAL_N)
        wm.train()
        return out

    curves = dict(train=[], val=[])
    start = 0
    cont_rec = dict(init=args.init, lr=args.lr, seed=args.seed)
    if args.resume and resume_p.exists():
        ck = torch.load(resume_p, map_location="cpu", weights_only=False)
        mi, un = wm.load_state_dict(ck["state_dict"], strict=False)
        assert not [k for k in mi if not k.startswith("dino.")] and not un
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, curves = ck["step"], ck["curves"]
        assert ck["steps_total"] == args.steps
        assert ck.get("cont", dict(init=None, lr=LR, seed=SEED)) == cont_rec, (ck.get("cont"), cont_rec)
        print(f"resumed at step {start}", flush=True)
    else:
        curves["val"].append(dict(step=0, elapsed_s=0.0, **evaluate()))
        print(f"step 0 val {json.dumps(curves['val'][-1])}", flush=True)

    def write_deliverable(t, step, train_h):
        """the drop-in --dino-wm-ckpt + json: the final one (t = tag) or a --snapshot-steps one (t = <tag>_at<k>k)"""
        path = deliverable_path(t)
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = C.predictor_state(wm)
        torch.save(sd, path)
        wm2 = load_dino_wm(path, device=dev, depth=depth, heads=heads, mlp_dim=mlp, heads_explicit=True)
        assert wm2.has_state_head, "loader rebuilt a different head"
        sd2 = {k: v for k, v in wm2.state_dict().items() if not k.startswith("dino.")}
        assert set(sd2) == set(sd), (set(sd2) ^ set(sd))
        for k, v in sd.items():
            assert torch.equal(v, sd2[k].cpu()), k
        meta = dict(tag=t, trunk=s["hf"], dino_size=s["px"], history=H_CTX, depth=depth, heads=heads, mlp_dim=mlp,
                    feat_loss="mse", dino_wm_arch=f"{depth},{heads},{mlp}", action_encoder=args.act, state_head=True,
                    state_w=args.state_w, state_delta_scale=scale.tolist(), predictor="transformer",
                    predictor_params=n_par, recipe=f"from scratch: AdamW peak lr {args.lr:g} wd {WD}, {WARM} warm-up + cosine "
                    f"to 0.1x, batch {B}, paired DrQ, rollout episodes < {C.TRAIN_EPISODES}",
                    steps=args.steps, train_hours=train_h, final_val=curves["val"][-1])
        if cont:
            assert curves["val"][-1]["step"] == step, (curves["val"][-1]["step"], step)
            init_steps = trained_steps(init_meta) if args.init else 0
            meta.update(steps=step, schedule_steps=args.steps,
                        recipe=(f"continuation of {init_meta.get('tag') or Path(args.init).name} ({init_steps} steps)"
                                if args.init else "from scratch")
                        + f": fresh AdamW peak lr {args.lr:g} wd {WD}, {WARM} warm-up + cosine to 0.1x over {args.steps} "
                        f"steps, data seed {args.seed}, paired DrQ; this file at step {step}",
                        init=Path(args.init).name if args.init else None, init_tag=init_meta.get("tag"),
                        init_steps=init_steps, lr=args.lr, lr_floor=0.1 * args.lr, warmup=WARM, seed=args.seed, step=step,
                        lr_at_step=float(sched.get_last_lr()[0]),
                        total_steps_incl_init=None if init_steps is None else init_steps + step,
                        tag_suffix=args.tag_suffix, snapshot_steps=snaps, snapshot=step != args.steps)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {path} (round trip OK)", flush=True)

    wm.train()
    acc, tl = {}, time.time()
    t_train = time.time() - (curves["train"][-1]["elapsed_s"] if curves["train"] else 0.0)
    for step, b in Prefetch(produce, start, args.steps):
        fr = torch.from_numpy(b["frames"]).to(dev, non_blocking=True)
        off = torch.from_numpy(b["off"]).to(dev)[:, None].expand(-1, H_CTX + 1, -1, -1)
        state = torch.from_numpy(b["state"]).to(dev)
        nstate = torch.from_numpy(b["nstate"]).to(dev)
        action = torch.from_numpy(b["action"]).to(dev)
        z = C.encode_u8(bb, fr, off, bf16=True)
        pred, delta, _ = fwd(z[:, :H_CTX], action, state)
        feat_loss = F.mse_loss(pred, z[:, H_CTX])
        tgt_n = (nstate - state) / wm.state_delta_scale
        state_loss = F.mse_loss(delta, tgt_n)
        loss = feat_loss + args.state_w * state_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        opt.step(); sched.step()
        with torch.no_grad():
            copy = F.mse_loss(z[:, H_CTX - 1], z[:, H_CTX])
            state_copy = tgt_n.pow(2).mean()
        for k, v in (("loss", loss), ("feat_loss", feat_loss), ("state_loss", state_loss), ("copy_mse", copy),
                     ("state_copy", state_copy), ("grad_norm", gn)):
            acc.setdefault(k, []).append(float(v.detach()))
        s1 = step + 1
        if s1 % 100 == 0:
            dt = (time.time() - tl) / 100; tl = time.time()
            row = {k: float(np.mean(v)) for k, v in acc.items()}
            row.update(step=s1, lr=float(sched.get_last_lr()[0]), s_per_step=dt, elapsed_s=time.time() - t_train,
                       batch_skill=1 - row["feat_loss"] / row["copy_mse"],
                       batch_state_skill=1 - row["state_loss"] / row["state_copy"])
            curves["train"].append(row); acc = {}
            if s1 % 500 == 0:
                print(f"step {s1:6d} feat {row['feat_loss']:.4f} (skill {row['batch_skill']:.3f}) state {row['state_loss']:.4f} "
                      f"(skill {row['batch_state_skill']:.3f}) gn {row['grad_norm']:.2f} lr {row['lr']:.2e} "
                      f"[{dt*1000:.0f} ms/step, rss {C.rss_gb():.1f} GB, {(time.time()-t0)/3600:.2f} h]", flush=True)
        if s1 % args.val_every == 0 or s1 == args.steps or s1 in snaps:
            v = dict(step=s1, elapsed_s=time.time() - t_train, **evaluate())
            curves["val"].append(v)
            c = v["clean"]
            print(f"  val@{s1}: clean skill {c['skill']:.4f} pooled {c['pooled_skill']:.4f} shuffled-a {c['skill_shuffled_action']:.4f} | "
                  f"state skill {c['state_skill']:.4f} (shuffled-a {c['state_skill_shuffled_action']:.4f}) eef {c['eef_mm']:.2f} mm "
                  f"(copy {c['eef_copy_mm']:.2f}) | shifted skill {v['shift']['skill']:.4f}", flush=True)
            curves_p.write_text(json.dumps(dict(args=args_json, curves=curves), indent=1))
        if s1 in snaps and not args.smoke:
            write_deliverable(snap_tag(s1), s1, (time.time() - t_train) / 3600)
        if (s1 % args.save_every == 0 or s1 == args.steps) and not args.smoke:
            resume_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=C.predictor_state(wm), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=s1, steps_total=args.steps, curves=curves, **(dict(cont=cont_rec) if cont else {})), resume_p)
    train_h = (time.time() - t_train) / 3600
    if not args.smoke:
        write_deliverable(tag, args.steps, train_h)
    curves_p.write_text(json.dumps(dict(args=args_json, curves=curves, train_hours=train_h), indent=1))
    rd.close()
    print(f"done: {train_h:.2f} h training, {(time.time()-t0)/3600:.2f} h total", flush=True)


if __name__ == "__main__":
    main()
