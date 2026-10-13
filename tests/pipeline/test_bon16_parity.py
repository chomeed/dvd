"""Parity test of eval/eval_bon16.py. No env, no prior: runs on decision records the evaluator dumped. NEEDS A GPU
(the recorded scores were computed under bf16 autocast on one) and prints SKIP without --dump / --build-only.

eval_bon16.py CALLS train_online.select_dice_exploration / dvd_candidate_scores, but the scorer objects it hands them
(DvdTerm + ValueEnsemble + readout + DINO-WM adapter, or the pixel critic ensemble) are assembled by the evaluator, since
train() builds them inline and has no builder function. This test proves the assembly on RECORDED decision states:

 A  REPLAY   rebuild with eval_bon16.build, re-encode the recorded frame, re-install the recorded [t-8, t] context and
             re-run the trainer's selection on the recorded 16 prior draws: the recorded state / actor input / DINO
             features / scores / chosen chunk must come back (the rollout's decision is reproducible from its inputs).
 B  INDEPENDENT  a from-scratch scorer that uses NONE of DvdTerm, ValueEnsemble, the pool modules, DinoWmAdapter,
             dvd_candidate_scores, select_dice_exploration, final_action or critics.reduce -- only the raw DinoDynamics
             predictor (load_dino_wm + the checkpoint's fine-tuned predictor tensors), the raw attention query / value /
             critic tensors applied with F.linear / F.layer_norm / F.mish, and the actor's forward (tanh(mean), no
             sampling): its scores must match the trainer path's and its argmax must be the trainer path's pick.
 C  CROSS    (--q-ckpt) a Q checkpoint OF THE SAME TASK scored on the same recorded observations and draws: trainer path
             (select_dice_exploration -> critics.reduce(bon_reduce)) vs the independent critic MLP.
 D  BUILD    (--build-only ckpt ...) build() succeeds for other tasks' checkpoints, and (with a dump) their scorer agrees
             with the independent one on a recorded context.

Checkpoints of the pre-rename research code are accepted (eval_bon16.load_payload upgrades their names).
usage: python eval/eval_bon16.py --ckpt <dvd ckpt> --out <json> --episodes 2 --dump d1.pt       (records decisions)
       python tests/pipeline/test_bon16_parity.py --dump d1.pt [--dump d2.pt] [--q-ckpt q.pt] [--build-only c1.pt c2.pt]
exit 0 iff every check passes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _pipeline_util as U

U.use_repo("eval")

import eval_bon16 as E  # noqa: E402
from online_rl_mimicgen import train_online as T  # noqa: E402
from online_rl_mimicgen.common import StateScaler, TASKS  # noqa: E402

FAILS = []
TOL_REL = 2e-4


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def mlp_ln_mish(x, sd, prefix, n_hidden, final):
    """Linear -> LayerNorm -> Mish, n_hidden times (module indices 0,1,2 | 3,4,5 | ...), then the final Linear."""
    for i in range(n_hidden):
        w, bb = sd[f"{prefix}{3 * i}.weight"], sd[f"{prefix}{3 * i}.bias"]
        x = F.linear(x, w, bb)
        ln_w, ln_b = sd[f"{prefix}{3 * i + 1}.weight"], sd[f"{prefix}{3 * i + 1}.bias"]
        x = F.mish(F.layer_norm(x, (x.shape[-1],), ln_w, ln_b))
    return F.linear(x, sd[f"{final}.weight"], sd[f"{final}.bias"])


def independent_actions(payload, actor_module, actor_in, bases, dev):
    cfg = payload["config"]
    with torch.no_grad():
        mean, _ = actor_module(actor_in.repeat(bases.shape[0], 1), bases, None)
        r = torch.tanh(mean)
        mx = float(cfg.get("residual_max_norm", 0.0))
        if mx > 0:
            nrm = r.norm(dim=-1, keepdim=True)
            r = r * (mx / nrm.clamp_min(mx))
        return (bases + float(cfg["residual_scale"]) * r).clamp(-1.0, 1.0)


def value_state_dict(payload):
    """extra.value_fn, or (a checkpoint written before any V update, e.g. 000000.pt) the value_model dir's seed files."""
    cfg, ex = payload["config"], payload["extra"]
    if "value_fn" in ex:
        return ex["value_fn"]
    import json
    root = Path(cfg["value_model"])
    seeds = json.loads((root / "config.json").read_text()).get("seeds", 3)
    seeds = range(int(seeds)) if isinstance(seeds, (int, float, str)) else tuple(seeds)
    sd = {}
    for i, seed in enumerate(seeds):
        for k, v in torch.load(root / "weights" / f"{cfg['value_variant']}_seed{seed}.pt", map_location="cpu").items():
            sd[f"nets.{i}.{k}"] = v
    return sd


def attn_query(payload):
    ex = payload["extra"]
    if "attn_pool" in ex:
        return ex["attn_pool"]["q"]
    return torch.load(Path(payload["config"]["value_model"]) / "attn_pool.pt", map_location="cpu", weights_only=False)["state_dict"]["q"]


def mqattn_state(payload, key="attn_pool"):
    ex = payload["extra"]
    if key in ex:
        return ex[key]
    assert key == "attn_pool", f"checkpoint has no extra.{key}"
    return torch.load(Path(payload["config"]["value_model"]) / "mqattn_pool.pt", map_location="cpu", weights_only=False)["state_dict"]


def independent_mqattn(payload, feats, dev, key="attn_pool"):
    """(B, V, P, d) -> pooled, from the raw q / out_proj tensors with an explicit per-(camera, query) loop; no
    MultiQueryAttnPool. Supports the layouts the campaign fitted: per-view or joint, no query norm, no shared queries,
    one flat projection (or none)."""
    sd = {k: v.to(dev) for k, v in mqattn_state(payload, key).items()}
    dim, k_q, per_view, n_views, out_dim, qnorm, share, _patches = (int(x) for x in sd["meta"].tolist())
    assert not qnorm and not share and "hidden_meta" not in sd, "independent mqattn: plain layout only"
    q = sd["q"].float()
    feats = feats.float()
    b, v, p, d = feats.shape
    assert (v, d) == (n_views, dim)
    parts = []
    if per_view:
        for vi in range(v):
            for ki in range(k_q):
                w = torch.softmax(feats[:, vi] @ q[vi, ki] / dim ** 0.5, dim=1)          # (B, P) over this camera
                parts.append((w.unsqueeze(-1) * feats[:, vi]).sum(dim=1))
    else:
        t = feats.reshape(b, v * p, d)
        for ki in range(k_q):
            w = torch.softmax(t @ q[ki] / dim ** 0.5, dim=1)
            parts.append((w.unsqueeze(-1) * t).sum(dim=1))
    z = torch.cat(parts, dim=-1)
    return F.linear(z, sd["out_proj.weight"].float(), sd["out_proj.bias"].float()) if out_dim else z


def independent_pool(payload, feats, dev, key="attn_pool"):
    """pool_features for the three observation/value pools this test knows: attnpool, mqattn, per-view mean."""
    mode = payload["config"].get("dino_pool", "mean")
    if mode == "attnpool":
        q = attn_query(payload).to(dev).float()
        t = feats.float().flatten(1, 2)
        w = torch.softmax((t @ q.T) / q.shape[-1] ** 0.5, dim=1)
        return (t * w).sum(dim=1)
    if mode == "mqattn":
        return independent_mqattn(payload, feats, dev, key)
    assert mode == "mean", f"independent pool: {mode!r} not implemented"
    return feats.float().mean(dim=2).flatten(1)


def independent_dvd_scores(payload, wm, ctx, state, actions, dev):
    cfg, ex = payload["config"], payload["extra"]
    scaler = StateScaler(Path(TASKS[cfg["task"]]["dataset_root"]))
    assert cfg.get("state_repr", "absolute") == "absolute", "independent scorer: absolute state_repr only"
    low = torch.as_tensor(np.asarray(scaler.low, np.float32), device=dev)
    span = torch.as_tensor(np.asarray(scaler.scale, np.float32), device=dev)
    n = actions.shape[0]
    raw = ((state + 1.0) * 0.5 * span + low).repeat(n, 1)
    z = ctx.expand(n, -1, -1, -1, -1)[:, -int(wm.history):]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.get("dino_bf16", True))):
        pred = wm.predict_from_features(z, actions.view(n, 8, 7), raw[:, :9])
    pred = pred.float()
    if wm.residual and cfg.get("dvd_residual_anchor", False):
        pred = z[:, -1] + pred
    pooled = independent_pool(payload, pred, dev)
    prop = (wm.last_state_pred[:, :9].float() if cfg.get("dvd_proprio_source", "current") == "predicted" else raw[:, :9])
    x = torch.cat([pooled, prop], dim=-1)
    sd = {k: v.to(dev).float() for k, v in value_state_dict(payload).items()}
    seeds = sorted({int(k.split(".")[1]) for k in sd if k.startswith("nets.")})
    vals = torch.stack([mlp_ln_mish(x, sd, f"nets.{i}.trunk.", 3, f"nets.{i}.v_head").squeeze(-1) for i in seeds])
    agg = cfg.get("value_aggregate", "mean")
    assert agg == "mean", f"independent scorer implements value_aggregate mean only (got {agg})"
    return (float(cfg["gamma"]) ** int(cfg["action_steps"])) * vals.mean(0), vals


def independent_q_scores(payload, feat, actions, dev):
    cfg = payload["config"]
    sd = {k: v.to(dev).float() for k, v in payload["critics"].items()}
    heads = sorted({int(k.split(".")[1]) for k in sd})
    x = torch.cat([feat.repeat(actions.shape[0], 1), actions], dim=-1)
    nl = int(cfg.get("critic_layers", 3))
    # PixelCriticEnsemble: nets.{i}.{j}; CriticEnsemble (--dino-obs Q arms): q.{i}.net.{j}
    pre = (lambda i: f"q.{i}.net.") if next(iter(sd)).startswith("q.") else (lambda i: f"nets.{i}.")
    vals = torch.stack([mlp_ln_mish(x, sd, pre(i), nl, f"{pre(i)}{3 * nl}").squeeze(-1) for i in heads])
    mode = cfg.get("bon_reduce", "min")
    red = {"min": vals.min(0).values, "max": vals.max(0).values, "mean": vals.mean(0)}[mode]
    return red, vals


def load_raw_wm(payload, dev):
    from online_rl_mimicgen.dino_wm import load_dino_wm
    cfg, ex = payload["config"], payload["extra"]
    assert not cfg.get("dino_wm_arch"), "independent WM loader: default arch only"
    wm = load_dino_wm(cfg["dino_wm_ckpt"], device=dev, state_head=bool(cfg.get("dino_wm_state_head", False)))
    sd = ex.get("dvd_dynamics")
    if sd is not None:
        sd = {k[len("wm."):]: v for k, v in sd.items() if k.startswith("wm.")}
        miss = wm.load_state_dict(sd, strict=False)
        assert not [k for k in miss.missing_keys if not k.startswith("dino.")] and not miss.unexpected_keys, miss
    return wm.eval()


def rel_close(a, b):
    d = float((a - b).abs().max())
    return d <= TOL_REL * max(1.0, float(b.abs().max())), d


def replay_one(bundle, rec, dev, label):
    """A: re-run the trainer path on one recorded decision. Returns (score, members, chosen) from the trainer path."""
    obs = {"agent_pos": rec["agent_pos"], "pixels": rec["pixels"]}
    state = torch.as_tensor(bundle["scaler"](obs["agent_pos"]), device=dev).unsqueeze(0)
    bases = rec["bases"].to(dev)
    hist = bundle["dino_history"]
    if hist is not None:
        from online_rl_mimicgen import dino_dvd
        cur = dino_dvd.encode_frames(hist.adapter, T.pixel_preprocess(obs["pixels"])[None])
        ok, d = rel_close(cur.cpu(), rec["ctx_cur"])
        check(f"{label} A DINO re-encode of the recorded frame == recorded context t", ok, f"max|d|={d:.2e}")
        hist.prev, hist.cur = rec["ctx_prev"].to(dev), rec["ctx_cur"].to(dev)
    # AFTER the context is installed: --dino-obs pools dino_history.cur (for pixel arms obs_parts is (None, None)).
    state_in = T.net_state(obs, state, bundle["encoder"], *E.obs_parts(bundle, obs))
    if hist is not None:
        with torch.inference_mode():
            hist.apply(bases.shape[0])
    with torch.inference_mode():
        chosen = E.select(bundle, state_in, state, bases)
    return state, state_in, bases, chosen, E.REC["score"].float().cpu(), E.REC.get("members")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", action="append", default=[], help="decision records written by eval_bon16.py --dump")
    ap.add_argument("--q-ckpt", default="", help="a Q checkpoint of the dumps' task")
    ap.add_argument("--build-only", nargs="*", default=[])
    args = ap.parse_args()
    if not args.dump and not args.build_only:
        print("SKIP test_bon16_parity: no --dump / --build-only given (GPU test on recorded decisions; see the docstring)")
        return
    if not torch.cuda.is_available():
        print("SKIP test_bon16_parity: no CUDA device (the recorded scores are bf16 GPU values)")
        return
    dev = torch.device("cuda")
    q_payload = q_bundle = None
    if args.q_ckpt:
        q_payload = E.load_payload(args.q_ckpt)[0]
        q_bundle = E.build(q_payload, dev)
        assert q_bundle["critics"] is not None, "--q-ckpt must be a bon_source q checkpoint"
    for dump_path in args.dump:
        dump = torch.load(dump_path, map_location="cpu", weights_only=False)
        ckpt = dump["ckpt"]
        print(f"\n=== {dump_path}: {len(dump['dumps'])} recorded decisions of {ckpt}", flush=True)
        payload = E.load_payload(ckpt)[0]
        bundle = E.build(payload, dev)
        wm = load_raw_wm(payload, dev) if bundle["cfg"].bon_source == "dvd" else None
        n_pick = n_top = 0
        for rec in dump["dumps"]:
            label = f"ep{rec['episode']}/dec{rec['decision']}"
            state, state_in, bases, chosen, score, members = replay_one(bundle, rec, dev, label)
            check(f"{label} A scaled state == recorded", torch.equal(state.float().cpu(), rec["state"]))
            ok, d = rel_close(state_in.float().cpu(), rec["state_in"])
            check(f"{label} A encoder features == recorded", ok, f"max|d|={d:.2e}")
            ok, d = rel_close(score, rec["score"])
            check(f"{label} A trainer-path scores == recorded rollout scores", ok, f"max|d|={d:.2e}")
            check(f"{label} A trainer-path chosen chunk == executed chunk", torch.equal(chosen.float().cpu(), rec["chosen"]),
                  f"(recorded idx {rec['rec']['idx']})")
            # ---- B: independent scorer on the recorded inputs ----
            if payload["config"].get("dino_obs"):
                # --dino-obs observation from raw tensors: pool(recorded DINO t) ++ raw proprio[:9]
                _okey = "obs_pool" if "obs_pool" in payload["extra"] else "attn_pool"   # separate critic-trained pool
                obs_ind = torch.cat([independent_pool(payload, rec["ctx_cur"].to(dev), dev, _okey),
                                     torch.as_tensor(np.asarray(rec["agent_pos"][:9], np.float32), device=dev)[None]], dim=-1)
                ok, d = rel_close(obs_ind.cpu(), rec["state_in"])
                check(f"{label} B independent dino-obs pool(DINO t) ++ raw proprio == recorded actor input", ok,
                      f"max|d|={d:.2e} width {obs_ind.shape[-1]}")
            actions_ind = independent_actions(payload, bundle["actor"], state_in, bases, dev)
            ok, d = rel_close(actions_ind.cpu(), rec["actions"])
            check(f"{label} B independent tanh(actor mean) + projection + clamp == trainer candidates", ok, f"max|d|={d:.2e}")
            if wm is not None:
                ctx = torch.stack([rec["ctx_prev"], rec["ctx_cur"]], dim=1).to(dev)     # (1, 2, V, P, d)
                s_ind, v_ind = independent_dvd_scores(payload, wm, ctx, state, actions_ind, dev)
            else:
                s_ind, v_ind = independent_q_scores(payload, state_in, actions_ind, dev)
            ok, d = rel_close(s_ind.cpu(), score)
            gap = float(score.topk(2).values[0] - score.topk(2).values[1])
            check(f"{label} B independent scores == trainer-path scores", ok, f"max|d|={d:.2e} (top gap {gap:.2e})")
            if members is not None:
                okm, dm = rel_close(v_ind.cpu() * (float(payload['config']['gamma']) ** 8 if wm is not None else 1.0),
                                    members.float().cpu())
                check(f"{label} B independent per-net values == trainer-path members", okm, f"max|d|={dm:.2e}")
            i_ind, i_tr = int(s_ind.argmax()), rec["rec"]["idx"]
            n_pick += int(i_ind == i_tr)
            n_top += 1
            check(f"{label} B independent argmax == executed candidate", i_ind == i_tr, f"({i_ind} vs {i_tr})")
            check(f"{label} B executed chunk == independent candidate[argmax]",
                  bool(torch.allclose(actions_ind[i_ind].cpu(), rec["chosen"][0], atol=1e-6)))
            # ---- C: the pixq checkpoint on the same observation and draws ----
            if q_bundle is not None:
                obs = {"agent_pos": rec["agent_pos"], "pixels": rec["pixels"]}
                qs = torch.as_tensor(q_bundle["scaler"](obs["agent_pos"]), device=dev).unsqueeze(0)
                qf = T.net_state(obs, qs, q_bundle["encoder"])
                with torch.inference_mode():
                    q_chosen = E.select(q_bundle, qf, qs, bases)
                q_score, q_actions = E.REC["score"].float().cpu(), E.REC["actions"]
                qa_ind = independent_actions(q_payload, q_bundle["actor"], qf, bases, dev)
                q_ind, _ = independent_q_scores(q_payload, qf, qa_ind, dev)
                ok, d = rel_close(q_ind.cpu(), q_score)
                check(f"{label} C pixq independent critic reduce == trainer path", ok, f"max|d|={d:.2e}")
                qi = int(q_ind.argmax())
                check(f"{label} C pixq selection == independent argmax",
                      bool(torch.allclose(q_chosen[0].float().cpu(), qa_ind[qi].cpu(), atol=1e-6)), f"(idx {qi})")
        print(f"  selections reproduced independently: {n_pick}/{n_top}", flush=True)
        del bundle, wm, payload
        torch.cuda.empty_cache()
    for ck in args.build_only:
        print(f"\n=== build-only {ck}", flush=True)
        name = f"{Path(ck).parent.parent.name}/{Path(ck).name}"
        try:
            p = E.load_payload(ck)[0]
            if "encoder" not in p.get("extra", {}):
                # 000000.pt (written before the first update) carries no extra: inject a FRESH encoder so the task's
                # scaler / V dir / WM / proprio-head / DvdTerm wiring can still be built and scored. Wiring only.
                from online_rl_mimicgen.pixel import PixelEncoder
                p.setdefault("extra", {})["encoder"] = PixelEncoder(out_dim=p["config"]["pixel_feature_dim"],
                                                                    shift_pad=p["config"]["pixel_shift_pad"]).state_dict()
                print("  (no extra in this checkpoint: fresh encoder injected; V from value_model dir, f from dino_wm_ckpt)")
            bb = E.build(p, dev)
            check(f"D build {name}", True, str(bb["info"].get("scorer")))
            if args.dump and bb["dvd_term"] is not None:
                # E SYNTHETIC WIRING PARITY: this task's scorer on a recorded decision's context features and draws,
                # at the mid-span state (scaled 0). Not a rollout state -- it checks the two scoring paths agree.
                rec = torch.load(args.dump[0], map_location="cpu", weights_only=False)["dumps"][-1]
                sdim = int(TASKS[bb["cfg"].task]["state_dim"])
                state = torch.zeros(1, sdim, device=dev)
                obs = {"agent_pos": rec["agent_pos"], "pixels": rec["pixels"]}
                feat = bb["encoder"](torch.as_tensor(T.pixel_preprocess(obs["pixels"]), device=dev)[None],
                                     state[:, :9]).detach()
                bases = rec["bases"].to(dev)
                hist = bb["dino_history"]
                hist.prev, hist.cur = rec["ctx_prev"].to(dev), rec["ctx_cur"].to(dev)
                with torch.inference_mode():
                    hist.apply(bases.shape[0])
                    chosen = E.select(bb, feat, state, bases)
                score = E.REC["score"].float().cpu()
                wm = load_raw_wm(p, dev)
                acts = independent_actions(p, bb["actor"], feat, bases, dev)
                ctx = torch.stack([rec["ctx_prev"], rec["ctx_cur"]], dim=1).to(dev)
                s_ind, _ = independent_dvd_scores(p, wm, ctx, state, acts, dev)
                ok, d = rel_close(s_ind.cpu(), score)
                check(f"E {name} synthetic: independent scores == trainer path", ok, f"max|d|={d:.2e}")
                check(f"E {name} synthetic: selection == independent argmax",
                      bool(torch.allclose(chosen[0].float().cpu(), acts[int(s_ind.argmax())].cpu(), atol=1e-6)))
                del wm
            del bb, p
            torch.cuda.empty_cache()
        except BaseException as exc:          # SystemExit from a refusal counts as a failure here
            check(f"D build {name}", False, repr(exc)[:300])
    print(f"\n{'ALL PARITY CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECKS FAILED: ' + '; '.join(FAILS[:10])}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
