"""Best-of-16 evaluation of a DICE-RL checkpoint with the trainer's OWN collection-time selection.

THE PROTOCOL of every DICE-RL number (DVD and Q arms): 100 episodes, env seeds 10000 + i, at every decision 16
independent draws of the frozen prior, each corrected by the deterministic residual actor, ranked by the run's own
critic, the best one executed for 8 primitive steps. In-loop evaluation is off in the campaign; every saved checkpoint
is evaluated by this script.

Every decision does what train_online's collection loop does when best-of-N fires:

    state    = scaler(agent_pos)                          scaled env state
    push     FrameHistory (DINO arms: context [t-8, t])   BEFORE net_state, the trainer's order
    actor_in = net_state(obs, state, encoder)             checkpoint IMPALA encoder (--dino-obs: the trainer's
                                                          _dino_obs_parts, pool(DINO t) ++ raw proprio)
    bases    = prior.action_chunk_samples(obs, 16)        bon_samples, or 16 independent prior draws
    apply    FrameHistory.apply(16)                       DINO arms
    chosen   = train_online.select_dice_exploration(actor, critics, actor_in, bases, residual_scale, dvd_term,
                                                    bon_reduce, dvd_state=state, bon_select, bon_quantile, None,
                                                    residual_max_norm)

select_dice_exploration is CALLED, not copied. The one difference from collection is the actor: it is wrapped so every
actor.sample() runs deterministic=True (bit-identical to the trainer's call for the deterministic_residual, latent_dim 0
arms of the campaign; for a stochastic actor it removes the exploration noise).

Scorers are rebuilt with the trainer's own classes and the arguments train() passes them:
  bon_source dvd : gamma^8 V(f(ctx, s, a)) through train_online.dvd_candidate_scores -- the world model (dino_dvd.build
                   + checkpoint extra.dvd_dynamics), ValueEnsemble + extra.value_fn, the V directory's readout
                   (mqattn_pool.pt / attn_pool.pt, + extra.attn_pool).
  bon_source q   : PixelCriticEnsemble + checkpoint critics, reduced with bon_reduce.
Modules keep the train/eval mode the trainer leaves them in; no module in training mode carries dropout (asserted).
dvd_candidate_scores is wrapped IN THIS PROCESS ONLY to record candidate scores for the gap statistics; the wrapper
returns the original's output.

Episodes: env seed seed_start + i; torch / numpy / random are seeded with the same number at episode start, so an episode
reproduces on its own (which makes the per-episode resume through <out stem>.partial.json exact). Success = any chunk
whose info["is_success"] fired. As everywhere in this code base, the frozen prior is conditioned on the task id string
and on a duplicated current frame (see eval/eval_prior.py); that is the as-run behaviour of every reference number.

Checkpoints of the pre-rename research code (the reference runs) load through online_rl_mimicgen.legacy, the one
module that knows the old names. The asset paths stored in a checkpoint are those of the training machine: a path that does
not exist here is looked up under the scripts/env.sh roots ($CKPT_ROOT/priors, $CKPT_ROOT/value, $CKPT_ROOT/wm), and
--base-checkpoint / --value-model / --dino-wm-ckpt / --dataset-root override them explicitly (an override that does not
exist is an error).

The run's configuration is read from the checkpoint itself (payload["config"]); no config.json is opened, so there is
no --config. --ckpt is resolved through symlinks, as in the research code, and the run directory (the "run" field of the
result, the optional code_version.json) is two levels above the REAL file: keep checkpoints as real files inside
<run>/checkpoints/ (fetch a run with local_dir, not out of the Hugging Face hub cache, whose snapshot files are links
into blobs/), or the result is labelled with the name of wherever the link points. A note is printed when that happens.

usage: python eval/eval_bon16.py --ckpt <run>/checkpoints/<step>.pt --out <json> [--episodes 100] [--seed-start 10000]
       [--base-checkpoint <prior>/checkpoints/<step>/pretrained_model] [--value-model <V dir>]
       [--dino-wm-ckpt <wm .pt>] [--dataset-root <demo dataset>]
       [--dump <file.pt> --dump-every 7 --dump-max 8]      decision records for tests/pipeline/test_bon16_parity.py
One GPU: CUDA_VISIBLE_DEVICES=g MUJOCO_EGL_DEVICE_ID=g. Compare two result files with eval/compare_evals.py.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1] / "rl"
for _p in (str(R.parent / "envs/lerobot_env_mimicgen/src"), str(R)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)                    # this repository's packages, whatever PYTHONPATH says

import online_rl_mimicgen  # noqa: E402

assert R in Path(online_rl_mimicgen.__file__).resolve().parents, (
    f"online_rl_mimicgen resolved to {online_rl_mimicgen.__file__}, not to {R}: another copy was imported first")

from online_rl_mimicgen import legacy  # noqa: E402
from online_rl_mimicgen import train_online as T  # noqa: E402
from online_rl_mimicgen.common import (FrozenPolicyPrior, GOAL_RELATIVE_MAPS, GoalRelativeScaler,  # noqa: E402
                                       CriticEnsemble, ResidualActor, StateScaler, TASKS, TrainConfig)
from online_rl_mimicgen.pixel import PixelCriticEnsemble, PixelEncoder  # noqa: E402
from online_rl_mimicgen.value_model import ValueEnsemble  # noqa: E402

CODE_FILES = ("train_online.py", "common.py", "robometer_distill.py", "pixel.py", "dino_dvd.py", "value_model.py")
STAT_KEYS = ("rank", "top_gap", "max_minus_median", "best_minus_worst", "member_std", "top_gap_over_std", "meaningful",
             "chosen_minus_mean", "chosen_minus_draw0", "chosen_score")
# checkpoint config field holding an asset path this evaluator opens -> the argparse dest that overrides it
ASSET_ARGS = {"base_checkpoint": "base_checkpoint", "value_model": "value_model", "dino_wm_ckpt": "dino_wm_ckpt"}

# ---- in-process recorders (no source file is touched) -----------------------------------------------------------------
REC: dict = {}
_ORIG_DVD_CANDIDATE_SCORES = T.dvd_candidate_scores


def _recording_dvd_candidate_scores(dvd_term, state, action, frozen_value_fn=None):
    out = _ORIG_DVD_CANDIDATE_SCORES(dvd_term, state, action, frozen_value_fn)
    REC["actions"], REC["score"], REC["members"] = action, out[0], out[2]
    return out


T.dvd_candidate_scores = _recording_dvd_candidate_scores     # select_dice_exploration resolves the name at call time


class DeterministicActor:
    """Forwards to the checkpoint actor; sample() always deterministic=True (no exploration noise)."""

    def __init__(self, actor):
        self.actor = actor

    def sample(self, state, base_action, deterministic=True):
        return self.actor.sample(state, base_action, deterministic=True)

    def __getattr__(self, name):
        return getattr(self.__dict__["actor"], name)


class RecordingCritics:
    """PixelCriticEnsemble view for bon_source q: reduce() is the ensemble's own; members/actions are recorded."""

    def __init__(self, critics):
        self.critics = critics

    def reduce(self, feature, action, mode):
        out = self.critics.reduce(feature, action, mode)
        REC["actions"], REC["score"] = action, out.squeeze(-1)
        REC["members"] = self.critics(feature, action).squeeze(-1)
        return out

    def __getattr__(self, name):
        return getattr(self.__dict__["critics"], name)


def _no_dropout(*modules):
    for mod in modules:
        if mod is None:
            continue
        for m in mod.modules():
            if not m.training:
                continue
            p = m.p if isinstance(m, torch.nn.modules.dropout._DropoutNd) else getattr(m, "dropout", 0.0)
            assert not (isinstance(p, float) and p > 0.0), f"{type(m).__name__} in training mode has dropout {p}"


def code_hashes():
    here = R / "online_rl_mimicgen"
    return {f: hashlib.md5((here / f).read_bytes()).hexdigest() for f in CODE_FILES if (here / f).exists()}


def load_payload(ckpt, overrides=None, verbose=True):
    """A checkpoint payload ready for build(): legacy names upgraded, asset paths resolved on this machine.

    overrides: {config field: path} for base_checkpoint / value_model / dino_wm_ckpt (None = no override). Only the
    fields this evaluator opens for the checkpoint's bon_source are resolved, so an asset the run recorded but the
    evaluation never reads cannot fail it. Returns (payload, was_legacy, resolved paths)."""
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    was_legacy = legacy.is_legacy(raw["config"]) or any(k in raw.get("extra", {}) for k in legacy.EXTRA_RENAMES)
    payload = legacy.upgrade_checkpoint(raw)
    cfg = payload["config"]
    needs_dvd = cfg.get("bon_source") == "dvd" or bool(cfg.get("dino_obs"))
    fields = ("base_checkpoint",) + (("value_model", "dino_wm_ckpt") if needs_dvd else ())
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    unused = sorted(set(overrides) - set(fields))
    if unused:
        raise SystemExit(f"[bon16] override(s) {unused} given, but a bon_source {cfg.get('bon_source')!r} checkpoint "
                         f"does not read them")
    if not cfg.get("base_checkpoint") and "base_checkpoint" not in overrides:
        raise SystemExit("[bon16] the checkpoint config has no base_checkpoint; pass --base-checkpoint")
    payload["config"] = legacy.remap_config_assets(cfg, overrides, fields=fields)
    resolved = {k: payload["config"].get(k) for k in fields}
    if verbose and was_legacy:
        print("[bon16] checkpoint of the pre-rename code: names upgraded by online_rl_mimicgen.legacy", flush=True)
    return payload, was_legacy, resolved


def build(payload, dev, verbose=True, dataset_root=None, allow_dropped=False):
    """Everything select_dice_exploration needs, from one checkpoint payload (no prior, no env; the parity test imports
    this). payload must carry this repository's names (load_payload / legacy.upgrade_checkpoint).
    Returns a dict: cfg, scaler, encoder, actor_encoder, actor, det_actor, dvd_term, dino_history, critics, info."""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    dropped = sorted(k for k in payload["config"] if k not in fields)
    if dropped and not allow_dropped:
        # A key TrainConfig does not know is a setting this code cannot honour. The research evaluator dropped such
        # keys silently; here that takes an explicit flag, and the keys are recorded in the result either way.
        raise SystemExit(f"[bon16] the checkpoint config has key(s) this TrainConfig does not know: {dropped}. They would "
                         f"be ignored; pass --allow-dropped-config-keys if that is intended")
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in fields})
    ex = payload.get("extra", {})
    g = lambda k, d=None: getattr(cfg, k, d)   # noqa: E731
    # -- refuse what this evaluator would otherwise silently mis-rebuild --
    refuse = []
    if cfg.method != "dice_rl":
        refuse.append(f"method {cfg.method} (collection best-of-N is dice_rl only)")
    if g("latent_obs", False) or g("seg_obs", False) or not (cfg.pixel_obs or cfg.dino_obs):
        refuse.append("observation is neither --pixel-obs nor --dino-obs (latent_obs/seg_obs/state arms are not wired here)")
    if cfg.dino_obs:
        # --dino-obs: the actor reads V's own input, pool(DINO t) ++ raw proprio, built by the trainer's _dino_obs_parts
        # on the rebuilt dvd_term. Only the ONE-shared-pool layout is rebuilt, so every layout that pools the
        # observation with a module other than dvd_term._pool is refused. With bon_source q the DVD stack is still
        # built (it hosts the DINO encoder, the frame history and the pool) but ranking is critics.reduce.
        if cfg.bon_source not in ("dvd", "q"):
            refuse.append(f"dino_obs with bon_source {cfg.bon_source}")
        if cfg.dvd_dynamics != "dino_wm":
            refuse.append(f"dino_obs with dvd_dynamics {cfg.dvd_dynamics} (the observation is DINO-WM features)")
        if g("dino_obs_pool_separate", False) and cfg.dino_pool != "mqattn":
            refuse.append("dino_obs_pool_separate on a non-mqattn pool (only the mqattn obs_pool is rebuilt)")
        if g("dino_obs_pool_separate", False) and "obs_pool" not in ex:
            refuse.append("dino_obs_pool_separate but the checkpoint has no extra.obs_pool (e.g. 000000.pt)")
        if g("dvd_actor_pool", False):
            refuse.append("dvd_actor_pool (the actor's pool copy / dvd_term._value_pool is not rebuilt)")
        if cfg.dino_pool not in ("mean", "attnpool", "mqattn") or g("value_pool", ""):
            refuse.append(f"dino_obs with dino_pool {cfg.dino_pool!r} / value_pool {g('value_pool', '')!r}")
        if g("actor_encoder_finetune", False):
            refuse.append("dino_obs with actor_encoder_finetune (there is no pixel encoder to copy)")
    if g("bon_axis", "prior") != "prior":
        refuse.append(f"bon_axis {g('bon_axis')} (the request is best-of-16 over prior draws)")
    if g("pose_from_seghead", False) or g("state_noise_mm", 0.0) > 0.0:
        refuse.append("pose_from_seghead / state_noise_mm corrupt the rollout state")
    if g("bon_dvd_until", 0) > 0:
        refuse.append("bon_dvd_until > 0 (scorer hand-off V(f) -> Q mid-run)")
    if g("bon_frozen_v", "off") != "off":
        refuse.append("bon_frozen_v (frozen-initial-V co-rank) is not rebuilt here")
    if int(g("bon_lookahead", 1) or 1) != 1:
        refuse.append(f"bon_lookahead {g('bon_lookahead')} (only the depth-1 scorer gamma^8 V(f(ctx, s, a)) is rebuilt here)")
    if cfg.bon_source == "dvd":
        if cfg.dvd_dynamics != "dino_wm":
            refuse.append(f"dvd_dynamics {cfg.dvd_dynamics} (only the DINO-WM feature path is rebuilt)")
        if g("dvd_on_encoder", False) or g("dvd_encoder_learned", False):
            refuse.append("dvd_on_encoder / dvd_encoder_learned")
        if g("value_td_model", ""):
            refuse.append("value_td_model (DualValue)")
        if g("value_residual", False):
            refuse.append("value_residual")
        if g("value_random_frozen", False):
            refuse.append("value_random_frozen (its random V is never checkpointed)")
        if not cfg.value_model:
            refuse.append("bon_source dvd without value_model")
        if "value_fn" not in ex and g("value_scratch", False) and not g("value_init"):
            refuse.append("value_scratch run without extra.value_fn")
    elif cfg.bon_source == "q":
        if not cfg.train_critic:
            refuse.append("bon_source q with --no-train-critic (collection never ranks with an untrained critic)")
        if cfg.value_intervention == "dueling":
            refuse.append("value_intervention dueling (DuelingCritic needs V)")
    else:
        refuse.append(f"bon_source {cfg.bon_source}")
    if not cfg.dino_obs and "encoder" not in ex:
        refuse.append("no extra.encoder in the payload (a step-0 checkpoint is written before training and holds no "
                      "trained encoder, V or world model; score a later checkpoint)")
    if refuse:
        raise SystemExit("[bon16] unsupported checkpoint: " + "; ".join(refuse))

    task_cfg = TASKS[cfg.task]
    state_dim = int(task_cfg["state_dim"])
    chunk_dim = cfg.action_steps * 7
    dataset_root = Path(dataset_root) if dataset_root else Path(task_cfg["dataset_root"])
    scaler = (GoalRelativeScaler(dataset_root, cfg.task) if cfg.state_repr == "goal_relative"
              else StateScaler(dataset_root))
    if cfg.dino_obs:
        # No pixel encoder exists on a --dino-obs arm; the actor is sized as train() sizes it
        # (critic_state_dim = pooled_dim(dino_pool) + PIXEL_PROPRIO), and the strict actor load below checks it.
        from online_rl_mimicgen.dino_wm import pooled_dim
        encoder = None
        actor_in_dim = pooled_dim(cfg.dino_pool) + T.PIXEL_PROPRIO
    else:
        encoder = PixelEncoder(out_dim=cfg.pixel_feature_dim, shift_pad=cfg.pixel_shift_pad).to(dev)
        encoder.load_state_dict(ex["encoder"], strict=True)
        encoder.eval()
        actor_in_dim = encoder.out_dim
    actor_encoder = None
    if g("actor_encoder_finetune", False):
        actor_encoder = PixelEncoder(out_dim=cfg.pixel_feature_dim, shift_pad=cfg.pixel_shift_pad).to(dev)
        actor_encoder.load_state_dict(ex["actor_encoder"], strict=True)
        actor_encoder.eval()
    actor = ResidualActor(
        actor_in_dim, chunk_dim, layernorm=cfg.method == "dice_rl", hidden=(cfg.actor_hidden,) * 3,
        initial_log_std=cfg.resolved_initial_log_std(), log_std_min=cfg.log_std_min,
        stochastic=not cfg.deterministic_residual, latent_dim=cfg.latent_dim, head=cfg.residual_head,
        action_steps=cfg.action_steps, conv_kernel=cfg.residual_conv_kernel, conv_channels=cfg.residual_conv_channels,
        conv_broadcast=cfg.residual_conv_broadcast).to(dev)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    info = dict(observation="dino_obs" if cfg.dino_obs else "pixel_obs", actor_in_dim=int(actor_in_dim),
                bon_source=cfg.bon_source, bon_select=g("bon_select", "max"), bon_quantile=g("bon_quantile", 0.75),
                n_candidates=int(g("bon_samples", 0) or 16), residual_scale=cfg.residual_scale,
                residual_max_norm=cfg.residual_max_norm, actor_stochastic=not cfg.deterministic_residual,
                actor_latent_dim=cfg.latent_dim, config_keys_not_in_trainconfig=dropped,
                dataset_root=str(dataset_root))
    dvd_term = dino_history = critics = None
    loaded = []
    if cfg.bon_source == "dvd" or cfg.dino_obs:
        from online_rl_mimicgen.dvd_term import DvdTerm
        from online_rl_mimicgen import dino_dvd as dino
        from online_rl_mimicgen.dino_wm_any import is_sweep_checkpoint, load_any
        if cfg.dino_wm_ckpt and is_sweep_checkpoint(cfg.dino_wm_ckpt):
            dino.assert_frame_size()
            adapter = load_any(cfg.dino_wm_ckpt, device=dev, bf16=cfg.dino_bf16)
        else:
            adapter = dino.build(cfg.dino_wm_ckpt or None, device=dev, bf16=cfg.dino_bf16, action_dim=7,
                                 arch=cfg.dino_wm_arch, flow_steps=cfg.flow_steps or None,
                                 flow_inverse_iters=cfg.flow_inverse_iters or None,
                                 finetune_dino_blocks=cfg.dvd_finetune_dino_blocks,
                                 dvd_checkpoint=cfg.dvd_checkpoint, state_head=g("dino_wm_state_head", False))
        adapter.mse_anchor = bool(cfg.dvd_residual_anchor)
        value_fn = ValueEnsemble(cfg.value_model, cfg.value_variant, scaler, state_dim=state_dim, device=dev,
                                 finetune=cfg.value_finetune, scratch=g("value_scratch", False),
                                 aggregate=cfg.value_aggregate, lcb_lambda=cfg.value_lcb_lambda)
        if "value_fn" in ex:
            value_fn.load_state_dict(ex["value_fn"]); loaded.append("value_fn")
        elif g("value_init"):
            value_fn.load_state_dict(legacy.load_checkpoint(cfg.value_init)["extra"]["value_fn"])
            loaded.append("value_fn<-value_init (frozen-V run)")
        else:
            loaded.append("value_fn<-value_model dir (frozen-V run)")
        attn_pool = sargmax_pool = None
        if cfg.dino_pool == "attnpool":
            from online_rl_mimicgen.dino_wm import AttnPool
            blob = torch.load(Path(cfg.value_model) / "attn_pool.pt", map_location="cpu", weights_only=False)
            attn_pool = AttnPool(dim=blob.get("dim", 384)).to(dev)
            attn_pool.load_state_dict(blob["state_dict"])
            if g("attn_pool_freeze", False):
                attn_pool.requires_grad_(False)
            if "attn_pool" in ex:
                attn_pool.load_state_dict(ex["attn_pool"]); loaded.append("attn_pool")
        elif cfg.dino_pool == "mqattn":
            # The multi-query attention readout ("pv4"). Its SHAPE comes from the blob's own meta (K, per_view,
            # out_dim), exactly as in the trainer, so one branch loads every fitted variant. Bound to the same
            # `attn_pool` name, so the checkpoint load, dvd_term._pool and _no_dropout reach it with no second path.
            from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool
            blob = torch.load(Path(cfg.value_model) / "mqattn_pool.pt", map_location="cpu", weights_only=False)
            attn_pool = MultiQueryAttnPool.from_blob(blob, device=dev)
            if g("attn_pool_freeze", False):
                attn_pool.requires_grad_(False)
            if "attn_pool" in ex:
                attn_pool.load_state_dict(ex["attn_pool"]); loaded.append("attn_pool (mqattn)")
        if g("value_pool", "") == "sargmax":
            from online_rl_mimicgen.dino_wm import SoftArgmaxPool
            sblob = torch.load(Path(cfg.value_model) / "soft_argmax.pt", map_location="cpu", weights_only=False)
            sargmax_pool = SoftArgmaxPool(dim=sblob.get("dim", 384), k=sblob.get("k", 16), hid=sblob.get("hid", 128)).to(dev)
            sargmax_pool.load_state_dict(sblob["state_dict"])
            sargmax_pool.requires_grad_(False).eval()
            if "sargmax_pool" in ex:
                sargmax_pool.load_state_dict(ex["sargmax_pool"]); loaded.append("sargmax_pool")
        proprio_dim = dino.resolve_proprio_dim(value_fn, g("value_pool", "") or cfg.dino_pool)
        dvd_term = DvdTerm(
            value_fn, scaler, chunk_dim, dev, discount=cfg.gamma, horizon=cfg.action_steps, chunks=cfg.dvd_chunks,
            model_tag=cfg.dvd_dynamics, finetune_dynamics=cfg.dvd_finetune_dynamics,
            value_aware_weight=cfg.dvd_value_aware_weight, identity_scaling=False, dynamics_lr=cfg.dvd_dynamics_lr,
            dynamics_scratch=cfg.dvd_dynamics_scratch, dynamics=adapter, feature_space=True, pool_mode=cfg.dino_pool,
            value_pool_mode=(g("value_pool", "") or None), proprio_dim=proprio_dim, static_grad_block=cfg.dvd_static_grad,
            goal_slice=(GOAL_RELATIVE_MAPS[cfg.task]["goal_pos"] if cfg.task in GOAL_RELATIVE_MAPS else None),
            fixed_scale=cfg.dvd_fixed_scale, proprio_source=g("dvd_proprio_source", "current"),
            state_loss_weight=g("dvd_state_loss_weight", 1.0))
        if attn_pool is not None:
            dvd_term._pool = attn_pool
        if sargmax_pool is not None:
            dvd_term._pool = sargmax_pool
        if "dvd_dynamics" in ex:
            miss = dvd_term.dynamics.load_state_dict(ex["dvd_dynamics"], strict=False)
            bad = [k for k in miss.missing_keys if not k.startswith("wm.dino.")]
            assert not bad and not miss.unexpected_keys, f"dvd_dynamics mismatch: missing={bad} unexpected={miss.unexpected_keys}"
            loaded.append("dvd_dynamics")
        elif g("dvd_dynamics_init"):
            dsd = legacy.load_checkpoint(cfg.dvd_dynamics_init)["extra"]["dvd_dynamics"]
            miss = dvd_term.dynamics.load_state_dict(dsd, strict=False)
            assert not [k for k in miss.missing_keys if not k.startswith("wm.dino.")] and not miss.unexpected_keys
            loaded.append("dvd_dynamics<-dvd_dynamics_init (frozen-f run)")
        else:
            loaded.append("dvd_dynamics<-dino_wm_ckpt (frozen-f run)")
        wm = getattr(dvd_term.dynamics, "wm", None)
        if getattr(wm, "has_state_head", False) and bool((wm.state_delta_scale == 1.0).all()):
            raise SystemExit("[bon16] proprio head with unset state_delta_scale: the trainer fills it from the demo buffer")
        if bool(getattr(wm, "flow", False)):
            print("[bon16] WARNING: flow WM -- every candidate is scored on a fresh noise draw (as in the trainer)", flush=True)
        dino_history = dino.FrameHistory(adapter)
        _no_dropout(adapter, value_fn, attn_pool, sargmax_pool)
        info.update(dino_wm_ckpt=cfg.dino_wm_ckpt, state_head=bool(getattr(wm, "has_state_head", False)),
                    action_encoder=getattr(wm, "action_encoder", "linear"),
                    dino_pool=cfg.dino_pool, attn_pool_freeze=g("attn_pool_freeze", False), value_pool=g("value_pool", ""),
                    dvd_proprio_source=dvd_term.proprio_source, proprio_dim=proprio_dim, value_model=cfg.value_model,
                    value_aggregate=cfg.value_aggregate, value_nets=len(value_fn.nets), discount_h=dvd_term.discount,
                    scorer="gamma^H V(f(ctx,s,a)) via train_online.dvd_candidate_scores")
        if cfg.dino_obs and g("dino_obs_pool_separate", False):
            # --dino-obs-pool-separate: the actor/critic observation is pooled by the run's OWN critic-trained module
            # (dvd_term._obs_pool, read by _dino_obs_parts), not by V's readout.
            from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool
            obs_pool = MultiQueryAttnPool.from_blob(ex["obs_pool"], device=dev).eval()
            dvd_term._obs_pool = obs_pool
            loaded.append("obs_pool")
            _no_dropout(obs_pool)
            info.update(obs_pool="extra.obs_pool (separate actor/critic pool)")
    if cfg.bon_source == "q":
        if cfg.dino_obs:     # train_online: CriticEnsemble(critic_state_dim) when not --pixel-obs
            critics = CriticEnsemble(actor_in_dim, chunk_dim, cfg.critic_ensemble_size,
                                     hidden=(cfg.critic_hidden,) * cfg.critic_layers, layernorm=cfg.method == "dice_rl").to(dev)
        else:
            critics = PixelCriticEnsemble(encoder.out_dim, chunk_dim, cfg.critic_ensemble_size,
                                          hidden=(cfg.critic_hidden,) * cfg.critic_layers, layernorm=cfg.method == "dice_rl").to(dev)
        critics.load_state_dict(payload["critics"], strict=True)
        loaded.append("critics")
        _no_dropout(critics)
        info.update(bon_reduce=cfg.bon_reduce, critic_nets=len(getattr(critics, "nets", None) or critics.q),
                    scorer=f"critics.reduce({'pooled DINO features' if cfg.dino_obs else 'encoder features'}, a, {cfg.bon_reduce!r})")
    _no_dropout(encoder, actor, actor_encoder)
    info["loaded_from_checkpoint"] = loaded
    if verbose:
        print(f"[bon16] built {cfg.bon_source} scorer; loaded {loaded}", flush=True)
    return dict(cfg=cfg, scaler=scaler, encoder=encoder, actor_encoder=actor_encoder, actor=actor,
                det_actor=DeterministicActor(actor), dvd_term=dvd_term, dino_history=dino_history,
                critics=critics, info=info, dev=dev)


def select(b, actor_in, state, bases):
    """The trainer's selection call on prepared inputs (the DINO context must already be applied)."""
    cfg = b["cfg"]
    REC.clear()
    return T.select_dice_exploration(
        b["det_actor"], RecordingCritics(b["critics"]) if b["critics"] is not None else None, actor_in, bases,
        cfg.residual_scale, dvd_term=(b["dvd_term"] if cfg.bon_source == "dvd" else None),   # trainer: dvd_term if bon_dvd
        bon_reduce=cfg.bon_reduce, dvd_state=state,
        bon_select=getattr(cfg, "bon_select", "max"), bon_quantile=getattr(cfg, "bon_quantile", 0.75),
        frozen_value_fn=None, max_norm=cfg.residual_max_norm)


def obs_parts(b, observation):
    """net_state's extra arguments: the trainer's own _dino_obs_parts on the rebuilt dvd_term -- (pooled DINO t, raw
    proprio) on a --dino-obs arm, (None, None) otherwise. Reads dino_history.cur, so push() must already have run.
    no_grad only drops the graph a trainable pool would build; the values are the call's."""
    with torch.no_grad():
        return T._dino_obs_parts(b["cfg"].dino_obs, b["dvd_term"], b["dino_history"], observation, b["dev"])


def decide(b, prior, observation, n):
    """One best-of-N decision in the trainer's collection order. Returns (chosen (1, D), stats, tensors)."""
    dev, hist = b["dev"], b["dino_history"]
    state = torch.as_tensor(b["scaler"](observation["agent_pos"]), device=dev).unsqueeze(0)
    if hist is not None:
        hist.push(observation)
    state_in = T.net_state(observation, state, b["encoder"], *obs_parts(b, observation))
    actor_in = T.net_state(observation, state, b["actor_encoder"]) if b["actor_encoder"] is not None else state_in
    bases = T.flatten_chunk(prior.action_chunk_samples(observation, n))
    with torch.inference_mode():
        if hist is not None and b["cfg"].bon_source == "dvd":     # trainer: `if bon_dvd and dino_history is not None`
            hist.apply(bases.shape[0])
        chosen = select(b, actor_in, state, bases)
    sde = T.select_dice_exploration
    score, members, actions = REC["score"].float(), REC.get("members"), REC["actions"]
    hits = (actions == chosen).all(dim=1).nonzero().flatten()
    idx = int(hits[0]) if hits.numel() else -1
    top2 = score.topk(2).values
    gap = float(top2[0] - top2[1])
    mstd = (float(members.float().std(dim=0).mean()) if members is not None and members.dim() == 2 and members.shape[0] > 1
            else None)
    rec = dict(idx=idx, rank=float(getattr(sde, "last_rank", float("nan"))), top_gap=gap,
               max_minus_median=float(score.max() - score.median()), best_minus_worst=float(score.max() - score.min()),
               member_std=mstd, top_gap_over_std=(gap / mstd if mstd else None),
               meaningful=(float(gap > mstd) if mstd else None),
               chosen_minus_mean=float(score[idx] - score.mean()) if idx >= 0 else None,
               chosen_minus_draw0=float(score[idx] - score[0]) if idx >= 0 else None,
               chosen_score=float(score[idx]) if idx >= 0 else None)
    tensors = dict(state=state, state_in=state_in, actor_in=actor_in, bases=bases, actions=actions, chosen=chosen,
                   score=REC["score"], members=members, agent_pos=np.array(observation["agent_pos"], copy=True),
                   pixels={k: np.array(v, copy=True) for k, v in observation["pixels"].items()},
                   ctx_prev=None if hist is None else hist.prev, ctx_cur=None if hist is None else hist.cur)
    return chosen, rec, tensors


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else None


def _atomic_write(path: Path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True,
                    help="<run>/checkpoints/<step>.pt, a real file (the run directory is two levels above the file the "
                         "path resolves to). The run's configuration is read from the checkpoint; there is no config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=10_000)
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing <out stem>.partial.json")
    ap.add_argument("--base-checkpoint", default="",
                    help="the prior's pretrained_model directory, overriding base_checkpoint of the checkpoint config "
                         "(default: that path if it exists here, else the same prior under $CKPT_ROOT/priors)")
    ap.add_argument("--value-model", default="",
                    help="V directory, overriding value_model (default: stored path, else $CKPT_ROOT/value/<name>). It "
                         "supplies the state scaler check and the readout's shape; the weights come from the checkpoint")
    ap.add_argument("--dino-wm-ckpt", default="",
                    help="world-model file, overriding dino_wm_ckpt (default: stored path, else below $CKPT_ROOT/wm). It "
                         "supplies the architecture; a fine-tuned predictor comes from the checkpoint")
    ap.add_argument("--dataset-root", default="",
                    help="the task's demo dataset (meta/stats.json feeds the state scaler); "
                         "default $DATA_ROOT/mimicgen_<task>_224x224")
    ap.add_argument("--allow-dropped-config-keys", action="store_true",
                    help="evaluate although the checkpoint config has keys this TrainConfig does not know")
    ap.add_argument("--dump", default="", help="save decision records (torch .pt) for tests/pipeline/test_bon16_parity.py")
    ap.add_argument("--dump-every", type=int, default=7)
    ap.add_argument("--dump-max", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()
    ckpt = Path(args.ckpt).resolve()
    if Path(os.path.abspath(args.ckpt)).parent.parent.name != ckpt.parent.parent.name:
        print(f"[bon16] NOTE: --ckpt {args.ckpt} is a link to {ckpt}; the run is taken to be {ckpt.parent.parent} "
              f"(result field 'run', code_version.json). Checkpoints should be real files inside <run>/checkpoints/",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.json")
    payload, was_legacy, assets = load_payload(
        ckpt, {field: getattr(args, arg) for field, arg in ASSET_ARGS.items()})
    env_steps = payload.get("env_steps")
    b = build(payload, dev, dataset_root=args.dataset_root or None, allow_dropped=args.allow_dropped_config_keys)
    del payload                                  # optimiser states etc. are not needed past build()
    cfg, info = b["cfg"], b["info"]
    n = info["n_candidates"]
    run_dir = ckpt.parent.parent
    try:
        code_run = json.loads((run_dir / "code_version.json").read_text())
    except Exception:
        code_run = None
    code_now = code_hashes()
    code_diff = sorted(k for k, v in code_now.items() if code_run is not None and code_run.get(k) not in (None, v))
    prior = FrozenPolicyPrior(cfg.task, Path(cfg.base_checkpoint), cfg.action_steps, cfg.inference_steps,
                              device=str(dev), prior_temperature=getattr(cfg, "prior_temperature", 1.0))
    insertion_cfg = ({"span": cfg.insertion_span, "joint_sigma": cfg.insertion_joint_sigma,
                      "holder_sigma": cfg.insertion_holder_sigma} if cfg.insertion_only else None)
    env = T.make_env(cfg.task, cfg.episode_horizon, False, insertion_cfg, holder_sigma=cfg.holder_sigma,
                     fixed_goal=cfg.fixed_goal, fixed_goal_rot=cfg.fixed_goal_rot, full_fixed=cfg.full_fixed,
                     jitter_goal_m=cfg.jitter_goal_mm / 1000.0)
    geom = T._PHASE_GEOM.get(cfg.task)
    episodes = []
    if partial.exists() and not args.no_resume:
        prev = json.loads(partial.read_text())
        if prev.get("ckpt_path") == str(ckpt) and prev.get("seed_start") == args.seed_start and prev.get("n_candidates") == n:
            episodes = prev["episodes_done"][: args.episodes]
            print(f"[bon16] resuming after {len(episodes)} finished episodes from {partial.name}", flush=True)
    print(f"[bon16] {ckpt} @{env_steps} env steps, task {cfg.task}, {args.episodes} episodes, best-of-{n} "
          f"({info['scorer']}, select {info['bon_select']}), env seeds {args.seed_start}+i"
          + (f"\n[bon16] NOTE: trainer source now differs from the run's launch-time code_version.json in {code_diff}"
             + (" (expected: the run was trained by the pre-rename code)" if was_legacy else "")
             if code_diff else ""), flush=True)
    dumps, decision_counter = [], 0
    try:
        for ep in range(len(episodes), args.episodes):
            seed = args.seed_start + ep
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            observation, _ = env.reset(seed=seed)
            if b["dino_history"] is not None:
                b["dino_history"].reset()
            z0 = float(observation["agent_pos"][11]) if geom is not None else 0.0
            ep_steps = {k: 0 for k in T.EVAL_PHASES}
            furthest, done, length, ret, success, recs = 0, False, 0, 0.0, False, []
            te = time.time()
            while not done:
                chosen, rec, tensors = decide(b, prior, observation, n)
                recs.append(rec)
                if args.dump and len(dumps) < args.dump_max and decision_counter % args.dump_every == 0:
                    dumps.append(dict(episode=ep, seed=seed, decision=len(recs) - 1, rec=rec,
                                      **{k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()}))
                    torch.save(dict(ckpt=str(ckpt), dumps=dumps), args.dump)
                decision_counter += 1
                action = chosen.view(1, prior.action_steps, prior.action_dim)
                observation, reward, done, chunk_success, steps = T.rollout_chunk(
                    env, action[0].cpu().numpy(), None, cfg.step_penalty)
                length += steps; ret += reward; success = success or chunk_success
                if geom is not None:
                    ph = T._eval_phase_of(observation["agent_pos"], geom, z0)
                    ep_steps[ph] += steps
                    furthest = max(furthest, ph)
            e = dict(seed=seed, success=bool(success), length=int(length), ret=float(ret), decisions=len(recs),
                     furthest=furthest, phase_steps=ep_steps if geom is not None else None,
                     unmatched_idx=int(sum(r["idx"] < 0 for r in recs)),
                     **{k: _mean([r[k] for r in recs]) for k in STAT_KEYS})
            episodes.append(e)
            sr = 100.0 * np.mean([x["success"] for x in episodes])
            ms = "-" if e["member_std"] is None else f"{e['member_std']:.4g}"
            print(f"ep {ep} seed {seed}: {'success' if success else 'fail'} len {length} decisions {len(recs)} "
                  f"top_gap {e['top_gap']:.4g} member_std {ms}  running {sr:.1f}% of {len(episodes)} "
                  f"[{time.time() - te:.0f}s, total {time.time() - t0:.0f}s]", flush=True)
            _atomic_write(partial, dict(ckpt_path=str(ckpt), seed_start=args.seed_start, n_candidates=n,
                                        episodes_done=episodes))
    finally:
        env.close()
        prior.close()

    def dw(k):   # decision-weighted mean of the per-episode decision means
        rows = [x for x in episodes if x[k] is not None]
        return float(sum(x[k] * x["decisions"] for x in rows) / sum(x["decisions"] for x in rows)) if rows else None

    succ = [x["success"] for x in episodes]
    res = dict(run=run_dir.name, ckpt=ckpt.name, ckpt_path=str(ckpt), env_steps=env_steps, task=cfg.task,
               episodes=len(episodes), strategy=f"bon{n}", seed_start=args.seed_start, seconds=time.time() - t0,
               success_rate=100.0 * float(np.mean(succ)), avg_length=float(np.mean([x["length"] for x in episodes])),
               avg_return=float(np.mean([x["ret"] for x in episodes])), per_episode=[bool(s) for s in succ],
               per_episode_length=[x["length"] for x in episodes], decisions=int(sum(x["decisions"] for x in episodes)),
               **{f"bon_mean_{k}": dw(k) for k in STAT_KEYS if k != "meaningful"},
               bon_meaningful_frac=dw("meaningful"),
               bon_top_gap_success_eps=_mean([x["top_gap"] for x in episodes if x["success"]]),
               bon_top_gap_failed_eps=_mean([x["top_gap"] for x in episodes if not x["success"]]),
               unmatched_selections=int(sum(x["unmatched_idx"] for x in episodes)),
               per_episode_stats=episodes, code_version_run=code_run, code_version_now=code_now,
               code_changed_since_launch=code_diff, legacy_checkpoint=was_legacy, assets=assets, **info)
    res["bon_mean_selected_rank"] = res.pop("bon_mean_rank")
    if geom is not None:
        failed = [x for x in episodes if not x["success"]]
        res["n_failed"] = float(len(failed))
        for k, name in T.EVAL_PHASES.items():
            res[f"fail_furthest/{name}"] = (sum(x["furthest"] == k for x in failed) / len(failed)) if failed else float("nan")
    _atomic_write(out, res)
    if partial.exists():
        partial.unlink()
    print("[bon16] " + json.dumps({k: v for k, v in res.items() if k not in (
        "per_episode", "per_episode_length", "per_episode_stats", "code_version_run", "code_version_now")}, default=float),
        flush=True)


if __name__ == "__main__":
    main()
