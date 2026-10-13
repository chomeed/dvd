#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import copy
import json
import math
import os
import time
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .pixel import (
    PROPRIO as PIXEL_PROPRIO, CAMERAS as PIXEL_CAMERAS, IMG as PIXEL_IMG,
    PixelEncoder, PixelCriticEnsemble, preprocess as pixel_preprocess, random_shift,
)
from .gradvar import gradient_snr, directional_return_probe
from .common import (
    CACHE_ROOT,
    GOAL_RELATIVE_MAPS,
    TASKS,
    CriticEnsemble,
    FrozenPolicyPrior,
    GoalRelativeScaler,
    ReplayBuffer,
    ResidualActor,
    StateScaler,
    TrainConfig,
    check_state_repr,
    concat_batches,
    fill_mc_return,
    fill_episode_len,
    fill_nstep,
    find_latest_checkpoint,
    save_checkpoint,
    save_video,
    set_seed,
    soft_update,
)
from .critical import CODE_TYPES, TYPE_CODES, OnlineAnnotator, StateAnnotator
from .expert import load_successful_expert_buffer
from .value_model import DualValue, DuelingCritic, RunningStats, ValueEnsemble


def make_env(task: str, horizon: int | None = None, reward_shaping: bool = False,
             insertion: dict | None = None, holder_sigma: float = 0.0,
             fixed_goal: bool = False, fixed_goal_rot: bool = False,
             full_fixed: bool = False, jitter_goal_m: float = 0.0):
    from lerobot_env_mimicgen.environment import MimicGenGymEnv, normalize_task

    spec = normalize_task(task)
    if sum(bool(x) for x in (fixed_goal, fixed_goal_rot, full_fixed)) > 1:
        raise ValueError(
            "--fixed-goal, --fixed-goal-rot and --full-fixed each name a different env variant "
            "(goal pinned / goal rotation only / goal AND object pinned). Pass exactly one."
        )
    if fixed_goal and fixed_goal_rot:
        raise ValueError(
            "--fixed-goal already pins the goal's z-rotation; --fixed-goal-rot pins ONLY the "
            "rotation and leaves the position random. Pass one or the other, not both."
        )
    if jitter_goal_m > 0.0:


        if fixed_goal or fixed_goal_rot or full_fixed:
            raise ValueError("--jitter-goal-mm randomises the goal; --fixed-goal* pin it")
        if insertion or holder_sigma > 0:
            raise ValueError("--jitter-goal-mm cannot combine with --insertion-only/--holder-sigma")
        from .fixed_goal_env import JitterGoalEnv
        return JitterGoalEnv(
            task=task,
            reward_shaping=reward_shaping,
            episode_length=horizon or spec.horizon,
            observation_height=224,
            observation_width=224,
            render_gpu_device_id=0,
            jitter_m=jitter_goal_m,
        )
    if fixed_goal or fixed_goal_rot or full_fixed:
        if insertion or holder_sigma > 0:
            raise ValueError(
                "--fixed-goal/--fixed-goal-rot name their own env variant; they cannot combine "
                "with --insertion-only or --holder-sigma"
            )
        from .fixed_goal_env import FixedGoalEnv, FixedRotEnv, FullFixedEnv

        variant = (FixedGoalEnv if fixed_goal else
                   FixedRotEnv if fixed_goal_rot else FullFixedEnv)
        return variant(
            task=task,
            reward_shaping=reward_shaping,
            episode_length=horizon or spec.horizon,
            observation_height=224,
            observation_width=224,
            render_gpu_device_id=0,
        )
    if insertion:


        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0811"))
        from insertion_env import InsertionOnlyEnv

        return InsertionOnlyEnv(
            task=task,
            reward_shaping=reward_shaping,
            episode_length=horizon or spec.horizon,
            observation_height=224,
            observation_width=224,
            render_gpu_device_id=0,
            span=insertion["span"],
            joint_sigma=insertion["joint_sigma"],
            holder_sigma=insertion["holder_sigma"],
        )
    if holder_sigma > 0:


        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0811"))
        from jitter_env import JitteredHolderEnv

        return JitteredHolderEnv(
            task=task,
            reward_shaping=reward_shaping,
            episode_length=horizon or spec.horizon,
            observation_height=224,
            observation_width=224,
            render_gpu_device_id=0,
            holder_sigma=holder_sigma,
        )
    return MimicGenGymEnv(
        task=task,
        reward_shaping=reward_shaping,
        episode_length=horizon or spec.horizon,
        observation_height=224,
        observation_width=224,
        render_gpu_device_id=0,


        insertion_camera="insertion" in PIXEL_CAMERAS,
    )


def _value_pool_of(dvd_term):
    """The module that pools V'S OWN FIT ROWS."""
    vp = getattr(dvd_term, "_value_pool", None)
    return dvd_term._pool if vp is None else vp


def _obs_pool_of(dvd_term):
    """The module that pools the --dino-obs OBSERVATION under --dvd-actor-pool."""
    return (_value_pool_of(dvd_term)
            if getattr(dvd_term, "_actor_pool_obs", "actor") == "value"
            else dvd_term._pool)


def _pool_grad_norm(module) -> float:
    """L2 norm over a pooling module's accumulated .grad; 0.0 when nothing landed on it."""
    gs = [p.grad for p in module.parameters() if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gs])))


@torch.no_grad()
def _pool_rel_distance(module, reference) -> float:
    """||theta_module - theta_reference|| / ||theta_reference|| over two modules of ONE architecture."""
    pairs = list(zip(module.parameters(), reference.parameters()))
    num = torch.stack([torch.linalg.vector_norm(a.detach().float() - b.detach().float())
                       for a, b in pairs])
    den = torch.stack([torch.linalg.vector_norm(b.detach().float()) for _, b in pairs])
    return float(torch.linalg.vector_norm(num)
                 / torch.linalg.vector_norm(den).clamp_min(1e-12))


@torch.no_grad()
def _grad_change_norm(module, before) -> float:
    """L2 norm of what changed on a module's .grad since `before` (per-parameter clones or None)."""
    parts = []
    for q, g0 in zip(module.parameters(), before):
        if q.grad is None:
            if g0 is not None:
                parts.append(torch.linalg.vector_norm(g0.float()))
            continue
        d = q.grad.detach().float() if g0 is None else q.grad.detach().float() - g0.float()
        parts.append(torch.linalg.vector_norm(d))
    return float(torch.linalg.vector_norm(torch.stack(parts))) if parts else 0.0


def _dino_obs_parts(enabled, dvd_term, dino_history, observation, device):
    """(pooled features, raw proprio) for --dino-obs, or (None, None) to leave net_state alone."""
    if not enabled or dino_history is None or dvd_term is None:
        return (None, None)
    if dino_history.cur is None:
        raise RuntimeError("--dino-obs read dino_history.cur before push(); the observation "
                           "would be built from the previous frame")


    _obs_pool = getattr(dvd_term, "_obs_pool", None)


    _actor_pool_on = getattr(dvd_term, "_value_pool", None) is not None
    if (_obs_pool is None and not getattr(dvd_term, "_pool_obs_grad", False)
            and not _actor_pool_on):
        feat = dvd_term._pool(dino_history.cur, dvd_term.pool_mode)
    else:
        with torch.no_grad():
            feat = (_obs_pool if _obs_pool is not None else _obs_pool_of(dvd_term))(
                dino_history.cur, dvd_term.pool_mode)
    prop = torch.as_tensor(observation["agent_pos"][:dvd_term.proprio_dim],
                           device=device, dtype=feat.dtype).unsqueeze(0)
    return (feat, prop)


def _pool_without_param_grad(pool, feats: torch.Tensor, mode) -> torch.Tensor:
    """pool(feats, mode) with the pool's PARAMETERS entering as detached constants."""
    if isinstance(pool, torch.nn.Module):
        params = dict(pool.named_parameters())
        if params:
            from torch.func import functional_call
            state = {n: q.detach() for n, q in params.items()}
            state.update(dict(pool.named_buffers()))
            return functional_call(pool, state, (feats, mode))
    return pool(feats, mode)


def dino_obs_from_features(dvd_term, feats: torch.Tensor, proprio_raw: torch.Tensor) -> torch.Tensor:
    """The --dino-obs ACTOR observation [pool(feats) || raw proprio] from (B, V, 196, 384) features that are already in the encoder's space (no re-encode) -- update_dice's --dino-obs batch rewrite, expression for expression:"""
    ob = getattr(dvd_term, "_obs_pool", None)
    pool = ob if ob is not None else _obs_pool_of(dvd_term)
    ctx = getattr(dvd_term.dynamics, "ctx_feats", None)
    if ctx is not None and feats.dtype != ctx.dtype:
        feats = feats.to(ctx.dtype)
    feat = _pool_without_param_grad(pool, feats, dvd_term.value_pool_mode)
    return torch.cat([feat, proprio_raw[:, :dvd_term.proprio_dim]], dim=-1)


def make_step2_policy(dvd_term, actor, residual_scale: float, residual_max_norm: float = 0.0):
    """--dvd-step2-action actor:"""
    def policy(zhat: torch.Tensor, proprio_raw: torch.Tensor, follow: torch.Tensor) -> torch.Tensor:
        follow = follow.detach()
        obs = dino_obs_from_features(dvd_term, zhat, proprio_raw)
        residual, _ = actor.sample(obs, follow, deterministic=True)
        dvd_term.last_step2_residual_norm = residual.detach().norm(dim=-1)
        return final_action(follow, residual, residual_scale, residual_max_norm)
    return policy


_LATENT_OBS_CTX: dict = {}


def _latent_obs_state(batch, key, cfg, scaler):
    """[z, proprio_normalised] for one side of a transition, from the SCALED env state."""
    from .latent56 import norm_prop as _lo_norm
    lo = torch.as_tensor(scaler.low[:PIXEL_PROPRIO], device=batch[key].device)
    sc = torch.as_tensor(scaler.scale[:PIXEL_PROPRIO], device=batch[key].device)
    raw = (batch[key][:, :PIXEL_PROPRIO] + 1.0) * sc / 2.0 + lo
    return torch.as_tensor(_lo_norm(raw, cfg), device=batch[key].device, dtype=torch.float32)


def _seg_obs_parts(enabled, seg_hist, observation, device):
    """(keypoint vector, raw proprio) for --seg-obs, or (None, None)."""
    if not enabled or seg_hist is None or seg_hist.cur is None:
        return (None, None)


    cur = seg_hist.cur.float()
    if cur.dim() == 4:
        cur = cur.unsqueeze(0)
    feat = _LATENT_OBS_CTX["seg_enc"](cur)
    prop = torch.as_tensor(observation["agent_pos"][:PIXEL_PROPRIO],
                           device=device, dtype=feat.dtype).unsqueeze(0)
    return (feat, prop)


def _lat_feat_shape(config):
    """(latent,) for the replay column, read from the world model's own config.json."""
    import json as _json
    return (int(_json.loads(
        (Path(config.dvd_latent_wm) / "config.json").read_text())["latent"]),)


def _tok_shape_from_ckpt(config):
    """(V, P, D) of one frame's DINOv2 tokens, read from the world model's own config."""
    import json as _json
    from .latent56 import token_shape as _ts
    return _ts(_json.loads((Path(config.dvd_latent_wm) / "config.json").read_text()))


def _shift_frames(img_u8, pad):
    """DrQ random shift on a stored (B,V,H,W,3) uint8 batch -> (B,V,3,H,W) float in [0,1]."""
    b, v, h, w = img_u8.shape[0], img_u8.shape[1], img_u8.shape[2], img_u8.shape[3]
    x = img_u8.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w).float()
    if pad > 0:
        x = random_shift(x, pad)
    return x.reshape(b, v, 3, h, w) / 255.0


def _latent_obs_parts(enabled, seg_hist, observation, device):
    """(z_t, normalised proprio) for --latent-obs, or (None, None) to leave net_state alone."""
    if not enabled or seg_hist is None:
        return (None, None)
    if seg_hist.cur is None:
        raise RuntimeError("--latent-obs read seg_hist.cur before push(); the observation would "
                           "be built from the previous frame")
    from .latent56 import norm_prop as _lo_norm
    prop = torch.as_tensor(
        _lo_norm(np.asarray(observation["agent_pos"][:PIXEL_PROPRIO], dtype=np.float32),
                 seg_hist.cfg), device=device, dtype=seg_hist.cur.dtype).unsqueeze(0)
    return (seg_hist.cur, prop)


def _learned_deferred_encode(encoder, device, drq: bool):
    """--dvd-encoder-learned:"""
    def _enc(frames_u8, props, grad):
        fr = torch.as_tensor(frames_u8, device=device)
        pr = torch.as_tensor(props, device=device, dtype=torch.float32)
        if grad:
            return encoder(fr, pr, augment=drq)
        with torch.no_grad():
            return encoder(fr, pr, augment=False)
    return _enc


def net_state(observation, scaled_state, encoder, dino_feat=None, raw_proprio=None):
    """What the actor and critic consume for one observation."""
    if dino_feat is not None:


        return torch.cat([dino_feat, raw_proprio], dim=-1)
    if encoder is None:
        return scaled_state
    img = torch.as_tensor(
        pixel_preprocess(observation["pixels"]), device=scaled_state.device).unsqueeze(0)
    with torch.no_grad():
        return encoder(img, scaled_state[:, :PIXEL_PROPRIO])


def final_action(base: torch.Tensor, residual: torch.Tensor, scale: float,
                 max_norm: float = 0.0) -> torch.Tensor:
    """base + scale*residual, optionally PROJECTED onto a ball of radius `max_norm`."""
    if max_norm > 0.0:
        norm = residual.norm(dim=-1, keepdim=True)
        residual = residual * (max_norm / norm.clamp_min(max_norm))
    return (base + scale * residual).clamp(-1.0, 1.0)


CRITICAL_DIAGNOSTIC_EVERY = 500


def critical_diagnostics(
    batch: dict[str, torch.Tensor],
    critics: CriticEnsemble,
    target: torch.Tensor,
    q_values: torch.Tensor,
) -> dict[str, float]:
    """TD error and critic action-gradient statistics, split critical vs noncritical."""
    mask = batch["is_critical"][:, 0] > 0.5
    result: dict[str, float] = {"critical_batch_fraction_actual": float(mask.float().mean())}
    with torch.no_grad():
        td_error = (q_values.mean(dim=0) - target).abs()[:, 0]

    action = batch["action"].detach().clone().requires_grad_(True)
    gradients = []
    with torch.enable_grad():
        state = batch["state"].detach()
        for critic in critics.q:
            value = critic(torch.cat((state, action), dim=-1)).sum()
            (gradient,) = torch.autograd.grad(value, action, retain_graph=False)
            gradients.append(gradient.detach())
    stacked = torch.stack(gradients, dim=0)
    norms = stacked.norm(dim=-1)
    mean_norm = norms.mean(dim=0)
    unit = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = torch.einsum("ibd,jbd->ijb", unit, unit)
    ensemble = stacked.shape[0]
    off_diagonal = ~torch.eye(ensemble, dtype=torch.bool, device=stacked.device)
    cosine = similarity[off_diagonal].view(-1, stacked.shape[1]).mean(dim=0)

    for name, selector in (("critical", mask), ("noncritical", ~mask)):
        if not bool(selector.any()):
            continue
        result[f"td_error_{name}"] = float(td_error[selector].mean())
        result[f"qgrad_norm_{name}"] = float(mean_norm[selector].mean())


        result[f"qgrad_norm_var_{name}"] = float(norms[:, selector].var(dim=0).mean())
        result[f"qgrad_cosine_{name}"] = float(cosine[selector].mean())


    for code, name in CODE_TYPES.items():
        if not code or name is None:
            continue
        selector = batch["critical_type"][:, 0] == code
        if not bool(selector.any()):
            continue
        result[f"td_error_{name}"] = float(td_error[selector].mean())
        result[f"qgrad_norm_{name}"] = float(mean_norm[selector].mean())
        result[f"qgrad_cosine_{name}"] = float(cosine[selector].mean())
    return result


def stamp_annotation(episode: list[dict], annotation, config) -> list[dict]:
    """Attach critical-state metadata to an episode and drop the post-catastrophe suffix."""
    critical = annotation.critical_chunks(config.critical_pre_window)
    retained = annotation.retained_chunks(config.truncate_after_catastrophe)
    catastrophic = annotation.catastrophic_chunk
    code = float(TYPE_CODES[annotation.critical_type])
    kept: list[dict] = []
    for index, transition in enumerate(episode):
        if index not in retained:
            continue
        transition["trajectory_id"] = float(annotation.trajectory_id)
        transition["timestep"] = float(index)
        transition["is_critical"] = float(index in critical)
        transition["is_catastrophic"] = float(index == catastrophic)


        transition["critical_type"] = code if index in critical else 0.0
        if catastrophic is None:
            transition["catastrophic_timestep"] = float("nan")
            transition["steps_to_catastrophe"] = float("nan")
        else:
            transition["catastrophic_timestep"] = float(catastrophic)
            transition["steps_to_catastrophe"] = float(index - catastrophic)
            if config.catastrophic_as_terminal and index == catastrophic:


                transition["done"] = 1.0
        kept.append(transition)
    return kept


def flatten_chunk(action: torch.Tensor) -> torch.Tensor:
    return action.reshape(action.shape[0], -1)


_SECT = defaultdict(float)


def _timed(fn, name):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _SECT[name] += time.perf_counter() - t0
    wrapper._timed_stage = name
    wrapper.__wrapped__ = fn
    return wrapper


def install_component_timers(prior=None, value_fn=None, dvd_term=None, rm_distiller=None, online=None,
                             backward=False):
    """Monkey-patch the stage entry points."""
    from . import dino_dvd as _dino
    done = []
    g = globals()
    for fname, stage in (("rollout_chunk", "env_step"), ("select_dice_exploration", "bon_select")):
        if fname in g and getattr(g[fname], "_timed_stage", None) is None:
            g[fname] = _timed(g[fname], stage); done.append(stage)
    for mod, attr, stage in ((_dino, "encode_frames", "dino_encode"), (_dino, "fit_batch", "wm_fit")):
        fn = getattr(mod, attr, None)
        if fn is not None and getattr(fn, "_timed_stage", None) is None:
            setattr(mod, attr, _timed(fn, stage)); done.append(stage)
    for obj, attr, stage in ((prior, "action_chunk", "prior_sample"), (prior, "action_chunk_samples", "prior_sample"),
                             (value_fn, "fit_step", "v_fit"), (dvd_term, "objective_terms", "dvd_obj_fwd"),
                             (rm_distiller, "maybe_fit", "distill"), (online, "sample", "replay_sample"),
                             (dvd_term, "objective", "dvd_obj_fwd"),
                             (rm_distiller, "add_episode", "rm_score_episode")):
        if obj is None:
            continue
        fn = getattr(obj, attr, None)
        if fn is not None and getattr(fn, "_timed_stage", None) is None:
            setattr(obj, attr, _timed(fn, stage)); done.append(stage)

    for mod, stage in ((getattr(dvd_term, "dynamics", None), "wm_fwd"),
                       (getattr(dvd_term, "value_fn", None), "v_fwd")):
        fwd = getattr(mod, "forward", None)
        if fwd is not None and getattr(fwd, "_timed_stage", None) is None:
            mod.forward = _timed(fwd, stage); done.append(stage)
    if backward:
        if getattr(torch.Tensor.backward, "_timed_stage", None) is None:
            torch.Tensor.backward = _timed(torch.Tensor.backward, "backward"); done.append("backward")
        for cls in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
            if getattr(cls.step, "_timed_stage", None) is None:
                cls.step = _timed(cls.step, "optim_step"); done.append("optim_step")
    return sorted(set(done))


def install_compiled_parts(parts: str, dvd_term=None, value_fn=None, encoder=None, critics=None, actor=None):
    """torch.compile the named forwards."""
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
    want = {p.strip() for p in parts.split(",") if p.strip()}
    if "all" in want:
        want = {"encoder", "wm", "v", "critic", "actor"}
    done = []
    wm = getattr(getattr(dvd_term, "dynamics", None), "wm", None)
    targets = []
    if wm is not None:
        if "encoder" in want and getattr(wm, "dino", None) is not None:
            targets.append((wm.dino, "encoder"))
        if "wm" in want and getattr(wm, "predictor", None) is not None:
            targets.append((wm.predictor, "wm"))
    if "v" in want:
        for obj in (value_fn, getattr(dvd_term, "_pool", None)):
            if isinstance(obj, torch.nn.Module):
                targets.append((obj, "v"))

    if "encoder" in want and isinstance(encoder, torch.nn.Module):
        targets.append((encoder, "pixel_encoder"))
    if "critic" in want:
        for c in (critics if isinstance(critics, (list, tuple, torch.nn.ModuleList)) else [critics]):
            if isinstance(c, torch.nn.Module):
                targets.append((c, "critic"))
    if "actor" in want and isinstance(actor, torch.nn.Module):
        targets.append((actor, "actor"))
    for mod, tag in targets:
        if getattr(mod.forward, "_compiled_tag", None) is not None:
            continue
        compiled = torch.compile(mod.forward, dynamic=True)
        compiled._compiled_tag = tag
        mod.forward = compiled
        done.append(tag)
    return sorted(set(done))


def rollout_chunk(
    env,
    actions: np.ndarray,
    frames: list[np.ndarray] | None = None,
    step_penalty: float = -0.01,
    observer=None,
    use_env_reward: bool = False,
):
    reward = 0.0
    success = False
    terminated = False
    truncated = False
    observation = None
    steps = 0
    for action in actions:
        observation, sparse_reward, terminated, truncated, info = env.step(action)
        steps += 1
        success = success or bool(info["is_success"])
        reward += float(sparse_reward)
        if frames is not None:
            frames.append(env.render().copy())
        if observer is not None:


            observer(observation)
        if terminated or truncated:
            break


    if use_env_reward:
        return observation, reward, terminated or truncated, success, steps
    shaped_reward = 1.0 if success else step_penalty * steps
    return observation, shaped_reward, terminated or truncated, success, steps


@torch.inference_mode()
def evaluate(
    task: str,
    prior: FrozenPolicyPrior,
    actor: ResidualActor,
    scaler: StateScaler,
    residual_scale: float,
    episodes: int,
    seed_start: int,
    step_penalty: float = -0.01,
    video_path: Path | None = None,
    video_episodes: int = 0,
    critics: CriticEnsemble | None = None,
    strategy: str = "greedy",
    samples: int = 16,
    insertion: dict | None = None,
    horizon: int | None = None,
    dvd_term=None,
    bon_source: str = "q",
    encoder=None,
    holder_sigma: float = 0.0,
    residual_max_norm: float = 0.0,
    fixed_goal: bool = False,
    fixed_goal_rot: bool = False,
    full_fixed: bool = False,
    jitter_goal_m: float = 0.0,
    dino_history=None,
    bon_reduce: str = "min",
    dino_obs: bool = False,
    latent_obs: bool = False,
    latent_hist=None,
    actor_encoder=None,
) -> dict[str, float]:


    env = make_env(task, horizon, False, insertion, holder_sigma=holder_sigma,
                   fixed_goal=fixed_goal, fixed_goal_rot=fixed_goal_rot,
                   full_fixed=full_fixed, jitter_goal_m=jitter_goal_m)


    geom = _PHASE_GEOM.get(task)
    fail_furthest = {k: 0 for k in EVAL_PHASES}
    fail_steps = {k: [] for k in EVAL_PHASES}
    succ_steps = {k: [] for k in EVAL_PHASES}
    n_failed = 0
    successes = []
    lengths = []
    returns = []
    frames: list[np.ndarray] = []


    if not (dvd_term is not None and bon_source == "dvd"):
        dino_history = None
    actor.eval()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_start + episode)


            z0 = float(observation["agent_pos"][11]) if geom is not None else 0.0
            ep_steps = {k: 0 for k in EVAL_PHASES}
            furthest = 0
            done = False
            length = 0
            episode_return = 0.0
            success = False
            capture = frames if video_path is not None and episode < video_episodes else None
            if dino_history is not None:
                dino_history.reset()
            while not done:
                state = torch.as_tensor(
                    scaler(observation["agent_pos"]), device=prior.device
                ).unsqueeze(0)
                if dino_history is not None:


                    dino_history.push(observation)
                if latent_hist is not None:


                    latent_hist.push(
                        torch.as_tensor(pixel_preprocess(observation["pixels"])[None]
                                        ).permute(0, 1, 4, 2, 3).float().div_(255.0))
                _scaled_state = state
                state = net_state(observation, state, encoder,
                                  *(_latent_obs_parts(True, latent_hist, observation,
                                                      prior.device)
                                    if latent_obs
                                    else _dino_obs_parts(dino_obs, dvd_term, dino_history,
                                                         observation, prior.device)))


                a_state = (net_state(observation, _scaled_state, actor_encoder)
                           if actor_encoder is not None else state)
                if strategy == "max_q_min" and (critics is not None or dvd_term is not None):


                    bases = flatten_chunk(prior.action_chunk_samples(observation, samples))
                    residual, _ = actor.sample(
                        a_state.repeat(samples, 1), bases, deterministic=True
                    )
                    candidates = final_action(bases, residual, residual_scale,
                                              residual_max_norm)


                    if dvd_term is not None and bon_source == "dvd":
                        if dino_history is not None:
                            dino_history.apply(samples)
                        scores = dvd_score(dvd_term, state.repeat(samples, 1),
                                           candidates).squeeze(-1)
                    elif critics is not None:
                        scores = critics.reduce(state.repeat(samples, 1),
                                                candidates, bon_reduce).squeeze(-1)
                    else:
                        raise RuntimeError("max_q_min eval needs a critic or an DVD term")


                    _sel = getattr(config, "bon_select", "max")
                    if _sel == "median" and scores.numel() > 1:
                        _idx = int(scores.argsort()[scores.numel() // 2])
                    elif _sel == "mean" and scores.numel() > 1:
                        _idx = int((scores - scores.mean()).abs().argmin())
                    elif _sel == "quantile" and scores.numel() > 1:


                        _q = float(getattr(config, "bon_quantile", 0.75))
                        _k = min(scores.numel() - 1,
                                 max(0, int(round(_q * (scores.numel() - 1)))))
                        _idx = int(scores.argsort()[_k])
                    else:
                        _idx = int(scores.argmax())
                    chosen = candidates[_idx].unsqueeze(0)
                else:
                    base = flatten_chunk(prior.action_chunk(observation))
                    residual, _ = actor.sample(a_state, base, deterministic=True)
                    chosen = final_action(base, residual, residual_scale,
                                          residual_max_norm)
                action = chosen.view(1, prior.action_steps, prior.action_dim)
                observation, reward, done, chunk_success, steps = rollout_chunk(
                    env, action[0].cpu().numpy(), capture, step_penalty
                )
                length += steps
                episode_return += reward
                success = success or chunk_success
                if geom is not None:
                    ph = _eval_phase_of(observation["agent_pos"], geom, z0)
                    ep_steps[ph] += steps
                    furthest = max(furthest, ph)
            successes.append(float(success))
            lengths.append(float(length))
            returns.append(float(episode_return))
            if geom is not None:
                bucket = succ_steps if success else fail_steps
                for k, n in ep_steps.items():
                    if n:
                        bucket[k].append(float(n))
                if not success:
                    n_failed += 1
                    fail_furthest[furthest] += 1
    finally:
        env.close()
        actor.train()
    if video_path is not None and frames:
        save_video(video_path, frames)
    out = {
        "success_rate": 100.0 * float(np.mean(successes)),
        "avg_length": float(np.mean(lengths)),
        "avg_return": float(np.mean(returns)),


        "per_episode": [bool(x) for x in successes],
    }
    if geom is not None:
        out["n_failed"] = float(n_failed)
        for k, name in EVAL_PHASES.items():

            out[f"fail_furthest/{name}"] = (fail_furthest[k] / n_failed
                                            if n_failed else float("nan"))
            out[f"fail_steps/{name}"] = (float(np.median(fail_steps[k]))
                                         if fail_steps[k] else float("nan"))
            out[f"succ_steps/{name}"] = (float(np.median(succ_steps[k]))
                                         if succ_steps[k] else float("nan"))
    return out


def update_policy_decorator(
    batch: dict[str, torch.Tensor],
    actor: ResidualActor,
    critics: CriticEnsemble,
    target_critics: CriticEnsemble,
    actor_optimizer,
    critic_optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer,
    config: TrainConfig,
) -> dict[str, float]:
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_residual, next_log_prob = actor.sample(batch["next_state"], batch["next_base"])
        next_action = final_action(batch["next_base"], next_residual, config.residual_scale, config.residual_max_norm)
        next_q = target_critics.conservative(batch["next_state"], next_action)
        target = batch["reward"] + (1.0 - batch["done"]) * batch["discount"] * (
            next_q - alpha * next_log_prob
        )

    q_values = critics(batch["state"], batch["action"])
    critic_loss = sum(F.mse_loss(value, target) for value in q_values)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(critics.parameters(), 10.0)
    critic_optimizer.step()

    residual, log_prob = actor.sample(batch["state"], batch["base"])
    action = final_action(batch["base"], residual, config.residual_scale, config.residual_max_norm)
    q_pi = critics.reduce(batch["state"], action, config.actor_q_reduce)
    actor_loss = (alpha * log_prob - q_pi).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
    actor_optimizer.step()

    target_entropy = -float(batch["action"].shape[-1])
    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()
    if config.train_critic:
        soft_update(critics, target_critics, config.tau)
    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "alpha": float(log_alpha.exp().detach()),
        "q_mean": float(q_values.mean().detach()),
        "critic_grad": float(critic_grad),
        "actor_grad": float(actor_grad),
    }


def _value_read(value_fn, dvd_term):
    """The V an auxiliary TARGET reads (--encoder-value-aux)."""
    _sv = getattr(dvd_term, "value_fn", None)
    if isinstance(_sv, DualValue):
        return _sv
    return value_fn if value_fn is not None else _sv


def _dual_member_action_grads(graph, action) -> dict[str, float]:
    """How much EACH DualValue member drives the actor, on the actor's own DVD rows."""
    if graph is None or action is None or not getattr(action, "requires_grad", False):
        return {}
    na, nb = graph
    if not (na.requires_grad and nb.requires_grad):
        return {}
    ga, = torch.autograd.grad(na.sum(), action, retain_graph=True, allow_unused=True)
    gb, = torch.autograd.grad(nb.sum(), action, retain_graph=True, allow_unused=True)
    if ga is None or gb is None:
        return {}
    ga = ga.detach().float().flatten(1)
    gb = gb.detach().float().flatten(1)
    cos = torch.nn.functional.cosine_similarity
    return {
        "dual_gnorm_primary": float(ga.norm(dim=1).mean()),
        "dual_gnorm_td": float(gb.norm(dim=1).mean()),
        "dual_gcos_row": float(cos(ga, gb, dim=1, eps=1e-12).mean()),
        "dual_gcos_batch": float(cos(ga.mean(0), gb.mean(0), dim=0, eps=1e-12)),
    }


def _check_row_chunk_ownership(dvd_term, actor_optimizer) -> None:
    """--dvd-row-chunks N > 1 hands the actor the DVD gradient through the ACTION ROWS only (DvdTerm._chunked_value)."""
    key = tuple(len(g["params"]) for g in actor_optimizer.param_groups)
    if getattr(dvd_term, "_row_chunk_owner_key", None) == key:
        return
    inside = {}
    for label, mod in (("value_fn", getattr(dvd_term, "value_fn", None)),
                       ("dynamics", getattr(dvd_term, "dynamics", None)),
                       ("pool", getattr(dvd_term, "_pool", None)),
                       ("value_pool", getattr(dvd_term, "_value_pool", None))):
        if isinstance(mod, torch.nn.Module):
            for p in mod.parameters():
                inside.setdefault(id(p), label)
    owned = sorted({inside[id(p)] for g in actor_optimizer.param_groups for p in g["params"]
                    if id(p) in inside})
    if owned:
        raise ValueError(
            f"--dvd-row-chunks > 1: the actor optimiser owns parameters "
            f"of {', '.join(owned)} inside the DVD graph (--dvd-actor-pool / --dvd-pool-actor-step?). The chunked "
            f"path gives the actor its gradient through the action rows only, so those parameters would lose "
            f"their L_DVD gradient. Use --dvd-row-chunks 1 for this configuration.")
    dvd_term._row_chunk_owner_key = key


def dvd_score(dvd_term, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """gamma^H V(f(s, a)), per sample -- the actor's own objective, usable as a ranking signal."""
    with torch.no_grad():
        predicted = dvd_term.dynamics(dvd_term.raw_state(state), action)
        return _one_chunk_discount(dvd_term) * dvd_term.value_fn(dvd_term.value_input(
            predicted, state, dvd_term.model_proprio(predicted)))


def _one_chunk_discount(dvd_term) -> float:
    """gamma^H for a ONE-chunk score."""
    fn = getattr(dvd_term, "lookahead_discount", None)
    return dvd_term.discount if fn is None else fn(1)


def check_bon_lookahead(dvd_term, lookahead: int) -> None:
    """--bon-lookahead 2 runs DvdTerm.feature_step2, which needs the DINO-WM proprio head and a V that reads predicted proprio (the same requirements as feature-space --dvd-chunks 2)."""
    if int(lookahead) == 1:
        return
    if int(lookahead) != 2:
        raise ValueError(f"--bon-lookahead {lookahead}: only 1 and 2 are defined")
    if not (getattr(dvd_term, "feature_space", False) and getattr(dvd_term, "state_head_on", False)
            and getattr(dvd_term, "proprio_source", "current") == "predicted"
            and hasattr(dvd_term.dynamics, "ctx_feats")):
        raise ValueError("--bon-lookahead 2 needs the DINO-WM feature path with its proprio head and "
                         "--dvd-proprio-source predicted (--dvd-dynamics dino_wm --dino-wm-state-head)")


def dvd_candidate_scores(dvd_term, state: torch.Tensor, action: torch.Tensor, frozen_value_fn=None,
                         cont: torch.Tensor | None = None, lookahead: int = 1):
    """Best-of-N scoring on ONE world-model pass -> (online, frozen or None, members or None), per candidate."""
    if int(lookahead) > 1:
        check_bon_lookahead(dvd_term, lookahead)
        if cont is None or cont.shape != action.shape:
            raise ValueError(f"--bon-lookahead 2: each candidate needs its continuation (got "
                             f"{None if cont is None else tuple(cont.shape)} for actions {tuple(action.shape)})")
    with torch.no_grad():
        predicted = dvd_term.dynamics(dvd_term.raw_state(state), action)
        disc = _one_chunk_discount(dvd_term)
        if int(lookahead) > 1:
            predicted = dvd_term.feature_step2(predicted, dvd_term.dynamics.last_state_pred, cont)
            disc = dvd_term.lookahead_discount(2)
        x = dvd_term.value_input(predicted, state, dvd_term.model_proprio(predicted))
        online = (disc * dvd_term.value_fn(x)).squeeze(-1)
        frozen = None if frozen_value_fn is None else (disc * frozen_value_fn(x)).squeeze(-1)
        members = None
        vf = dvd_term.value_fn
        if isinstance(vf, ValueEnsemble) and getattr(vf, "_delta", None) is None:
            members = disc * torch.stack([net(x) for net in vf.nets]).squeeze(-1)
        return online, frozen, members


BON_TARGET_TANH_LIMIT = 0.95


def bon_draw_targets(score_members, state: torch.Tensor, base: torch.Tensor, samples: int,
                     rows: torch.Tensor | None, expert: torch.Tensor | None, residual_scale: float,
                     max_rows: int = 256, max_norm: float = 0.0, ramp: float = 1.0):
    """--bc-target bon:"""
    k = int(samples)
    n_states, dim = int(state.shape[0]), int(base.shape[-1])
    if k < 2:
        raise ValueError(f"--bc-target bon needs >= 2 prior draws per state to choose between, got {k}")
    if base.shape[0] != n_states * k:
        raise ValueError(
            f"--bc-target bon: base has {base.shape[0]} tiled rows but state has {n_states} rows x {k} draws; "
            "the (state r // K, draw r % K) layout does not hold")
    if not float(residual_scale) > 0.0:
        raise ValueError(f"--bc-target bon: residual_scale must be > 0, got {residual_scale}")
    if not 0.0 <= float(ramp) <= 1.0:
        raise ValueError(f"--bc-target bon: ramp must be in [0, 1], got {ramp}")
    scale, cap = float(residual_scale), float(max_norm)
    with torch.no_grad():
        dev = base.device
        rows = (torch.arange(n_states * k, device=dev) if rows is None
                else rows.to(device=dev, dtype=torch.long).reshape(-1))
        n_rows = int(rows.numel())
        own = rows % k
        is_expert = (torch.zeros(n_rows, dtype=torch.bool, device=dev) if expert is None
                     else expert.to(dev).reshape(-1)[rows].bool())
        states, inv = torch.unique(torch.div(rows, k, rounding_mode="floor"), sorted=True, return_inverse=True)
        n_u = int(states.numel())

        ctx_pos = torch.full((n_u,), n_rows, dtype=torch.long, device=dev)
        ctx_pos.scatter_reduce_(0, inv, torch.arange(n_rows, device=dev), reduce="amin")

        scored = torch.zeros(n_u, dtype=torch.bool, device=dev)
        scored[inv[~is_expert]] = True
        pick = scored.nonzero().squeeze(1)
        n_scored = int(pick.numel())
        draws = base.detach().reshape(n_states, k, dim)[states]
        state_u = state.detach()[states]
        centered = torch.full((n_u, k), float("-inf"), device=dev)
        if n_scored:
            per_call = max(1, int(max_rows) // k)
            parts = []
            for lo in range(0, n_scored, per_call):
                sel = pick[lo:lo + per_call]
                n_sel = int(sel.numel())
                m = score_members(ctx_pos[sel], state_u[sel].repeat_interleave(k, dim=0),
                                  draws[sel].reshape(n_sel * k, dim))
                m = m.detach().float()
                parts.append((m.unsqueeze(0) if m.dim() == 1 else m).reshape(-1, n_sel, k))
            members = torch.cat(parts, dim=1)
            cen = (members - members.mean(dim=2, keepdim=True)).mean(dim=0)
            centered[pick] = torch.where(torch.isfinite(cen), cen, torch.full_like(cen, float("-inf")))
        best = centered.argmax(dim=1)[inv]
        score_rows = centered[inv]
        own_score = score_rows.gather(1, own.unsqueeze(1)).squeeze(1)
        best_score = score_rows.gather(1, best.unsqueeze(1)).squeeze(1)
        keep = (own_score >= best_score) | is_expert
        chosen = torch.where(keep, own, best)
        delta = draws[inv, chosen] - base.detach()[rows]
        reach = delta / scale
        if cap > 0.0:
            reach = reach * (cap / reach.norm(dim=-1, keepdim=True).clamp_min(cap))
        clipped = reach.abs() > BON_TARGET_TANH_LIMIT
        reach = reach.clamp(-BON_TARGET_TANH_LIMIT, BON_TARGET_TANH_LIMIT)
        target = torch.where(keep.unsqueeze(-1), torch.zeros_like(reach), float(ramp) * reach)
        stats: dict[str, float] = {}
        online = ~is_expert
        if bool(online.any()):
            changed, zero = ~keep, torch.zeros_like(own_score)
            stats["bc_target_changed"] = float(changed[online].float().mean())
            stats["bc_target_dist"] = float(delta.norm(dim=-1)[online].mean())
            stats["bc_target_exec_dist"] = float(
                torch.where(changed, (scale * reach).norm(dim=-1), zero)[online].mean())
            stats["bc_target_clip_frac"] = float(
                torch.where(changed, clipped.float().mean(dim=-1), zero)[online].mean())
            stats["bc_target_score_gap"] = float(torch.where(keep, zero, best_score - own_score)[online].mean())
            stats["bc_target_states"] = float(n_scored)
    return target, stats


def dino_bon_member_scorer(dvd_term, context_rows: int):
    """score_members for bon_draw_targets on --dvd-dynamics dino_wm:"""
    def score(ctx_pos: torch.Tensor, state_rep: torch.Tensor, action_rep: torch.Tensor) -> torch.Tensor:
        adapter = dvd_term.dynamics
        if getattr(adapter, "flow", False):
            raise RuntimeError(
                "--bc-target bon: the world model is a flow head, whose forward draws fresh noise per candidate, so "
                "the K-draw ranking would rank noise (train() should have refused it)")
        ctx = getattr(adapter, "ctx_feats", None)
        if ctx is None or int(ctx.shape[0]) != int(context_rows):
            raise RuntimeError(
                f"--bc-target bon: the installed world-model context has "
                f"{None if ctx is None else int(ctx.shape[0])} rows, expected the {int(context_rows)} DVD rows")
        n_states = int(ctx_pos.shape[0])
        if n_states == 0 or action_rep.shape[0] % n_states or state_rep.shape[0] != action_rep.shape[0]:
            raise RuntimeError("--bc-target bon: the candidates are not a whole number of draws per state")
        modes = [(m, m.training) for owner in (adapter, getattr(dvd_term, "value_fn", None))
                 if isinstance(owner, torch.nn.Module) for m in owner.modules()]
        adapter.ctx_feats = ctx[ctx_pos].repeat_interleave(action_rep.shape[0] // n_states, dim=0)
        try:
            for m, _ in modes:
                m.training = False
            _, _, members = dvd_candidate_scores(dvd_term, state_rep, action_rep)
        finally:
            adapter.ctx_feats = ctx
            for m, was in modes:
                m.training = was
        if members is None:
            raise RuntimeError(
                "--bc-target bon ranks the prior draws by the V ensemble's per-net scores, and this V has none "
                "(a residual delta head, or not a plain ValueEnsemble); check_bc_target should have refused it")
        return members
    return score


_PHASE_WARNED: list = []
EVAL_PHASES = {1: "1_REACH", 2: "2_GRASP", 3: "3_TRANSPORT", 4: "4_INSERT"}
_PHASE_GEOM = {"threading_d0": dict(grasp_mm=90.0, lift_mm=20.0, goal_mm=150.0)}


_SEG_EEF, _SEG_NEEDLE, _SEG_TRIPOD = slice(0, 3), slice(9, 12), slice(23, 26)
_SEG_GRASP_MM, _SEG_HOLD, _SEG_HOLD_FRAC = 80.0, 40, 0.9

_SEG_REF = np.array([0.0007828745245933533, 0.03100725555419922, 0.08943326568603516], np.float32)


class _SegTerminator:
    """Causal segment cut."""

    def __init__(self, patience: int, stride: int = 8):
        self.stride = max(1, int(stride))
        self.patience = int(patience)
        self._hold_calls = max(1, round(_SEG_HOLD / self.stride))
        self._patience_calls = max(1, round(self.patience / self.stride))
        self.reset()

    def reset(self) -> None:
        self._near: list[bool] = []
        self._grasped = False
        self._best = float("inf")
        self._stale = 0

    def __call__(self, raw) -> bool:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not self._grasped:
            d = float(np.linalg.norm(raw[_SEG_EEF] - raw[_SEG_NEEDLE])) * 1000.0
            self._near.append(d < _SEG_GRASP_MM)
            if len(self._near) > self._hold_calls:
                self._near.pop(0)
            if (len(self._near) == self._hold_calls
                    and np.mean(self._near) > _SEG_HOLD_FRAC):
                self._grasped = True
            return False
        r = float(np.linalg.norm((raw[_SEG_NEEDLE] - raw[_SEG_TRIPOD]) - _SEG_REF))
        if r < self._best - 1e-9:
            self._best, self._stale = r, 0
            return False
        self._stale += 1
        return self._stale >= self._patience_calls


def _eval_phase_of(raw, geom, z0: float) -> int:
    """Phase of one RAW 37-dim env state, given the needle height at episode start."""
    import numpy as _np
    d_en = float(_np.linalg.norm(raw[0:3] - raw[9:12])) * 1000.0
    d_nt = float(_np.linalg.norm(raw[9:12] - raw[23:26])) * 1000.0
    lift = (float(raw[11]) - z0) * 1000.0
    if lift < geom["lift_mm"]:
        return 2 if d_en < geom["grasp_mm"] else 1
    return 4 if d_nt < geom["goal_mm"] else 3


def _phase_of_rows(dvd_term, batch, samples):
    """Coarse task phase per row, for bucketing the gradient-SNR statistics."""
    if dvd_term is None or not hasattr(dvd_term, "raw_state"):
        return None


    key = "env_state" if "env_state" in batch else "state"
    scaled = batch.get(key)
    if scaled is None:
        return None


    if not (26 <= scaled.shape[-1] <= 64):
        if not _PHASE_WARNED:
            print(f"[gradvar] per-phase buckets OFF: batch['{key}'] is "
                  f"{scaled.shape[-1]}-wide, need 26..64 (a raw env state, not features)",
                  flush=True)
            _PHASE_WARNED.append(1)
        return None
    if samples > 1:
        scaled = scaled.repeat_interleave(samples, dim=0)
    try:
        raw = dvd_term.raw_state(scaled)
        d_en = (raw[:, 0:3] - raw[:, 9:12]).norm(dim=-1) * 1000.0
        d_nt = (raw[:, 9:12] - raw[:, 23:26]).norm(dim=-1) * 1000.0
    except Exception as exc:
        if not _PHASE_WARNED:
            print(f"[gradvar] per-phase buckets OFF: {type(exc).__name__}: {exc}", flush=True)
            _PHASE_WARNED.append(1)
        return None
    phase = torch.full_like(d_en, 1.0)
    phase = torch.where(d_en < 90.0, torch.full_like(phase, 2.0), phase)
    phase = torch.where(d_nt < 150.0, torch.full_like(phase, 3.0), phase)
    return phase.detach()


def _dvd_follow(batch, config, dvd_term, samples: int, chunk_dim: int, base_from_samples: bool):
    """The chunk DVD(k>1) executes after the optimised one, row-aligned with update_dice's tiled `base`."""
    if dvd_term.chunks <= 1:
        return None
    if getattr(config, "dvd_follow", "recorded") == "continuation":
        if base_from_samples:
            return batch["base_cont_samples"].view(-1, chunk_dim)
        return batch["base_cont"].repeat_interleave(samples, dim=0)
    return batch["next_base"].repeat_interleave(samples, dim=0)


def _tile_rows(batch_size: int, samples: int, device) -> torch.Tensor:
    """Row of `batch` that each row of a repeat_interleave(samples) tiling came from."""
    return torch.arange(batch_size, device=device).repeat_interleave(samples)


TRUST_RADIUS = 0.05


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation between two 1-D tensors; nan when it is undefined."""
    a, b = a.flatten().float(), b.flatten().float()
    n = a.numel()
    if n < 3:
        return float("nan")


    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    ra = torch.empty(n, device=a.device); ra[a.argsort()] = torch.arange(n, dtype=torch.float32, device=a.device)
    rb = torch.empty(n, device=b.device); rb[b.argsort()] = torch.arange(n, dtype=torch.float32, device=b.device)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) < 1e-12:
        return float("nan")
    return float((ra @ rb) / denom)


class TrainingGate:
    """Feedback-gated training:"""

    def __init__(self, skill_tau, hyst, td_tau, td_hyst, dwell, slow,
                 actor_opt, value_opt, dvd_opt):
        self.skill_tau, self.hyst = skill_tau, hyst
        self.td_tau, self.td_hyst = td_tau, td_hyst
        self.dwell, self.slow = int(dwell), float(slow)
        self.mode, self.since = "dyn", 0
        self.skill, self.td = None, None
        self._opts = {"pi": actor_opt, "val": value_opt, "dyn": dvd_opt}
        for opt in self._opts.values():
            for g in opt.param_groups:
                g.setdefault("_gate_base_lr", g["lr"])

    def observe(self, feat_mse, copy_mse, td_loss):
        if feat_mse is not None and copy_mse:
            sk = 1.0 - feat_mse / max(copy_mse, 1e-8)
            self.skill = sk if self.skill is None else 0.98 * self.skill + 0.02 * sk
        if td_loss is not None:
            self.td = td_loss if self.td is None else 0.98 * self.td + 0.02 * td_loss

    def step(self):
        self.since += 1
        sk = self.skill if self.skill is not None else -1.0
        td = self.td if self.td is not None else float("inf")
        if self.since >= self.dwell:
            new = self.mode
            if self.mode != "dyn" and sk < self.skill_tau - self.hyst:
                new = "dyn"
            elif self.mode == "dyn" and sk >= self.skill_tau:
                new = "val"
            elif self.mode == "val" and td <= self.td_tau:
                new = "pol"
            elif self.mode == "pol" and td > self.td_tau + self.td_hyst:
                new = "val"
            if new != self.mode:
                print(f"[gate] {self.mode} -> {new}  skill={sk:.3f} td={td:.5f}", flush=True)
                self.mode, self.since = new, 0
        gf = 1.0 if self.mode == "dyn" else (self.slow if self.mode == "pol" else 0.0)
        gv = 1.0 if self.mode == "val" else (self.slow if self.mode == "pol" else 0.0)
        gp = 1.0 if self.mode == "pol" else 0.0
        for key, g in (("dyn", gf), ("val", gv), ("pi", gp)):
            for grp in self._opts[key].param_groups:
                grp["lr"] = grp["_gate_base_lr"] * g
        return {"gate_mode": {"dyn": 0.0, "val": 1.0, "pol": 2.0}[self.mode],
                "gate_skill": sk, "gate_td": td if td != float("inf") else -1.0,
                "gate_gf": gf, "gate_gv": gv, "gate_gpi": gp}


def update_dice(
    batch: dict[str, torch.Tensor],
    actor: ResidualActor,
    critics: CriticEnsemble,
    target_critics: CriticEnsemble,
    actor_optimizer,
    critic_optimizer,
    config: TrainConfig,
    updates: int = 0,
    expert_ratio: float = 0.0,
    expert_count: int = 0,
    value_fn=None,
    dvd_term=None,
    encoder=None,
    target_encoder=None,
    mbpo_constants=None,
    prior_surrogate=None,
    dino=None,
    target_actor: ResidualActor | None = None,
    env_steps_now: int = 0,
    probe_grads: dict | None = None,


    v_aux_head=None,
    actor_encoder=None,
) -> dict[str, float]:


    chunk_dim = batch["base"].shape[-1]


    actor_state_key = "state"


    dvd_state_key, dvd_next_key = "state", "next_state"
    if encoder is not None:
        batch = dict(batch)
        batch["env_state"] = batch["state"]
        batch["env_next_state"] = batch["next_state"]
        dvd_state_key, dvd_next_key = "env_state", "env_next_state"
        if getattr(config, "dvd_on_encoder", False):


            dvd_state_key, dvd_next_key = "state", "next_state"
        proprio = batch["state"][:, :PIXEL_PROPRIO]
        next_proprio = batch["next_state"][:, :PIXEL_PROPRIO]


        _aug_c = config.pixel_augment and config.aug_target in ("both", "critic")
        _aug_a = config.pixel_augment and config.aug_target in ("both", "actor")
        batch["state"] = encoder(batch["image"], proprio, augment=_aug_c)
        if getattr(config, "encoder_value_aux_only", False):


            batch["enc_state"] = batch["state"]
            batch["state"] = batch["state"].detach()
        with torch.no_grad():


            batch["next_state"] = target_encoder(batch.get("next_image_boot", batch["next_image"]),
                                                 next_proprio, augment=_aug_c)
        if actor_encoder is not None:


            batch["actor_state"] = actor_encoder(batch["image"], proprio, augment=_aug_a)
        elif config.actor_train_encoder and _aug_a == _aug_c:


            batch["actor_state"] = batch["state"]
        elif _aug_a == _aug_c:
            batch["actor_state"] = batch["state"].detach()
        else:


            with torch.no_grad():
                batch["actor_state"] = encoder(batch["image"], proprio, augment=_aug_a).detach()
        actor_state_key = "actor_state"
    elif config.dino_obs:


        batch = dict(batch)
        batch["env_state"] = batch["state"]
        batch["env_next_state"] = batch["next_state"]
        dvd_state_key, dvd_next_key = "env_state", "env_next_state"


        def _shift(img):
            if config.dino_shift_pad <= 0 and config.dino_photometric <= 0:
                return img
            b, v, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
            x = img.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w).float()
            if config.dino_shift_pad > 0:
                x = random_shift(x, config.dino_shift_pad)
            if config.dino_photometric > 0:


                st = config.dino_photometric
                gain = 1.0 + (torch.rand(b * v, 1, 1, 1, device=x.device) * 2 - 1) * st
                bias = (torch.rand(b * v, 1, 1, 1, device=x.device) * 2 - 1) * st * 0.5 * 255
                x = (x - 127.5) * gain + 127.5 + bias
            return (x.round().clamp_(0, 255).to(torch.uint8)
                     .reshape(b, v, 3, h, w).permute(0, 1, 3, 4, 2).contiguous())


        _obs_pool = getattr(dvd_term, "_obs_pool", None)
        _pool_grad = _obs_pool is not None or getattr(dvd_term, "_pool_obs_grad", False)
        if not _pool_grad:
            with torch.no_grad():


                f_now = _obs_pool_of(dvd_term)(
                    dino.encode_frames(dvd_term.dynamics, _shift(batch["image"])),
                    dvd_term.value_pool_mode)
                f_next = _obs_pool_of(dvd_term)(
                    dino.encode_frames(dvd_term.dynamics, _shift(batch["next_image"])),
                    dvd_term.value_pool_mode)
        else:


            _pool_fn = _obs_pool if _obs_pool is not None else _obs_pool_of(dvd_term)


            _pool_tgt = getattr(dvd_term, "_obs_pool_target", None)
            with torch.no_grad():
                _z_now = dino.encode_frames(dvd_term.dynamics, _shift(batch["image"]))
            f_now = _pool_fn(_z_now, dvd_term.value_pool_mode)
            del _z_now
            with torch.no_grad():
                _z_next = dino.encode_frames(dvd_term.dynamics, _shift(batch["next_image"]))
                f_next = (_pool_tgt if _pool_tgt is not None else _pool_fn)(
                    _z_next, dvd_term.value_pool_mode)
            del _z_next
        p_now = dvd_term.raw_state(batch["env_state"])[:, :dvd_term.proprio_dim]
        p_next = dvd_term.raw_state(batch["env_next_state"])[:, :dvd_term.proprio_dim]
        batch["state"] = torch.cat([f_now, p_now], dim=-1)
        batch["next_state"] = torch.cat([f_next, p_next], dim=-1)
        if config.state_augment > 0.0:


            sc_ = batch["state"].std(0, keepdim=True).clamp_min(1e-6)
            sig = config.state_augment * sc_
            batch["state"] = batch["state"] + sig * torch.randn_like(batch["state"])
            batch["next_state"] = batch["next_state"] + sig * torch.randn_like(batch["next_state"])


        batch["actor_state"] = (batch["state"].detach() if _pool_grad
                                else batch["state"])
        actor_state_key = "actor_state"
    elif getattr(config, "seg_obs", False):


        batch = dict(batch)
        batch["env_state"] = batch["state"]
        batch["env_next_state"] = batch["next_state"]
        dvd_state_key, dvd_next_key = "env_state", "env_next_state"
        if updates < 3 and float(batch["next_feat"].abs().max()) == 0.0:
            raise RuntimeError("--seg-obs: next_feat is all zeros in the first batch, so the TD "
                               "target reads a blank map. Rebuild the expert cache and do not "
                               "resume a pre-next_feat replay.")
        _se = _LATENT_OBS_CTX["seg_enc"]
        _to_map = SegHeadAdapter.to_map
        _pn = batch["env_state"][:, :PIXEL_PROPRIO]
        _px = batch["env_next_state"][:, :PIXEL_PROPRIO]
        if getattr(config, "seg_obs_finetune", False):


            from .seghead import encode_grad as _sg_encode, norm_feat as _sg_norm
            _dn, _hd = _LATENT_OBS_CTX["seg_dino"], _LATENT_OBS_CTX["seg_head"]
            _pad = getattr(config, "obs_shift_pad", 0)
            _im = _shift_frames(batch["image"], _pad)
            _imx = _shift_frames(batch["next_image"], _pad)
            _m = _sg_norm(_sg_encode(_dn, _hd, _im, _pn, _im.device), _im.device)
            batch["state"] = torch.cat([_se(_m), _pn], dim=-1)
            with torch.no_grad():
                _mx = _sg_norm(_sg_encode(_dn, _hd, _imx, _px, _im.device), _im.device)
                batch["next_state"] = torch.cat([_se(_mx), _px], dim=-1)
        else:
            batch["state"] = torch.cat([_se(_to_map(batch["seg_feat"].float())), _pn], dim=-1)
            with torch.no_grad():
                batch["next_state"] = torch.cat(
                    [_se(_to_map(batch["next_feat"].float())), _px], dim=-1)
        if config.state_augment > 0.0:
            _sc = batch["state"].std(0, keepdim=True).clamp_min(1e-6)
            _sg = config.state_augment * _sc
            batch["state"] = batch["state"] + _sg * torch.randn_like(batch["state"])
            batch["next_state"] = batch["next_state"] + _sg * torch.randn_like(batch["next_state"])


        batch["actor_state"] = batch["state"].detach()
        actor_state_key = "actor_state"
    elif getattr(config, "latent_obs", False):


        batch = dict(batch)
        batch["env_state"] = batch["state"]
        batch["env_next_state"] = batch["next_state"]
        dvd_state_key, dvd_next_key = "env_state", "env_next_state"
        if "next_feat" not in batch:
            raise RuntimeError("--latent-obs needs the buffer's next_feat column; this buffer "
                               "was built without it (see ReplayBuffer.seg_feat)")
        if updates < 3 and float(batch["next_feat"].abs().max()) == 0.0:


            raise RuntimeError("--latent-obs: next_feat is all zeros in the first batch, so the "
                               "TD target would read a blank z'. Rebuild the expert cache "
                               "(the key gains '_nf') and do not resume a pre-next_feat replay.")
        _cfg = _LATENT_OBS_CTX["cfg"]
        _scl = _LATENT_OBS_CTX["scaler"]
        _pn = _latent_obs_state(batch, "env_state", _cfg, _scl)
        _px = _latent_obs_state(batch, "env_next_state", _cfg, _scl)
        _tpad = getattr(config, "tok_shift_pad", 0)
        _pad = getattr(config, "obs_shift_pad", 0)
        if _tpad > 0:


            _en = _LATENT_OBS_CTX["enc"]
            _ft = getattr(config, "latent_obs_finetune", False)

            def _sh(tk):
                b, v, pp, dd = tk.shape
                g = int(round(pp ** 0.5))
                x = tk.reshape(b * v, g, g, dd).permute(0, 3, 1, 2).float()
                x = random_shift(x, _tpad)
                return x.permute(0, 2, 3, 1).reshape(b, v, pp, dd)

            _z = _en(_sh(batch["tok"]))
            if not _ft:
                _z = _z.detach()
            with torch.no_grad():
                _zx = _en(_sh(batch["next_tok"]))
            batch["state"] = torch.cat([_z, _pn], dim=-1)
            batch["next_state"] = torch.cat([_zx, _px], dim=-1)
        elif _pad > 0:


            from .latent56 import tokens as _lo_tok
            _dn, _en = _LATENT_OBS_CTX["dino"], _LATENT_OBS_CTX["enc"]
            _ft = getattr(config, "latent_obs_finetune", False)
            _tk = _lo_tok(_dn, _shift_frames(batch["image"], _pad), _cfg, batch["state"].device)
            _z = _en(_tk) if _ft else _en(_tk).detach()
            with torch.no_grad():
                _zx = _en(_lo_tok(_dn, _shift_frames(batch["next_image"], _pad), _cfg,
                                  batch["state"].device))
            batch["state"] = torch.cat([_z, _pn], dim=-1)
            batch["next_state"] = torch.cat([_zx, _px], dim=-1)
        elif getattr(config, "latent_obs_finetune", False):


            _enc = _LATENT_OBS_CTX["enc"]
            _z = _enc(batch["tok"].float())
            with torch.no_grad():
                _zx = _enc(batch["next_tok"].float())
            batch["state"] = torch.cat([_z, _pn], dim=-1)
            batch["next_state"] = torch.cat([_zx, _px], dim=-1)

        else:
            batch["state"] = torch.cat([batch["seg_feat"].float(), _pn], dim=-1)
            batch["next_state"] = torch.cat([batch["next_feat"].float(), _px], dim=-1)
        if config.state_augment > 0.0:


            _sc = batch["state"].std(0, keepdim=True).clamp_min(1e-6)
            _sg = config.state_augment * _sc
            batch["state"] = batch["state"] + _sg * torch.randn_like(batch["state"])
            batch["next_state"] = batch["next_state"] + _sg * torch.randn_like(batch["next_state"])


        batch["actor_state"] = (batch["state"].detach()
                                if getattr(config, "latent_obs_finetune", False)
                                else batch["state"])
        actor_state_key = "actor_state"
    elif config.state_augment > 0.0:


        batch = dict(batch)
        batch["env_state"] = batch["state"]
        batch["env_next_state"] = batch["next_state"]
        dvd_state_key, dvd_next_key = "env_state", "env_next_state"
        sigma = config.state_augment
        batch["state"] = batch["state"] + sigma * torch.randn_like(batch["state"])
        batch["next_state"] = batch["next_state"] + sigma * torch.randn_like(batch["next_state"])
        batch["actor_state"] = batch["state"].detach()
        actor_state_key = "actor_state"
    k = config.multi_sample_k


    need_critic_targets = (config.train_critic
                           or config.critical_enabled()
                           or config.value_intervention == "penalty"
                           or config.mbpo_weight > 0.0)
    with torch.no_grad():


        next_base = batch["next_base_samples"].view(-1, chunk_dim) if k > 1 else batch["next_base"]
        next_state = batch["next_state"].repeat_interleave(k, dim=0) if k > 1 else batch["next_state"]


        bootstrap_actor = target_actor if target_actor is not None else actor
        if need_critic_targets:
            next_residual, _ = bootstrap_actor.sample(next_state, next_base)
            next_action = final_action(next_base, next_residual, config.residual_scale,
                                       config.residual_max_norm)
            next_q = target_critics.conservative(next_state, next_action)
        else:

            next_q = torch.zeros((next_state.shape[0], 1), device=next_state.device)
        if k > 1:
            next_q = next_q.view(-1, k, 1).mean(dim=1)
        if getattr(config, "rosmoq", False) and dvd_term is not None and dino is not None:


            _n = batch["reward"].shape[0]
            _ridx = dino.select_rows(_n, config.rosmoq_rows, batch["reward"].device)
            if _ridx is None:
                _ridx = torch.arange(_n, device=batch["reward"].device)
            _prev = dino.encode_frames(dvd_term.dynamics, batch["image"][_ridx])
            _cur = dino.encode_frames(dvd_term.dynamics, batch["next_image"][_ridx])
            dino.set_context(dvd_term.dynamics, _prev, _cur)
            _tiled = _ridx * k
            _a_next = next_action[_tiled] if k > 1 else next_action[_ridx]
            _s_next = batch["next_state"][_ridx]
            _t, _ = dvd_term.objective_terms(_s_next, _a_next, None)
            _score = (-_t.reshape(-1, 1)) * config.dvd_fixed_scale
            _w = config.rosmoq_weight
            next_q[_ridx] = (1.0 - _w) * next_q[_ridx] + _w * _score
            rolling["rosmoq_score_mean"].append(float(_score.mean()))
            rolling["rosmoq_vs_q_gap"].append(float((_score - next_q[_ridx]).mean()))
        target = batch["reward"] + (1.0 - batch["done"]) * batch["discount"] * next_q

    value_diagnostics: dict[str, float] = {}
    if config.zero_floor:


        with torch.no_grad():
            live = batch["done"] <= 0.5
            floored = torch.where(live, target.clamp_min(0.0), target)
            binds = floored > target
            value_diagnostics = {
                "zero_floor_binding_fraction": float(binds.float().mean()),
                "zero_floor_lift": float((floored - target)[binds].mean()) if binds.any() else 0.0,
            }
            target = floored
    if value_fn is not None and config.value_intervention == "floor":


        with torch.no_grad():
            raw_floor = batch["discount"] * value_fn.frozen(batch["next_state"])
            live = batch["done"] <= 0.5
            floor = torch.where(live, raw_floor, target)
            binds = floor > target
            value_diagnostics = {


                "v_floor_binding_fraction": float(binds.float().mean()),
                "v_floor_lift": float((floor - target)[binds].mean()) if binds.any() else 0.0,
                "v_next_mean": float(raw_floor[live].mean()) if live.any() else 0.0,
            }
            target = torch.maximum(target, floor)

    if need_critic_targets:
        q_values = critics(batch["state"], batch["action"])
        critic_loss = sum(F.smooth_l1_loss(value_, target)
                          for value_ in q_values) / q_values.shape[0]
    else:


        q_values = torch.zeros((2, batch["state"].shape[0], 1), device=batch["state"].device)
        critic_loss = torch.zeros((), device=batch["state"].device)
    critical_stats: dict[str, float] = {}
    if config.critical_enabled() and updates % CRITICAL_DIAGNOSTIC_EVERY == 0:

        critical_stats = critical_diagnostics(batch, critics, target, q_values)
    if value_fn is not None and config.value_intervention == "penalty":


        with torch.no_grad():
            v_state = value_fn.frozen(batch["state"])
        deficit = F.relu(v_state - critics.conservative(batch["state"], batch["base"]))
        critic_loss = critic_loss + config.value_penalty_beta * deficit.mean()
        value_diagnostics = {
            "v_penalty": float(deficit.mean().detach()),
            "v_penalty_active_fraction": float((deficit > 0).float().mean()),
            "v_state_mean": float(v_state.mean()),
        }
    mbpo_stats: dict[str, float] = {}
    if config.mbpo_weight > 0.0 and mbpo_constants is not None and dvd_term is not None:


        from online_rl_mimicgen_mbpo import mbpo

        rows = mbpo.branch_rollout(
            model=dvd_term,
            actor=actor,
            critics=critics,
            batch=batch,
            constants=mbpo_constants,
            chunk_dim=chunk_dim,
            residual_scale=config.residual_scale,
            residual_max_norm=config.residual_max_norm,
            step_penalty=config.step_penalty,
            discount=config.gamma,


            samples=0,
            best_of_n=config.mbpo_best_of_n,
            final_action_fn=final_action,
            chunks=config.mbpo_chunks,
            prior_surrogate=prior_surrogate,
            max_sigma_mm=config.mbpo_max_sigma_mm,
            success_margin_mm=config.mbpo_success_margin_mm,
        )
        if rows.get("empty"):


            mbpo_stats = {"mbpo_gated_fraction": 1.0, "mbpo_rows_per_branch": 0.0}
        else:
            synthetic_loss = mbpo.model_critic_loss(rows, critics, value_fn, F.smooth_l1_loss)
            critic_loss = critic_loss + config.mbpo_weight * synthetic_loss
            mbpo_stats = {
                "mbpo_critic_loss": float(synthetic_loss.detach()),
                "mbpo_reward_mean": float(rows["reward"].mean()),


                "mbpo_success_fraction": rows["success_fraction"],


                "mbpo_rows_per_branch": rows["rows_per_branch"],
                "mbpo_depth_mean": float(rows["depth"].mean()),
                "mbpo_gated_fraction": rows["gated_fraction"],


                "mbpo_baseline_success_fraction": rows["baseline_success_fraction"],
                "mbpo_selection_gap": rows["selection_gap"],


                "mbpo_real_success_fraction": rows["real_success_fraction"],
                "mbpo_calibration_gap": rows["calibration_gap"],
            }
    if not config.train_critic:


        critic_loss = torch.zeros((), device=batch["state"].device)
        critic_grad = torch.zeros((), device=batch["state"].device)
    elif config.freeze_init_critic:


        critic_grad = torch.zeros((), device=critic_loss.device)
    else:


        _vaux_critic = (getattr(config, "encoder_value_aux", 0.0) > 0.0
                        and v_aux_head is not None and encoder is not None
                        and not config.actor_train_encoder)
        if _vaux_critic:
            _vaux_v_c = _value_read(value_fn, dvd_term)
        if _vaux_critic and _vaux_v_c is not None:
            with torch.no_grad():
                if dvd_term is not None and getattr(dvd_term, "feature_space", False):
                    from . import dino_dvd as _ddvd_c
                    _cur_c = _ddvd_c.encode_frames(dvd_term.dynamics, batch["image"]).float()
                    _yv_c = _vaux_v_c(dvd_term.value_input(
                        _cur_c, batch.get("env_state", batch["state"])))
                else:
                    _yv_c = _vaux_v_c(batch.get("env_state", batch["state"]))
                _yv_c = _yv_c.detach()
            # ATTACHED encoding: this is the whole point -- the gradient reaches the encoder,

            _xaux_c = batch["state"]
            if getattr(config, "encoder_value_aux_image_only", False):
                _xaux_c = _xaux_c[:, :-PIXEL_PROPRIO]
            _laux_c = (v_aux_head(_xaux_c) - _yv_c).pow(2).mean()
            critic_loss = critic_loss + config.encoder_value_aux * _laux_c


            _vy = _yv_c.detach()
            _vaux_critic_metric = {
                "encoder_value_aux": float(_laux_c.detach()),
                "encoder_value_aux_unexpl": float(
                    _laux_c.detach() / _vy.var().clamp_min(1e-8)),
            }
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(critics.parameters(), 10.0)
        critic_optimizer.step()


    batch_size = batch["state"].shape[0]
    actor_k = config.resolved_actor_k()
    if actor_k > 1:
        samples = actor_k
        state = batch[actor_state_key].repeat_interleave(actor_k, dim=0)
        base = batch["base_samples"].view(-1, chunk_dim)
    elif k > 1:
        samples = k
        state = batch[actor_state_key].repeat_interleave(k, dim=0)
        base = batch["base_samples"].view(-1, chunk_dim)
    else:


        samples = 1 if (config.deterministic_residual and not config.latent_dim) else 4
        state = batch[actor_state_key].repeat_interleave(samples, dim=0)
        base = batch["base"].repeat_interleave(samples, dim=0)


    _base_from_samples = actor_k > 1 or k > 1
    residual, _ = actor.sample(state, base)
    action = final_action(base, residual, config.residual_scale, config.residual_max_norm)
    if not config.train_critic and config.dvd_weight >= 1.0:

        q_pi = torch.zeros((state.shape[0], 1), device=state.device)
        q_scale = torch.ones((), device=state.device)
    else:
        q_pi = critics.reduce(state, action, config.actor_q_reduce)
        q_scale = q_pi.detach().abs().mean().clamp_min(1e-3)
    if getattr(config, "q_gap_weight", 0.0) and "dvd_gap" in batch:


        _g = batch["dvd_gap"].repeat_interleave(samples, dim=0).squeeze(-1)
        _pos = _g > 0
        if _pos.any():
            _w = torch.where(_pos, _g / _g[_pos].mean().clamp_min(1e-8),
                             torch.ones_like(_g)).clamp(0.25, 4.0)
            _w = 1.0 + config.q_gap_weight * (_w - 1.0)
            q_pi = q_pi * _w.unsqueeze(-1)
    if getattr(config, "q_rows", 0) and 0 < config.q_rows < q_pi.shape[0]:


        _qidx = torch.randperm(q_pi.shape[0], device=q_pi.device)[:config.q_rows]
        q_pi = q_pi[_qidx]
        q_scale = q_pi.detach().abs().mean().clamp_min(1e-3)
    else:
        _qidx = None
    if config.disable_q_loss_for_expert:


        online_mask = (batch["data_source"] < 0.5).float().repeat_interleave(samples, dim=0)
        if _qidx is not None:
            online_mask = online_mask[_qidx]
        q_objective = -(q_pi * online_mask).sum() / online_mask.sum().clamp_min(1.0) / q_scale
    elif config.actor_k_reduce == "min" and samples > 1:


        q_objective = -q_pi.view(-1, samples).min(dim=1).values.mean() / q_scale
    elif config.actor_k_reduce == "max" and samples > 1:


        q_objective = -q_pi.view(-1, samples).max(dim=1).values.mean() / q_scale
    else:
        q_objective = -q_pi.mean() / q_scale


    q_only, dvd_only = q_objective, None
    _qdvd_objective = None


    _rc = ({"row_chunks": int(config.dvd_row_chunks)}
           if int(getattr(config, "dvd_row_chunks", 1)) > 1 and dvd_term is not None else {})
    if _rc:
        _check_row_chunk_ownership(dvd_term, actor_optimizer)


    _dual_v = getattr(dvd_term, "value_fn", None)
    if not isinstance(_dual_v, DualValue):
        _dual_v = None
    if _dual_v is not None:
        _dual_v.arm_scale_update(keep_graph=_dual_v.claim_grad_diag(env_steps_now))


    if dvd_term is not None and config.dvd_weight > 0.0 and config.dvd_mode == "replay":


        if getattr(dvd_term, "feature_space", False) and "next_feat_obs" not in batch:


            from . import dino_dvd as _dino
            with torch.no_grad():
                batch["next_feat_obs"] = _dino.encode_frames(
                    dvd_term.dynamics, batch["next_image"]).float()
        dvd_objective, dvd_value, dvd_ratio, dvd_valid, dvd_clip = dvd_term.replay_objective(
            actor, batch, config.residual_scale)
        dvd_only = dvd_objective
        q_objective = (1.0 - config.dvd_weight) * q_objective + config.dvd_weight * dvd_objective
    elif dvd_term is not None and config.dvd_weight > 0.0 and config.dvd_phase_gate:
        dvd_ratio = dvd_valid = dvd_clip = None


        follow = _dvd_follow(batch, config, dvd_term, samples, chunk_dim, _base_from_samples)
        if dino is not None:


            dino.context_for_rows(dvd_term.dynamics, batch,
                                  _tile_rows(batch_size, samples, state.device))
        dvd_terms, dvd_value = dvd_term.objective_terms(state, action, follow, **_rc)
        raw = dvd_term.raw_state(state)
        if config.dvd_gate_source == "obj_goal":


            near = ((raw[:, 9:12] - raw[:, 23:26]).norm(dim=-1, keepdim=True) * 1000.0
                    < config.dvd_gate_mm).float()
        else:
            near = ((raw[:, 0:3] - raw[:, 9:12]).norm(dim=-1, keepdim=True) * 1000.0
                    < config.dvd_held_mm).float()
        held = near
        lam = held * config.dvd_weight
        if config.dvd_gate_target == "q":


            q_only = (lam * (-q_pi / q_scale)).mean()
            q_objective = q_only
        else:


            q_only = ((1.0 - lam) * (-q_pi / q_scale)).mean()
            dvd_only = (lam * dvd_terms).mean()
            q_objective = q_only + dvd_only
        gate_frac = float(held.mean())
    elif dvd_term is not None and config.dvd_weight > 0.0:
        dvd_ratio = dvd_valid = dvd_clip = None
        follow = _dvd_follow(batch, config, dvd_term, samples, chunk_dim, _base_from_samples)


        anchor = None
        anchor_action = None
        if getattr(dvd_term, "anchored", False):
            anchor_action = (batch["action"].repeat_interleave(samples, dim=0)
                             if samples > 1 else batch["action"])
            if not dvd_term.feature_space:
                anchor = (anchor_action,
                          batch[dvd_next_key].repeat_interleave(samples, dim=0)
                          if samples > 1 else batch[dvd_next_key])
        if dvd_term is not None and getattr(dvd_term.dynamics, "is_latent56", False):

            _pz = batch["prev_feat"].float()
            _cz = batch["seg_feat"].float()
            _rp = dvd_term.raw_env_state(batch.get("env_state", batch["state"]))[:, :9]
            if samples > 1:
                _pz = _pz.repeat_interleave(samples, dim=0)
                _cz = _cz.repeat_interleave(samples, dim=0)
                _rp = _rp.repeat_interleave(samples, dim=0)
            from .latent56 import norm_prop as _l56_norm_prop
            dvd_term.dynamics.set_context(_pz, _cz, _l56_norm_prop(_rp, dvd_term.dynamics.cfg))
        elif dvd_term is not None and getattr(dvd_term.dynamics, "ctx_prop", "missing") != "missing":


            _pm = SegHeadAdapter.to_map(batch["prev_feat"].float())
            _cm = SegHeadAdapter.to_map(batch["seg_feat"].float())
            if samples > 1:
                _pm = _pm.repeat_interleave(samples, dim=0)
                _cm = _cm.repeat_interleave(samples, dim=0)


            _rp = dvd_term.raw_state(batch.get("env_state", batch["state"]))[:, :9]
            dvd_term.dynamics.set_context(_pm, _cm,
                                          _rp.repeat_interleave(samples, 0) if samples > 1 else _rp)


        _dvd_src = batch[dvd_state_key]
        if getattr(config, "dvd_on_encoder", False):
            _dvd_src = _dvd_src.detach()
        dvd_state = (_dvd_src.repeat_interleave(samples, dim=0)
                     if samples > 1 else _dvd_src)
        dvd_action = action
        if dino is not None:


            if (config.actor_lr_drop_at and env_steps_now >= config.actor_lr_drop_at
                    and actor_optimizer.param_groups[0]["lr"] > config.actor_lr * config.actor_lr_late_factor * 1.001):
                for g in actor_optimizer.param_groups:
                    g["lr"] = config.actor_lr * config.actor_lr_late_factor
                print(f"[actor-lr] dropped to {config.actor_lr * config.actor_lr_late_factor:g} "
                      f"at {env_steps_now} env steps", flush=True)
            if config.dvd_weight_ramp_end > 0:
                t = min(max((env_steps_now - config.dvd_weight_ramp_start) /
                            max(config.dvd_weight_ramp_end - config.dvd_weight_ramp_start, 1), 0.0), 1.0)
                config.dvd_weight = ((1 - t) * config._dvd_weight_base
                                     + t * config.dvd_weight_late)
            _rows = (config.dvd_rows_late
                     if config.dvd_rows_switch and env_steps_now >= config.dvd_rows_switch
                     else config.dvd_rows)


            from .dino_dvd import select_rows as _select_rows
            dvd_rows_idx = _select_rows(
                dvd_state.shape[0], _rows, dvd_state.device,
                group=samples if config.dvd_rows_grouped else 1)
            batch_rows = _tile_rows(batch_size, samples, dvd_state.device)


            dino_rows = batch_rows if dvd_rows_idx is None else batch_rows[dvd_rows_idx]
            dino.context_for_rows(dvd_term.dynamics, batch, dino_rows)
            if dvd_rows_idx is not None:
                dvd_state, dvd_action = dvd_state[dvd_rows_idx], action[dvd_rows_idx]
                follow = None if follow is None else follow[dvd_rows_idx]
                anchor = None if anchor is None else tuple(a[dvd_rows_idx] for a in anchor)
            if anchor_action is not None and dvd_term.feature_space:


                anchor = (anchor_action if dvd_rows_idx is None else anchor_action[dvd_rows_idx],
                          dino.encode_frames(dvd_term.dynamics, batch["next_image"][dino_rows]))
        if getattr(config, "rosmo", False):


            with torch.no_grad():
                K = config.rosmo_k
                _b, _d = dvd_action.shape[0], dvd_action.shape[-1]
                cand = dvd_action.detach().unsqueeze(1).repeat(1, K, 1)
                noise = torch.randn(_b, K, _d, device=dvd_action.device) * (
                    config.rosmo_sigma * config.residual_scale)
                noise[:, 0] = 0.0
                cand = cand + noise
                scores = []
                for _k in range(K):
                    _t, _ = dvd_term.objective_terms(dvd_state, cand[:, _k], follow, anchor, **_rc)
                    scores.append(-_t.reshape(_b))
                S = torch.stack(scores, 1)
                w = torch.softmax(config.rosmo_beta * (S - S.mean(1, keepdim=True)), dim=1)
                target = (w.unsqueeze(-1) * cand).sum(1)
                rosmo_sel_gap = float((S.max(1).values - S[:, 0]).mean())
            _mse = ((dvd_action - target) ** 2).mean(-1)
            dvd_objective = (_mse / _mse.detach().mean().clamp_min(1e-10)).mean()
            dvd_value = float(S[:, 0].mean())
            _rosmo_metrics = {"rosmo_selection_gap": rosmo_sel_gap,
                              "rosmo_weight_max": float(w.max(1).values.mean())}
        elif config.qdvd_combine != "off":


            _rows, dvd_value = dvd_term.objective_terms(dvd_state, dvd_action, follow, anchor, **_rc)
            dvd_objective = _rows.mean()


            _v_dvd = (-_rows).reshape(-1)
            _q_sel = q_pi.reshape(-1)


            _sri = locals().get("dvd_rows_idx")
            if _sri is not None:
                _q_sel = _q_sel[_sri]


            if getattr(config, "qdvd_normalise", "meanabs") == "zscore":


                _v_q = ((_q_sel - _q_sel.detach().mean())
                        / _q_sel.detach().std().clamp_min(1e-6))
                _v_dvd = ((_v_dvd - _v_dvd.detach().mean())
                          / _v_dvd.detach().std().clamp_min(1e-6))
            else:
                _v_q = _q_sel / _q_sel.detach().abs().mean().clamp_min(1e-3)
            if _v_dvd.shape != _v_q.shape:
                raise ValueError(
                    f"--qdvd-combine needs the two estimators on the same rows, got "
                    f"dvd {tuple(_v_dvd.shape)} vs q {tuple(_v_q.shape)}. The startup guards "
                    f"cover --q-rows/--dvd-rows; this means a new row path was added.")
            if config.qdvd_combine == "min":
                _v = torch.minimum(_v_q, _v_dvd)
            elif config.qdvd_combine == "max":
                _v = torch.maximum(_v_q, _v_dvd)
            else:
                _v = 0.5 * (_v_q + _v_dvd)
            if config.actor_k_reduce == "max" and samples > 1:
                _v = _v.view(-1, samples).max(dim=1).values
            _qdvd_objective = -_v.mean()


            _qdvd_metrics = {
                "qdvd_dvd_win_frac": float((_v_dvd > _v_q).float().mean()),
                "qdvd_v_q_std": float(_v_q.detach().std()),
                "qdvd_v_dvd_std": float(_v_dvd.detach().std()),
                "qdvd_v_q_mean": float(_v_q.detach().mean()),
                "qdvd_v_dvd_mean": float(_v_dvd.detach().mean()),
            }
        else:
            if getattr(config, "actor_k_reduce", "mean") in ("min", "max") and samples > 1:


                _srows, dvd_value = dvd_term.objective_terms(
                    dvd_state, dvd_action, follow, anchor, **_rc)
                _sv = (-_srows).reshape(-1, samples)
                _sv = (_sv.min(dim=1).values if config.actor_k_reduce == "min"
                       else _sv.max(dim=1).values)
                dvd_objective = -_sv.mean()
            else:
                dvd_objective, dvd_value = dvd_term.objective(dvd_state, dvd_action, follow, anchor, **_rc)
        dvd_only = dvd_objective
        if _qdvd_objective is not None:
            q_objective = _qdvd_objective
        else:
            q_objective = (1.0 - config.dvd_weight) * q_objective + config.dvd_weight * dvd_objective
    else:
        dvd_value = dvd_ratio = dvd_valid = dvd_clip = None
    gate_frac = locals().get("gate_frac")


    dvd_rows_idx = locals().get("dvd_rows_idx")
    if _dual_v is not None:


        _dual_v.disarm_scale_update(expected=dvd_value is not None)
        _dual_graph = _dual_v.take_graph()
        if _dual_graph is not None:


            _dual_gstats = _dual_member_action_grads(
                _dual_graph, locals().get("dvd_action", action))
        del _dual_graph


    per_sample = residual.square().mean(dim=-1, keepdim=True)


    with torch.no_grad():
        _rn = residual.norm(dim=-1)
        trust_frac = None if dino is not None else float((_rn < TRUST_RADIUS).float().mean())
        residual_norm = float(_rn.mean())


        residual_capped_frac = (float((_rn > config.residual_max_norm).float().mean())
                                if config.residual_max_norm > 0.0 else None)


    with torch.no_grad():
        if not config.train_critic and config.dvd_weight >= 1.0:


            _adv_src = batch[dvd_state_key]
            if getattr(config, "dvd_on_encoder", False):
                _adv_src = _adv_src.detach()
            adv_state = (_adv_src.repeat_interleave(samples, dim=0)
                         if samples > 1 else _adv_src)
            adv_action, adv_base = action, base
            if dvd_rows_idx is not None:
                adv_state = adv_state[dvd_rows_idx]
                adv_action, adv_base = action[dvd_rows_idx], base[dvd_rows_idx]
            if config.bc_filter and config.bc_filter_source == "dvd" and dino is not None:


                _on_a, _, _mem_a = dvd_candidate_scores(dvd_term, adv_state, adv_action)
                _on_b, _, _mem_b = dvd_candidate_scores(dvd_term, adv_state, adv_base)
                advantage_all = (_on_a - _on_b).unsqueeze(-1)
                dvd_member_advantage = None if _mem_a is None else _mem_a - _mem_b
            else:
                advantage_all = (dvd_score(dvd_term, adv_state, adv_action)
                                 - dvd_score(dvd_term, adv_state, adv_base))
            diagnostics: dict[str, float] = {
                "dvd_advantage": float(advantage_all.mean()),
                "better_than_prior": float((advantage_all > 0.0).float().mean()),
            }
        else:


            _adv_state = state if _qidx is None else state[_qidx]
            _adv_base = base if _qidx is None else base[_qidx]
            advantage_all = q_pi.detach() - critics.reduce(
                _adv_state, _adv_base, config.actor_q_reduce)
            diagnostics = {
                "q_advantage": float(advantage_all.mean()),
                "better_than_prior": float((advantage_all > 0.0).float().mean()),
            }


    if "mc_return" in batch:
        with torch.no_grad():
            have = batch["mc_return"] > -1e8
            if bool(have.any()):
                mc = batch["mc_return"][have]
                if config.train_critic:
                    q_here = critics.conservative(batch["state"], batch["action"])[have]
                    diagnostics["q_vs_mc_mae"] = float((q_here - mc).abs().mean())
                    diagnostics["q_vs_mc_rank"] = _spearman(q_here, mc)


                _vd = int(getattr(dvd_term.value_fn, "input_dim", 0)) if dvd_term is not None else 0
                if (dvd_term is not None and dino is None and not dvd_term.feature_space
                        and _vd == int(batch[dvd_state_key].shape[-1])):
                    v_here = dvd_term.value_fn(batch[dvd_state_key])[have]
                    diagnostics["v_vs_mc_mae"] = float((v_here - mc).abs().mean())


                    diagnostics["v_vs_mc_rank"] = _spearman(v_here, mc)


    bon_metrics: dict[str, float] = {}
    bon_contraction = None
    if getattr(config, "bc_target", "base") == "bon":
        bon_metrics["bc_target_active"] = 0.0
        if env_steps_now >= config.bc_target_after:
            if dino is None or dvd_term is None or locals().get("dino_rows") is None:
                raise RuntimeError(
                    "--bc-target bon: the DINO DVD branch installed no context on this update, so there are no DVD "
                    "rows to score the prior draws on (check_bc_target should have refused this config)")
            _bon_ramp = int(getattr(config, "bc_target_ramp", 0))
            _bon_lam = (min(max((env_steps_now - config.bc_target_after) / _bon_ramp, 0.0), 1.0)
                        if _bon_ramp > 0 else 1.0)
            if not getattr(config, "_bc_target_banner", False):
                config._bc_target_banner = True
                print(f"[bc-target] bon ON at {env_steps_now} env steps: the DVD rows' BC anchor pulls the residual "
                      f"toward the online-V best of each state's {samples} prior draws, full from "
                      f"{config.bc_target_after + _bon_ramp}"
                      + (f"; --residual-max-norm {config.residual_max_norm:g} lets the executed action move at most "
                         f"{config.residual_scale * config.residual_max_norm:g} toward it (bc_target_exec_dist vs "
                         f"bc_target_dist)" if config.residual_max_norm > 0.0 else ""), flush=True)
            from time import perf_counter as _bon_clock
            if base.is_cuda:
                torch.cuda.synchronize(base.device)
            _bon_t0 = _bon_clock()
            _bon_target, _bon_stats = bon_draw_targets(
                dino_bon_member_scorer(dvd_term, base.shape[0] if dvd_rows_idx is None else dvd_rows_idx.numel()),
                batch[dvd_state_key], base, samples, dvd_rows_idx,
                (batch["data_source"] > 0.5).repeat_interleave(samples, dim=0).reshape(-1),
                config.residual_scale, max_norm=config.residual_max_norm, ramp=_bon_lam)
            if base.is_cuda:
                torch.cuda.synchronize(base.device)
            bon_metrics["bc_target_ms"] = 1000.0 * (_bon_clock() - _bon_t0)
            _bon_residual = residual if dvd_rows_idx is None else residual[dvd_rows_idx]
            bon_contraction = (_bon_residual - _bon_target).square().mean(dim=-1).mean()
            bon_metrics["bc_target_active"] = 1.0
            bon_metrics["bc_target_lambda"] = float(_bon_lam)
            bon_metrics.update(_bon_stats)
    if config.bc_filter and updates > config.bc_filter_warmup_updates:


        with torch.no_grad():
            dvd_rows_filter = (config.bc_filter_source == "dvd" and dvd_term is not None
                               and dino is not None)
            if dvd_rows_filter:


                advantage = advantage_all
                q_prior = None
            elif config.bc_filter_source == "dvd" and dvd_term is not None:


                advantage = dvd_score(dvd_term, state, action) - dvd_score(dvd_term, state, base)
                q_prior = None
            else:
                q_prior = critics.conservative(state, base)
                advantage = q_pi.detach() - q_prior
            if config.bc_filter_uncertainty > 0.0 and q_prior is None and not dvd_rows_filter:
                raise ValueError(
                    "--bc-filter-uncertainty needs the per-critic spread and so requires "
                    "--bc-filter-source q; the DVD score is a single number with no ensemble."
                )
            if config.bc_filter_uncertainty > 0.0 and dvd_rows_filter:


                if dvd_member_advantage is None:
                    raise ValueError(
                        "--bc-filter-uncertainty with --bc-filter-source dvd needs per-member V scores "
                        "(a plain ValueEnsemble); this V has a residual delta head.")
                advantage_se = dvd_member_advantage.std(dim=0).unsqueeze(-1)
                better = (advantage > config.bc_filter_uncertainty * advantage_se).float()
                diagnostics["advantage_se"] = float(advantage_se.mean())
                diagnostics["advantage_over_se"] = float(
                    (advantage / advantage_se.clamp_min(1e-9)).mean()
                )
            elif config.bc_filter_uncertainty > 0.0:


                per_critic = (critics(state, action)[:, :, 0]
                              - critics(state, base)[:, :, 0])
                advantage_se = per_critic.std(dim=0).unsqueeze(-1)
                better = (advantage > config.bc_filter_uncertainty * advantage_se).float()
                diagnostics["advantage_se"] = float(advantage_se.mean())
                diagnostics["advantage_over_se"] = float(
                    (advantage / advantage_se.clamp_min(1e-9)).mean()
                )
            else:
                better = (advantage > 0.0).float()
            if config.bc_filter_soft:


                baseline = (
                    value_fn.frozen(batch["state"])
                    if value_fn is not None and config.value_intervention == "filter"
                    else batch["mc_return"]
                )
                overestimation = critics.conservative(
                    batch["state"], batch["action"]
                ) - baseline
                underestimated = (
                    overestimation < config.q_underestimation_threshold
                ).float().repeat_interleave(samples, dim=0)
                bc_filter = 1.0 - better * underestimated
            else:
                bc_filter = 1.0 - better
            if dvd_rows_filter:
                diagnostics["bc_release_share_scored"] = float(better.mean())
                if dvd_rows_idx is not None:
                    _full = torch.ones_like(per_sample)
                    _full[dvd_rows_idx] = bc_filter
                    bc_filter = _full
            if config.retain_bc_for_expert:
                expert_mask = (
                    batch["data_source"] > 0.5
                ).float().repeat_interleave(samples, dim=0)
                bc_filter = expert_mask + (1.0 - expert_mask) * bc_filter
            if dvd_rows_filter:
                diagnostics["bc_filter"] = float(bc_filter.mean())
            else:
                diagnostics = {
                    "bc_filter": float(bc_filter.mean()),
                    "q_advantage": float(advantage.mean()),
                    "better_than_prior": float(better.mean()),
                    "mc_return": float(batch["mc_return"].mean()),
                }
            if config.bc_filter_soft:
                diagnostics["q_overestimation"] = float(overestimation.mean())
        contraction = (bc_filter * per_sample).mean()
    else:
        contraction = per_sample.mean()
    if bon_contraction is not None:

        contraction = bon_contraction


    if env_steps_now < config.actor_value_after:
        q_objective = q_objective.detach() * 0.0
    actor_loss = q_objective + config.bc_loss_weight * contraction


    if (getattr(config, "encoder_drift", 0.0) > 0.0 and encoder is not None
            and target_encoder is not None and config.actor_train_encoder):
        with torch.no_grad():
            _ref = target_encoder(batch["image"], batch["state"][:, :PIXEL_PROPRIO],
                                  augment=False)
        _live = encoder(batch["image"], batch["state"][:, :PIXEL_PROPRIO], augment=False)
        _drift = (_live - _ref).pow(2).mean()
        actor_loss = actor_loss + config.encoder_drift * _drift


        try:
            diagnostics["encoder_drift"] = float(_drift.detach())
        except NameError:
            diagnostics = {"encoder_drift": float(_drift.detach())}


    _vaux_v = _value_read(value_fn, dvd_term)
    if (getattr(config, "encoder_value_aux", 0.0) > 0.0 and v_aux_head is not None
            and _vaux_v is not None and encoder is not None):
        with torch.no_grad():
            if dvd_term is not None and getattr(dvd_term, "feature_space", False):


                from . import dino_dvd as _ddvd
                _cur = _ddvd.encode_frames(dvd_term.dynamics, batch["image"]).float()


                _yv = _vaux_v(dvd_term.value_input(
                    _cur, batch.get("env_state", batch["state"])))
            else:


                _yv = _vaux_v(batch.get("env_state", batch["state"]))
            _yv = _yv.detach()


        _xaux = (batch["enc_state"] if getattr(config, "encoder_value_aux_only", False)
                 else batch[actor_state_key])
        if getattr(config, "encoder_value_aux_image_only", False):
            _xaux = _xaux[:, :-PIXEL_PROPRIO]
        _pred = v_aux_head(_xaux)
        _laux = (_pred - _yv).pow(2).mean()
        actor_loss = actor_loss + config.encoder_value_aux * _laux
        try:
            diagnostics["encoder_value_aux"] = float(_laux.detach())
        except NameError:
            diagnostics = {"encoder_value_aux": float(_laux.detach())}


    gradvar_stats: dict[str, float] = {}
    if config.gradvar_every and updates % config.gradvar_every == 0:
        gate = (config.dvd_phase_gate and config.dvd_weight > 0.0 and dvd_term is not None)


        _w = 0.0 if env_steps_now < config.actor_value_after else 1.0
        gradvar_stats = gradient_snr(
            {"q": q_only, "dvd": dvd_only, "bc": contraction},
            residual,
            _tile_rows(batch_size, samples, state.device),
            batch_size,
            [prm for prm in actor.parameters() if prm.requires_grad],
            action=action,
            composite={
                "q": _w * (1.0 if gate else 1.0 - config.dvd_weight),
                "dvd": _w * (1.0 if gate else config.dvd_weight),
                "bc": config.bc_loss_weight,
            },
            phase=_phase_of_rows(dvd_term, batch, samples) if config.gradvar_by_phase else None,
            out_grads=probe_grads,
        )


    _pool_diag = None
    _vpool = getattr(dvd_term, "_value_pool", None) if dvd_term is not None else None
    if (_vpool is not None and getattr(config, "dvd_actor_pool_log_every", 0) > 0
            and updates % config.dvd_actor_pool_log_every == 0):
        _pool_diag = {}
        for _vp in _vpool.parameters():
            _vp.grad = None

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
    if (_vpool is None and dvd_term is not None
            and getattr(config, "dvd_pool_actor_step", False)):


        torch.nn.utils.clip_grad_norm_(dvd_term._pool.parameters(), config.actor_grad_clip)
    if _vpool is not None:


        _apg = torch.nn.utils.clip_grad_norm_(dvd_term._pool.parameters(), config.actor_grad_clip)
        if _pool_diag is not None:
            _pool_diag["pool_grad_actor_dvd"] = float(_apg)
            _pool_diag["pool_grad_value_dvd"] = _pool_grad_norm(_vpool)
            _pool_diag["pool_drift_actor_vs_value"] = _pool_rel_distance(dvd_term._pool, _vpool)
    actor_optimizer.step()
    if not config.freeze_init_critic and config.train_critic:
        soft_update(critics, target_critics, config.tau)
        if target_actor is not None:
            soft_update(actor, target_actor, config.tau)
        if encoder is not None:
            soft_update(encoder, target_encoder, config.tau)
        if getattr(dvd_term, "_obs_pool_target", None) is not None:

            soft_update(dvd_term._obs_pool, dvd_term._obs_pool_target, config.tau)
    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "q_objective": float(q_objective.detach()),
        **({"dvd_value": dvd_value} if dvd_value is not None else {}),

        **(_pool_diag or {}),
        **(locals().get("_rosmo_metrics") or {}),
        **(locals().get("_qdvd_metrics") or {}),
        **(locals().get("_vaux_critic_metric") or {}),
        **(locals().get("_dual_gstats") or {}),


        **({"dvd_scale": float(dvd_term.last_scale)}
           if (dvd_term is not None and dvd_value is not None) else {}),
        **({"dvd_amp": float(dvd_value) / float(dvd_term.last_scale)}
           if (dvd_term is not None and dvd_value is not None
               and dvd_term.last_scale and dvd_term.last_scale == dvd_term.last_scale) else {}),
        **({"dvd_is_ratio": dvd_ratio} if dvd_ratio is not None else {}),


        **({"dvd_v_proprio_shift_mm": dvd_term.last_proprio_shift_mm}
           if getattr(dvd_term, "last_proprio_shift_mm", None) is not None else {}),


        **({"dvd_step2_residual_norm": float(dvd_term.last_step2_residual_norm.mean())}
           if getattr(dvd_term, "last_step2_residual_norm", None) is not None else {}),


        **({"dvd_is_valid_frac": dvd_valid} if dvd_valid is not None else {}),


        **({"dvd_xi_clipped": dvd_clip} if dvd_clip is not None else {}),

        **({"dvd_gate_frac": gate_frac} if gate_frac is not None else {}),
        "contraction_loss": float(contraction.detach()),


        **bon_metrics,
        **gradvar_stats,


        "residual_sq": float(per_sample.mean().detach()),
        "residual_norm": residual_norm,


        **({"residual_capped_frac": residual_capped_frac}
           if residual_capped_frac is not None else {}),
        **({"in_trust_region": trust_frac} if trust_frac is not None else {}),


        "expert_ratio": float(expert_ratio),
        "expert_count": int(expert_count),
        "q_mean": float(q_values.mean().detach()),
        "q_std": float(q_values.std(dim=0).mean().detach()),
        "critic_grad": float(critic_grad),
        "actor_grad": float(actor_grad),
        **diagnostics,
        **value_diagnostics,
        **critical_stats,
        **mbpo_stats,
    }


@torch.inference_mode()
def select_dice_exploration(
    actor: ResidualActor,
    critics: CriticEnsemble,
    state: torch.Tensor,
    bases: torch.Tensor,
    scale: float,
    dvd_term=None,
    bon_reduce: str = "min",
    dvd_state: torch.Tensor | None = None,
    bon_select: str = "max",
    bon_quantile: float = 0.75,
    frozen_value_fn=None,
    max_norm: float = 0.0,
    conts: torch.Tensor | None = None,
    lookahead: int = 1,
) -> torch.Tensor:
    """Pick one of the supplied candidate bases (argmax by default), one residual conditioned on each."""
    select_dice_exploration.last_frozen_changed = None
    select_dice_exploration.last_frozen_rank_corr = None
    select_dice_exploration.last_top_gap = None
    select_dice_exploration.last_member_std = None
    select_dice_exploration.last_top_gap_over_std = None
    select_dice_exploration.last_meaningful = None
    members = None
    states = state.repeat(bases.shape[0], 1)
    residual, _ = actor.sample(states, bases)


    actions = final_action(bases, residual, scale, max_norm)
    if dvd_term is not None:


        ss = (dvd_state if dvd_state is not None else state).repeat(bases.shape[0], 1)
        if int(lookahead) > 1:


            score, score_frozen, members = dvd_candidate_scores(
                dvd_term, ss, actions, frozen_value_fn, cont=conts, lookahead=lookahead)
        else:
            score, score_frozen, members = dvd_candidate_scores(dvd_term, ss, actions, frozen_value_fn)
    else:
        score = critics.reduce(states, actions, bon_reduce).squeeze(-1)


    select_dice_exploration.last_gap = float((score.max() - score.median()).item())
    if dvd_term is not None and score.numel() > 1:

        _top2 = score.topk(2).values
        select_dice_exploration.last_top_gap = float(_top2[0] - _top2[1])
        if members is not None and members.dim() == 2 and members.shape[0] > 1:
            _std = float(members.std(dim=0).mean())
            select_dice_exploration.last_member_std = _std
            if _std > 0:
                select_dice_exploration.last_top_gap_over_std = select_dice_exploration.last_top_gap / _std
                select_dice_exploration.last_meaningful = float(select_dice_exploration.last_top_gap > _std)
    online_score = None
    if dvd_term is not None and frozen_value_fn is not None and score.numel() > 1:
        _n = score.numel()
        r_on = score.argsort().argsort().double() / (_n - 1)
        r_fr = score_frozen.argsort().argsort().double() / (_n - 1)
        _c = torch.corrcoef(torch.stack([r_on, r_fr]))[0, 1]
        select_dice_exploration.last_frozen_rank_corr = float(_c) if torch.isfinite(_c) else None
        online_score = score


        score = torch.minimum(r_on, r_fr) + 1e-3 * (r_on + r_fr) + 1e-6 * r_on


    def _pick(s):
        m = s.numel()
        if bon_select == "median" and m > 1:
            return int(s.argsort()[m // 2])
        if bon_select == "mean" and m > 1:
            return int((s - s.mean()).abs().argmin())
        if bon_select == "quantile" and m > 1:
            k = min(m - 1, max(0, int(round(float(bon_quantile) * (m - 1)))))
            return int(s.argsort()[k])
        return int(s.argmax())
    n = score.numel()
    idx = _pick(score)


    ref = online_score if online_score is not None else score
    select_dice_exploration.last_rank = (
        float((ref < ref[idx]).sum().item()) / max(1, n - 1))
    if online_score is not None:
        select_dice_exploration.last_frozen_changed = float(idx != _pick(online_score))
    return actions[idx].unsqueeze(0)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def check_dvd_row_chunks(config) -> None:
    """--dvd-row-chunks N:"""
    n = getattr(config, "dvd_row_chunks", 1)
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError(f"--dvd-row-chunks {n!r}: must be an integer >= 1 (1 = the single-pass DVD term)")
    if int(n) == 1:
        return
    why = []
    if getattr(config, "method", "dice_rl") != "dice_rl":
        why.append("--method dice_rl (the chunked term lives in update_dice's actor update)")
    if getattr(config, "dvd_dynamics", "") != "dino_wm":
        why.append("--dvd-dynamics dino_wm (the chunks slice the DINO-WM adapter's ctx_feats context; the other "
                   "f's are cheap per row or keep their context elsewhere)")
    if float(getattr(config, "dvd_weight", 0.0)) <= 0.0:
        why.append("--dvd-weight > 0 (with no DVD term the flag would be inert)")
    if getattr(config, "dvd_mode", "dpg") == "replay":
        why.append("--dvd-mode dpg (the replay objective is not chunked)")
    if getattr(config, "dvd_step2_action", "prior") == "actor":
        why.append("--dvd-step2-action prior (the closed-loop step-2 residual puts actor-parameter gradient "
                   "through a path other than the action rows, which the chunked surrogate cannot carry)")
    if getattr(config, "value_td_model", ""):
        why.append("no --value-td-model (DualValue normalises by a batch statistic and updates its EMA scales on "
                   "the first large read, so a row's value would depend on its chunk)")
    if getattr(config, "dvd_actor_pool", False) or getattr(config, "dvd_pool_actor_step", False):
        why.append("no --dvd-actor-pool / --dvd-pool-actor-step (the actor optimiser steps a pool inside the DVD "
                   "graph; the chunked path gives the actor gradient through the action rows only)")
    if getattr(config, "flow_steps", 0) or getattr(config, "flow_inverse_iters", 0):
        why.append("no flow head (--flow-steps/--flow-inverse-iters: the anchored rollout is not verified "
                   "under chunking)")
    if why:
        raise ValueError(f"--dvd-row-chunks {int(n)} needs " + "; ".join(why))


def check_dvd_k2(config) -> None:
    """Startup guard for --dvd-chunks 2 on the DINO WM, --dvd-follow and --bon-lookahead."""
    follow = getattr(config, "dvd_follow", "recorded")
    look = int(getattr(config, "bon_lookahead", 1))
    chunks = int(getattr(config, "dvd_chunks", 1))
    if follow not in ("recorded", "continuation"):
        raise ValueError(f"--dvd-follow {follow!r}: expected recorded or continuation")
    if look not in (1, 2):
        raise ValueError(f"--bon-lookahead {look}: only 1 and 2 are defined")
    step2 = getattr(config, "dvd_step2_action", "prior")
    if step2 not in ("prior", "actor"):
        raise ValueError(f"--dvd-step2-action {step2!r}: expected prior or actor")
    if step2 == "actor":


        need = []
        if chunks != 2 or getattr(config, "dvd_dynamics", "") != "dino_wm":
            need.append("--dvd-chunks 2 --dvd-dynamics dino_wm (there is no chunk 2 to act in otherwise)")
        if follow != "continuation":
            need.append("--dvd-follow continuation (the actor corrects the row's OWN prior continuation; "
                        "the recorded next_base is a draw at the REAL t+8 state, not the imagined one)")
        if not getattr(config, "dino_obs", False):
            need.append("--dino-obs (the imagined observation is built like the actor's real one: pooled "
                        "DINO-WM features + proprio; any other actor input cannot be imagined from zhat)")
        if not getattr(config, "dino_wm_state_head", False) or \
                getattr(config, "dvd_proprio_source", "current") != "predicted":
            need.append("--dino-wm-state-head --dvd-proprio-source predicted (the imagined proprio is the "
                        "chunk-1 proprio-head prediction)")
        if getattr(config, "dvd_mode", "dpg") == "replay":
            need.append("--dvd-mode dpg (replay never rolls past chunk 1)")
        if getattr(config, "method", "dice_rl") != "dice_rl":
            need.append("--method dice_rl (the residual actor's objective is update_dice's)")
        if float(getattr(config, "dvd_weight", 0.0)) <= 0.0:
            need.append("--dvd-weight > 0 (with no DVD term the flag would be inert)")
        if getattr(config, "value_pool", ""):
            need.append("no --value-pool (the rollout observation pools with --dino-pool's mode and the "
                        "update's with the value pool's; the imagined observation would match only one)")
        if need:
            raise ValueError("--dvd-step2-action actor needs " + "; ".join(need))
    if follow == "continuation" and chunks < 2:
        raise ValueError("--dvd-follow continuation picks what DVD's chunk 2 executes: it needs --dvd-chunks 2 "
                         "(with --dvd-chunks 1 nothing follows and the flag would be inert)")
    if follow == "continuation" and getattr(config, "dvd_mode", "dpg") == "replay":
        raise ValueError("--dvd-follow continuation is read by the DPG DVD path only (--dvd-mode replay never rolls "
                         "past chunk 1)")
    feature_k2 = chunks > 1 and getattr(config, "dvd_dynamics", "") == "dino_wm"
    if feature_k2 or look > 1:
        what = "--dvd-chunks 2 on --dvd-dynamics dino_wm" if feature_k2 else "--bon-lookahead 2"
        if chunks > 2:
            raise ValueError(f"--dvd-chunks {chunks} on the DINO WM: only k=1 and k=2 are defined")
        if look > 1 and getattr(config, "dvd_dynamics", "") != "dino_wm":
            raise ValueError("--bon-lookahead 2 is wired for --dvd-dynamics dino_wm only")
        if not getattr(config, "dino_wm_state_head", False):
            raise ValueError(f"{what} needs --dino-wm-state-head: chunk 2's raw proprio input is the chunk-1 "
                             "PREDICTED proprio, which only the proprio head supplies")
        if getattr(config, "dvd_proprio_source", "current") != "predicted":
            raise ValueError(f"{what} needs --dvd-proprio-source predicted: V next to predicted t+16 features must "
                             "read the predicted t+16 proprio, not s_t's")
    if look > 1:
        if getattr(config, "bon_source", "q") != "dvd":
            raise ValueError("--bon-lookahead 2 scores candidates with V(f(f(.))): it needs --bon-source dvd")
        if int(getattr(config, "bon_dvd_until", 0)) > 0:
            raise ValueError("--bon-lookahead 2 with --bon-dvd-until: after the hand-off the critic ranks one chunk; "
                             "not combined")


def check_dvd_fit_sampler(config: TrainConfig) -> None:
    """Startup guard for --dvd-fit-sampler / --dvd-fit-recent-* / --dvd-fit-probe-every."""
    sampler = config.dvd_fit_sampler
    if sampler not in ("batch", "recent", "mix"):
        raise ValueError(f"--dvd-fit-sampler {sampler!r}: expected batch, recent or mix")
    if sampler != "batch":
        if not config.dvd_finetune_dynamics:
            raise ValueError(
                f"--dvd-fit-sampler {sampler} chooses the rows f FINE-TUNES on: it needs "
                "--dvd-finetune-dynamics (a frozen f has no fit batch to redirect)")
        if config.dvd_dynamics != "dino_wm":
            raise ValueError(
                f"--dvd-fit-sampler {sampler} is wired for --dvd-dynamics dino_wm only (got "
                f"{config.dvd_dynamics!r}): seghead_wm, latent56 and the state-space f build their "
                "fit batches on a branch that never reads the sampler")
        if config.method != "dice_rl":
            raise ValueError("--dvd-fit-sampler only reaches the dice_rl update loop")
        if config.gated_training:


            raise ValueError(
                f"--dvd-fit-sampler {sampler} with --gated-training: the gate reads f's skill off "
                "its fit rows, which this sampler moves to the recent window, so gate_skill_tau "
                "would be compared against a different statistic than it was set on")
    if not 0.0 < config.dvd_fit_recent_frac <= 1.0:
        raise ValueError(
            f"--dvd-fit-recent-frac {config.dvd_fit_recent_frac} must be in (0, 1]: it is the "
            "share of f's rows drawn from the recent window (1.0 is --dvd-fit-sampler recent)")
    if sampler == "mix":


        rows, count = config.dvd_fit_batch_rows(), config.dvd_fit_recent_count()
        if count == 0:
            raise ValueError(
                f"--dvd-fit-sampler mix with --dvd-fit-recent-frac {config.dvd_fit_recent_frac} "
                f"draws round({config.dvd_fit_recent_frac} x {rows}) = 0 recent rows: that is "
                "uniform over the whole online buffer, not a mix. Raise the frac or the row count")
        if count == rows and config.dvd_fit_recent_frac < 1.0:
            raise ValueError(
                f"--dvd-fit-sampler mix with --dvd-fit-recent-frac {config.dvd_fit_recent_frac} "
                f"draws round({config.dvd_fit_recent_frac} x {rows}) = {rows} recent rows, i.e. all "
                "of them: that is --dvd-fit-sampler recent under a mix label")
    if config.dvd_fit_recent_steps < config.action_steps:
        raise ValueError(
            f"--dvd-fit-recent-steps {config.dvd_fit_recent_steps} is shorter than one action "
            f"chunk ({config.action_steps} env steps). The replay stores one row per chunk, so the "
            "window is steps // action_steps rows and this would be less than one of them")
    if config.dvd_fit_probe_every < 0:
        raise ValueError(
            f"--dvd-fit-probe-every {config.dvd_fit_probe_every}: it is an update interval, "
            "0 = off; negative means nothing")
    if config.dvd_fit_probe_every > 0:
        if config.dvd_dynamics != "dino_wm":
            raise ValueError(
                f"--dvd-fit-probe-every reads f's feature MSE through dino_dvd.fit_batch: it "
                f"needs --dvd-dynamics dino_wm (got {config.dvd_dynamics!r})")
        if config.method != "dice_rl":
            raise ValueError("--dvd-fit-probe-every only reaches the dice_rl update loop")


def check_bc_target(config) -> None:
    """Startup guard for --bc-target / --bc-target-after / --bc-target-ramp."""
    target = getattr(config, "bc_target", "base")
    if target not in ("base", "bon"):
        raise ValueError(f"--bc-target must be 'base' or 'bon', got {target!r}")
    if target != "bon":
        return
    if config.method != "dice_rl":
        raise ValueError(
            f"--bc-target bon only reaches the dice_rl actor loss (update_dice); --method {config.method} would run "
            "as a duplicate of its control")
    if int(config.bc_target_after) < 0:
        raise ValueError(f"--bc-target-after is an env-step count, got {config.bc_target_after}")
    if int(getattr(config, "bc_target_ramp", 0)) < 0:
        raise ValueError(f"--bc-target-ramp is an env-step count (0 = at once), got {config.bc_target_ramp}")
    if not (config.value_finetune or config.value_mc_weight > 0.0):
        raise ValueError(
            "--bc-target bon ranks the prior draws by the ONLINE V, and this V never trains online (no "
            "--value-finetune, no --value-mc-weight): the pretrained progress-distilled V's ranking of prior draws "
            "is uninformative, so the target would be noise for the whole run")
    if int(config.bc_target_after) <= int(config.value_finetune_after):
        raise ValueError(
            f"--bc-target-after {config.bc_target_after} must be later than --value-finetune-after "
            f"{config.value_finetune_after}: the ranking is only informative once V has trained online (and the "
            "offline warm start runs update_dice at env step 0). A smoke can use --bc-target-after 1")
    if config.dvd_dynamics != "dino_wm":
        raise ValueError(
            f"--bc-target bon is implemented for --dvd-dynamics dino_wm only (got {config.dvd_dynamics!r}): it "
            "scores the prior draws against the context the DINO DVD branch installs")
    if config.dvd_weight <= 0.0 or config.dvd_mode != "dpg" or config.dvd_phase_gate:
        raise ValueError(
            "--bc-target bon scores the rows of the DPG DVD term: it needs --dvd-weight > 0 with --dvd-mode dpg "
            "and no --dvd-phase-gate")
    if config.dvd_weight_ramp_end > 0 and config.dvd_weight_late <= 0.0:
        raise ValueError(
            f"--bc-target bon needs the DPG DVD branch on every update from --bc-target-after on: "
            f"--dvd-weight-ramp-end with --dvd-weight-late {config.dvd_weight_late} ramps --dvd-weight to <= 0, "
            "which skips that branch (and the ramp inside it) for good, so the next bon update would raise")
    if config.dvd_rows <= 0 or (config.dvd_rows_switch and config.dvd_rows_late <= 0):
        raise ValueError(
            "--bc-target bon retargets only the DVD term's rows: set --dvd-rows > 0 (and --dvd-rows-late > 0 with "
            "--dvd-rows-switch); 0 would score K draws for every tiled row of the batch")
    if config.resolved_actor_k() < 2:
        raise ValueError(
            "--bc-target bon picks the best of each state's prior draws: needs --multi-sample-k >= 2 "
            "(or --actor-sample-k >= 2)")
    tiled = int(config.batch_size) * int(config.resolved_actor_k())
    if config.dvd_rows >= tiled or (config.dvd_rows_switch and config.dvd_rows_late >= tiled):
        raise ValueError(
            f"--bc-target bon with --dvd-rows {config.dvd_rows}"
            + (f" / --dvd-rows-late {config.dvd_rows_late}" if config.dvd_rows_switch else "")
            + f" >= batch_size x K = {tiled}: select_rows then takes every tiled row, so bon would score K draws "
            "for every state of the batch on each update. Use a real subset")
    if getattr(config, "value_td_model", ""):
        raise ValueError(
            "--bc-target bon ranks by a plain ValueEnsemble's per-net scores; --value-td-model makes V a DualValue")
    if getattr(config, "value_residual", False):
        raise ValueError(
            "--bc-target bon ranks by per-net V scores; --value-residual adds a delta head that has none")
    if getattr(config, "actor_k_reduce", "mean") != "mean":
        raise ValueError(
            f"--bc-target bon with --actor-k-reduce {config.actor_k_reduce}: min/max needs --dvd-rows-grouped, which "
            "leaves the retargeted BC term rows // K states per update, and it reduces over the same K draws bon "
            "pulls toward one a*")
    if config.bc_filter:
        raise ValueError(
            "--bc-target bon replaces the BC term on the DVD rows and --bc-filter gates that same term: pick one")


def check_bc_target_resume(config, output_dir) -> None:
    """--resume guard for --bc-target, run by train() BEFORE it rewrites config.json."""
    if not getattr(config, "resume", ""):
        return
    try:
        prev = json.loads((Path(output_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(prev, dict):
        return
    was, now = prev.get("bc_target") or "base", getattr(config, "bc_target", "base")
    if was == "base" and now == "base":
        return
    keys = ("bc_target", "bc_target_after", "bc_target_ramp")
    diff = {key: (prev.get(key), getattr(config, key, None)) for key in keys if prev.get(key) != getattr(config, key, None)}
    if diff:
        raise ValueError(
            f"--resume into {output_dir}: its config.json and this launch disagree on {diff} (previous, now). "
            "Resume with the run's own --bc-target / --bc-target-after / --bc-target-ramp, or use a fresh output dir")


def check_dvd_actor_pool_resume(config, output_dir) -> None:
    """--resume guard for --dvd-actor-pool, run by train() BEFORE it rewrites config.json."""
    if not getattr(config, "resume", ""):
        return
    try:
        prev = json.loads((Path(output_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(prev, dict):
        return
    if not prev.get("dvd_actor_pool", False) and not getattr(config, "dvd_actor_pool", False):
        return
    defaults = {"dvd_actor_pool": False, "dvd_actor_pool_obs": "actor"}
    diff = {key: (prev.get(key, d), getattr(config, key, d)) for key, d in defaults.items()
            if prev.get(key, d) != getattr(config, key, d)}
    if diff:
        raise ValueError(
            f"--resume into {output_dir}: its config.json and this launch disagree on {diff} "
            "(previous, now). Resume with the run's own --dvd-actor-pool / --dvd-actor-pool-obs, or "
            "use a fresh output dir")


MQATTN_OUT_DIM_UNSET = -1


def _mqattn_val_text(key: str, value) -> str:
    """How one element of the mqattn shape reads ON THE COMMAND LINE."""
    if key == "per_view":
        return "per_view" if value else "joint"
    return repr(value)


def _mqattn_flag_text(key: str, value) -> str:
    """The flag an operator would actually TYPE to assert `value` for `key`."""
    flag = "--mqattn-scope" if key == "per_view" else f"--mqattn-{key.replace('_', '-')}"
    return f"{flag} {_mqattn_val_text(key, value)}"


def _mqattn_flag_spec(config) -> dict:
    """What the --mqattn-* flags ASSERT about the fitted module; only the keys actually set."""
    spec = {}
    if int(getattr(config, "mqattn_queries", 0) or 0):
        spec["queries"] = int(config.mqattn_queries)
    scope = str(getattr(config, "mqattn_scope", "") or "")
    if scope:
        spec["per_view"] = (scope == "per_view")
    out_dim = int(getattr(config, "mqattn_out_dim", MQATTN_OUT_DIM_UNSET))
    if out_dim != MQATTN_OUT_DIM_UNSET:
        spec["out_dim"] = out_dim
    return spec


def _check_mqattn_flag_ranges(config) -> None:
    """Value ranges for the --mqattn-* assertions."""
    queries = int(getattr(config, "mqattn_queries", 0) or 0)
    if queries < 0:
        raise ValueError(f"--mqattn-queries is a count, 0 = take the fitted checkpoint's, got "
                         f"{queries}")
    out_dim = int(getattr(config, "mqattn_out_dim", MQATTN_OUT_DIM_UNSET))
    if out_dim < MQATTN_OUT_DIM_UNSET:
        raise ValueError(f"--mqattn-out-dim is a width, 0 = assert NO projection (the raw "
                         f"concatenation), {MQATTN_OUT_DIM_UNSET} = take the fitted checkpoint's, "
                         f"got {out_dim}")


def _check_mqattn_value_dir(value_model, dino_pool: str, value_pool: str, flag_spec: dict) -> None:
    """Refuse an mqattn V directory under a different --dino-pool, and vice versa."""
    if not value_model:
        return
    try:
        with open(Path(value_model) / "config.json") as _fh:
            vcfg = json.load(_fh)
        if not isinstance(vcfg, dict):
            return
    except (OSError, ValueError):
        return
    recorded = vcfg.get("dino_pool")
    if recorded == "mqattn" and "mqattn" not in (dino_pool, value_pool):
        raise ValueError(
            f"{value_model}/config.json records dino_pool='mqattn' (a fitted "
            f"pool_mqattn.MultiQueryAttnPool readout) but this arm passes --dino-pool "
            f"{dino_pool!r}" + (f" --value-pool {value_pool!r}" if value_pool else "") + ". The "
            "per-view K=4 fit is 777-dim -- EXACTLY the mean-pool width -- so nothing downstream "
            "would catch it and the arm would spend its whole budget reading V through the wrong "
            "reduction while carrying the control's name. Pass --dino-pool mqattn, or point "
            "--value-model at the V fitted for the pooling you asked for.")
    if dino_pool == "mqattn" and recorded not in (None, "mqattn"):
        raise ValueError(
            f"--dino-pool mqattn but {value_model}/config.json records dino_pool={recorded!r}: "
            "that V's head was fitted on a different pooled vector. Point --value-model at an "
            "mqattn fit (the directory also needs mqattn_pool.pt beside it).")
    if dino_pool != "mqattn" or not flag_spec:
        return


    fitted = {k: vcfg[j] for k, j in (("queries", "mqattn_queries"),
                                      ("per_view", "mqattn_per_view"),
                                      ("out_dim", "mqattn_out_dim")) if j in vcfg}
    if "out_dim" in fitted:


        fitted["out_dim"] = int(fitted["out_dim"] or 0)
    bad = {k: (v, fitted[k]) for k, v in flag_spec.items() if k in fitted and fitted[k] != v}
    if bad:
        raise ValueError(
            f"{value_model}/config.json records a "
            + ", ".join(f"{'scope' if k == 'per_view' else k}={_mqattn_val_text(k, fitted[k])}"
                        for k in sorted(fitted))
            + " multi-query pool, but this arm asserted "
            + ", ".join(_mqattn_flag_text(k, want) for k, (want, _got) in sorted(bad.items()))
            + ". The --mqattn-* flags name the module the arm THINKS it is running and appear in "
              "its config.json; a mismatch would mislabel the whole cell.")


def check_dino_config(config: TrainConfig) -> None:
    """Startup guard for --dvd-dynamics dino_wm."""

    check_dvd_fit_sampler(config)
    check_dvd_k2(config)
    check_dvd_row_chunks(config)

    check_bc_target(config)
    if config.dvd_dynamics != "dino_wm":
        if config.dvd_rows > 0 and config.dvd_dynamics not in ("seghead_wm", "latent56"):
            raise ValueError(
                "--dvd-rows only subsamples the DINO-WM path (--dvd-dynamics dino_wm). The "
                "state-space f costs microseconds per row, so a state arm passing this would "
                "think it had changed something and would not have.")
        if config.flow_steps or config.flow_inverse_iters:
            raise ValueError(
                f"--flow-steps/--flow-inverse-iters configure the DINO-WM rectified-flow ODE "
                f"(--dvd-dynamics dino_wm), but this arm is --dvd-dynamics {config.dvd_dynamics}. "
                f"The state-space flow f has its own solver and does not read these, so the arm "
                f"would run its own control while its name said otherwise.")
        if config.dvd_dynamics == "seghead_wm":
            if config.dvd_finetune_dynamics and not config.pixel_obs:
                raise ValueError(
                    "--dvd-finetune-dynamics on seghead_wm needs --pixel-obs: f's target is the "
                    "encoder's own map of next_image, and the replay buffer only allocates the "
                    "image/next_image columns under --pixel-obs (or a DINO adapter). Without it "
                    "the fit batch KeyErrors on next_image at the first update.")
            if config.dvd_mode == "replay":
                raise ValueError(
                    "--dvd-mode replay is not implemented for a feature-space f "
                    "(DvdTerm.replay_objective has no seghead_wm branch); use --dvd-mode dpg")
            if config.dvd_phase_gate and config.pixel_obs:
                raise ValueError(
                    "--dvd-phase-gate reads raw_state(batch['state']); under --pixel-obs "
                    "batch['state'] is the IMPALA encoder output, not the 37-dim env state "
                    "raw_state's low/span were fit against")
        return
    if config.flow_steps < 0 or config.flow_inverse_iters < 0:
        raise ValueError(
            f"--flow-steps {config.flow_steps} --flow-inverse-iters {config.flow_inverse_iters}: "
            f"both are loop counts. 0 means 'the dino_wm default'; negative means nothing.")
    from .dino_wm import HORIZON

    if config.dvd_weight <= 0.0 and config.bon_source != "dvd" and not config.dino_obs:


        raise ValueError("--dvd-dynamics dino_wm with no DVD consumer: pass --dvd-weight > 0 "
                         "or --bon-source dvd or --dino-obs, or the world model is loaded and "
                         "never read")
    if config.task != "threading_d0":


        _want = f"mimicgen_{config.task}_224x224_dinowm"
        if _want not in str(getattr(config, "dino_wm_ckpt", "") or ""):
            raise ValueError(
                f"the DINO-WM checkpoint {getattr(config, 'dino_wm_ckpt', None) or 'mimicgen_threading_d0_224x224_dinowm (default)'} "
                f"is not {_want}; task is {config.task!r}. Its DINOv2 features and its 8-step lead are task-specific: "
                f"pass --dino-wm-ckpt <...{_want}.../dino_step_*.pt>.")
    if config.action_steps != HORIZON:
        raise ValueError(
            f"--action-steps {config.action_steps} but the world model's action token is "
            f"{HORIZON} x 7 and its image lead time is {HORIZON} frames")
    if config.exploration_gate_mm > 0.0:


        raise ValueError(
            "--exploration-gate-mm needs the dynamics model's object error in MILLIMETRES "
            "(dvd_pod_mm). A feature-space f reports dvd_feat_mse, which is not a distance "
            "and must never be compared against a mm threshold.")
    if config.dvd_mode == "replay":
        raise ValueError("--dvd-mode replay is not implemented for a feature-space f")
    if config.dvd_phase_gate and config.dvd_rows > 0:
        raise ValueError(
            "--dvd-phase-gate blends the DVD term with Q row by row, so every row needs its "
            "term; --dvd-rows would leave most of them without one. Pick one.")
    if config.bc_filter and config.bc_filter_source == "dvd":


        if config.dvd_dynamics != "dino_wm":
            raise ValueError("--bc-filter-source dvd on a feature-space f is implemented for --dvd-dynamics dino_wm only.")
        if config.dvd_rows <= 0:
            raise ValueError("--bc-filter-source dvd on the DINO f scores only the DVD term's rows: set --dvd-rows > 0.")
        if config.train_critic or config.dvd_weight < 1.0:
            raise ValueError("--bc-filter-source dvd on the DINO f reuses the dvd_advantage rows, which exist only with "
                             "--no-train-critic and --dvd-weight 1.0.")
        if config.bc_filter_soft:
            raise ValueError("--bc-filter-soft compares a critic against mc_return; it has no meaning with "
                             "--bc-filter-source dvd on the DINO f.")
    if config.mbpo_weight > 0.0:
        raise ValueError(
            "--mbpo-weight reads f's output as a raw state and thresholds its geometry to mint "
            "synthetic reward; a feature-space f has no geometry to threshold.")
    if config.value_intervention != "none":


        raise ValueError(
            "a feature-space V is fed pooled world-model features, but --value-intervention "
            f"{config.value_intervention!r} calls it on the 37-dim scaled state. Run this arm "
            "with no intervention.")


    if config.dino_obs:


        bad = [n for n, on in (("--eval-strategy max_q_min", config.eval_strategy == "max_q_min"),
                               ("--dvd-phase-gate", config.dvd_phase_gate),
                               ("--bc-filter-source dvd", config.bc_filter_source == "dvd"))
               if on]
        if bad:
            raise ValueError(
                f"--dino-obs with {', '.join(bad)}: these paths call dvd_score/raw_state on "
                "`state`, which --dino-obs makes the 777-dim observation rather than the 37-dim "
                "env state f needs. Supported combination is greedy eval with the default "
                "bc-filter source and no phase gate.")
    if getattr(config, "seg_obs", False):
        if config.dvd_dynamics != "seghead_wm":
            raise ValueError(
                f"--seg-obs needs --dvd-dynamics seghead_wm (got {config.dvd_dynamics!r}): the "
                "observation is the SegHead world model's own map, and the backbone that "
                "produces it is only loaded on that path.")
        if getattr(config, "seg_obs_finetune", False) and getattr(config, "seg_obs_freeze", False):
            raise ValueError("--seg-obs-finetune with --seg-obs-freeze: one trains the SegHead "
                             "head by the critic loss, the other freezes the readout. Pick one.")
        if config.pixel_obs or config.dino_obs or getattr(config, "latent_obs", False):
            raise ValueError("--seg-obs with another observation flag: each overwrites "
                             "batch['state'], so the actor would read whichever ran last.")
    if getattr(config, "latent_obs", False):
        if config.dvd_dynamics != "latent56":
            raise ValueError(
                f"--latent-obs needs --dvd-dynamics latent56 (got {config.dvd_dynamics!r}): the "
                "observation is the world model's own latent, and the encoder that produces it "
                "is only loaded on that path.")
        if config.pixel_obs or config.dino_obs:
            raise ValueError("--latent-obs with --pixel-obs/--dino-obs: each overwrites "
                             "batch['state'] with its own encoder's output, so the actor would "
                             "read whichever branch ran last. Pick one observation.")
        bad = [n for n, on in (("--eval-strategy max_q_min", config.eval_strategy == "max_q_min"),
                               ("--dvd-phase-gate", config.dvd_phase_gate),
                               ("--bc-filter-source dvd", config.bc_filter_source == "dvd"))
               if on]
        if bad:
            raise ValueError(
                f"--latent-obs with {', '.join(bad)}: these paths call dvd_score/raw_state on "
                "`state`, which --latent-obs makes the 65-dim observation rather than the 37-dim "
                "env state f needs.")
    if config.dino_obs and config.pixel_obs:
        raise ValueError(
            "--dino-obs with --pixel-obs: both overwrite batch['state'] with their own encoder's "
            "output, so the actor would read whichever branch ran last. Pick one observation.")
    if config.dino_obs and config.dvd_dynamics != "dino_wm":
        raise ValueError(
            f"--dino-obs needs --dvd-dynamics dino_wm (got {config.dvd_dynamics!r}): the "
            "observation is built from the world model's own pooled features, and there is no "
            "such feature space without it.")
    if config.dino_pool == "xattn":


        if config.value_pool not in ("", "xattn"):
            raise ValueError(
                f"--dino-pool xattn with --value-pool {config.value_pool!r}: CrossAttnPool is the "
                "pooling for BOTH the observation and V's input and ignores the mode argument, so "
                "the --value-pool would have no effect while naming a different reduction. Drop it "
                "(or pass --value-pool xattn, which is the same thing).")
        if config.xattn_queries < 1 or config.xattn_heads < 1:
            raise ValueError(f"--xattn-queries/--xattn-heads are counts: got "
                             f"{config.xattn_queries}/{config.xattn_heads}")
        from .dino_wm import FEATURE_DIM as _XF, pooled_dim as _xpd
        if _xpd("xattn") % config.xattn_queries:
            raise ValueError(
                f"--xattn-queries {config.xattn_queries} does not divide the pooled width "
                f"{_xpd('xattn')}: each query owns out_dim//K dims and they are concatenated, so "
                f"K must divide it (4, 8 and 16 all do).")
        if _XF % config.xattn_heads:
            raise ValueError(f"--xattn-heads {config.xattn_heads} does not divide the patch "
                             f"dimension {_XF}")
    elif config.xattn_queries != 8 or config.xattn_heads != 6:
        raise ValueError(
            f"--xattn-queries/--xattn-heads configure dino_wm.CrossAttnPool (--dino-pool xattn), "
            f"but this arm is --dino-pool {config.dino_pool!r}. They would be recorded in "
            f"config.json and change nothing, which is how three knobs in the last sweep were "
            f"silently inert.")
    if config.dino_pool == "mqattn":


        if config.value_pool not in ("", "mqattn"):
            raise ValueError(
                f"--dino-pool mqattn with --value-pool {config.value_pool!r}: "
                "MultiQueryAttnPool is the pooling for BOTH the observation and V's input and "
                "ignores the mode argument, so the --value-pool would have no effect while naming "
                "a different reduction. Drop it (or pass --value-pool mqattn, which is the same "
                "thing).")
        _check_mqattn_flag_ranges(config)
    else:
        if config.value_pool == "mqattn":
            raise ValueError(
                f"--value-pool mqattn needs --dino-pool mqattn (got {config.dino_pool!r}): the "
                "MultiQueryAttnPool module is loaded on that path only, so V's input would be "
                "built by dino_wm.pool_features, which has no 'mqattn' mode.")
        if _mqattn_flag_spec(config):
            raise ValueError(
                f"--mqattn-queries/--mqattn-scope/--mqattn-out-dim assert the shape of the fitted "
                f"pool_mqattn.MultiQueryAttnPool (--dino-pool mqattn), but this arm is "
                f"--dino-pool {config.dino_pool!r}. They would be recorded in config.json and "
                f"change nothing, which is how three knobs in the last sweep were silently inert.")

    _check_mqattn_value_dir(config.value_model, config.dino_pool, config.value_pool,
                            _mqattn_flag_spec(config))
    if getattr(config, "attn_pool_scratch", False):
        if config.dino_pool != "mqattn":
            raise ValueError(f"--attn-pool-scratch is implemented for --dino-pool mqattn (got {config.dino_pool!r})")
        if getattr(config, "attn_pool_freeze", False) or not config.value_finetune:
            raise ValueError("--attn-pool-scratch needs a TRAINED readout (--value-finetune, no --attn-pool-freeze): "
                             "a frozen random readout would be a different, unlabelled experiment")
    if getattr(config, "dino_obs_pool_separate", False):


        if config.dino_pool not in ("xattn", "mqattn"):
            raise ValueError(
                f"--dino-obs-pool-separate builds a second trainable pool (dino_wm.CrossAttnPool or "
                f"pool_mqattn.MultiQueryAttnPool): it needs --dino-pool xattn or mqattn (got "
                f"{config.dino_pool!r}).")
        if getattr(config, "dino_obs_pool_scratch", False) and config.dino_pool != "mqattn":
            raise ValueError(
                f"--dino-obs-pool-scratch is implemented for --dino-pool mqattn (got "
                f"{config.dino_pool!r}).")
        if not config.dino_obs:
            raise ValueError(
                "--dino-obs-pool-separate splits the ACTOR/CRITIC observation pool from V's, but "
                "without --dino-obs the actor and critic read the 37-dim env state and there is no "
                "observation pool to split. Pass --dino-obs, or drop this flag.")
        if config.attn_pool_freeze:
            raise ValueError(
                "--dino-obs-pool-separate with --attn-pool-freeze: variant 3 is ONE shared pool "
                "frozen at the offline fit, variant 2 is TWO pools of which the actor/critic's "
                "trains. Pick one.")
        if not config.train_critic:
            raise ValueError(
                "--dino-obs-pool-separate needs a trained critic (--train-critic): the second "
                "pool's parameters live in the critic optimiser and the actor reads a detached "
                "copy, so with --no-train-critic nothing would ever train it -- the arm would be "
                "an exact, slower duplicate of the shared-pool run.")
        if getattr(config, "freeze_init_critic", False):
            raise ValueError(
                "--dino-obs-pool-separate with --freeze-init-critic: that arm skips "
                "critic_loss.backward() entirely, so the second pool would sit at its offline fit "
                "forever -- the same silently-inert cell --no-train-critic is refused for.")
        if config.method != "dice_rl":


            raise ValueError(
                f"--dino-obs-pool-separate is implemented for --method dice_rl (got "
                f"{config.method!r}): only that update path applies the --dino-obs observation "
                f"rewrite and reads the actor's state through the detached copy.")
    elif getattr(config, "dino_obs_pool_scratch", False) or getattr(config, "dino_obs_pool_ema_target", False):
        raise ValueError(
            "--dino-obs-pool-scratch / --dino-obs-pool-ema-target modify variant 2's actor/critic "
            "pool: they need --dino-obs-pool-separate, or they would be recorded and change nothing.")
    if getattr(config, "dvd_actor_pool", False):


        if config.dino_pool not in ("attnpool", "xattn", "mqattn"):
            raise ValueError(
                f"--dvd-actor-pool copies the FITTED pooling module: it needs --dino-pool "
                f"attnpool, xattn or mqattn (got {config.dino_pool!r}). Mean pooling is a "
                f"parameter-free function -- there is nothing for the actor to own.")
        if not config.value_model:
            raise ValueError("--dvd-actor-pool copies the readout fitted beside V: it needs "
                             "--value-model")
        if config.dvd_dynamics != "dino_wm":
            raise ValueError(
                f"--dvd-actor-pool is implemented for --dvd-dynamics dino_wm (got "
                f"{config.dvd_dynamics!r}): DvdTerm.value_input only pools in feature space, and "
                f"the seghead/latent56 paths bind dvd_term._pool differently.")
        if config.dvd_weight <= 0.0:
            raise ValueError(
                "--dvd-actor-pool with --dvd-weight 0: L_DVD is the ONLY loss that reaches the "
                "actor's copy (DvdTerm.value_input is the one un-no_grad'd pool call), so nothing "
                "would train it and the run would be byte-identical to the shared-pool arm while "
                "being labelled a two-module cell.")
        if config.method != "dice_rl":
            raise ValueError(
                f"--dvd-actor-pool is implemented for --method dice_rl (got {config.method!r}): "
                f"only update_dice evaluates the DVD objective and applies the --dino-obs "
                f"observation rewrite.")
        if getattr(config, "dino_obs_pool_separate", False):
            raise ValueError(
                "--dvd-actor-pool with --dino-obs-pool-separate: two different second-module "
                "stories. This one gives the ACTOR a copy trained by L_DVD at --actor-lr; that "
                "one gives the actor/critic OBSERVATION a copy trained by the critic's TD loss. "
                "Pick one.")
        if config.value_pool == "sargmax":
            raise ValueError(
                "--dvd-actor-pool with --value-pool sargmax: that path rebinds dvd_term._pool to "
                "the keypoint readout after the copy is taken, so the actor would differentiate a "
                "module nothing reads. Pick one.")
        if getattr(config, "bon_frozen_v", "off") != "off":
            raise ValueError(
                "--dvd-actor-pool with --bon-frozen-v: the frozen-V co-rank exists to score "
                "candidates through a FIXED lens, and its own guard demands --attn-pool-freeze for "
                "exactly that reason. Best-of-N ranks through value_input, i.e. through the "
                "actor's MOVING copy, so that guarantee is void even with the freeze.")
        if config.dino_obs and config.value_intervention in ("floor", "penalty"):
            raise ValueError(
                f"--dvd-actor-pool --dino-obs with --value-intervention {config.value_intervention}: "
                f"that intervention evaluates value_fn on batch[\"state\"], which under "
                f"--dvd-actor-pool-obs actor is pooled by the ACTOR'S copy -- V would be read "
                f"through a lens it was never fitted in. Use --value-intervention none, or "
                f"--dvd-actor-pool-obs value.")
        if getattr(config, "dvd_actor_pool_obs", "actor") not in ("actor", "value"):
            raise ValueError(
                f"--dvd-actor-pool-obs must be 'actor' or 'value' (got "
                f"{config.dvd_actor_pool_obs!r})")
        if getattr(config, "dvd_actor_pool_obs", "actor") == "value" and not config.dino_obs:
            raise ValueError(
                "--dvd-actor-pool-obs value picks the module that pools the --dino-obs "
                "OBSERVATION, but this arm has no feature observation to pool. Drop it.")
        if getattr(config, "dvd_pool_actor_step", False):
            raise ValueError(
                "--dvd-actor-pool (a separate actor COPY) with --dvd-pool-actor-step (ONE shared "
                "pool in both optimisers): different designs. Pick one.")
        if (config.dino_pool == "xattn" and config.dino_obs
                and not getattr(config, "attn_pool_freeze", False)
                and config.train_critic and not getattr(config, "freeze_init_critic", False)):


            raise ValueError(
                "--dvd-actor-pool on CAP variant 1 (--dino-pool xattn --dino-obs, trainable pool, "
                "a critic that trains): the critic's TD gradient would reach the actor's copy, "
                "which the critic optimiser does not own, so the critic would train no pool while "
                "the arm is labelled variant 1. Freeze V's instance (--attn-pool-freeze) or run "
                "it with --no-train-critic.")
        if getattr(config, "dvd_actor_pool_log_every", 200) < 0:
            raise ValueError("--dvd-actor-pool-log-every must be >= 0 (0 disables the diagnostic)")
    elif getattr(config, "dvd_actor_pool_obs", "actor") != "actor":
        raise ValueError(
            "--dvd-actor-pool-obs only has a meaning with --dvd-actor-pool (there is one module "
            "otherwise). It would be recorded in config.json and change nothing.")
    elif getattr(config, "dvd_actor_pool_log_every", 200) != 200:
        raise ValueError(
            "--dvd-actor-pool-log-every only has a meaning with --dvd-actor-pool (there is no "
            "diagnostic otherwise). It would be recorded in config.json and change nothing.")
    if (config.dino_pool == "xattn" and config.dino_obs
            and not getattr(config, "attn_pool_freeze", False)
            and not getattr(config, "dino_obs_pool_separate", False)):

        _critic_trains = (config.method == "dice_rl" and config.train_critic
                          and not getattr(config, "freeze_init_critic", False))
        if not _critic_trains and not config.value_finetune:
            raise ValueError(
                "--dino-pool xattn --dino-obs (CAP variant 1) with nothing that can train the "
                f"shared pool: method={config.method!r} train_critic={config.train_critic} "
                f"freeze_init_critic={getattr(config, 'freeze_init_critic', False)} "
                "value_finetune=False. No optimiser would hold the module, so the run would be "
                "byte-identical to variant 3 while being labelled variant 1. Add --value-finetune, "
                "run it on a critic-training dice_rl arm, or say what you mean with "
                "--attn-pool-freeze.")
    if config.pixel_obs and not config.train_critic:


        if not (config.actor_train_encoder
                or getattr(config, "encoder_value_aux_only", False)
                or getattr(config, "encoder_init", "")):


            raise ValueError(
                "--pixel-obs with --dvd-dynamics dino_wm and --no-train-critic: the encoder is 2M "
                "parameters nothing trains, and dvd_advantage would score encoder features against "
                "a world model defined on the env state. Either drop --pixel-obs (the DINO path "
                "stores the frames itself), train the critic (the vfpix arm), or pass "
                "--actor-train-encoder (the pure-DVD arm: the actor loss trains the encoder), "
                "or pass --encoder-value-aux-only (L_V alone trains it).")
        if getattr(config, "encoder_value_aux_only", False):
            print("[aux-only] no critic; L_V is the encoder's only trainer", flush=True)
        else:
            print("[pure-dvd] no critic; the actor loss is the encoder's only trainer", flush=True)


def train(config: TrainConfig) -> None:
    import wandb

    device = torch.device("cuda")
    set_seed(config.seed)


    check_state_repr(config)
    check_dino_config(config)

    check_bc_target_resume(config, Path(config.output_dir))

    check_dvd_actor_pool_resume(config, Path(config.output_dir))
    output_dir = Path(config.output_dir)


    output_dir.mkdir(parents=True, exist_ok=bool(config.resume))
    (output_dir / "checkpoints").mkdir(exist_ok=bool(config.resume))


    try:
        import hashlib
        _here = Path(__file__).resolve().parent
        (output_dir / "code_version.json").write_text(json.dumps({
            f: hashlib.md5((_here / f).read_bytes()).hexdigest()
            for f in ("train_online.py", "common.py", "robometer_distill.py", "pixel.py",
                      "dino_dvd.py", "value_model.py") if (_here / f).exists()}, indent=1))
    except Exception as _e:
        print(f"[code-version] not recorded: {_e}", flush=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    metrics_path = output_dir / "metrics.jsonl"

    task_cfg = TASKS[config.task]
    state_dim = int(task_cfg["state_dim"])
    chunk_dim = config.action_steps * 7


    scaler = (
        GoalRelativeScaler(task_cfg["dataset_root"], config.task)
        if config.state_repr == "goal_relative"
        else StateScaler(task_cfg["dataset_root"])
    )
    prior = FrozenPolicyPrior(
        config.task,
        Path(config.base_checkpoint),
        config.action_steps,
        config.inference_steps,
        prior_temperature=config.prior_temperature,
    )


    flowdpg_wrapper = flowdpg_cond = flowdpg_executed = None
    if config.method == "flowdpg":
        from .flowdpg import attach_prior, executed_slice as _executed_slice
        flowdpg_wrapper, flowdpg_cond = attach_prior(
            prior, mode=config.flowdpg_mode, hidden=config.flowdpg_hidden,
            layers=config.flowdpg_layers, rank=config.flowdpg_rank)
        flowdpg_executed = _executed_slice(prior.policy)


        if config.residual_scale != 0.0:
            print(f"[flowdpg] forcing residual_scale {config.residual_scale} -> 0.0: the policy "
                  "is prior + r_phi, and the ResidualActor is never trained by this method",
                  flush=True)
            config.residual_scale = 0.0
        n_train = sum(p.numel() for p in flowdpg_wrapper.trainable_parameters())
        n_frozen = sum(p.numel() for p in flowdpg_wrapper.frozen.parameters())
        print(f"[flowdpg] mode={config.flowdpg_mode}  r_phi {n_train/1e6:.3f}M trainable vs "
              f"{n_frozen/1e6:.3f}M frozen  executed slice {flowdpg_executed}", flush=True)

    insertion_cfg = ({"span": config.insertion_span,
                      "joint_sigma": config.insertion_joint_sigma,
                      "holder_sigma": config.insertion_holder_sigma}
                     if config.insertion_only else None)
    seg_term = None
    if getattr(config, "seg_terminate_patience", 0):
        if config.task != "threading_d0":
            raise SystemExit(
                "--seg-terminate-patience encodes threading_d0's needle/tripod state slices and "
                f"its measured endpoint reference; it is not calibrated for {config.task!r}.")
        seg_term = _SegTerminator(config.seg_terminate_patience, config.action_steps)
        print(f"[seg-term] causal segment cut on, patience={config.seg_terminate_patience} "
              "primitive steps (TRAINING rollouts only; eval is untouched)", flush=True)

    env = make_env(config.task, config.episode_horizon, config.env_dense_reward, insertion_cfg,
                   holder_sigma=config.holder_sigma, fixed_goal=config.fixed_goal,
                   fixed_goal_rot=config.fixed_goal_rot, full_fixed=config.full_fixed,
                   jitter_goal_m=config.jitter_goal_mm / 1000.0)
    layernorm = config.method == "dice_rl"
    ensemble_size = config.critic_ensemble_size if config.method == "dice_rl" else 2
    encoder = target_encoder = None
    encoder_optimizer = None
    actor_encoder = None
    critic_state_dim = state_dim
    if config.dino_obs:


        from .dino_wm import pooled_dim as _pooled_dim
        critic_state_dim = _pooled_dim(config.dino_pool) + PIXEL_PROPRIO
        print(f"[dino-obs] actor/critic read V's input: {_pooled_dim(config.dino_pool)} pooled "
              f"+ {PIXEL_PROPRIO} raw proprio -> {critic_state_dim}", flush=True)
    if getattr(config, "seg_obs", False):
        from .seghead import SegObsEncoder as _SegObsEnc
        _seg_obs_enc = _SegObsEnc(n_views=len(PIXEL_CAMERAS)).to(device)
        if getattr(config, "seg_obs_freeze", False):
            if not config.value_model:
                raise ValueError(
                    "--seg-obs-freeze needs --value-model: the frozen readout is LOADED from that "
                    "checkpoint's fitted ConvV trunk. Without it the arm would freeze a random "
                    "projection, which tests neither hypothesis.")


            import glob as _glob
            _w = sorted(_glob.glob(str(Path(config.value_model) / "weights" / "v_td_seed*.pt")))[0]
            _sd = torch.load(_w, map_location=device)
            _m = _seg_obs_enc.load_state_dict(_sd, strict=False)
            _loaded = [k for k in _sd if k.startswith(("trunk.", "to_heat."))]
            if not _loaded:
                raise RuntimeError(f"{_w} carries no trunk/to_heat weights to freeze")
            _seg_obs_enc.requires_grad_(False).eval()
            print(f"[seg-obs] readout FROZEN from {Path(_w).parent.parent.name}: "
                  f"{len(_loaded)} tensors loaded, unexpected={len(_m.unexpected_keys)}", flush=True)
        _LATENT_OBS_CTX.update(seg_enc=_seg_obs_enc, scaler=scaler)
        critic_state_dim = _seg_obs_enc.out_dim + PIXEL_PROPRIO
        print(f"[seg-obs] actor/critic read {_seg_obs_enc.out_dim} spatial-softmax keypoints "
              f"+ {PIXEL_PROPRIO} proprio -> {critic_state_dim}; "
              f"{sum(q.numel() for q in _seg_obs_enc.parameters())} trainable params", flush=True)
    if getattr(config, "latent_obs", False):


        import json as _json
        _lw = int(_json.loads(
            (Path(config.dvd_latent_wm) / "config.json").read_text())["latent"])
        critic_state_dim = _lw + PIXEL_PROPRIO
        print(f"[latent-obs] actor/critic read V's input: {_lw} latent + {PIXEL_PROPRIO} "
              f"normalised proprio -> {critic_state_dim}", flush=True)
    if config.pixel_obs:


        encoder = PixelEncoder(out_dim=config.pixel_feature_dim,
                               shift_pad=config.pixel_shift_pad).to(device)
        target_encoder = PixelEncoder(out_dim=config.pixel_feature_dim,
                                      shift_pad=config.pixel_shift_pad).to(device)
        if getattr(config, "encoder_init", ""):


            _eblob = torch.load(config.encoder_init, map_location=device, weights_only=False)
            _esd = _eblob.get("extra", {}).get("encoder")
            if _esd is None:
                raise SystemExit(f"--encoder-init {config.encoder_init}: no extra.encoder in checkpoint")
            encoder.load_state_dict(_esd, strict=True)
            encoder.requires_grad_(False)
            print(f"[encoder-init] pixel encoder loaded from {config.encoder_init} "
                  f"(@{_eblob.get('env_steps')} env steps) and FROZEN", flush=True)
            del _eblob, _esd
        target_encoder.load_state_dict(encoder.state_dict())
        for q in target_encoder.parameters():
            q.requires_grad_(False)
        if getattr(config, "actor_encoder_finetune", False):


            import copy as _copy_ae
            if getattr(config, "actor_encoder_scratch", False):


                actor_encoder = PixelEncoder(out_dim=config.pixel_feature_dim,
                                             shift_pad=config.pixel_shift_pad).to(device)
            else:
                actor_encoder = _copy_ae.deepcopy(encoder)
            actor_encoder.requires_grad_(True)
            print(f"[actor-encoder] actor reads its own TRAINABLE "
                  f"{'FRESHLY INITIALISED encoder (--actor-encoder-scratch)' if getattr(config, 'actor_encoder_scratch', False) else 'copy of the --encoder-init encoder'} "
                  f"({sum(q.numel() for q in actor_encoder.parameters())/1e6:.2f}M params); "
                  f"f/V/BoN keep the frozen original", flush=True)
        if getattr(config, "dvd_encoder_learned", False):


            encoder.requires_grad_(True)
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.value_finetune_lr)
            print(f"[encoder-learned] V's IMPALA encoder trains from scratch "
                  f"({sum(q.numel() for q in encoder.parameters())/1e6:.2f}M params, lr {config.value_finetune_lr}, "
                  f"EMA tau {config.tau}, DrQ {config.pixel_augment})", flush=True)
        critic_state_dim = encoder.out_dim
        print(f"[pixel] IMPALA encoder: {len(PIXEL_CAMERAS)} cameras at {PIXEL_IMG}x{PIXEL_IMG} "
              f"+ {PIXEL_PROPRIO}-dim proprio -> {critic_state_dim} features "
              f"({sum(q.numel() for q in encoder.parameters())/1e6:.2f}M params)", flush=True)


    dino = dino_adapter = dino_history = None
    if config.dvd_dynamics == "dino_wm":
        from . import dino_dvd as dino


        _sweep_wm = False
        if config.dino_wm_ckpt:
            from .dino_wm_any import is_sweep_checkpoint
            _sweep_wm = is_sweep_checkpoint(config.dino_wm_ckpt)
        if _sweep_wm:


            if (config.dino_wm_arch or config.flow_steps or config.flow_inverse_iters
                    or config.dvd_finetune_dino_blocks or config.dvd_checkpoint
                    or config.dino_wm_state_head):
                raise SystemExit(f"{config.dino_wm_ckpt} is a sweep checkpoint (MSE head, arch in its "
                                 "metadata): drop --dino-wm-arch, --flow-steps/--flow-inverse-iters, "
                                 "--dvd-finetune-dino-blocks and --dvd-checkpoint")
            from .dino_wm_any import load_any
            dino.assert_frame_size()
            dino_adapter = load_any(config.dino_wm_ckpt, device=device, bf16=config.dino_bf16)
        else:
            dino_adapter = dino.build(config.dino_wm_ckpt or None, device=device,
                                      bf16=config.dino_bf16, action_dim=7,
                                      arch=config.dino_wm_arch,
                                      flow_steps=config.flow_steps or None,
                                      flow_inverse_iters=config.flow_inverse_iters or None,
                                      finetune_dino_blocks=config.dvd_finetune_dino_blocks,
                                      dvd_checkpoint=config.dvd_checkpoint,
                                      state_head=config.dino_wm_state_head)
        dino_adapter.mse_anchor = bool(config.dvd_residual_anchor)
        if getattr(config, "bc_target", "base") == "bon" and getattr(dino_adapter, "flow", False):

            raise SystemExit(
                "--bc-target bon ranks K prior draws through f's forward, which on a FLOW world model is one draw "
                "from fresh noise per candidate (it moves the prediction 14.9x more than the action, "
                "dino_wm.DinoWmAdapter.forward): the argmax would rank noise. Use an MSE world model")
        if config.dvd_residual_anchor:
            print("[dino-wm] residual anchor ON: adapter.forward returns anchor + delta "
                  "(fit target and V input are absolute-scale)", flush=True)
        dino_history = dino.FrameHistory(dino_adapter)

    pixel_demo_cache = None


    if (config.pixel_obs or dino_adapter is not None
            or config.dvd_dynamics in ("latent56", "seghead_wm")):

        cache_name = f"pixel_cache_{Path(task_cfg['dataset_root']).name}_{PIXEL_IMG}.npy"
        cache_path = Path(os.environ.get(
            "PIXEL_CACHE", CACHE_ROOT / "pixel_cache" / cache_name))
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{cache_path} is missing. Build it once with "
                "`python -m online_rl_mimicgen.pixel_cache` -- the expert buffer needs demo "
                "images and decoding them per run would cost ~10 minutes each.")
        pixel_demo_cache = np.load(cache_path, mmap_mode="r")


        if pixel_demo_cache.shape[1:] != (len(PIXEL_CAMERAS), PIXEL_IMG, PIXEL_IMG, 3):
            raise RuntimeError(
                f"{cache_path} holds {pixel_demo_cache.shape[1:]} per frame but PIXEL_CAMERAS="
                f"{PIXEL_CAMERAS} at {PIXEL_IMG}px needs "
                f"{(len(PIXEL_CAMERAS), PIXEL_IMG, PIXEL_IMG, 3)}. Rebuild the cache with the "
                "same PIXEL_CAMERAS the run uses.")
        print(f"[pixel] demo cache {cache_path.name}: {pixel_demo_cache.shape}", flush=True)
    make_critics = (
        (lambda: PixelCriticEnsemble(critic_state_dim, chunk_dim, ensemble_size,
                                     hidden=(config.critic_hidden,) * config.critic_layers,
                                     layernorm=layernorm))
        if config.pixel_obs else


        (lambda: CriticEnsemble(critic_state_dim, chunk_dim, ensemble_size,
                                hidden=(config.critic_hidden,) * config.critic_layers,
                                layernorm=layernorm)))
    actor = ResidualActor(
        critic_state_dim,
        chunk_dim,
        layernorm=layernorm,
        hidden=(config.actor_hidden,) * 3,
        initial_log_std=config.resolved_initial_log_std(),
        log_std_min=config.log_std_min,
        stochastic=not config.deterministic_residual,
        latent_dim=config.latent_dim,
        head=config.residual_head,
        action_steps=config.action_steps,
        conv_kernel=config.residual_conv_kernel,
        conv_channels=config.residual_conv_channels,
        conv_broadcast=config.residual_conv_broadcast,
    ).to(device)
    critics = make_critics().to(device)
    target_critics = make_critics().to(device)
    target_critics.load_state_dict(critics.state_dict())

    target_actor = None
    if config.target_actor:
        import copy as _copy
        target_actor = _copy.deepcopy(actor).to(device)
        target_actor.load_state_dict(actor.state_dict())
        target_actor.requires_grad_(False)
        target_actor.eval()
        print(f"[target-actor] enabled, Polyak tau={config.tau}", flush=True)
    if config.init_actor_from:
        donor = torch.load(config.init_actor_from, map_location=device, weights_only=False)
        actor.load_state_dict(donor["actor"])
        dc = donor.get("config", {})
        print(f"Warm-started ACTOR from {config.init_actor_from} "
              f"(mode={dc.get('pretrain_mode')}, updates={donor.get('updates')}, "
              f"data={dc.get('buffer')}); critic and everything else are fresh", flush=True)
        for key, want in (("actor_hidden", config.actor_hidden),
                          ("latent_dim", config.latent_dim),
                          ("deterministic_residual", config.deterministic_residual)):
            got = dc.get(key)
            if got is not None and got != want:
                raise SystemExit(
                    f"actor donor has {key}={got} but this run uses {want}; the shapes or the "
                    "sampling path would not match")
    if config.init_critic_from:
        donor = torch.load(config.init_critic_from, map_location=device, weights_only=False)


        critics.load_state_dict(donor["critics"])
        target_critics.load_state_dict(donor["target_critics"])
        donor_config = donor.get("config", {})
        print(
            f"Warm-started critic from {config.init_critic_from} "
            f"(donor env_steps={donor.get('env_steps')}, updates={donor.get('updates')}, "
            f"task={donor_config.get('task')}, critic_hidden={donor_config.get('critic_hidden')}, "
            f"frozen={config.freeze_init_critic}); actor is fresh",
            flush=True,
        )
        if donor_config.get("task") not in (None, config.task):
            raise SystemExit(
                f"donor critic was trained on {donor_config['task']}, this run is {config.task}"
            )
        if encoder is not None:


            _enc_sd = donor.get("extra", {}).get("encoder")
            _tenc_sd = donor.get("extra", {}).get("target_encoder")
            if _enc_sd is None:
                raise SystemExit(
                    f"--init-critic-from {config.init_critic_from} has no extra.encoder, but this "
                    f"run has a pixel encoder. The donor's critic is defined on ITS encoder's "
                    f"features and cannot be warm-started without it.")
            encoder.load_state_dict(_enc_sd)
            if _tenc_sd is not None and target_encoder is not None:
                target_encoder.load_state_dict(_tenc_sd)
            print(f"[init-critic] encoder warm-started from the same donor "
                  f"({sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params)", flush=True)
        if config.freeze_init_critic:
            for parameter in critics.parameters():
                parameter.requires_grad_(False)
    v_aux_head = None
    if getattr(config, "encoder_value_aux_image_only", False) and (
            encoder is None or getattr(config, "encoder_value_aux", 0.0) <= 0.0):
        raise SystemExit("--encoder-value-aux-image-only needs --encoder-value-aux > 0 and a pixel "
                         "encoder (it slices the proprio tail off the pixel encoding)")
    if getattr(config, "encoder_value_aux", 0.0) > 0.0:
        if encoder is None:
            raise SystemExit("--encoder-value-aux needs a pixel encoder (--pixel-obs)")


        _vaux_in = critic_state_dim - (PIXEL_PROPRIO if getattr(config, "encoder_value_aux_image_only", False) else 0)
        v_aux_head = (torch.nn.Linear(_vaux_in, 1).to(device)
                      if getattr(config, "encoder_value_aux_head", "mlp") == "linear"
                      else torch.nn.Sequential(
                          torch.nn.Linear(_vaux_in, 256), torch.nn.ReLU(),
                          torch.nn.Linear(256, 1)).to(device))
        print(f"[value-aux] head on the {critic_state_dim}-d actor observation, "
              f"weight={config.encoder_value_aux}", flush=True)


    _vaux_only = getattr(config, "encoder_value_aux_only", False)
    _vaux_on_actor = (v_aux_head is not None
                      and (config.actor_train_encoder or _vaux_only))
    _actor_params = list(actor.parameters()) + (
        list(v_aux_head.parameters()) if _vaux_on_actor else [])
    if (config.actor_train_encoder or _vaux_only) and encoder is not None:
        _actor_params += list(encoder.parameters())
        print("[pure-dvd] actor optimizer also trains the pixel encoder "
              f"({sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params)", flush=True)
    if actor_encoder is not None:
        _actor_params += list(actor_encoder.parameters())
    actor_optimizer = torch.optim.Adam(_actor_params, lr=config.actor_lr)


    flowdpg_critic = flowdpg_target = flowdpg_actor_opt = flowdpg_critic_opt = None
    if config.method == "flowdpg":
        import copy as _copy
        from .flowdpg import IQLValue
        flowdpg_critic = IQLValue(state_dim, chunk_dim, hidden=config.critic_hidden,
                                  tau=config.flowdpg_expectile).to(device)
        flowdpg_target = _copy.deepcopy(flowdpg_critic)
        for _p in flowdpg_target.parameters():
            _p.requires_grad_(False)
        flowdpg_actor_opt = torch.optim.Adam(flowdpg_wrapper.trainable_parameters(),
                                             lr=config.actor_lr)
        flowdpg_critic_opt = torch.optim.Adam(flowdpg_critic.parameters(), lr=config.critic_lr)
    if (encoder is not None and not config.train_critic
            and not config.actor_train_encoder
            and not getattr(config, "encoder_value_aux_only", False)
            and not getattr(config, "encoder_init", "")
            and not getattr(config, "dvd_encoder_learned", False)):


        raise SystemExit(
            "--no-train-critic with a pixel encoder would leave the IMPALA encoder at its random "
            "initialisation for the entire run -- its parameters live in the critic optimiser and "
            "the actor's input is detached. Use ARM=vfpix (DVD actor, critic still training) or "
            "drop --pixel-obs.")
    critic_optimizer = torch.optim.Adam(
        list(critics.parameters())
        + (list(encoder.parameters())
           if (encoder is not None and not _vaux_only
               and not getattr(config, "encoder_init", "")
               and not getattr(config, "dvd_encoder_learned", False)) else [])
        + (list(v_aux_head.parameters())
           if (v_aux_head is not None and not _vaux_on_actor) else [])
        + (list(_LATENT_OBS_CTX["seg_enc"].parameters())
           if (getattr(config, "seg_obs", False)
               and not getattr(config, "seg_obs_freeze", False)) else []),
        lr=config.critic_lr)
    log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.critic_lr)
    prior_draws = max(
        config.resolved_actor_k(),
        config.multi_sample_k,
        (config.bon_samples or 16) if (config.method == "dice_rl"
                                       and config.bon_axis == "prior") else 1,
    )


    flowdpg_cond_dim = (int(prior.policy.noise_predictor.cond_dim)
                        - int(prior.policy.noise_predictor.frozen.time_mlp(
                            torch.zeros(1, device=device)).shape[-1])
                        if config.method == "flowdpg" else None)
    _segwm = config.dvd_dynamics == "seghead_wm"


    _lat = config.dvd_dynamics == "latent56"
    online = ReplayBuffer(
        config.replay_capacity, state_dim, chunk_dim,
        samples=config.resolved_actor_k(), next_samples=config.multi_sample_k,


        image_shape=((len(PIXEL_CAMERAS), PIXEL_IMG, PIXEL_IMG, 3)
                     if (config.pixel_obs or dino_adapter is not None
                         or getattr(config, "seg_obs_finetune", False)
                         or getattr(config, "obs_shift_pad", 0) > 0) else None),
        cond_dim=flowdpg_cond_dim,


        tok_shape=(_tok_shape_from_ckpt(config)
                   if (getattr(config, "latent_obs_finetune", False)
                       or getattr(config, "tok_shift_pad", 0) > 0) else None),


        feat_shape=(_lat_feat_shape(config) if _lat else
                    (SEG_FEAT_SHAPE if _segwm else
                     (None if dino is None else dino.FRAME_FEAT_SHAPE))),
        seg_feat=_segwm or _lat,

        boot_image=config.nstep > 1,

        continuation=config.store_prior_continuation(),
    )
    _prior_cont = config.need_prior_continuation()
    _store_cont = config.store_prior_continuation()


    try:
        import pyarrow.dataset as _pads
        _t = _pads.dataset(task_cfg["dataset_root"] / "data", format="parquet").to_table(
            columns=["observation.state"]).slice(0, 20000)
        _s = np.asarray(_t["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        _audit = scaler.audit(_s)
        if _audit["saturated_dims"]:
            raise RuntimeError(
                f"StateScaler saturates dims {_audit['saturated_dims']} on >50% of demo frames "
                f"while those dims genuinely vary -- the policy cannot see them. Repair the "
                f"observation.state q01 / q99 of meta/stats.json (recompute them from the data) "
                f"before training."
            )
        print(f"[scaler audit] no saturated dims (max clip fraction "
              f"{_audit['clip_fraction'].max():.3f})", flush=True)
    except FileNotFoundError:
        pass

    seg_hist = None
    seg_encode = None
    tok_encode = None
    seg_adapter = None
    seghead_ema = None
    seghead_guard = None
    if config.dvd_dynamics == "latent56":


        from .latent56 import (load_all as _l56_load, LatentHistory as _LatHist,
                               make_latent_encode as _mk_lat_enc)
        _l56_dino, _l56_enc_net, seg_adapter, _l56_cfg = _l56_load(
            config.dvd_latent_wm, device=device)


        _vcfg_p = Path(config.value_model) / "config.json"
        if _vcfg_p.exists():
            _vwm = json.loads(_vcfg_p.read_text()).get("wm")
            if _vwm is not None:
                _a, _b = Path(_vwm).resolve(), Path(config.dvd_latent_wm).resolve()
                if _a != _b:
                    raise SystemExit(
                        f"--value-model was fitted on latent WM {_a}\n"
                        f"but --dvd-latent-wm is {_b}.\n"
                        "Both are input_dim=65, so this would load without error and train "
                        "against the wrong latent. Pass the matched pair.")


        from .latent56 import n_views as _l56_views
        if _l56_views(_l56_cfg) != len(PIXEL_CAMERAS):
            raise RuntimeError(
                f"{config.dvd_latent_wm} was trained on {_l56_views(_l56_cfg)} camera view(s) but "
                f"PIXEL_CAMERAS is {PIXEL_CAMERAS} ({len(PIXEL_CAMERAS)}). Set PIXEL_CAMERAS to "
                f"{_l56_cfg.get('cameras', 'the checkpoint camera list')} in the launcher, and "
                "make sure the pixel cache was built with the same value.")
        seg_hist = _LatHist(_l56_dino, _l56_enc_net, _l56_cfg, device)
        if getattr(config, "latent_obs", False):
            _LATENT_OBS_CTX.update(cfg=_l56_cfg, enc=_l56_enc_net, scaler=scaler,
                                   dino=_l56_dino)
        if getattr(config, "latent_obs_finetune", False):


            _ft_head = list(_l56_enc_net.parameters())
            critic_optimizer.add_param_group({"params": _ft_head})
            print(f"[latent-obs-ft] conv head joined the critic optimiser: "
                  f"{sum(q.numel() for q in _ft_head)} params", flush=True)

        _l56_tok_shape = None
        from .latent56 import tokens as _l56_tokens
        if (getattr(config, "latent_obs_finetune", False)
                or getattr(config, "tok_shift_pad", 0) > 0):
            from .latent56 import token_shape as _l56_ts
            _l56_tok_shape = _l56_ts(_l56_cfg)


            _l56_enc_net.requires_grad_(True).train()
            _mb = int(np.prod(_l56_tok_shape)) * 2 / 1e6
            print(f"[latent-obs-ft] conv head trainable; caching DINOv2 tokens "
                  f"{_l56_tok_shape} = {_mb:.2f} MB/frame, {2 * _mb:.2f} MB/replay row",
                  flush=True)

        _l56_tok_needed = (getattr(config, "latent_obs_finetune", False)
                           or getattr(config, "tok_shift_pad", 0) > 0)

        def _latent_tokens(obs):
            im = torch.as_tensor(pixel_preprocess(obs["pixels"]),
                                 device=device).unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
            return _l56_tokens(_l56_dino, im, _l56_cfg,
                               device)[0].detach().cpu().numpy().astype(np.float16)

        def _latent_next_feat(next_obs):
            """z_{t+8} for the transition about to be stored -- a BARE encode."""
            from .latent56 import encode as _lnf_encode
            im = torch.as_tensor(pixel_preprocess(next_obs["pixels"]),
                                 device=device).unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
            return _lnf_encode(_l56_dino, _l56_enc_net, im, _l56_cfg,
                               device)[0].detach().cpu().numpy().astype(np.float32)
        seg_encode = _mk_lat_enc(_l56_dino, _l56_enc_net, _l56_cfg, device)
        if getattr(config, "latent_obs_finetune", False):
            @torch.no_grad()
            def _tok_encode(frames_uint8):
                im = torch.as_tensor(frames_uint8).permute(0, 1, 4, 2, 3).float().div_(255.)
                return (_l56_tokens(_l56_dino, im, _l56_cfg, device)
                        .cpu().numpy().astype(np.float16))
            tok_encode = _tok_encode
        print(f"[latent56] f = shipped dynamics over z in R^{_l56_cfg['latent']} "
              f"(enc={_l56_cfg['enc']}, ratio={_l56_cfg['ratio']:.3f}); encoder frozen", flush=True)
    if config.dvd_dynamics == "seghead_wm":


        from .seghead import (load_all as _sh_load, SegHistory as _SegHistory,
                              make_seg_encode as _mk_seg_enc, assert_frame_size as _seg_assert)
        _seg_assert()
        _sd, _shh, _sro, _swm, _scfg = _sh_load(device=device, with_wm=True)
        seghead_stack = (_sd, _shh, _swm)
        seg_hist = _SegHistory(_sd, _shh, device)
        seg_encode = _mk_seg_enc(_sd, _shh, device)
        if getattr(config, "seg_obs", False):
            _LATENT_OBS_CTX.update(seg_dino=_sd, seg_head=_shh)
        if getattr(config, "seg_obs_finetune", False):
            _shh.requires_grad_(True).train()
            _hp = [q for q in _shh.parameters() if q.requires_grad]
            critic_optimizer.add_param_group({"params": _hp})
            print(f"[seg-obs-ft] SegHead head trained by the CRITIC loss: "
                  f"{sum(q.numel() for q in _hp)} params (DINOv3 stays frozen)", flush=True)

        def _seg_next_feat(next_obs):
            """The SegHead map at t+8, for the TD target."""
            return np.asarray(seg_encode(
                pixel_preprocess(next_obs["pixels"])[None],
                np.asarray(next_obs["agent_pos"][:9], dtype=np.float32)[None])[0],
                dtype=np.float32)
        from .seghead import SegHeadAdapter as _SHAdapter
        seg_adapter = _SHAdapter(_swm, device=device).to(device).eval()
        seghead_ema = None
        seghead_guard = None
        if config.dvd_finetune_seghead:
            if not config.dvd_finetune_dynamics:
                raise ValueError("--dvd-finetune-seghead needs --dvd-finetune-dynamics: the head "
                                 "trains under f's fit loss, and with f frozen there is no loss "
                                 "to train it with (drop FREEZEDYN).")
            from .seghead import EmaHead as _EmaHead, CollapseGuard as _Guard
            for _p in _shh.parameters():
                _p.requires_grad_(True)
            _shh.train()
            seghead_ema = _EmaHead(_shh, tau=config.dvd_seghead_ema)
            seghead_guard = _Guard()
            print(f"[seghead-ft] SegHead head TRAINABLE "
                  f"({sum(p.numel() for p in _shh.parameters())/1e6:.1f}M) lr={config.dvd_seghead_lr} "
                  f"| target from EMA copy tau={config.dvd_seghead_ema} | collapse guard ON",
                  flush=True)


        _varch = "?"
        try:
            _varch = str(json.loads((Path(config.value_model) / "config.json").read_text())
                         .get("arch", "convv")).split(":")[0].strip()
        except Exception:
            pass
        print(f"[seghead-wm] f=Predictor over {_shh.c}x{_shh.out_size}x{_shh.out_size} map, "
              f"V={_varch} on the same map "
              f"({'head TRAINABLE' if config.dvd_finetune_seghead else 'frozen encoder'})", flush=True)
    expert = None
    if config.method == "dice_rl":
        expert = load_successful_expert_buffer(
            task_cfg["dataset_root"],
            scaler,
            config.action_steps,
            config.gamma,
            config.step_penalty,
            pixel_cache=pixel_demo_cache,
            samples=config.resolved_actor_k(),
            next_samples=config.multi_sample_k,


            dino_encode=None if dino is None else (
                lambda frames: dino.encode_frames(dino_adapter, frames)
                .to(torch.float16).cpu().numpy()),
            seg_encode=seg_encode,


            next_feat=(getattr(config, "latent_obs", False)
                       or getattr(config, "seg_obs", False)),
            tok_encode=tok_encode,


            store_images=bool(config.pixel_obs or dino_adapter is not None
                              or getattr(config, "seg_obs_finetune", False)
                              or getattr(config, "obs_shift_pad", 0) > 0),
            boot_image=config.nstep > 1,

            **({"continuation": True} if config.store_prior_continuation() else {}),
        )
        print(f"Loaded {expert.size} successful-demo chunk transitions", flush=True)

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_name or f"{config.method}_{config.task}",
        dir=str(output_dir),
        config=config.to_dict(),
        save_code=False,
    )
    save_checkpoint(
        output_dir / "checkpoints/000000.pt", config, actor, critics, target_critics, 0, 0
    )
    video_dir = output_dir / "eval_videos"

    def run_eval(step: int) -> None:
        """Evaluate at fixed seeds, log scalars, and push the first episodes as a video."""


        if config.eval_episodes <= 0:
            return
        video_path = video_dir / f"eval_{step:07d}.mp4" if config.eval_video_episodes else None
        values = evaluate(
            config.task,
            prior,
            actor,
            scaler,
            config.residual_scale,
            config.eval_episodes,
            10_000,
            config.step_penalty,
            video_path,
            config.eval_video_episodes,
            critics,
            config.eval_strategy,
            config.eval_samples,
            insertion_cfg,
            config.episode_horizon,
            dvd_term,
            config.bon_source,
            encoder,
            config.holder_sigma,
            config.residual_max_norm,
            config.fixed_goal,
            config.fixed_goal_rot,
            config.full_fixed,
            jitter_goal_m=config.jitter_goal_mm / 1000.0,


            dino_history=None if dino is None else dino.FrameHistory(dino_adapter),
            dino_obs=config.dino_obs,
            latent_obs=getattr(config, "latent_obs", False),
            latent_hist=(_LatHist(_l56_dino, _l56_enc_net, _l56_cfg, device)
                         if getattr(config, "latent_obs", False) else None),
            actor_encoder=actor_encoder,
            bon_reduce=config.bon_reduce,
        )
        append_jsonl(
            metrics_path,
            {"kind": "eval", "env_steps": step, "strategy": config.eval_strategy, **values},
        )
        payload = {f"eval/{key}": value for key, value in values.items() if key != "per_episode"}
        if video_path is not None and video_path.exists():
            payload["eval/video"] = wandb.Video(
                str(video_path),
                caption=f"{config.method} {config.task} @ {step:,} env steps",
                format="mp4",
            )
        wandb.log(payload, step=step)

    if getattr(config, "value_distill_after", 0) > 0 or getattr(config, "value_finetune_after", 0) > 0:
        print(f"[value-freeze] V's distillation fits start at {config.value_distill_after} env steps and its TD "
              f"update (with the Polyak target) at {config.value_finetune_after}; episodes are scored and pooled "
              f"from step 0", flush=True)
    if getattr(config, "update_on_episode_end", False):
        print("[update-on-episode-end] no training during an episode; at each boundary one update per collected "
              f"chunk (x utd={config.utd}) runs in a burst -- same total gradient steps, frozen actor/V/f (and "
              "best-of-N scorer) within an episode", flush=True)
    if getattr(config, "update_every", 1) > 1:
        print(f"[update-every] one update round (utd={config.utd}) every {config.update_every} collected chunks "
              f"= {config.utd / config.update_every:g} gradient steps per chunk", flush=True)
    if getattr(config, "actor_utd", 1) > 1:
        print(f"[actor-utd] the ACTOR takes {config.actor_utd} gradient steps per collected chunk "
              f"(fresh batch each); V, f and the distiller stay at 1 -- the mirror of --value-utd",
              flush=True)
    if getattr(config, "dvd_fit_sampler", "batch") != "batch":


        _fr, _fw, _fk = (config.dvd_fit_batch_rows(), config.dvd_fit_recent_rows(),
                         config.dvd_fit_recent_count())
        print(f"[dvd-fit-sampler] {config.dvd_fit_sampler}: f fine-tunes on its OWN draw of {_fr} ONLINE "
              f"rows per update -- {_fk} uniform over the newest {_fw} rows ({config.dvd_fit_recent_steps} "
              f"env steps / {config.action_steps} per row), {_fr - _fk} uniform over the whole online "
              f"buffer -- once the online buffer holds {config.dvd_fit_min_online_rows()} rows; until "
              f"then the batch sampler's path, verbatim (dvd_fit_own_draw logs 0 -> 1). The actor, V "
              f"and the --actor-utd/--value-utd loops keep the RL batch"
              + ("; --dvd-fit-online-only is redundant here (no expert rows) and has no effect"
                 if config.dvd_fit_online_only else ""),
              flush=True)


    dvd_fit_own_draw_started = False
    if getattr(config, "dvd_fit_probe_every", 0) > 0:
        print(f"[dvd-fit-probe] every {config.dvd_fit_probe_every} updates: f's feature MSE on the newest "
              f"{config.dvd_fit_batch_rows()} online rows (no grad, no step, torch RNG restored) -> "
              f"dvd_feat_mse_recent", flush=True)
    value_fn = None
    frozen_value_fn = None


    value_td = None
    value_td_optimizer = None
    value_dual = None


    attn_pool = None


    obs_pool = None


    obs_pool_target = None


    actor_pool = None


    pool_obs_grad = False


    sargmax_pool = None
    value_optimizer = None
    value_stats = RunningStats()
    if config.value_model:
        if (config.value_intervention == "none" and config.dvd_weight <= 0.0
                and config.mbpo_weight <= 0.0 and not getattr(config, "rosmoq", False)
                and getattr(config, "encoder_value_aux", 0.0) <= 0.0
                and config.bon_source != "dvd"
                and not getattr(config, "latent_obs", False)
                and not getattr(config, "seg_obs", False)
                and not config.dino_obs):


            raise ValueError("--value-model given with --value-intervention none; pick one")


        if config.dvd_dynamics == "latent56":


            from .latent56 import Latent56Value
            _vraw = ValueEnsemble(config.value_model, config.value_variant, scaler,
                                  state_dim=state_dim, device=device,
                                  finetune=config.value_finetune)
            value_fn = Latent56Value(_vraw, seg_adapter)
            print(f"[latent56] V = {Path(config.value_model).name} on [z(56), proprio(9)]; "
                  f"input_dim={_vraw.input_dim}", flush=True)
        _seg_v = config.dvd_dynamics == "seghead_wm"
        if _seg_v:


            from .seghead import load_map_value
            value_fn = load_map_value(config.value_model, device=device,
                                      finetune=config.value_finetune)
            print(f"[seghead-wm] V = ConvV on the map from {Path(config.value_model).name} "
                  f"({'FROZEN' if not config.value_finetune else 'finetuned'})", flush=True)


        _keep_v = _seg_v or config.dvd_dynamics == "latent56"
        value_fn = value_fn if _keep_v else ValueEnsemble(
            config.value_model,
            config.value_variant,
            scaler,
            state_dim=state_dim,
            device=device,
            finetune=config.value_finetune,
            scratch=config.value_scratch or getattr(config, "value_random_frozen", False),
            aggregate=config.value_aggregate,
            lcb_lambda=config.value_lcb_lambda,
        )
        if (not _seg_v and config.value_scratch and not config.value_finetune
                and config.value_mc_weight <= 0.0 and not config.value_distill_robometer):


            raise ValueError("--value-scratch needs --value-finetune: a randomly initialised V "
                             "that is never trained is pure noise in the DVD objective")


        if getattr(config, "value_random_frozen", False):
            if (config.value_finetune or config.value_scratch
                    or getattr(config, "value_distill_robometer", "") or config.value_init):
                raise ValueError("--value-random-frozen is a frozen control: it cannot be combined "
                                 "with --value-finetune, --value-scratch, --value-init or "
                                 "--value-distill-robometer (each would train or replace V)")
            print("[value] RANDOM FROZEN V: offline weights discarded, V is a fixed random "
                  "function (control for value-function quality)", flush=True)
        if config.value_init and not _seg_v:


            _vi = torch.load(config.value_init, map_location=device, weights_only=False)
            _vsd = _vi.get("extra", {}).get("value_fn")
            assert _vsd is not None, (
                f"--value-init {config.value_init} has no extra.value_fn -- it was written by a "
                f"run with a FROZEN V (nothing to save). Point at a --value-finetune run.")
            value_fn.load_state_dict(_vsd)
            print(f"[value-init] loaded V from {config.value_init} "
                  f"(@{_vi.get('env_steps')} steps)", flush=True)
            del _vi, _vsd
        if config.value_finetune and config.value_target_tau > 0.0:


            if getattr(config, "value_nets", 0) > 0 and hasattr(value_fn, "nets"):


                _k = min(int(config.value_nets), len(value_fn.nets))
                if _k < len(value_fn.nets):
                    import torch.nn as _vnn
                    _was = len(value_fn.nets)
                    value_fn.nets = _vnn.ModuleList(list(value_fn.nets)[:_k])
                    print(f"[value-nets] V ensemble cut from {_was} seeds to {_k}", flush=True)
            value_fn.enable_target(device)
            print(f"[value-target] enabled, Polyak tau={config.value_target_tau}", flush=True)
        if getattr(config, "value_loss", "smooth_l1") != "smooth_l1":
            for _vm in (value_fn, value_td):
                if _vm is not None:
                    _vm.value_loss = config.value_loss
            print(f"[value-loss] V TD regression loss = {config.value_loss}", flush=True)


        if getattr(config, "bon_frozen_v", "off") != "off":
            if not isinstance(value_fn, ValueEnsemble):
                raise ValueError("--bon-frozen-v needs a plain ValueEnsemble V (not latent56 / seghead / DualValue)")

            frozen_value_fn = copy.deepcopy(value_fn).requires_grad_(False).train(value_fn.training)
            print("[bon-frozen-v] frozen copy of the initial V kept for best-of-N "
                  f"(rank-normalised, combine={config.bon_frozen_v})", flush=True)


        if config.dino_pool == "attnpool" and not _seg_v:
            from online_rl_mimicgen.dino_wm import AttnPool
            _ap = Path(config.value_model) / "attn_pool.pt"
            if not _ap.exists():
                raise FileNotFoundError(
                    f"--dino-pool attnpool needs {_ap}: the readout saved beside the V it was fitted "
                    "with. This repository ships no fitter for it (value/fit_v_mqattn.py fits "
                    "--dino-pool mqattn); a V whose pooling is fused into the value head will NOT load")
            _blob = torch.load(_ap, map_location=device, weights_only=False)
            attn_pool = AttnPool(dim=_blob.get("dim", 384)).to(device)
            attn_pool.load_state_dict(_blob["state_dict"])
            if getattr(config, "attn_pool_freeze", False):
                attn_pool.requires_grad_(False)
            print(f"[attnpool] query loaded from {_ap}; "
                  f"{'FROZEN (--attn-pool-freeze)' if getattr(config, 'attn_pool_freeze', False) else 'TRAINABLE' if config.value_finetune else 'frozen'}",
                  flush=True)


        if config.dino_pool == "xattn" and not _seg_v:
            from online_rl_mimicgen.dino_wm import CrossAttnPool
            _xp = Path(config.value_model) / "xattn_pool.pt"
            if not _xp.exists():
                raise FileNotFoundError(
                    f"--dino-pool xattn needs {_xp}: the readout saved beside the V it was fitted "
                    "with. This repository ships no fitter for it (value/fit_v_mqattn.py fits "
                    "--dino-pool mqattn). A V directory fitted for mean or attnpool will not do: its "
                    "value head was trained on a different pooled vector.")
            _blob = torch.load(_xp, map_location=device, weights_only=False)


            attn_pool = CrossAttnPool.from_blob(_blob, device=device)
            if (config.xattn_queries, config.xattn_heads) != (attn_pool.k, attn_pool.heads):
                raise ValueError(
                    f"{_xp} holds a K={attn_pool.k} heads={attn_pool.heads} cross-attention pool, "
                    f"but this arm passed --xattn-queries {config.xattn_queries} --xattn-heads "
                    f"{config.xattn_heads}. The flags name the module the arm THINKS it is running "
                    f"and appear in config.json; a mismatch would mislabel the whole cell.")
            if getattr(config, "attn_pool_freeze", False):
                attn_pool.requires_grad_(False)


            pool_obs_grad = bool(
                config.dino_obs
                and not getattr(config, "attn_pool_freeze", False)
                and not getattr(config, "dino_obs_pool_separate", False)
                and config.method == "dice_rl"
                and config.train_critic
                and not getattr(config, "freeze_init_critic", False))
            if pool_obs_grad:
                attn_pool.train()
                critic_optimizer.add_param_group({"params": list(attn_pool.parameters())})
                print(f"[xattn] VARIANT 1 (one shared pool): the critic's TD loss trains "
                      f"dvd_term._pool -- {sum(q.numel() for q in attn_pool.parameters())} params "
                      f"added to the CRITIC optimiser (lr={config.critic_lr})"
                      + (f" AND kept in the value optimiser (lr={config.value_finetune_lr})"
                         if config.value_finetune else
                         " (--value-finetune is OFF, so the critic is its only trainer)"),
                      flush=True)
            elif (config.dino_obs and not getattr(config, "attn_pool_freeze", False)
                  and not getattr(config, "dino_obs_pool_separate", False)):
                print("[xattn] VARIANT 1: the critic CANNOT train the shared pool on this arm "
                      f"(method={config.method!r} train_critic={config.train_critic} "
                      f"freeze_init_critic={getattr(config, 'freeze_init_critic', False)}); it is "
                      f"trained by V alone"
                      + ("" if config.value_finetune else " -- and --value-finetune is OFF, so by "
                         "NOTHING, which is variant 3 without the label"), flush=True)
            if getattr(config, "dino_obs_pool_separate", False):


                obs_pool = copy.deepcopy(attn_pool).requires_grad_(True).to(device)
                obs_pool.train()


                critic_optimizer.add_param_group({"params": list(obs_pool.parameters())})
                print(f"[xattn] SEPARATE actor/critic pool (--dino-obs-pool-separate): copy of the "
                      f"offline fit, {sum(q.numel() for q in obs_pool.parameters())} params in the "
                      f"CRITIC optimiser (lr={config.critic_lr}); V keeps its own",
                      flush=True)
            _frozen = getattr(config, "attn_pool_freeze", False)
            print(f"[xattn] cross-attention pool loaded from {_xp}: K={attn_pool.k} "
                  f"heads={attn_pool.heads} d={attn_pool.dim} -> {attn_pool.out_dim} "
                  f"({attn_pool.per_query}/query); V-side "
                  f"{'FROZEN (--attn-pool-freeze)' if _frozen else 'TRAINABLE' if config.value_finetune else 'frozen'}; "
                  f"{sum(q.numel() for q in attn_pool.parameters())} params", flush=True)


        if config.dino_pool == "mqattn" and not _seg_v:
            from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool
            _mp = Path(config.value_model) / "mqattn_pool.pt"
            if not _mp.exists():
                raise FileNotFoundError(
                    f"--dino-pool mqattn needs {_mp}; fit it with "
                    "value/fit_v_mqattn.py (it writes the module beside the V "
                    "it was fitted with, exactly as --pool attnpool writes attn_pool.pt and "
                    "--pool xattn writes xattn_pool.pt). A V directory fitted for mean, attnpool "
                    "or xattn will NOT do: its value head was trained on a different pooled "
                    "vector, and the per-view K=4 fit is the same 777 width as the mean one.")
            _blob = torch.load(_mp, map_location=device, weights_only=False)


            attn_pool = MultiQueryAttnPool.from_blob(_blob, device=device)
            _got = dict(queries=attn_pool.k, per_view=attn_pool.per_view,
                        out_dim=(attn_pool.out_dim if attn_pool.projected else 0))
            _bad = {k: v for k, v in _mqattn_flag_spec(config).items() if _got[k] != v}
            if _bad:
                raise ValueError(
                    f"{_mp} holds queries={_got['queries']} "
                    f"scope={_mqattn_val_text('per_view', _got['per_view'])} "
                    f"out_dim={_got['out_dim'] or 'none (raw concatenation)'}, but this arm "
                    "asserted "
                    + ", ".join(f"{_mqattn_flag_text(k, want)} "
                                f"(checkpoint {_mqattn_val_text(k, _got[k])})"
                                for k, want in sorted(_bad.items()))
                    + ". The --mqattn-* flags name the module the arm THINKS it is running and "
                      "appear in config.json; a mismatch would mislabel the whole cell.")
            from .dino_wm import pooled_dim as _mq_pooled_dim
            if int(attn_pool.pooled_dim) != int(_mq_pooled_dim("mqattn")):


                raise ValueError(
                    f"{_mp} pools to {attn_pool.pooled_dim} dims but dino_wm.pooled_dim('mqattn') "
                    f"is {_mq_pooled_dim('mqattn')}, which is what sizes V's first Linear and the "
                    "--dino-obs critic. Refit the readout at the mean-pool width, or move "
                    "MQATTN_OUT_DIM on both sides.")
            if getattr(config, "attn_pool_scratch", False):


                _fit_q = attn_pool.q.detach().float().flatten(0, -2)
                attn_pool = MultiQueryAttnPool(
                    **MultiQueryAttnPool.spec_from_state_dict(attn_pool.state_dict())).to(device)
                print(f"[mqattn] V-side pool re-initialised (--attn-pool-scratch): query cosine to the "
                      f"fitted pool {float(torch.nn.functional.cosine_similarity(attn_pool.q.detach().float().flatten(0, -2), _fit_q, dim=-1).mean()):.3f}",
                      flush=True)
            if getattr(config, "attn_pool_freeze", False):
                attn_pool.requires_grad_(False)
            _frozen = getattr(config, "attn_pool_freeze", False)


            print(f"[mqattn] multi-query attention pool loaded from {_mp}: "
                  f"queries={attn_pool.k} per_view={attn_pool.per_view} "
                  f"out_dim={attn_pool.out_dim if attn_pool.projected else 0} "
                  f"({attn_pool.k} raw queries "
                  f"{'PER CAMERA, each a softmax over that camera 196 patches' if attn_pool.per_view else 'joint over both views, softmax over all 392 patches'}, "
                  f"d={attn_pool.dim}, concat={attn_pool.concat_dim} -> "
                  f"{attn_pool.out_dim if attn_pool.projected else 'raw, no projection'}); V-side "
                  f"{'FROZEN (--attn-pool-freeze)' if _frozen else 'TRAINABLE' if config.value_finetune else 'frozen'}; "
                  f"{sum(_q.numel() for _q in attn_pool.parameters())} params", flush=True)
            if getattr(config, "dino_obs_pool_separate", False):


                if getattr(config, "dino_obs_pool_scratch", False):
                    obs_pool = MultiQueryAttnPool(
                        **MultiQueryAttnPool.spec_from_state_dict(attn_pool.state_dict())).to(device)
                    _src = "FRESH initialisation (--dino-obs-pool-scratch), same architecture as the offline fit"
                else:
                    obs_pool = copy.deepcopy(attn_pool)
                    _src = "copy of the offline fit"
                obs_pool.requires_grad_(True).train()
                critic_optimizer.add_param_group({"params": list(obs_pool.parameters())})
                with torch.no_grad():
                    _qcos = float(torch.nn.functional.cosine_similarity(
                        obs_pool.q.detach().float().flatten(0, -2),
                        attn_pool.q.detach().float().flatten(0, -2), dim=-1).mean())
                print(f"[mqattn] SEPARATE actor/critic pool (--dino-obs-pool-separate): {_src}; "
                      f"query cosine to the fitted pool {_qcos:.3f}; "
                      f"{sum(_q.numel() for _q in obs_pool.parameters())} params in the CRITIC "
                      f"optimiser (lr={config.critic_lr}); V keeps its own", flush=True)

        if obs_pool is not None and getattr(config, "dino_obs_pool_ema_target", False):


            obs_pool_target = copy.deepcopy(obs_pool).requires_grad_(False)
            print(f"[dino-obs] TD target pooled by a Polyak copy of the actor/critic pool "
                  f"(--dino-obs-pool-ema-target, tau={config.tau})", flush=True)

        if getattr(config, "dvd_pool_actor_step", False):


            if not all(q.requires_grad for q in attn_pool.parameters()):
                raise RuntimeError("--dvd-pool-actor-step needs a TRAINABLE pool, but attn_pool is "
                                   "frozen: the actor step would move nothing")
            actor_optimizer.add_param_group({"params": list(attn_pool.parameters())})
            print(f"[pool-mix] --dvd-pool-actor-step: SHARED pool trained by L_DVD (actor_optimizer, "
                  f"lr={config.actor_lr}) AND V's losses (value_optimizer, "
                  f"lr={config.value_finetune_lr}); {sum(q.numel() for q in attn_pool.parameters())} "
                  f"params in BOTH optimisers", flush=True)
        if getattr(config, "dvd_actor_pool", False):


            if attn_pool is None:
                raise ValueError(
                    "--dvd-actor-pool copies the FITTED pooling module, but no module was loaded "
                    f"(--dino-pool {config.dino_pool!r}, --value-model {config.value_model!r}). It "
                    "needs --dino-pool attnpool, xattn or mqattn with the matching fitted pool "
                    "beside V; with mean pooling there is no parameter for the actor to own.")


            actor_pool = copy.deepcopy(attn_pool).requires_grad_(True).to(device)
            actor_pool.train()


            if not all(q.requires_grad for q in actor_pool.parameters()):
                raise RuntimeError(
                    "--dvd-actor-pool built a FROZEN copy: --attn-pool-freeze must reach V's own "
                    "instance ONLY. A frozen copy receives no L_DVD gradient and the cell would be "
                    "byte-identical to the shared-frozen arm under a different name.")


            actor_optimizer.add_param_group({"params": list(actor_pool.parameters())})
            _apn = sum(q.numel() for q in actor_pool.parameters())
            _vfrozen = getattr(config, "attn_pool_freeze", False)


            if _vfrozen:
                _vowner = "NOBODY (--attn-pool-freeze: frozen at the offline fit)"
            elif config.value_finetune or config.value_mc_weight > 0.0:
                _vowner = (f"value_optimizer (lr={config.value_finetune_lr}), moved only by V's "
                           + " + ".join(_nm for _nm, _on in (
                               ("TD fit_step", config.value_finetune),
                               ("MC regression", config.value_mc_weight > 0.0)) if _on)
                           + ("; the RoboMeter distiller never reaches it (its rows are cached "
                              "detached)" if getattr(config, "value_distill_robometer", "") else ""))
            else:
                _vowner = "NOBODY (--value-finetune is OFF)"
            _obs_owner = getattr(config, "dvd_actor_pool_obs", "actor")
            print(f"[actor-pool] --dvd-actor-pool: TWO modules, both from "
                  f"{Path(config.value_model).name}, deepcopy-identical at birth", flush=True)
            print(f"[actor-pool]   ACTOR copy -> dvd_term._pool       : {_apn} params, "
                  f"trainable=True, owner=actor_optimizer (lr={config.actor_lr}), trained by "
                  f"L_DVD through DvdTerm.value_input", flush=True)
            print(f"[actor-pool]   V instance -> dvd_term._value_pool : "
                  f"{sum(q.numel() for q in attn_pool.parameters())} params, "
                  f"trainable={any(q.requires_grad for q in attn_pool.parameters())}, "
                  f"owner={_vowner}", flush=True)
            print("[actor-pool]   V READS dvd_term._value_pool: online TD fit rows, the RoboMeter "
                  "distiller's cached rows and the --value-distill-pred rows", flush=True)
            print(f"[actor-pool]   OBSERVATION (--dino-obs={config.dino_obs}) pooled by "
                  f"{'dvd_term._pool (the ACTOR copy)' if _obs_owner == 'actor' else 'dvd_term._value_pool (V instance)'} "
                  f"[--dvd-actor-pool-obs {_obs_owner}]", flush=True)
            print(f"[actor-pool]   grad diagnostic every "
                  f"{getattr(config, 'dvd_actor_pool_log_every', 0)} updates -> "
                  f"pool_grad_actor_dvd (L_DVD on the copy, must be > 0) / pool_grad_value_dvd "
                  f"(L_DVD on V's instance, must be 0.0) / pool_grad_value_fit (V's TD fit on V's "
                  f"instance, same update: > 0 iff V's instance trains) / pool_grad_actor_fit (V's "
                  f"TD fit on the copy, must be 0.0) / pool_drift_actor_vs_value; one "
                  f"'[actor-pool] grad-diag' line each", flush=True)

        if config.value_pool == "sargmax" and not _seg_v:


            from online_rl_mimicgen.dino_wm import SoftArgmaxPool
            if attn_pool is not None:
                raise ValueError("--value-pool sargmax and --dino-pool attnpool both replace "
                                 "dvd_term._pool; pick one")
            _sp = Path(config.value_model) / "soft_argmax.pt"
            if not _sp.exists():
                raise FileNotFoundError(
                    f"--value-pool sargmax needs {_sp} (the fitted keypoint readout beside V); "
                    "the soft-argmax readout is not part of this release")
            _sblob = torch.load(_sp, map_location=device, weights_only=False)
            sargmax_pool = SoftArgmaxPool(dim=_sblob.get("dim", 384), k=_sblob.get("k", 16),
                                          hid=_sblob.get("hid", 128)).to(device)
            sargmax_pool.load_state_dict(_sblob["state_dict"])


            sargmax_pool.requires_grad_(False)
            sargmax_pool.eval()
            print(f"[sargmax] keypoint readout loaded from {_sp} "
                  f"(k={sargmax_pool.k}, out={_sblob.get('pooled_dim')}); FROZEN "
                  f"(the distiller caches encoded rows, a moving readout would stale them)",
                  flush=True)

        if config.value_finetune or config.value_mc_weight > 0.0:
            if not config.value_finetune and config.value_mc_weight > 0.0:


                _nmc = sum(q.numel() for q in value_fn.parameters() if not q.requires_grad)
                value_fn.requires_grad_(True)
                print(f"[value-mc] unfroze {_nmc} V params for the MC regression "
                      f"(--value-finetune is off)", flush=True)
            if config.value_intervention == "shaping":


                raise ValueError(
                    "--value-finetune with --value-intervention shaping voids the "
                    "policy-invariance guarantee (Ng et al. 1999 requires a fixed potential)"
                )
            if config.value_residual:
                value_fn.enable_residual(device, freeze_base=not config.value_residual_unfrozen,
                                         condition_on_base=config.value_residual_cond)
                params = value_fn.residual_parameters()
                print(f"[value-residual] V = V_base + dV; base "
                      f"{'FROZEN' if not config.value_residual_unfrozen else 'TRAINABLE (capacity control)'}; "
                      f"optimising {sum(p.numel() for p in params)} params", flush=True)
            else:
                params = list(value_fn.parameters())
            if attn_pool is not None and not getattr(config, "attn_pool_freeze", False):


                params = params + list(attn_pool.parameters())
            value_optimizer = torch.optim.Adam(params, lr=config.value_finetune_lr)
    rm_distiller = None
    if getattr(config, "value_distill_robometer", ""):
        from online_rl_mimicgen.robometer_distill import RoboMeterDistiller
        if value_fn is None:
            raise ValueError("--value-distill-robometer needs a --value-model to distil INTO")


        _v_dist = value_fn.value_fn if config.dvd_dynamics == "latent56" else value_fn
        _vopt = value_optimizer
        if _vopt is None:


            _n_unfrozen = sum(p.numel() for p in _v_dist.parameters() if not p.requires_grad)
            _v_dist.requires_grad_(True)
            if _n_unfrozen:
                print(f"[rm-distill] unfroze {_n_unfrozen} V params for distillation "
                      f"(--value-finetune is off, so V arrived frozen)", flush=True)
            _vopt = torch.optim.Adam(list(_v_dist.parameters()), lr=config.value_finetune_lr)
            value_optimizer = _vopt
        _mix = None
        if getattr(config, "value_distill_mix", "") == "natural":
            _mix = "natural"
        elif getattr(config, "value_distill_mix", ""):
            _mix = {}
            for _kv in config.value_distill_mix.split(","):
                _k, _, _v = _kv.partition("=")
                if not _v:
                    raise ValueError(f"--value-distill-mix expects pool=ratio pairs, got {_kv!r}")
                _mix[_k.strip()] = float(_v)
            if sum(_mix.values()) <= 0:
                raise ValueError("--value-distill-mix ratios must sum to something positive")
        rm_distiller = RoboMeterDistiller(
            _v_dist, _vopt, spool=config.value_distill_robometer, device=device,
            frames_per_clip=config.value_distill_frames, every=config.value_distill_every,
            batch_size=config.value_distill_batch, capacity=config.value_distill_capacity,
            loss_weight=config.value_distill_weight, mix=_mix,
            target=config.value_distill_target,
            aug_views=int(getattr(config, "value_distill_aug_views", 0)),
            min_pool=getattr(config, "value_distill_min_pool", "0"),
            **({} if encoder_optimizer is None else dict(
                deferred_encode=_learned_deferred_encode(encoder, device, config.pixel_augment),
                extra_optimizer=encoder_optimizer,
                after_step=(lambda: soft_update(encoder, target_encoder, config.tau)))))
        if rm_distiller.aug_views > 0 and not (getattr(config, "value_distill_shift_pad", 0) > 0
                                               or getattr(config, "value_distill_photometric", 0.0) > 0):

            raise SystemExit("--value-distill-aug-views > 0 needs --value-distill-shift-pad > 0 "
                             "and/or --value-distill-photometric > 0")
        rm_distiller.value_loss = getattr(config, "value_loss", "smooth_l1")
        print(f"[rm-distill] ON spool={config.value_distill_robometer} "
              f"every={config.value_distill_every} w={config.value_distill_weight} "
              f"mix={rm_distiller.mix} target={rm_distiller.target} loss={rm_distiller.value_loss} "
              f"min_pool={rm_distiller.min_pool} need={rm_distiller._gate_need() if rm_distiller.min_pool else {}}",
              flush=True)
        print(
            f"value model: {config.value_model} variant={config.value_variant} "
            f"intervention={config.value_intervention} beta={config.value_penalty_beta} "
            f"finetune={config.value_finetune} scratch={config.value_scratch} "
            f"seeds={len(getattr(value_fn, 'nets', [1]))}"
        )

    if getattr(config, "value_td_model", ""):


        if value_fn is None or not isinstance(value_fn, ValueEnsemble):
            raise ValueError("--value-td-model needs a --value-model ValueEnsemble as the primary "
                             f"(got {type(value_fn).__name__})")
        if config.dvd_dynamics != "dino_wm":
            raise ValueError("--value-td-model is implemented for --dvd-dynamics dino_wm only")
        if config.nstep > 1:
            raise ValueError("--value-td-model with --nstep > 1: the dino_wm V fit batch encodes "
                             "the 1-step t+8 frame against an n-step target")


        with torch.random.fork_rng(devices=[]):
            value_td = ValueEnsemble(
                config.value_td_model,
                config.value_variant,
                scaler,
                state_dim=state_dim,
                device=device,
                finetune=True,
                aggregate=config.value_aggregate,
                lcb_lambda=config.value_lcb_lambda,
            )
        if int(value_td.input_dim) != int(value_fn.input_dim):
            raise ValueError(
                f"--value-td-model input_dim={value_td.input_dim} != the primary's "
                f"{value_fn.input_dim}: both read the same vector")
        if config.value_target_tau > 0.0:

            value_td.enable_target(device)
        value_td_optimizer = torch.optim.Adam(list(value_td.parameters()),
                                              lr=config.value_finetune_lr)
        value_dual = DualValue(value_fn, value_td, combine=config.value_dual_combine)
        print(f"[value-td] second V from {config.value_td_model} "
              f"(variant={config.value_variant} aggregate={config.value_aggregate} "
              f"seeds={len(value_td.nets)} input_dim={value_td.input_dim}); TD-finetuned "
              f"lr={config.value_finetune_lr} after={config.value_finetune_after} "
              f"target_tau={config.value_target_tau}; policy reads DualValue "
              f"combine={config.value_dual_combine} (EMA scales updated once per update from the "
              f"actor's >= {DualValue.MIN_ROWS}-row DVD read, read-only everywhere else; rescaled "
              f"to the primary's level); primary TD fit "
              f"{'ON' if value_fn.finetune else 'OFF'}", flush=True)

    dvd_term = None


    if (config.dvd_weight > 0.0 or config.bon_source == "dvd" or config.mbpo_weight > 0.0
            or getattr(config, "rosmoq", False) or config.dino_obs):


        from .dvd_term import DvdTerm
        if value_fn is None:
            raise ValueError(
                f"needs a value model: pass --value-model as well (dvd_weight="
                f"{config.dvd_weight}, bon_source={config.bon_source!r}, "
                f"mbpo_weight={config.mbpo_weight})")
        if not config.train_critic:
            needs = []
            if config.dvd_weight < 1.0:
                needs.append(f"--dvd-weight {config.dvd_weight} (< 1 keeps dQ/da in the actor loss)")
            if config.bc_filter and config.bc_filter_source == "q":
                needs.append("--bc-filter with --bc-filter-source q")
            if config.bon_source == "q" and config.eval_strategy == "max_q_min":
                needs.append("--eval-strategy max_q_min with --bon-source q")
            if config.bon_source == "q" and config.method == "dice_rl":
                needs.append("Q-scored best-of-N exploration (pass --bon-source dvd)")
            if config.value_intervention != "none":
                needs.append(f"--value-intervention {config.value_intervention}")
            if needs:
                raise ValueError("--no-train-critic but the critic is still read by: "
                                 + "; ".join(needs))
            bon = ("ranked by gamma^H V(f(s,a))" if config.bon_source == "dvd"
                   else "DISABLED -> plain sampling")
            print(f"[dvd] critic ensemble NOT trained; best-of-16 exploration {bon}; "
                  "q_advantage/q_vs_mc_mae replaced by dvd_advantage", flush=True)
        if config.dvd_mode == "replay":
            if config.deterministic_residual:
                raise ValueError(
                    "--dvd-mode replay infers eta_k from the policy's noise, which is degenerate "
                    "with a deterministic residual. Pass --stochastic-residual."
                )
            print("[dvd] mode=replay (Heess Alg. 2): inferred eta_k / xi_k + importance ratio",
                  flush=True)


        proprio_dim = (0 if dino is None else
                       dino.resolve_proprio_dim(value_fn, config.value_pool or config.dino_pool))
        if config.dvd_dynamics == "latent56":


            from .latent56 import Latent56Value as _L56V
            assert isinstance(value_fn, _L56V), (
                f"latent56 needs the Latent56Value wrapper (f emits 56-d, V takes 65-d); "
                f"value_fn is {type(value_fn).__name__}")

        dvd_term = DvdTerm(


            value_dual if value_dual is not None else value_fn, scaler, chunk_dim, device,
            discount=config.gamma, horizon=config.action_steps,
            chunks=config.dvd_chunks,
            model_tag=config.dvd_dynamics,
            finetune_dynamics=config.dvd_finetune_dynamics,
            value_aware_weight=config.dvd_value_aware_weight,
            identity_scaling=bool(getattr(config, 'dvd_on_encoder', False))
                             or config.dvd_dynamics == "latent56",
            dynamics_lr=config.dvd_dynamics_lr,
            dynamics_scratch=config.dvd_dynamics_scratch,


            dynamics=seg_adapter if seg_adapter is not None else dino_adapter,


            feature_space=((dino is not None) or seg_adapter is not None)
                          and config.dvd_dynamics != "latent56",
            pool_mode="flat" if seg_adapter is not None else config.dino_pool,
            value_pool_mode=("flat" if seg_adapter is not None
                             else (config.value_pool or None)),
            proprio_dim=9 if seg_adapter is not None else proprio_dim,
            static_grad_block=config.dvd_static_grad,


            goal_slice=(GOAL_RELATIVE_MAPS[config.task]["goal_pos"]
                        if config.task in GOAL_RELATIVE_MAPS else None),
            fixed_scale=config.dvd_fixed_scale,
            proprio_source=config.dvd_proprio_source,
            state_loss_weight=config.dvd_state_loss_weight,
        )
        _swm = getattr(dvd_term.dynamics, "wm", None)
        if getattr(_swm, "has_state_head", False) and bool((_swm.state_delta_scale == 1.0).all()):


            if expert is None or expert.size == 0:
                raise SystemExit("--dino-wm-state-head with a fresh head needs the demo buffer to set "
                                 "state_delta_scale (method dice_rl)")
            _n = expert.size
            _lo = np.asarray(scaler.low, np.float32)[:9]
            _sp = np.asarray(scaler.scale, np.float32)[:9]
            _nx_key = "next_state_1step" if "next_state_1step" in expert.arrays else "next_state"
            _p0 = (expert.arrays["state"][:_n, :9] + 1.0) * 0.5 * _sp + _lo
            _p1 = (expert.arrays[_nx_key][:_n, :9] + 1.0) * 0.5 * _sp + _lo
            _dsc = np.maximum((_p1 - _p0).std(0), 1e-4).astype(np.float32)
            _swm.state_delta_scale.copy_(torch.as_tensor(_dsc, device=_swm.state_delta_scale.device))
            print(f"[dino-wm] proprio head state_delta_scale from {_n} demo transitions ({_nx_key}): "
                  f"{[round(float(x), 5) for x in _dsc]}", flush=True)
        if config.value_distill_pred:
            if rm_distiller is None or dino_adapter is None or seg_adapter is not None:
                raise SystemExit("--value-distill-pred needs --value-distill-robometer on the DINO-WM feature path")
            if config.dvd_proprio_source != "predicted":
                raise SystemExit("--value-distill-pred needs --dvd-proprio-source predicted (V's predicted "
                                 "input carries the proprio head's prediction)")
            if config.value_distill_aug_views:
                raise SystemExit("--value-distill-pred with --value-distill-aug-views is not implemented")
            print("[rm-distill] PREDICTED inputs ON: each scored frame also adds [pool(f) | f's proprio] "
                  "with the same teacher target", flush=True)
        if getattr(config, "dvd_finetune_last_k", 0) > 0 and dvd_term is not None and dvd_term.optimiser is not None:


            _wm_inner = getattr(dvd_term.dynamics, "wm", dvd_term.dynamics)
            _layers = list(getattr(getattr(_wm_inner, "predictor", None), "layers", []))
            _k = min(int(config.dvd_finetune_last_k), len(_layers))
            if not _layers:
                raise SystemExit("--dvd-finetune-last-k needs a transformer predictor with .layers")
            for _i, _layer in enumerate(_layers):
                _layer.requires_grad_(_i >= len(_layers) - _k)
            _train = [p for p in dvd_term.dynamics.parameters() if p.requires_grad]
            dvd_term.optimiser = torch.optim.Adam(_train, lr=config.dvd_dynamics_lr)
            _n = sum(p.numel() for p in _train)
            print(f"[f-partial] online f fine-tune restricted to the last {_k} of {len(_layers)} predictor layers "
                  f"(+ state head and action encoder): {_n/1e6:.2f}M trainable params; the actor's gradient still "
                  f"traverses all {len(_layers)} layers", flush=True)
        if config.dvd_finetune_seghead and dvd_term.optimiser is not None:


            dvd_term.optimiser.add_param_group(
                {"params": [p for p in seghead_stack[1].parameters() if p.requires_grad],
                 "lr": config.dvd_seghead_lr})
            print(f"[seghead-ft] head params added to f's optimiser as a 2nd group "
                  f"(lr={config.dvd_seghead_lr}); predictor stays at {config.dvd_dynamics_lr}",
                  flush=True)


        _f_module = getattr(dvd_term.dynamics, "wm", dvd_term.dynamics)
        _f_is_flow = bool(getattr(_f_module, "flow", False)) or hasattr(_f_module, "t_mlp")
        if (dvd_term.feature_space and (dvd_term.generative or _f_is_flow)
                and not dvd_term.anchored):
            raise SystemExit(
                f"[dvd] {type(dvd_term.dynamics).__name__} is a feature-space STOCHASTIC f "
                f"(generative={dvd_term.generative} flow={_f_is_flow}) but exposes no "
                "sample_anchored(s, a_theta, a_data, z_next), so DvdTerm.anchored is False and "
                "every rollout would integrate from FRESH NOISE -- a silently wrong actor "
                "gradient behind healthy-looking metrics. Give the adapter sample_anchored "
                "(infer xi under no_grad, subtract the DETACHED offset z_next - f(s, a_data, xi), "
                "then re-sample at a_theta with the SAME xi through a GRAD-CARRYING integrator), "
                "or run the deterministic MSE head instead.")
        if attn_pool is not None:


            dvd_term._pool = attn_pool
        if actor_pool is not None:


            dvd_term._pool = actor_pool
            dvd_term._value_pool = attn_pool

            dvd_term._actor_pool_obs = getattr(config, "dvd_actor_pool_obs", "actor")
        if pool_obs_grad:


            dvd_term._pool_obs_grad = True
        if obs_pool_target is not None:


            dvd_term._obs_pool_target = obs_pool_target
        if obs_pool is not None:


            dvd_term._obs_pool = obs_pool
        if sargmax_pool is not None:


            dvd_term._pool = sargmax_pool
        if getattr(config, "dvd_step2_action", "prior") == "actor":


            if not getattr(dvd_term, "two_step_features", False):
                raise ValueError("--dvd-step2-action actor needs the feature-space DVD(k=2) path "
                                 "(DvdTerm.two_step_features is False)")
            dvd_term.step2_policy = make_step2_policy(dvd_term, actor, config.residual_scale,
                                                      config.residual_max_norm)
            print(f"[dvd] --dvd-step2-action actor: chunk 2 = continuation + {config.residual_scale} * "
                  f"actor(obs_hat_t+8, continuation) (deterministic residual, max_norm="
                  f"{config.residual_max_norm}); obs_hat = [pool(zhat_t+8) || predicted proprio], pool "
                  f"parameters detached on that path", flush=True)

        if config.dvd_dynamics_scratch and not config.dvd_finetune_dynamics:
            raise ValueError("--dvd-dynamics-scratch needs --dvd-finetune-dynamics: a randomly "
                             "initialised f that is never trained makes the DVD gradient noise")
        if config.dvd_dynamics_init:


            _di = torch.load(config.dvd_dynamics_init, map_location=device, weights_only=False)
            from .legacy import refuse_unconverted_checkpoint
            refuse_unconverted_checkpoint(_di, config.dvd_dynamics_init, "--dvd-dynamics-init")
            _dsd = _di.get("extra", {}).get("dvd_dynamics")
            assert _dsd is not None, (
                f"--dvd-dynamics-init {config.dvd_dynamics_init} has no extra.dvd_dynamics -- that "
                f"key is only written when --dvd-finetune-dynamics is on. Pick a finetuned run.")
            _m = dvd_term.dynamics.load_state_dict(_dsd, strict=False)
            _bad = [k for k in _m.missing_keys if not k.startswith("wm.dino.")]
            assert not _bad and not _m.unexpected_keys, (
                f"dvd_dynamics_init mismatch: missing={_bad} unexpected={_m.unexpected_keys}")
            print(f"[dynamics-init] loaded f from {config.dvd_dynamics_init} "
                  f"(@{_di.get('env_steps')} steps, {len(_dsd)} tensors)", flush=True)
            del _di, _dsd


    mbpo_constants = None
    prior_surrogate = None
    if config.mbpo_weight > 0.0:
        from online_rl_mimicgen_mbpo import mbpo as mbpo_module

        if not config.train_critic:
            raise ValueError(
                "--mbpo-weight with --no-train-critic: the synthetic loss trains the critic and "
                "nothing else, so this combination computes a gradient that goes nowhere."
            )
        if config.mbpo_chunks < 1 or config.mbpo_chunks > 3:
            raise ValueError(
                f"--mbpo-chunks {config.mbpo_chunks}: 1..3 only. The dynamics model is 4.33mm at "
                "one chunk and 12.85-23mm by twelve, so beyond ~3 the synthetic success bit -- "
                "which is the entire reward signal -- is fiction."
            )


        mbpo_constants = mbpo_module.load_constants(config.task)
        if config.mbpo_chunks > 1:
            from online_rl_mimicgen_mbpo.prior_surrogate import PriorSurrogate

            if not config.mbpo_prior_surrogate:
                raise ValueError(
                    f"--mbpo-chunks {config.mbpo_chunks} needs --mbpo-prior-surrogate: the "
                    "observation-conditioned flow cannot be queried at a synthetic state, so "
                    "depth >= 2 has no base chunk to apply the residual to."
                )
            payload = torch.load(config.mbpo_prior_surrogate, map_location=device)
            prior_surrogate = PriorSurrogate(payload["state_dim"], payload["chunk_dim"]).to(device)
            prior_surrogate.load_state_dict(payload["state_dict"])
            prior_surrogate.eval()
            prior_surrogate.requires_grad_(False)
            if payload["chunk_dim"] != chunk_dim:
                raise ValueError(
                    f"prior surrogate was fit at chunk_dim={payload['chunk_dim']} but this run is "
                    f"{chunk_dim}"
                )
            report = payload.get("report", {})


            print(f"[mbpo] prior surrogate std_ratio={report.get('std_ratio', float('nan')):.3f} "
                  f"gap_score={report.get('gap_score', float('nan')):.3f} "
                  f"test_nll={report.get('test_nll', float('nan')):.3f}", flush=True)
            if not 0.5 <= report.get("std_ratio", 1.0) <= 2.0:
                print("[mbpo] WARNING: surrogate spread is off by >2x from the real prior's; "
                      "best-of-N at depth >= 2 will not resemble the deployed policy", flush=True)
        print(f"[mbpo] weight={config.mbpo_weight} chunks={config.mbpo_chunks} "
              f"best_of_n={config.mbpo_best_of_n} max_sigma_mm={config.mbpo_max_sigma_mm} "
              f"task={config.task}", flush=True)
        if not config.dvd_finetune_dynamics:
            print("[mbpo] f is FROZEN on the distribution it was fit on (no "
                  "--dvd-finetune-dynamics). This is the staleness CONTROL arm; watch "
                  "mbpo_success_fraction against the real eval rate.", flush=True)

    if value_fn is not None and config.value_intervention == "dueling":


        critics_q = DuelingCritic(critics, value_fn)
        target_critics_q = DuelingCritic(target_critics, value_fn)
        print(
            f"dueling critic: Q = V_{config.value_variant}(s) + A(s,a); "
            f"only A is trained ({sum(p.numel() for p in critics.parameters()):,} params)"
        )
    else:
        critics_q = critics
        target_critics_q = target_critics


    if config.eval_freq < config.total_env_steps:
        run_eval(0)
    else:
        print("[eval] disabled (eval_freq >= total steps); trailing episode success is the metric, "
              "checkpoints still save at save_freq for offline eval later", flush=True)

    reward_model = None
    if config.reward_model:
        from online_rl_mimicgen.reward_model import SarmRewardModel

        reward_model = SarmRewardModel(
            config.reward_model,
            TASKS[config.task]["dataset_root"],
            device,
            weight=config.reward_model_weight,
        )
        print(
            f"reward model: {config.reward_model}  weight={config.reward_model_weight}  "
            f"window={reward_model.deltas}  target_index={reward_model.target_index}  "
            f"terminate_on_drop={config.terminate_on_reward_drop}"
        )


    if getattr(config, "encoder_value_aux", 0.0) > 0.0:
        _vaux_target = _value_read(value_fn, dvd_term)
        if _vaux_target is None:
            raise SystemExit(
                "--encoder-value-aux has no value function to regress onto: pass --value-model "
                "(for a critic-owned pixel encoder with no dvd_term, a STATE-space V -- it is fed "
                "the 37-dim env state). Without one the aux loss is skipped at every update.")
    observation, _ = env.reset(seed=config.seed)
    if reward_model is not None:
        reward_model.reset()
        reward_model.observe(observation)
    episode_index = 0
    episode_steps = 0


    annotator = None
    if config.critical_enabled():
        annotator = (StateAnnotator(config.task) if config.critical_ladder == "geometric"
                     else OnlineAnnotator(config.critical_ladder))
    if annotator is not None:
        for source in (config.actor_gradient_source_normal, config.actor_gradient_source_critical):


            if source == "dvd1" and config.dvd_weight <= 0.0:
                raise ValueError(
                    "actor_gradient_source='dvd1' but --dvd-weight is 0, which would run the Q "
                    "baseline under an DVD label. Set --dvd-weight > 0."
                )
            if source not in ("q", "dvd1"):
                raise NotImplementedError(f"actor_gradient_source={source!r} is not implemented")
        print(
            f"critical annotations: ladder={config.critical_ladder} "
            f"truncate={config.truncate_after_catastrophe} "
            f"sampling={config.critical_sampling}@{config.critical_batch_fraction} "
            f"pre_window={config.critical_pre_window} "
            f"terminal={config.catastrophic_as_terminal}"
        )

    generated = 0
    retained = 0
    pending: list[dict] = []

    pending_centres: list[int | None] = []
    causal_scores: list[float] = []

    previous_score: float | None = None
    _seghead = None
    if getattr(config, "pose_from_seghead", False):


        from .seghead import (load_all as _sh_load, encode as _seghead_encode,
                              read_pose as _seghead_read, assert_frame_size as _seg_assert2)
        _seg_assert2()
        _d, _h, _r, _w, _cfg = _sh_load(device=device, with_wm=False)
        _seghead = (_d, _h, _r)
        print(f"[seghead] perceived pose ON: c={_h.c} feat={_h.out_size} "
              f"(needle ~9.0mm / tripod ~1.55mm measured)", flush=True)
    episode_success = False


    ep_frames: list[np.ndarray] = []
    ep_actions: list[np.ndarray] = []


    v_reload_mtime = 0.0
    env_steps = 0
    updates = 0
    chunks_since_update = 0
    update_s_window = 0.0
    last_eval = 0
    earned_steps = 0
    last_env_steps = 0
    model_error_ema = None
    training_gate = None
    last_save = 0

    if config.resume:


        if config.resume == "auto":
            cks = sorted((output_dir / "checkpoints").glob("*.pt"))

            cks = [c for c in cks if c.stem != "000000"]
            ck_path = cks[-1] if cks else None
        else:
            ck_path = Path(config.resume)
        if ck_path is None:
            print("[resume] auto: no checkpoint past 000000 -- starting fresh", flush=True)
        else:
            pay = torch.load(ck_path, map_location=device, weights_only=False)
            from .legacy import refuse_unconverted_checkpoint
            refuse_unconverted_checkpoint(pay, ck_path, "--resume")
            ex = pay.get("extra", {})
            actor.load_state_dict(pay["actor"])
            critics.load_state_dict(pay["critics"])
            target_critics.load_state_dict(pay["target_critics"])
            env_steps = int(pay["env_steps"]); updates = int(pay["updates"])
            last_eval = last_save = env_steps
            restored = ["actor", "critics"]
            if "log_alpha" in ex:
                with torch.no_grad():
                    log_alpha.fill_(float(ex["log_alpha"]))
                restored.append("log_alpha")
            for key, obj in (("encoder", encoder), ("target_encoder", target_encoder),
                             ("actor_encoder", actor_encoder)):
                if key in ex and obj is not None:
                    obj.load_state_dict(ex[key]); restored.append(key)
            if "value_fn" in ex and value_fn is not None:
                value_fn.load_state_dict(ex["value_fn"]); restored.append("value_fn")
            if "attn_pool" in ex and attn_pool is not None:


                attn_pool.load_state_dict(ex["attn_pool"]); restored.append("attn_pool")
            if "obs_pool" in ex and obs_pool is None:


                raise ValueError(
                    "this checkpoint holds an 'obs_pool' (--dino-obs-pool-separate, CAP variant "
                    "2) but the resume command did not pass --dino-obs-pool-separate. Both pools "
                    "are 768-dim, so nothing downstream would catch it and the arm would silently "
                    "become variant 1. Re-add the flag, or resume from a variant-1 checkpoint.")
            if "obs_pool" in ex and obs_pool is not None:


                obs_pool.load_state_dict(ex["obs_pool"]); restored.append("obs_pool")
            if "obs_pool_target" in ex and obs_pool_target is None:
                raise ValueError(
                    "this checkpoint holds an 'obs_pool_target' (--dino-obs-pool-ema-target) but the "
                    "resume command did not pass that flag: the TD target would silently switch to "
                    "the live pool. Re-add --dino-obs-pool-ema-target.")
            if obs_pool_target is not None:
                if "obs_pool_target" in ex:
                    obs_pool_target.load_state_dict(ex["obs_pool_target"]); restored.append("obs_pool_target")
                elif "obs_pool" in ex:

                    obs_pool_target.load_state_dict(ex["obs_pool"]); restored.append("obs_pool_target<-obs_pool")
            if "actor_pool" in ex and actor_pool is None:


                raise ValueError(
                    "this checkpoint holds an 'actor_pool' (--dvd-actor-pool: the actor's own "
                    "DVD-trained copy of the readout) but the resume command did not pass "
                    "--dvd-actor-pool. The two modules are the same width, so nothing downstream "
                    "would catch it and the arm would silently become the shared-pool run. "
                    "Re-add the flag, or resume from a shared-pool checkpoint.")
            if actor_pool is not None and "actor_pool" not in ex:


                raise ValueError(
                    "--dvd-actor-pool was passed but this checkpoint has no 'actor_pool': it "
                    "predates the flag. Its actor optimiser state also has one parameter group "
                    "fewer, so the resume would die in opt.load_state_dict. Resume it without "
                    "--dvd-actor-pool, or start a fresh run.")
            if "actor_pool" in ex and actor_pool is not None:


                actor_pool.load_state_dict(ex["actor_pool"]); restored.append("actor_pool")
            if "sargmax_pool" in ex and sargmax_pool is not None:


                sargmax_pool.load_state_dict(ex["sargmax_pool"]); restored.append("sargmax_pool")
            if ("value_target" in ex and value_fn is not None
                    and getattr(value_fn, "_target", None) is not None):
                getattr(value_fn, "_target").load_state_dict(ex["value_target"])
                restored.append("value_target")
            if value_td is not None:


                if "value_td" in ex:
                    value_td.load_state_dict(ex["value_td"]); restored.append("value_td")
                if ("value_td_target" in ex
                        and getattr(value_td, "_target", None) is not None):
                    getattr(value_td, "_target").load_state_dict(ex["value_td_target"])
                    restored.append("value_td_target")
                if "value_dual" in ex:
                    value_dual.load_scale_state(ex["value_dual"]); restored.append("value_dual")
            if "dvd_dynamics" in ex and dvd_term is not None:


                miss = dvd_term.dynamics.load_state_dict(ex["dvd_dynamics"], strict=False)
                bad = [k for k in miss.missing_keys if not k.startswith("wm.dino.")]
                assert not bad and not miss.unexpected_keys, (
                    f"dvd_dynamics mismatch: missing={bad} unexpected={miss.unexpected_keys}")
                restored.append("dvd_dynamics")
            for key, opt in (("actor", actor_optimizer), ("critic", critic_optimizer),
                             ("alpha", alpha_optimizer), ("value", value_optimizer),
                             ("value_td", value_td_optimizer),
                             ("dvd", dvd_term.optimiser if dvd_term is not None else None)):
                st = ex.get("opt", {}).get(key)
                if st is not None and opt is not None:
                    opt.load_state_dict(st); restored.append(f"opt.{key}")
            rp = output_dir / f"replay/{env_steps:06d}.npz"
            if rp.exists():
                online = ReplayBuffer.load(rp); restored.append(f"replay({online.size})")
                if config.store_prior_continuation():
                    online.ensure_continuation()
            else:
                print(f"[resume] no replay at {rp.name} -- the buffer restarts EMPTY, so the "
                      f"first {config.learning_starts} steps after resume re-fill it", flush=True)
            print(f"[resume] {ck_path.name} @ env_steps={env_steps} updates={updates}; "
                  f"restored: {', '.join(restored)}", flush=True)

    grad_probe = None
    if config.grad_probe_freq > 0:
        if dvd_term is None or not getattr(dvd_term, "feature_space", False):
            raise ValueError("--grad-probe-freq needs a feature-space DVD term "
                             "(--dvd-dynamics dino_wm): it differentiates V through the DINO "
                             "encoder, which a state-space V has no path to.")
        from .grad_probe import GradProbe
        grad_probe = GradProbe(device)
    last_grad_probe = 0

    rolling = defaultdict(list)
    started = time.perf_counter()
    try:
        _pre = getattr(config, "offline_pretrain_steps", 0)
        if _pre > 0:


            print(f"[warm-start] {_pre} offline updates on demonstrations before any env step",
                  flush=True)
            for _p in range(_pre):
                _b = expert.sample(config.batch_size, device)
                _l = update_dice(
                    _b, actor, critics_q, target_critics_q, actor_optimizer, critic_optimizer,
                    config, updates, 1.0, config.batch_size, value_fn, dvd_term, encoder,
                    target_encoder, mbpo_constants, prior_surrogate, dino,
                    v_aux_head=v_aux_head, actor_encoder=actor_encoder,
                    target_actor=target_actor, env_steps_now=0, probe_grads=None)
                updates += 1
                if _p % 2000 == 0 or _p == _pre - 1:
                    print(f"  [warm-start] {_p+1}/{_pre} critic {_l.get('critic_loss', float('nan')):.5f} "
                          f"actor {_l.get('actor_loss', float('nan')):.4f}", flush=True)
            if value_dual is not None:


                value_dual.pop_stats()
            save_checkpoint(output_dir / "checkpoints/warmstart.pt", config, actor, critics,
                            target_critics, 0, updates,
                            {"log_alpha": float(log_alpha.detach())})
        window_t0 = time.perf_counter()
        if getattr(config, "compile_parts", ""):
            _cp = install_compiled_parts(config.compile_parts, dvd_term=dvd_term, value_fn=value_fn,
                                         encoder=encoder, critics=critics, actor=actor)
            print(f"[compile] torch.compile on: {', '.join(_cp) or 'nothing matched'} "
                  "(dynamic=True; the first batches of each shape pay the compile)", flush=True)
        if getattr(config, "time_components", False):
            _stages = install_component_timers(prior=prior, value_fn=value_fn, dvd_term=dvd_term,
                                               rm_distiller=rm_distiller, online=online,
                                               backward=getattr(config, "time_backward", False))
            print(f"[time-components] timing {', '.join(_stages)} (two cuda syncs per call; "
                  "nested stages are not additive)", flush=True)
        while env_steps < config.total_env_steps:
            _ap = observation["agent_pos"]
            if getattr(config, "pose_from_seghead", False):


                _ap = np.array(_ap, dtype=np.float64, copy=True)
                _im = torch.as_tensor(pixel_preprocess(observation["pixels"])[None]
                                      ).permute(0, 1, 4, 2, 3).float().div_(255.0)
                _fine = _seghead_encode(_seghead[0], _seghead[1], _im, _ap[None, :9], device)
                _xyz = _seghead_read(_seghead[2], _fine, _ap[None, :9], device)[0].cpu().numpy()
                _ap[9:12] = _xyz[0]
                _ap[23:26] = _xyz[1]
            if getattr(config, "state_noise_mm", 0.0) > 0.0:

                _ap = np.array(_ap, dtype=np.float64, copy=True)
                _sig = config.state_noise_mm / 1000.0
                if getattr(config, "state_noise_mode", "iid") == "episode":

                    if getattr(config, "_ep_bias", None) is None:
                        config._ep_bias = {sl: np.random.normal(0.0, _sig, int(sl.split(":")[1]) - int(sl.split(":")[0]))
                                           for sl in config.state_noise_slots.split(",")}
                    for _sl, _b_ in config._ep_bias.items():
                        _a, _b = (int(v) for v in _sl.split(":"))
                        _ap[_a:_b] += _b_
                else:
                    for _sl in config.state_noise_slots.split(","):
                        _a, _b = (int(v) for v in _sl.split(":"))
                        _ap[_a:_b] += np.random.normal(0.0, _sig, _b - _a)
            state = torch.as_tensor(scaler(_ap), device=device).unsqueeze(0)
            if seg_hist is not None:


                _im = torch.as_tensor(pixel_preprocess(observation["pixels"])[None]
                                      ).permute(0, 1, 4, 2, 3).float().div_(255.0)
                seg_hist.push(_im, np.asarray(observation["agent_pos"][:9], dtype=np.float32))
            if dino_history is not None:


                dino_history.push(observation)


            state_in = net_state(
                observation, state, encoder,
                *(_seg_obs_parts(True, seg_hist, observation, device)
                  if getattr(config, "seg_obs", False)
                  else _latent_obs_parts(True, seg_hist, observation, device)
                  if getattr(config, "latent_obs", False)
                  else _dino_obs_parts(config.dino_obs, dvd_term, dino_history,
                                       observation, device)))


            actor_in = (net_state(observation, state, actor_encoder)
                        if actor_encoder is not None else state_in)


            if _prior_cont:


                if prior_draws > 1:
                    base_samples, base_cont_samples = (flatten_chunk(t) for t in prior.action_chunk_samples(
                        observation, prior_draws, continuation=True))
                    base, base_cont = base_samples[:1], base_cont_samples[:1]
                else:
                    base, base_cont = (flatten_chunk(t) for t in prior.action_chunk(observation, continuation=True))
                    base_samples, base_cont_samples = base, base_cont
            elif prior_draws > 1:
                base_samples = flatten_chunk(prior.action_chunk_samples(observation, prior_draws))
                base = base_samples[:1]
            else:
                base_samples = base = flatten_chunk(prior.action_chunk(observation))
            with torch.inference_mode():


                if config.exploration_gate_mm > 0.0:
                    if model_error_ema is not None and model_error_ema < config.exploration_gate_mm:
                        earned_steps += env_steps - last_env_steps
                    ramp_clock = earned_steps
                else:
                    ramp_clock = env_steps
                last_env_steps = env_steps
                progress = min(1.0, ramp_clock / max(config.progressive_exploration_steps, 1))
                enabled = np.random.random() < progress


                bon_dvd = config.bon_source == "dvd" and dvd_term is not None
                if bon_dvd and config.bon_dvd_until > 0 and env_steps >= config.bon_dvd_until:


                    if not getattr(select_dice_exploration, "_handed_off", False):
                        print(f"[bon] scorer hand-off V(f) -> min-Q at {env_steps} env steps", flush=True)
                        select_dice_exploration._handed_off = True
                    bon_dvd = False
                if (config.method == "dice_rl" and (config.train_critic or bon_dvd)
                        and updates > 250 and enabled):
                    _bon_n = config.bon_samples or 16
                    bases = (
                        base_samples[:_bon_n] if config.bon_axis == "prior"
                        else base.repeat(_bon_n, 1)
                    )
                    if bon_dvd and dino_history is not None:
                        dino_history.apply(bases.shape[0])
                    if (bon_dvd and config.dvd_dynamics == "latent56"
                            and seg_hist is not None and seg_hist.cur is not None):


                        from .latent56 import norm_prop as _l56_np
                        _bn = bases.shape[0]
                        _p9 = np.asarray(observation["agent_pos"][:9], dtype=np.float32)
                        seg_adapter.set_context(
                            seg_hist.prev.expand(_bn, -1),
                            seg_hist.cur.expand(_bn, -1),
                            np.repeat(_l56_np(_p9, seg_adapter.cfg)[None], _bn, axis=0),
                        )
                    elif bon_dvd and seg_hist is not None and seg_hist.cur is not None:


                        _bn = bases.shape[0]
                        _p9 = np.asarray(observation["agent_pos"][:9], dtype=np.float32)
                        seg_adapter.set_context(
                            seg_hist.prev.expand(_bn, -1, -1, -1, -1),
                            seg_hist.cur.expand(_bn, -1, -1, -1, -1),
                            np.repeat(_p9[None], _bn, axis=0),
                        )
                    chosen = select_dice_exploration(
                        actor, critics_q, actor_in, bases, config.residual_scale,
                        dvd_term=dvd_term if bon_dvd else None,
                        bon_reduce=config.bon_reduce,


                        dvd_state=(state_in if getattr(config, "dvd_on_encoder", False)
                                   else state),
                        bon_select=getattr(config, "bon_select", "max"),
                        bon_quantile=getattr(config, "bon_quantile", 0.75),
                        frozen_value_fn=frozen_value_fn if bon_dvd else None,
                        max_norm=config.residual_max_norm,

                        **({"conts": (base_cont_samples[:_bon_n] if config.bon_axis == "prior"
                                      else base_cont.repeat(_bon_n, 1)),
                            "lookahead": config.bon_lookahead}
                           if (bon_dvd and config.bon_lookahead > 1) else {}),
                    )
                    from_actor = False
                    rolling["bon_selected_rank"].append(
                        getattr(select_dice_exploration, "last_rank", 1.0))
                    if frozen_value_fn is not None and bon_dvd:

                        for _key in ("frozen_changed", "frozen_rank_corr"):
                            _val = getattr(select_dice_exploration, "last_" + _key, None)
                            if _val is not None and math.isfinite(_val):
                                rolling["bon_" + _key].append(_val)
                    if bon_dvd:


                        for _key in ("top_gap", "member_std", "top_gap_over_std", "meaningful"):
                            _val = getattr(select_dice_exploration, "last_" + _key, None)
                            if _val is not None and math.isfinite(_val):
                                rolling["bon_" + _key].append(_val)
                        _g = getattr(select_dice_exploration, "last_gap", None)
                        if _g is not None and math.isfinite(_g):
                            rolling["bon_max_minus_median"].append(_g)
                elif enabled:
                    residual, _ = actor.sample(actor_in, base)
                    chosen = final_action(base, residual, config.residual_scale, config.residual_max_norm)
                    from_actor = True
                else:
                    chosen = base
                    from_actor = False


            if config.dvd_mode == "replay" and from_actor:
                with torch.no_grad():
                    behaviour_logp = float(actor.log_prob_of(
                        actor_in, base, ((chosen - base) / config.residual_scale).clamp(-1.0, 1.0))[0])
            else:
                behaviour_logp = float("nan")
            action_chunk = chosen.view(1, config.action_steps, 7)
            next_observation, reward, done, success, primitive_steps = rollout_chunk(
                env,
                action_chunk[0].cpu().numpy(),
                step_penalty=config.step_penalty,
                observer=None if reward_model is None else reward_model.observe,
                use_env_reward=config.env_dense_reward,
            )


            reward *= config.sparse_reward_weight
            reward_centre = None
            seg_cut = False
            if reward_model is not None:


                reward_centre = len(reward_model.images) - 1
                if config.reward_model_causal:


                    causal_scores.append(float(reward_model.score_latest()))
                if config.terminate_on_reward_drop > 0.0:


                    score = reward_model.score_latest()
                    if (
                        previous_score is not None
                        and previous_score - score > config.terminate_on_reward_drop
                    ):
                        done = True
                    previous_score = score
            if seg_term is not None and not done:


                seg_cut = bool(seg_term(next_observation["agent_pos"]))
                done = done or seg_cut
            next_state = torch.as_tensor(
                scaler(next_observation["agent_pos"]), device=device
            ).unsqueeze(0)


            _v_takes_state = (value_fn is not None
                              and int(getattr(value_fn, "input_dim", state.shape[-1]))
                              == int(state.shape[-1]))
            if value_fn is not None and dino is None and seg_adapter is None and _v_takes_state:


                value_stats.update(value_fn.frozen(state))
            if value_fn is not None and config.value_intervention == "shaping":


                chunk_discount = config.gamma**primitive_steps
                phi = float(value_fn.frozen(state))
                phi_next = 0.0 if done else float(value_fn.frozen(next_state))
                reward += chunk_discount * phi_next - phi


            timeout_boot = bool(getattr(config, "bootstrap_timeouts", False) and done and not success)
            if done and not timeout_boot:
                next_base_samples = torch.zeros((config.multi_sample_k, base.shape[1]),
                                                device=base.device)
            elif config.multi_sample_k > 1:
                next_base_samples = flatten_chunk(
                    prior.action_chunk_samples(next_observation, config.multi_sample_k)
                )
            else:
                next_base_samples = flatten_chunk(prior.action_chunk(next_observation))
            next_base = next_base_samples[:1]
            transition = dict(
                state=state[0].cpu().numpy(),
                base=base[0].cpu().numpy(),
                action=chosen[0].cpu().numpy(),
                behaviour_logp=float(behaviour_logp),
                next_state=next_state[0].cpu().numpy(),
                next_base=next_base[0].cpu().numpy(),
                base_samples=base_samples[: config.resolved_actor_k()].reshape(-1).cpu().numpy(),
                next_base_samples=next_base_samples[: config.multi_sample_k]
                .reshape(-1).cpu().numpy(),
                **({"base_cont": base_cont[0].cpu().numpy(),
                    "base_cont_samples": base_cont_samples[: config.resolved_actor_k()].reshape(-1).cpu().numpy()}
                   if _store_cont else {}),
                reward=reward,


                done=float(done and not seg_cut and not timeout_boot),
                discount=config.gamma**primitive_steps,
                data_source=0.0,
                **({"image": pixel_preprocess(observation["pixels"]),
                    "next_image": pixel_preprocess(next_observation["pixels"])}
                   if (config.pixel_obs or dino_history is not None) else {}),
                **({"prev_feat": dino_history.prev_feat()}
                   if dino_history is not None else {}),
                **({"prev_feat": seg_hist.as_grid("prev"),
                    "seg_feat": seg_hist.as_grid("cur"),


                    **({"next_feat": _latent_next_feat(next_observation)}
                       if getattr(config, "latent_obs", False) else {}),
                    **({"next_feat": _seg_next_feat(next_observation)}
                       if getattr(config, "seg_obs", False) else {}),
                    **({"tok": _latent_tokens(observation),
                        "next_tok": _latent_tokens(next_observation)}
                       if _l56_tok_needed else {})}
                   if seg_hist is not None else {}),


                **({"cond": flowdpg_cond()[0].float().cpu().numpy()}
                   if config.method == "flowdpg" and flowdpg_cond() is not None else {}),
            )
            if (getattr(config, "dump_episodes", False) or rm_distiller is not None) \
                    and "image" in transition:


                ep_frames.append((transition["image"],
                                  np.asarray(observation["agent_pos"], dtype=np.float32).copy()))
                ep_actions.append(np.asarray(transition["action"], dtype=np.float32).copy())
            if annotator is not None:


                if config.critical_ladder == "geometric":

                    annotator.observe(next_observation["agent_pos"], rewarded=bool(reward > 0.0))
                else:
                    annotator.observe(env)
            if config.needs_pending():


                transition["dvd_gap"] = getattr(select_dice_exploration, "last_gap", 0.0)
                pending.append(transition)
                pending_centres.append(reward_centre)
            else:
                online.add(mc_return=0.0, **transition)
            env_steps += primitive_steps
            episode_steps += primitive_steps
            episode_success = episode_success or success

            can_update = online.size > 0 and (
                config.method == "dice_rl" or env_steps >= config.learning_starts
            )


            if getattr(config, "update_every", 1) > 1:
                chunks_since_update += 1
                if chunks_since_update < config.update_every:
                    can_update = False
                elif can_update:
                    chunks_since_update = 0


            update_rounds = int(config.utd)
            if getattr(config, "update_on_episode_end", False):
                chunks_since_update += 1
                if done and can_update:
                    update_rounds = chunks_since_update * int(config.utd)
                    chunks_since_update = 0
                else:
                    can_update = False
            if getattr(config, "freeze_training_at", 0) and env_steps >= config.freeze_training_at:


                if not getattr(config, "_froze_banner", False):
                    config._froze_banner = True
                    print(f"[freeze-and-roll] training stopped at {env_steps}; "
                          f"collection continues", flush=True)
                can_update = False
            if can_update:
                _t_upd = time.perf_counter()


                for _ in range(update_rounds):
                    if config.method == "dice_rl":
                        if getattr(config, "expert_gate_success", 0.0) and rolling["episode_success"]:
                            config.note_success(float(np.mean(rolling["episode_success"][-30:])))
                        expert_ratio = config.current_expert_ratio(env_steps)
                        expert_count = int(config.batch_size * expert_ratio)
                        online_count = config.batch_size - expert_count


                        for _ai in range(max(1, int(getattr(config, "actor_utd", 1))) - 1):
                            update_dice(
                                concat_batches(
                                    online.sample(
                                        online_count, device,
                                        config.critical_batch_fraction if config.critical_sampling else 0.0,
                                        episode_uniform=config.replay_episode_uniform,
                                    ),
                                    expert.sample(expert_count, device),
                                ),
                                actor, critics_q, target_critics_q, actor_optimizer,
                                critic_optimizer, config, updates, expert_ratio, expert_count,
                                value_fn, dvd_term, encoder, target_encoder, mbpo_constants,
                                prior_surrogate, dino, target_actor=target_actor,
                                env_steps_now=env_steps, probe_grads=None,
                                v_aux_head=v_aux_head, actor_encoder=actor_encoder,
                            )


                        batch = concat_batches(
                            online.sample(
                                online_count,
                                device,
                                config.critical_batch_fraction if config.critical_sampling else 0.0,
                                episode_uniform=config.replay_episode_uniform,
                            ),
                            expert.sample(expert_count, device),
                        )
                        probe_grads = ({} if (config.probe_step_eps > 0.0
                                               and config.gradvar_every
                                               and updates % config.gradvar_every == 0)
                                       else None)
                        losses = update_dice(
                            batch,
                            actor,
                            critics_q,
                            target_critics_q,
                            actor_optimizer,
                            critic_optimizer,
                            config,
                            updates,
                            expert_ratio,
                            expert_count,
                            value_fn,
                            dvd_term,
                            encoder,
                            target_encoder,
                            mbpo_constants,
                            prior_surrogate,
                            dino,
                            target_actor=target_actor,
                            env_steps_now=env_steps,
                            probe_grads=probe_grads,
                            v_aux_head=v_aux_head,
                            actor_encoder=actor_encoder,
                        )
                        if probe_grads and config.probe_step_eps > 0.0:


                            _prm = [q for q in actor.parameters() if q.requires_grad]
                            _dirs = {k: -v for k, v in probe_grads.items()
                                     if k in ("q", "dvd")}
                            losses.update(directional_return_probe(
                                _prm, _dirs,
                                lambda: evaluate(
                                    config.task, prior, actor, scaler, config.residual_scale,
                                    config.probe_episodes, 90_000, config.step_penalty,
                                    critics=critics_q, strategy="greedy",
                                    samples=config.eval_samples, insertion=insertion,
                                    horizon=config.horizon, dvd_term=dvd_term,
                                    bon_source=config.bon_source, encoder=encoder,
                                    residual_max_norm=config.residual_max_norm,
                                    dino_history=None if dino is None else dino.FrameHistory(dino_adapter),
                                    dino_obs=config.dino_obs,
                                    actor_encoder=actor_encoder,
                                )["success_rate"],
                                epsilon=config.probe_step_eps))
                            probe_grads.clear()


                        fit_batch = batch
                        if config.pixel_obs and "env_state" in batch:
                            fit_batch = dict(batch)
                            fit_batch["state"] = batch["env_state"]
                            fit_batch["next_state"] = batch["env_next_state"]


                        _actor_batch, _actor_fit_batch = batch, fit_batch
                        for _vi in range(max(1, int(getattr(config, "value_utd", 1)))):
                            if _vi > 0:
                                batch = concat_batches(
                                    online.sample(
                                        online_count,
                                        device,
                                        config.critical_batch_fraction if config.critical_sampling else 0.0,
                                        episode_uniform=config.replay_episode_uniform,
                                    ),
                                    expert.sample(expert_count, device),
                                )
                                fit_batch = batch
                                if config.pixel_obs and "env_state" in batch:
                                    fit_batch = dict(batch)
                                    fit_batch["state"] = batch["env_state"]
                                    fit_batch["next_state"] = batch["env_next_state"]
                            if (value_fn is not None
                                    and (value_fn.finetune or value_td is not None
                                         or config.value_mc_weight > 0.0)
                                    and env_steps >= config.value_finetune_after):


                                v_fit_batch = fit_batch
                                if config.value_fit_online_only and online_count >= 8:
                                    _vn = fit_batch["state"].shape[0]
                                    _vk = min(online_count, _vn)
                                    v_fit_batch = {k: (v[:_vk] if hasattr(v, "shape")
                                                       and len(getattr(v, "shape", ())) > 0
                                                       and v.shape[0] == _vn else v)
                                                   for k, v in fit_batch.items()}
                                if getattr(config, "dvd_on_encoder", False):


                                    if encoder_optimizer is not None:


                                        encoder_optimizer.zero_grad(set_to_none=True)
                                        _enc_now = encoder(v_fit_batch["image"], v_fit_batch["state"][:, :PIXEL_PROPRIO],
                                                           augment=config.pixel_augment)
                                        with torch.no_grad():
                                            _enc_next = target_encoder(
                                                v_fit_batch.get("next_image_boot", v_fit_batch["next_image"]),
                                                v_fit_batch["next_state"][:, :PIXEL_PROPRIO], augment=config.pixel_augment)
                                    else:
                                        with torch.no_grad():
                                            _enc_now = encoder(v_fit_batch["image"],
                                                               v_fit_batch["state"][:, :PIXEL_PROPRIO], augment=False)
                                            _enc_next = encoder(v_fit_batch.get("next_image_boot", v_fit_batch["next_image"]),
                                                                v_fit_batch["next_state"][:, :PIXEL_PROPRIO], augment=False)
                                    v_fit_batch = dict(v_fit_batch)
                                    v_fit_batch["state"] = _enc_now
                                    v_fit_batch["next_state"] = _enc_next
                                elif dvd_term is not None and dvd_term.feature_space and dino is not None:


                                    _scaled_now = (batch["env_state"] if "env_state" in batch
                                                   else batch["state"])
                                    _scaled_next = (batch["env_next_state"] if "env_next_state" in batch
                                                    else batch["next_state"])
                                    raw_now = dvd_term.raw_state(_scaled_now)
                                    raw_next = dvd_term.raw_state(_scaled_next)


                                    feat_now = _value_pool_of(dvd_term)(
                                        dino.encode_frames(dino_adapter, batch["image"]),
                                        dvd_term.value_pool_mode)


                                    feat_next = _value_pool_of(dvd_term)(
                                        dino.encode_frames(dino_adapter,
                                                           batch.get("next_image_boot", batch["next_image"])),
                                        dvd_term.value_pool_mode)
                                    if dvd_term.proprio_dim:
                                        feat_now = torch.cat(
                                            [feat_now, raw_now[:, :dvd_term.proprio_dim]], dim=-1)
                                        feat_next = torch.cat(
                                            [feat_next, raw_next[:, :dvd_term.proprio_dim]], dim=-1)
                                    v_fit_batch = dict(fit_batch)
                                    v_fit_batch["state"] = feat_now
                                    v_fit_batch["next_state"] = feat_next
                                elif config.dvd_dynamics == "latent56":


                                    from .latent56 import encode as _l56_enc, norm_prop as _l56_np
                                    _sn = (batch["env_state"] if "env_state" in batch
                                           else batch["state"])
                                    _sx = (batch["env_next_state"] if "env_next_state" in batch
                                           else batch["next_state"])
                                    _rn = dvd_term.raw_state(_sn)[:, :9]
                                    _rx = dvd_term.raw_state(_sx)[:, :9]
                                    _im = batch["next_image"].permute(0, 1, 4, 2, 3).float() / 255.0
                                    _zn = batch["seg_feat"].float()
                                    _zx = _l56_enc(_l56_dino, _l56_enc_net, _im, _l56_cfg, device)
                                    _pn = torch.as_tensor(_l56_np(_rn, _l56_cfg), device=device)
                                    _px = torch.as_tensor(_l56_np(_rx, _l56_cfg), device=device)
                                    v_fit_batch = dict(fit_batch)
                                    v_fit_batch["state"] = torch.cat([_zn, _pn], dim=-1)
                                    v_fit_batch["next_state"] = torch.cat([_zx, _px], dim=-1)
                                elif dvd_term is not None and dvd_term.feature_space and seg_adapter is not None:


                                    _scaled_now = (batch["env_state"] if "env_state" in batch
                                                   else batch["state"])
                                    _scaled_next = (batch["env_next_state"] if "env_next_state" in batch
                                                    else batch["next_state"])
                                    raw_now = dvd_term.raw_state(_scaled_now)
                                    raw_next = dvd_term.raw_state(_scaled_next)
                                    from .seghead import encode as _seg_enc_fn, norm_feat as _seg_nf
                                    with torch.no_grad():
                                        _next_im = (batch["next_image"].permute(0, 1, 4, 2, 3).float()
                                                   / 255.0)
                                        _next_map = SegHeadAdapter.to_grid(_seg_nf(
                                            _seg_enc_fn(seghead_stack[0], seghead_stack[1], _next_im,
                                                       raw_next[:, :9].detach().cpu(), device),
                                            device))
                                    v_fit_batch = dict(fit_batch)
                                    v_fit_batch["state"] = torch.cat(
                                        [batch["seg_feat"].float().flatten(1), raw_now[:, :9]], dim=-1)
                                    v_fit_batch["next_state"] = torch.cat(
                                        [_next_map.flatten(1), raw_next[:, :9]], dim=-1)
                                if value_fn.finetune:


                                    _pool_fit_diag = _vi == 0 and "pool_grad_actor_dvd" in losses
                                    if _pool_fit_diag:
                                        _copy_g0 = [None if _q.grad is None else _q.grad.detach().clone()
                                                    for _q in dvd_term._pool.parameters()]
                                    rolling["v_fit_loss"].append(
                                        value_fn.fit_step(v_fit_batch, value_optimizer)
                                    )
                                    if _pool_fit_diag:


                                        losses["pool_grad_value_fit"] = _pool_grad_norm(
                                            dvd_term._value_pool)
                                        losses["pool_grad_actor_fit"] = _grad_change_norm(
                                            dvd_term._pool, _copy_g0)
                                        del _copy_g0
                                    if config.value_target_tau > 0.0:
                                        value_fn.soft_update_target(config.value_target_tau)
                                    if encoder_optimizer is not None and getattr(config, "dvd_on_encoder", False):
                                        rolling["encoder_grad"].append(float(
                                            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10.0)))
                                        encoder_optimizer.step()
                                        encoder_optimizer.zero_grad(set_to_none=True)
                                        soft_update(encoder, target_encoder, config.tau)
                                if config.value_mc_weight > 0.0:


                                    _mc = value_fn.mc_fit_step(
                                        v_fit_batch, value_optimizer, config.value_mc_weight)
                                    if _mc == _mc:
                                        rolling["v_mc_loss"].append(_mc)
                                if value_td is not None:


                                    if int(v_fit_batch["state"].shape[-1]) != int(value_td.input_dim):
                                        raise RuntimeError(
                                            f"td V fit batch is {v_fit_batch['state'].shape[-1]}-wide, "
                                            f"V takes {value_td.input_dim}: the feature-space "
                                            "v_fit_batch branch was not taken")
                                    _td_fit = dict(v_fit_batch)
                                    _td_fit["state"] = v_fit_batch["state"].detach()
                                    _td_fit["next_state"] = v_fit_batch["next_state"].detach()
                                    rolling["v_td_fit_loss"].append(
                                        value_td.fit_step(_td_fit, value_td_optimizer))
                                    if config.value_target_tau > 0.0:
                                        value_td.soft_update_target(config.value_target_tau)
                            if value_dual is not None and _vi == 0:


                                for _dk, _dv in value_dual.pop_stats().items():
                                    rolling[_dk].append(_dv)


                            if rm_distiller is not None and env_steps >= config.value_distill_after:
                                _dl = rm_distiller.maybe_fit(updates)
                                if _dl == _dl:
                                    rolling["rm_distill_loss"].append(_dl)
                        batch, fit_batch = _actor_batch, _actor_fit_batch


                        if (config.dvd_fit_probe_every > 0 and dino is not None
                                and dvd_term is not None
                                and updates % config.dvd_fit_probe_every == 0):
                            with torch.no_grad(), torch.random.fork_rng(
                                    devices=[device] if device.type == "cuda" else []):
                                _pr = config.dvd_fit_batch_rows()
                                _pb = online.gather(online.recent_indices(_pr), device)
                                if config.pixel_obs and "env_state" in _pb:
                                    _pb = dict(_pb)
                                    _pb["state"] = _pb["env_state"]
                                    _pb["next_state"] = _pb["env_next_state"]
                                _pb = dino.fit_batch(dino_adapter, _pb, _pr)
                                for _pk, _pv in dvd_term.probe_features(_pb).items():
                                    rolling[f"{_pk}_recent"].append(_pv)


                        dvd_fit_batch = fit_batch
                        if (dino is not None and config.dvd_finetune_dynamics
                                and config.dvd_fit_sampler != "batch"):


                            rolling["dvd_fit_own_draw"].append(
                                float(config.dvd_fit_own_draw(online.size)))
                        if (dino is not None and config.dvd_finetune_dynamics
                                and config.dvd_fit_sampler != "batch"
                                and config.dvd_fit_own_draw(online.size)):


                            if not dvd_fit_own_draw_started:
                                dvd_fit_own_draw_started = True
                                print(f"[dvd-fit-sampler] own draw takes over at env step {env_steps} "
                                      f"(update {updates}, online buffer {online.size} rows >= "
                                      f"{config.dvd_fit_min_online_rows()})", flush=True)
                            _fit_r = config.dvd_fit_batch_rows()
                            _fit_idx, _ = online.recent_mix_indices(
                                _fit_r, config.dvd_fit_recent_rows(), config.dvd_fit_recent_count())
                            _fb = online.gather(_fit_idx, device)


                            if config.pixel_obs and "env_state" in _fb:
                                _fb = dict(_fb)
                                _fb["state"] = _fb["env_state"]
                                _fb["next_state"] = _fb["env_next_state"]


                            rolling["dvd_fit_row_age_rows"].append(
                                float(online.row_age(_fit_idx).mean()))
                            dvd_fit_batch = dino.fit_batch(dino_adapter, _fb, _fit_r)
                        elif dino is not None and config.dvd_finetune_dynamics:


                            _fb = fit_batch
                            if config.dvd_fit_online_only and online_count >= 8:
                                _n = fit_batch["state"].shape[0]
                                _k = min(online_count, _n)
                                _fb = {k: (v[:_k] if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0
                                           and v.shape[0] == _n else v)
                                       for k, v in fit_batch.items()}
                            dvd_fit_batch = dino.fit_batch(
                                dino_adapter, _fb,
                                config.dvd_fit_rows or config.dvd_rows)
                        elif getattr(config, "dvd_on_encoder", False) and config.dvd_finetune_dynamics:


                            with torch.no_grad():
                                _nx1 = fit_batch.get("next_state_1step", fit_batch["next_state"])
                                _zs_enc = encoder(fit_batch["image"], fit_batch["state"][:, :PIXEL_PROPRIO], augment=False)
                                _zn_enc = encoder(fit_batch["next_image"], _nx1[:, :PIXEL_PROPRIO], augment=False)
                            dvd_fit_batch = dict(fit_batch)
                            dvd_fit_batch["state"] = _zs_enc
                            dvd_fit_batch["next_state"] = _zn_enc
                            dvd_fit_batch["next_state_1step"] = _zn_enc
                        elif seg_adapter is not None and config.dvd_finetune_dynamics:


                            _fb = fit_batch
                            if config.dvd_fit_online_only and online_count >= 8:
                                _n = fit_batch["state"].shape[0]
                                _k = min(online_count, _n)
                                _fb = {k: (v[:_k] if hasattr(v, "shape")
                                           and len(getattr(v, "shape", ())) > 0
                                           and v.shape[0] == _n else v)
                                       for k, v in fit_batch.items()}
                            _rn = dvd_term.raw_state(
                                _fb["env_state"] if "env_state" in _fb else _fb["state"])[:, :9]
                            _rx = dvd_term.raw_state(
                                _fb["env_next_state"] if "env_next_state" in _fb
                                else _fb["next_state"])[:, :9]
                            if config.dvd_dynamics == "latent56":


                                from .latent56 import (encode as _lfe, norm_prop as _lnp)
                                _im = _fb["next_image"].permute(0, 1, 4, 2, 3).float() / 255.0


                                _rn_env = dvd_term.raw_env_state(
                                    _fb["env_state"] if "env_state" in _fb else _fb["state"])[:, :9]
                                seg_adapter.set_context(
                                    _fb["prev_feat"].float(), _fb["seg_feat"].float(),
                                    _lnp(_rn_env, _l56_cfg))
                                dvd_fit_batch = dict(_fb)
                                dvd_fit_batch["target_feat"] = _lfe(
                                    _l56_dino, _l56_enc_net, _im, _l56_cfg, device)
                            else:
                              from .seghead import seg_fit_batch as _seg_fit_batch
                              dvd_fit_batch = _seg_fit_batch(
                                  seg_adapter, seghead_stack, _fb,
                                  config.dvd_fit_rows or config.dvd_rows or 32,
                                  _rn, _rx, device,
                                  train_head=config.dvd_finetune_seghead, ema=seghead_ema,
                                  map_pad=config.dvd_map_shift)
                        if dvd_term is not None:
                            _fs = dvd_term.fit_step(dvd_fit_batch)
                            if seghead_ema is not None:
                                seghead_ema.update(seghead_stack[1])
                                _r = seghead_guard.observe(_fs.get("dvd_copy_feat_mse"))
                                if _r is not None:
                                    rolling["seghead_copy_ratio"].append(_r)
                            for key, value in _fs.items():
                                if key == "dvd_pod_mm":
                                    model_error_ema = (value if model_error_ema is None
                                                       else 0.99 * model_error_ema + 0.01 * value)
                                rolling[key].append(value)
                        if config.gated_training and dvd_term is not None:
                            if training_gate is None:
                                if not (config.value_finetune and config.dvd_finetune_dynamics):
                                    raise SystemExit(
                                        "--gated-training needs --value-finetune and "
                                        "--dvd-finetune-dynamics: the controller gates their "
                                        "optimisers, and a module with no optimiser cannot be gated")
                                if config.actor_lr_drop_at:
                                    raise SystemExit("--gated-training and --actor-lr-drop-at both "
                                                     "rewrite the actor lr; pick one")
                                training_gate = TrainingGate(
                                    config.gate_skill_tau, config.gate_hyst,
                                    config.gate_td_tau, config.gate_td_hyst,
                                    config.gate_dwell, config.gate_slow,
                                    actor_optimizer, value_optimizer, dvd_term.optimiser)
                                print(f"[gate] ON  skill_tau={config.gate_skill_tau} "
                                      f"td_tau={config.gate_td_tau} dwell={config.gate_dwell} "
                                      f"slow={config.gate_slow}", flush=True)
                            training_gate.observe(
                                rolling["dvd_feat_mse"][-1] if rolling["dvd_feat_mse"] else None,
                                rolling["dvd_copy_feat_mse"][-1] if rolling["dvd_copy_feat_mse"] else None,
                                rolling["v_fit_loss"][-1] if rolling["v_fit_loss"] else None)
                            for key, value in training_gate.step().items():
                                rolling[key].append(value)
                    elif config.method == "flowdpg":
                        from .flowdpg import update_flowdpg


                        batch = online.sample(config.batch_size, device)
                        losses = update_flowdpg(
                            batch, flowdpg_wrapper, flowdpg_critic, flowdpg_target,
                            flowdpg_actor_opt, flowdpg_critic_opt,
                            step=updates,
                            horizon=int(prior.policy.config.horizon),
                            action_dim=int(prior.action_dim),
                            executed=flowdpg_executed,
                            lam_max=config.flowdpg_lambda_max,
                            warmup=config.flowdpg_warmup,
                            alpha=config.flowdpg_alpha,
                            cons_weight=config.flowdpg_cons_weight,
                            tau_polyak=config.tau,
                        )
                    else:
                        batch = online.sample(config.batch_size, device)
                        losses = update_policy_decorator(
                            batch,
                            actor,
                            critics,
                            target_critics,
                            actor_optimizer,
                            critic_optimizer,
                            log_alpha,
                            alpha_optimizer,
                            config,
                        )
                    updates += 1
                    for key, value in losses.items():
                        rolling[key].append(value)
                    if "pool_grad_actor_dvd" in losses:


                        print(f"[actor-pool] grad-diag update={updates - 1} env_steps={env_steps} "
                              f"actor_dvd={losses['pool_grad_actor_dvd']:.6g} "
                              f"value_dvd={losses['pool_grad_value_dvd']:.6g} "
                              f"value_fit={losses.get('pool_grad_value_fit', float('nan')):.6g} "
                              f"actor_fit={losses.get('pool_grad_actor_fit', float('nan')):.6g} "
                              f"drift_actor_vs_value={losses['pool_drift_actor_vs_value']:.6g}",
                              flush=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                update_s_window += time.perf_counter() - _t_upd

            if done and pending:
                if reward_model is not None:


                    if config.reward_model_causal:
                        scores = list(causal_scores)
                    else:
                        centres = [c for c in pending_centres if c is not None]
                        scores = list(reward_model.rescore_episode(centres))
                    rolling["reward_model_score"].append(float(np.mean(scores)) if len(scores) else 0.0)
                    weight = config.reward_model_weight
                    if config.reward_shaping_form == "potential":


                        for index, entry in enumerate(pending):
                            phi = float(scores[index - 1]) if index > 0 else 0.0
                            phi_next = (
                                float(scores[index])
                                if index < min(len(scores), len(pending) - 1) else 0.0
                            )
                            entry["reward"] += weight * (
                                float(entry["discount"]) * phi_next - phi
                            )
                    else:
                        for entry, score in zip(pending, scores, strict=False):
                            entry["reward"] += weight * float(score)


                episode = fill_mc_return(pending)


                episode = fill_episode_len(episode)
                if config.nstep > 1:


                    _traces = None
                    if getattr(config, "vtrace_sigma", 0.0) > 0.0:
                        with torch.no_grad():
                            _st = torch.as_tensor(
                                np.stack([t["state"] for t in episode]), device=device).float()
                            _bs = torch.as_tensor(
                                np.stack([t["base"] for t in episode]), device=device).float()
                            _ac = torch.as_tensor(
                                np.stack([t["action"] for t in episode]), device=device).float()
                            _res, _ = actor.sample(_st, _bs, deterministic=True)
                            _now = final_action(_bs, _res, config.residual_scale,
                                                config.residual_max_norm)
                            _d2 = (_ac - _now).pow(2).sum(-1)
                            _c = torch.exp(-_d2 / (2.0 * config.vtrace_sigma ** 2))
                            _traces = _c.clamp(max=config.vtrace_clip).cpu().tolist()
                        rolling["vtrace_c"].append(float(np.mean(_traces)))
                    episode = fill_nstep(episode, config.nstep, traces=_traces,
                                         floor=config.vtrace_floor)
                if annotator is not None:


                    annotation = annotator.finish(episode_index, episode_success)
                    episode = stamp_annotation(episode, annotation, config)
                    annotator.reset()
                    rolling["critical_transitions"].append(
                        float(sum(t["is_critical"] for t in episode))
                    )
                    rolling["suffix_fraction"].append(annotation.suffix_fraction)
                    rolling["has_catastrophe"].append(
                        float(annotation.catastrophic_frame is not None)
                    )
                    generated += len(pending)
                    retained += len(episode)
                    rolling["replay_removed_fraction"].append(
                        1.0 - len(episode) / max(len(pending), 1)
                    )
                for stamped in episode:
                    online.add(**stamped)
                pending = []
                pending_centres = []
                causal_scores = []

            if done:
                record = {
                    "kind": "episode",
                    "env_steps": env_steps,
                    "episode": episode_index,
                    "length": episode_steps,
                    "success": episode_success,
                }
                append_jsonl(metrics_path, record)
                if getattr(config, "dump_episodes", False) and ep_frames:


                    _imgs = [f for f, _ in ep_frames]
                    _prop = [p for _, p in ep_frames]
                    _last = transition.get("next_image")
                    if _last is not None:
                        _imgs = _imgs + [_last]
                        _prop = _prop + [np.asarray(next_observation["agent_pos"],
                                                    dtype=np.float32)]
                    _rdir = output_dir / "rollouts"
                    _rdir.mkdir(exist_ok=True)


                    np.savez_compressed(
                        _rdir / f"ep_{env_steps:07d}_{episode_index:05d}.npz",
                        frames=np.stack(_imgs),
                        proprio=np.stack(_prop),
                        success=np.array(episode_success),
                        env_steps=np.array(env_steps),
                        length=np.array(episode_steps))
                if rm_distiller is not None and ep_frames:


                    _eps_all = ep_frames
                    _last_im = transition.get("next_image")
                    if _last_im is not None:
                        _eps_all = ep_frames + [
                            (_last_im, np.asarray(next_observation["agent_pos"], dtype=np.float32))]


                    class _AV:
                        """Lazy agentview view over ep_frames."""
                        __slots__ = ("s",)
                        def __init__(self, src): self.s = src
                        def __len__(self): return len(self.s)
                        def __getitem__(self, i): return self.s[i][0][0]

                    def _encode(_clip, _props, idx, _eps=_eps_all, aug=False):
                        """Clip frames -> V's input space, via the SAME path V was fit on."""


                        if config.dvd_dynamics == "latent56":
                            from .latent56 import encode as _lz_enc, norm_prop as _lz_np
                            _im = (torch.as_tensor(np.stack([_eps[i][0] for i in idx]), device=device)
                                   .permute(0, 1, 4, 2, 3).float() / 255.0)
                            if aug:
                                raise RuntimeError("--value-distill-aug-views is not implemented for "
                                                   "the latent world model")
                            with torch.no_grad():
                                _z = _lz_enc(_l56_dino, _l56_enc_net, _im, _l56_cfg, device).float()
                            _pn = torch.as_tensor(_lz_np(np.stack(_props)[:, :9], _l56_cfg),
                                                  device=device, dtype=_z.dtype)
                            return torch.cat([_z, _pn], dim=-1)
                        if getattr(config, "dvd_on_encoder", False):


                            if aug:
                                raise RuntimeError("--value-distill-aug-views is not implemented "
                                                   "for --dvd-on-encoder")
                            if encoder_optimizer is not None:


                                return (np.stack([_eps[i][0] for i in idx]).astype(np.uint8),
                                        scaler(np.stack(_props))[:, :PIXEL_PROPRIO].astype(np.float32))
                            _im = torch.as_tensor(np.stack([_eps[i][0] for i in idx]), device=device)
                            _pr = torch.as_tensor(scaler(np.stack(_props))[:, :PIXEL_PROPRIO],
                                                  device=device)
                            with torch.no_grad():
                                return encoder(_im, _pr, augment=False)
                        if dvd_term is None or not getattr(dvd_term, "feature_space", False):
                            raise RuntimeError(
                                "--value-distill-robometer needs a FEATURE-SPACE V (the vf/vfpix arms). "
                                "A state-space V takes the 37-dim scaled state, which this path does "
                                "not build.")
                        both = torch.as_tensor(np.stack([_eps[i][0] for i in idx]), device=device)
                        if aug:


                            from .pixel import random_shift as _rs
                            _b, _v, _h, _w = both.shape[:4]
                            _x = both.permute(0, 1, 4, 2, 3).reshape(_b * _v, 3, _h, _w).float()
                            _sp = int(getattr(config, "value_distill_shift_pad", 0))
                            _ph = float(getattr(config, "value_distill_photometric", 0.0))
                            if _sp > 0:
                                _x = _rs(_x, _sp)
                            if _ph > 0:
                                _g = 1.0 + (torch.rand(_b * _v, 1, 1, 1, device=_x.device) * 2 - 1) * _ph
                                _o = (torch.rand(_b * _v, 1, 1, 1, device=_x.device) * 2 - 1) * _ph * 0.5 * 255
                                _x = (_x - 127.5) * _g + 127.5 + _o
                            both = (_x.round().clamp_(0, 255).to(torch.uint8)
                                    .reshape(_b, _v, 3, _h, _w).permute(0, 1, 3, 4, 2).contiguous())
                        feat = dino.encode_frames(dvd_term.dynamics, both)


                        pooled = _value_pool_of(dvd_term)(feat, dvd_term.value_pool_mode)
                        pd = int(getattr(dvd_term, "proprio_dim", 0) or 0)
                        if not pd:
                            return pooled
                        pr = torch.as_tensor(np.stack(_props)[:, :pd], device=device, dtype=pooled.dtype)
                        return torch.cat([pooled, pr], dim=-1)


                    def _encode_pred(idx, _eps=_eps_all, _acts=list(ep_actions)):
                        """PREDICTED V inputs for the scored frames (--value-distill-pred)."""
                        pos = [k for k, i in enumerate(idx) if 1 <= i <= len(_acts)]
                        if not pos:
                            return np.zeros((0, 1), np.float32), []
                        ci_all = [idx[k] - 1 for k in pos]
                        out = []
                        with torch.no_grad():
                            for s0 in range(0, len(ci_all), 32):
                                ci = ci_all[s0:s0 + 32]
                                pi = [max(i - 1, 0) for i in ci]
                                _cur = dino.encode_frames(dino_adapter, torch.as_tensor(
                                    np.stack([_eps[i][0] for i in ci]), device=device))
                                _prv = dino.encode_frames(dino_adapter, torch.as_tensor(
                                    np.stack([_eps[i][0] for i in pi]), device=device))
                                dino.set_context(dino_adapter, _prv, _cur)
                                _st = torch.as_tensor(np.stack([_eps[i][1][:9] for i in ci]), device=device)
                                _ac = torch.as_tensor(np.stack([_acts[i] for i in ci]), device=device)
                                _pf = dino_adapter(_st, _ac)
                                out.append(torch.cat(
                                    [_value_pool_of(dvd_term)(_pf, dvd_term.value_pool_mode),
                                     dino_adapter.last_state_pred[:, :dvd_term.proprio_dim].float()], dim=-1))
                        return torch.cat(out), pos

                    rm_distiller.add_episode(_AV(_eps_all),
                                             [p for _, p in _eps_all], _encode,
                                             success=bool(episode_success),
                                             pred_fn=(_encode_pred if config.value_distill_pred else None))
                    _st = rm_distiller.stats(probe_n=0)
                    if int(_st["rm_distill_episodes"]) % 10 == 0:
                        append_jsonl(metrics_path, {"kind": "rm_distill", "env_steps": env_steps, **_st})
                if getattr(config, "dump_episodes", False) or rm_distiller is not None:
                    ep_frames = []
                    ep_actions = []
                rolling["episode_success"].append(float(episode_success))
                rolling["episode_length"].append(float(episode_steps))
                episode_index += 1
                episode_steps = 0
                episode_success = False
                config._ep_bias = None
                if annotator is not None:


                    annotator.reset()
                observation, _ = env.reset(seed=config.seed + episode_index)
                if seg_hist is not None:
                    seg_hist.reset()
                if seg_term is not None:
                    seg_term.reset()
                if dino_history is not None:


                    dino_history.reset()
                if reward_model is not None:
                    reward_model.reset()
                    reward_model.observe(observation)
                    previous_score = None
            else:
                observation = next_observation

            if (getattr(config, "value_reload", "") and value_fn is not None
                    and env_steps // 2000 > (env_steps - primitive_steps) // 2000):


                _stamp = Path(config.value_reload) / "STAMP"
                if _stamp.exists() and _stamp.stat().st_mtime > v_reload_mtime:
                    try:
                        _new = ValueEnsemble(
                            config.value_reload, config.value_variant, scaler,
                            state_dim=state_dim, device=device, finetune=False,
                            aggregate=config.value_aggregate,
                            lcb_lambda=config.value_lcb_lambda)
                        value_fn.load_state_dict(_new.state_dict())
                        v_reload_mtime = _stamp.stat().st_mtime
                        del _new
                        print(f"[value-reload] hot-swapped V from {config.value_reload} "
                              f"at env_steps={env_steps}", flush=True)
                    except Exception as _e:


                        print(f"[value-reload] failed ({_e}); keeping current V", flush=True)
            if env_steps // 500 > (env_steps - primitive_steps) // 500:
                log_values = {
                    key: float(np.mean(value)) for key, value in rolling.items() if value
                }
                log_values.update(
                    {
                        "env_steps": env_steps,
                        "updates": updates,
                        "buffer_size": online.size,
                        "elapsed_s": time.perf_counter() - started,


                        "update_s_frac": update_s_window / max(time.perf_counter() - window_t0, 1e-9),
                        "window_s": time.perf_counter() - window_t0,
                        **{f"t_{_k}": _v for _k, _v in _SECT.items()},
                    }
                )
                if annotator is not None:
                    log_values.update(
                        {


                            "replay_generated": generated,
                            "replay_retained": retained,
                            "replay_removed": generated - retained,
                            "replay_removed_fraction_total":
                                (generated - retained) / max(generated, 1),
                            "buffer_critical_size": online.critical_size,
                            "buffer_critical_fraction":
                                online.critical_size / max(online.size, 1),
                        }
                    )


                log_values.update(value_stats.summary("v_online"))


                if rm_distiller is not None:
                    log_values.update(rm_distiller.stats())
                append_jsonl(metrics_path, {"kind": "train", **log_values})
                wandb.log({f"train/{key}": value for key, value in log_values.items()}, step=env_steps)
                print(json.dumps(log_values), flush=True)
                rolling.clear()
                update_s_window = 0.0
                _SECT.clear()
                window_t0 = time.perf_counter()

            if env_steps - last_eval >= config.eval_freq or env_steps >= config.total_env_steps:
                run_eval(env_steps)
                last_eval = env_steps

            if (grad_probe is not None
                    and env_steps - last_grad_probe >= config.grad_probe_freq):
                last_grad_probe = env_steps

                gp = grad_probe.measure(value_dual if value_dual is not None else value_fn,
                                        dvd_term)
                if gp:
                    wandb.log({**gp, "env_steps": env_steps})
                    print(f"[grad-probe] {env_steps} excess {gp['grad_ref_excess']:+.4f} "
                          f"CALIBRATED {gp['grad_ref_calibrated']:+.3f} | "
                          f"4x4 excess {gp['grad_ref_excess_4x4']:+.4f} "
                          f"CALIBRATED {gp['grad_ref_calibrated_4x4']:+.3f}   "
                          f"(1.0 = matches the s42/s43 seed control)", flush=True)

            if env_steps - last_save >= config.save_freq or env_steps >= config.total_env_steps:
                step = min(env_steps, config.total_env_steps)
                if config.save_replay:


                    online.save(output_dir / f"replay/{step:06d}.npz")
                save_checkpoint(
                    output_dir / f"checkpoints/{step:06d}.pt",
                    config,
                    actor,
                    critics,
                    target_critics,
                    env_steps,
                    updates,


                    {"log_alpha": float(log_alpha.detach()),
                     **({"seg_obs_enc": _LATENT_OBS_CTX["seg_enc"].state_dict()}
                        if getattr(config, "seg_obs", False) else {}),
                     **({"latent_enc": _LATENT_OBS_CTX["enc"].state_dict()}
                        if getattr(config, "latent_obs_finetune", False) else {}),
                     **({"seg_head": _LATENT_OBS_CTX["seg_head"].state_dict()}
                        if getattr(config, "seg_obs_finetune", False) else {}),


                     **({"value_fn": value_fn.state_dict()}
                        if value_fn is not None
                        and (value_fn.finetune or rm_distiller is not None) else {}),


                     **({"attn_pool": attn_pool.state_dict()}
                        if attn_pool is not None else {}),


                     **({"obs_pool": obs_pool.state_dict()}
                        if obs_pool is not None else {}),

                     **({"obs_pool_target": obs_pool_target.state_dict()}
                        if obs_pool_target is not None else {}),


                     **({"actor_pool": actor_pool.state_dict()}
                        if actor_pool is not None else {}),
                     **({"sargmax_pool": sargmax_pool.state_dict()}
                        if sargmax_pool is not None else {}),


                     **({"dvd_dynamics": {k: v for k, v in dvd_term.dynamics.state_dict().items()
                                          if not k.startswith("wm.dino.")}}
                        if dvd_term is not None and dvd_term.finetune_dynamics else {}),


                     **({"encoder": encoder.state_dict()} if encoder is not None else {}),


                     **({"actor_encoder": actor_encoder.state_dict()}
                        if actor_encoder is not None else {}),


                     **({"target_encoder": target_encoder.state_dict()}
                        if target_encoder is not None else {}),


                     "opt": {
                         "actor": actor_optimizer.state_dict(),
                         "critic": critic_optimizer.state_dict(),
                         "alpha": alpha_optimizer.state_dict(),
                         **({"value": value_optimizer.state_dict()}
                            if value_optimizer is not None else {}),
                         **({"value_td": value_td_optimizer.state_dict()}
                            if value_td_optimizer is not None else {}),
                         **({"dvd": dvd_term.optimiser.state_dict()}
                            if dvd_term is not None and dvd_term.optimiser is not None else {}),
                     },


                     **({"value_target": getattr(value_fn, "_target").state_dict()}
                        if value_fn is not None and getattr(value_fn, "_target", None) is not None
                        else {}),


                     **({"value_td": value_td.state_dict(),
                         "value_dual": value_dual.scale_state_dict()}
                        if value_td is not None else {}),
                     **({"value_td_target": getattr(value_td, "_target").state_dict()}
                        if value_td is not None and getattr(value_td, "_target", None) is not None
                        else {})},
                )
                last_save = env_steps
    finally:
        env.close()
        prior.close()
        wandb.finish()


def _dvd_actor_rows(args, deterministic_residual: bool, rows: int) -> int:
    """Rows the actor's DVD read gets on the dino_wm DPG path, EXACTLY as update_dice builds it."""
    _ak = args.multi_sample_k if args.actor_sample_k is None else args.actor_sample_k
    if _ak > 1:
        k = _ak
    elif args.multi_sample_k > 1:
        k = args.multi_sample_k
    else:
        k = 1 if (deterministic_residual and not args.latent_dim) else 4
    total = args.batch_size * k
    if args.dvd_phase_gate or rows <= 0 or rows >= total:
        return total
    if args.dvd_rows_grouped and k > 1:
        return min(max(rows // k, 1), total // k) * k
    return rows


def _check_value_td_args(parser, args, deterministic_residual: bool) -> None:
    """Parse-time refusals for --value-td-model, before any run dir exists."""
    from .value_model import SEEDS as _V_SEEDS

    td = args.value_td_model
    if not args.value_model:
        parser.error("--value-td-model needs --value-model: the td V is the SECOND ensemble, "
                     "combined with the primary one")
    if args.dvd_dynamics != "dino_wm":
        parser.error(f"--value-td-model needs --dvd-dynamics dino_wm for now (got "
                     f"{args.dvd_dynamics!r}): the latent56 / seghead V wrappers re-attach their "
                     "inputs differently and are not wired for a second ensemble")
    if args.value_random_frozen:
        parser.error("--value-td-model with --value-random-frozen: the random-V control must not "
                     "have a trained second V read alongside it")
    if args.value_scratch:
        parser.error("--value-td-model with --value-scratch: the td V is defined by its offline "
                     "initialisation; a scratch primary would leave nothing offline in the pair")
    if args.dino_pool in ("attnpool", "xattn", "mqattn"):
        parser.error(f"--value-td-model with --dino-pool {args.dino_pool}: the attention pooling "
                     "is the PRIMARY's trained readout, so the td V would read features pooled by "
                     "another V's queries")
    if args.gated_training:
        parser.error("--value-td-model with --gated-training: the gate multiplies the primary's "
                     "optimiser only, so the td V would keep training through the 'dyn' phase")
    if args.nstep > 1:
        parser.error(f"--value-td-model with --nstep {args.nstep}: the dino_wm V fit batch encodes "
                     "next_image (always the t+8 frame; fill_nstep never touches images) while "
                     "reward/discount/done are n-step, so the td V would bootstrap a 1-step "
                     "successor against an n-step target")
    if args.dvd_weight <= 0.0:
        parser.error("--value-td-model needs --dvd-weight > 0: the DualValue scales learn only "
                     "from the actor's DVD read, so without one they never initialise")
    if args.dvd_weight_ramp_end > 0 and args.dvd_weight_late <= 0.0:
        parser.error(f"--value-td-model with --dvd-weight-ramp-end and --dvd-weight-late "
                     f"{args.dvd_weight_late}: once the ramp reaches <= 0 the actor's DVD read "
                     "stops, and with it every DualValue scale update, while the td V keeps "
                     "TD-training")
    reads = [("--dvd-rows", args.dvd_rows)]
    if args.dvd_rows_switch:
        reads.append(("--dvd-rows-late", args.dvd_rows_late))
    for flag, rows in reads:
        eff = _dvd_actor_rows(args, deterministic_residual, rows)
        if eff < DualValue.MIN_ROWS:
            parser.error(
                f"--value-td-model with {flag} {rows}: the actor's DVD read would have {eff} rows "
                f"(batch {args.batch_size}, K per state as update_dice resolves it"
                f"{', grouped' if args.dvd_rows_grouped else ''}), and the DualValue scales only "
                f"learn from a >= {DualValue.MIN_ROWS}-row actor read -- they would "
                f"{'never initialise' if flag == '--dvd-rows' else 'freeze at the switch'}")

    def _json(path: Path, what: str) -> dict:
        try:
            out = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            parser.error(f"--value-td-model: cannot read the {what} {path}: {e}")
        if not isinstance(out, dict):
            parser.error(f"--value-td-model: the {what} {path} is not a JSON object")
        return out

    tdp, prp = Path(td), Path(args.value_model)
    if not (tdp / "config.json").is_file():
        parser.error(f"--value-td-model {td}: no config.json there (not a V dir)")
    if not (tdp / "state_scaler.json").is_file():
        parser.error(f"--value-td-model {td}: no state_scaler.json")
    if not (prp / "config.json").is_file() or not (prp / "state_scaler.json").is_file():
        parser.error(f"--value-td-model needs the primary --value-model {args.value_model} to be a "
                     "V dir with config.json and state_scaler.json, to check the two input spaces "
                     "match")
    cfg, pcfg = _json(tdp / "config.json", "td V config"), _json(prp / "config.json", "primary V config")
    try:
        _sd = cfg.get("seeds", _V_SEEDS)
        seeds = range(int(_sd)) if isinstance(_sd, (int, float, str)) else tuple(_sd)
        td_in = int(cfg.get("input_dim", cfg["state_dim"]))
        pr_in = int(pcfg.get("input_dim", pcfg["state_dim"]))
        td_sd, pr_sd = int(cfg["state_dim"]), int(pcfg["state_dim"])
    except (KeyError, TypeError, ValueError) as e:
        parser.error(f"--value-td-model: malformed V config.json ({type(e).__name__}: {e}); "
                     "both dirs need state_dim (and input_dim for a feature-space V)")
    missing = [str(tdp / "weights" / f"{args.value_variant}_seed{s}.pt") for s in seeds
               if not (tdp / "weights" / f"{args.value_variant}_seed{s}.pt").is_file()]
    if not seeds or missing:
        parser.error(f"--value-td-model {td} has no weights for --value-variant "
                     f"{args.value_variant}: missing {missing or 'all seeds'}")
    if td_in != pr_in:
        parser.error(f"--value-td-model input_dim={td_in} but --value-model input_dim={pr_in}: "
                     "the two ensembles are evaluated on the SAME vector")
    if td_sd != pr_sd:
        parser.error(f"--value-td-model state_dim={td_sd} but --value-model state_dim={pr_sd}: "
                     "both are built against the same task StateScaler")


    for key in ("input", "world_model", "task"):
        if key in cfg and key in pcfg and cfg[key] != pcfg[key]:
            parser.error(f"--value-td-model {key}={cfg[key]!r} but --value-model {key}="
                         f"{pcfg[key]!r}: the two ensembles must read the same input space")
    tsc = _json(tdp / "state_scaler.json", "td V scaler")
    psc = _json(prp / "state_scaler.json", "primary V scaler")
    for key in ("q01", "q99"):
        try:
            a_, b_ = (np.asarray(tsc[key], dtype=np.float32), np.asarray(psc[key], dtype=np.float32))
        except (KeyError, TypeError, ValueError) as e:
            parser.error(f"--value-td-model: state_scaler.json without a usable {key} ({e})")
        if a_.shape != b_.shape or not np.allclose(a_, b_, atol=0.0):
            parser.error(f"--value-td-model state_scaler.json {key} differs from the primary's: "
                         "ValueEnsemble would refuse it inside train(), after the run dir exists")


def parse_args() -> TrainConfig:


    from .dino_wm import FLOW_STEPS as _FLOW_STEPS, FLOW_INVERSE_ITERS as _FLOW_ITERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dice_rl", "policy_decorator", "flowdpg"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-output", type=Path)
    parser.add_argument("--wandb-project", default="mimicgen_online_rl")
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=10,
        help="Euler integration steps for the flow prior (denoising steps for a diffusion prior). "
        "The coffee/mugcleanup prior checkpoints declare num_integration_steps=100; this study has "
        "always run them at 10, so every baseline is measured against a coarsely integrated prior",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-env-steps", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-freq", type=int, help="defaults to --eval-freq")
    parser.add_argument(
        "--progressive-exploration-steps",
        type=int,
        help="residual exploration ramp; defaults to a quarter of --total-env-steps",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=-0.01,
        help="reward per primitive action in a chunk that does not reach success; 0 for sparse",
    )
    parser.add_argument(
        "--eval-video-episodes",
        type=int,
        default=1,
        help="number of eval episodes recorded and logged to W&B per evaluation; 0 disables",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.10,
        help="kappa in a = clamp(base + kappa * residual); 1.0 matches upstream dice-rl",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount PER PRIMITIVE STEP: transitions store gamma**primitive_steps and the DVD term uses "
        "gamma**action_steps. 0.99 is the campaign default. On long tasks only gamma^L of the success reward "
        "reaches the episode start (hammer_cleanup_d1, ~273-step successes: 0.99 -> 0.064, 0.995 -> 0.25).",
    )
    parser.add_argument(
        "--bc-loss-weight",
        type=float,
        default=100.0,
        help="weight on mean(residual^2). Note this penalizes the PRE-scale residual, so the "
        "effective action-space weight is --bc-loss-weight / --residual-scale^2",
    )
    parser.add_argument(
        "--bc-filter-uncertainty",
        type=float,
        default=0.0,
        help="release the BC anchor only where the residual's advantage over the prior exceeds "
        "this many ensemble standard errors of that advantage. 0 = dice-rl's bare advantage>0",
    )
    parser.add_argument(
        "--reward-model-causal",
        action="store_true",
        help="score each chunk from frames up to that chunk only, instead of re-scoring the "
        "episode with a bidirectional window at flush. Less accurate, but Markovian and therefore "
        "representable by the critic",
    )
    parser.add_argument(
        "--reward-shaping-form",
        choices=("additive", "potential"),
        default="additive",
        help="how the learned reward enters. 'additive' adds w*score to every chunk and CHANGES "
        "the optimal policy; 'potential' uses w*(discount*Phi(s') - Phi(s)), which provably does "
        "not, and telescopes so Q does not grow with episode length",
    )
    parser.add_argument(
        "--sparse-reward-weight",
        type=float,
        default=1.0,
        help="scale on the environment's own reward. 0.0 trains on the reward model ALONE, which "
        "is the reward-hacking test: success is still logged but no longer optimised",
    )
    parser.add_argument(
        "--residual-head",
        choices=("mlp", "conv"),
        default="mlp",
        help="residual actor head. mlp = one flat Linear over the whole chunk (the default every "
        "arm before CV1 used); conv = Conv1d stack over the chunk's TIME axis",
    )
    parser.add_argument(
        "--residual-conv-kernel",
        type=int,
        default=3,
        help="temporal kernel for --residual-head conv. 1 gives weight sharing across timesteps "
        "with no temporal coupling, which is what isolates sharing from locality",
    )
    parser.add_argument(
        "--residual-conv-channels",
        type=int,
        default=16,
        help="hidden channels in the conv head",
    )
    parser.add_argument(
        "--residual-conv-broadcast",
        action="store_true",
        help="fully shared conv head: project to ONE channel vector for the whole chunk "
        "and repeat it, so only the learned positional bias distinguishes timesteps",
    )
    parser.add_argument(
        "--init-actor-from",
        default="",
        help="path to a checkpoint whose ACTOR weights warm-start this run (e.g. an offline "
             "actor-pretraining run). Loads the actor only; the critic is untouched unless "
             "--init-critic-from is also given.",
    )
    parser.add_argument(
        "--init-critic-from",
        default="",
        help="path to a finished run's .pt checkpoint; loads its critic ensemble and targets "
        "while the residual actor starts from scratch",
    )
    parser.add_argument(
        "--freeze-init-critic",
        action="store_true",
        help="hold the warm-started critic fixed instead of continuing to train it",
    )
    parser.add_argument(
        "--initial-log-std",
        type=float,
        help="actor log_std bias at init; defaults to -3.0 for dice_rl, -1.5 for policy_decorator",
    )
    parser.add_argument(
        "--log-std-min",
        type=float,
        default=-5.0,
        help="hard floor on the actor's log_std; raise it to stop the residual collapsing below "
        "the critic's action resolution",
    )
    parser.add_argument(
        "--bc-filter",
        action="store_true",
        help="gate the BC/contraction term by Q-advantage over the prior action, after dice-rl's "
        "filtered_bc_loss; also enables end-of-episode buffer flushing so mc_return is available",
    )
    parser.add_argument(
        "--bc-filter-warmup-updates",
        type=int,
        default=5_000,
        help="updates before the filter engages. Upstream uses 25k of its own update budget; "
        "this run does ~37.5k total (one update per 8-step chunk), so 25k would leave the "
        "filter active for only a third of training",
    )
    parser.add_argument(
        "--bc-filter-soft",
        action="store_true",
        help="additionally require Q - mc_return < --q-underestimation-threshold before "
        "releasing the BC anchor",
    )
    parser.add_argument("--q-underestimation-threshold", type=float, default=0.0)
    parser.add_argument(
        "--no-retain-bc-for-expert",
        dest="retain_bc_for_expert",
        action="store_false",
        help="let the filter release the BC anchor on expert transitions too",
    )


    parser.add_argument(
        "--stochastic-residual",
        dest="deterministic_residual",
        action="store_false",
        help="keep the actor's Gaussian head. On dice_rl this "
        "makes the contraction term penalize mean^2 + sigma^2, which collapses sigma onto "
        "--log-std-min; the sigma-related flags only do anything in this mode",
    )
    parser.add_argument(
        "--target-actor",
        action="store_true",
        help="use a Polyak copy of the actor to pick the next action in the TD target. Without "
        "it the target is Q'(s', pi_theta(s')) with the LIVE actor, so it moves every actor step "
        "-- the DDPG failure mode. This policy is DETERMINISTIC, so SAC's reason for omitting a "
        "target actor (resampling a stochastic policy) does not apply here",
    )
    parser.add_argument(
        "--value-residual-cond",
        action="store_true",
        help="dV(s, V_frozen(s)) instead of dV(s). Adds no information but makes corrections "
        "relative to the base cheap -- and cancelling the base equally cheap, so it anchors far "
        "more weakly. A LEARNED RECALIBRATION of the pretrained value rather than a residual on "
        "it. Separate arm, not an upgrade",
    )
    parser.add_argument(
        "--value-residual-unfrozen",
        action="store_true",
        help="CAPACITY CONTROL for --value-residual: identical dV architecture and parameter "
        "count, but the pretrained ensemble is ALSO trained. The only difference from "
        "--value-residual is whether the pretrained landscape is preserved, so a gap between "
        "the two isolates the residual STRUCTURE from the extra capacity",
    )
    parser.add_argument(
        "--value-residual",
        action="store_true",
        help="V(s) = V_frozen(s) + dV_theta(s): freeze the pretrained value and learn only the "
        "correction. Online TD currently OVERWRITES the offline fit -- on threading, V initialised "
        "from the MC fit and from the RoboMeter distillation converge to functions correlating "
        "r=+0.862 with each other, higher than either does with its own init (+0.695 / +0.526). "
        "dV is zero-initialised, so step 0 is bit-identical to the frozen V",
    )
    parser.add_argument(
        "--bootstrap-timeouts",
        action="store_true",
        help="store an episode that hits the step limit WITHOUT success as a TRUNCATION: done=0, so V (and Q) bootstrap "
        "from the final observation instead of treating the timeout as a zero-value terminal, and the next-state prior "
        "draws are sampled at that observation instead of zero-filled. Default off (historical: timeouts are terminals). "
        "Matters in proportion to the timeout share: ~15%% of episodes on square/threading, ~57%% on hammer",
    )
    parser.add_argument(
        "--value-loss",
        choices=("smooth_l1", "mse"),
        default="smooth_l1",
        help="regression loss for V's online TD step AND the RoboMeter distillation step. smooth_l1 (Huber, beta=1) is the "
        "historical default, inherited from the first commit and never ablated; mse tests whether it matters",
    )
    parser.add_argument(
        "--value-target-tau",
        type=float,
        default=0.0,
        help="Polyak rate for a TARGET network on V (0 = off, the historical behaviour). V "
        "currently bootstraps from itself while the critic has had a target network all along",
    )
    parser.add_argument(
        "--value-td-model",
        default="",
        help="SECOND value ensemble (a V dir like --value-model: config.json, state_scaler.json, "
        "weights/<variant>_seed*.pt), loaded with the primary's variant/scaler/aggregate and "
        "TD-finetuned online on the primary's v_fit_batch with its own Adam "
        "(--value-finetune-lr), gate (--value-finetune-after) and Polyak target "
        "(--value-target-tau). Everything the policy reads V through (DVD objective, best-of-N, "
        "dvd_advantage, value-aux target) then reads DualValue(primary, td), see "
        "--value-dual-combine. The primary is unchanged: its own TD fit still needs "
        "--value-finetune and the RoboMeter distiller still trains it alone. Empty = off. "
        "--dvd-dynamics dino_wm only",
    )
    parser.add_argument(
        "--value-dual-combine",
        choices=("min", "mean", "max"),
        default=None,
        help="how DualValue combines the primary and --value-td-model V: each is divided by its "
        "own EMA of mean|V| (momentum 0.99, updated once per update from the actor's >= 64-row "
        "DVD read), combined per row, and rescaled to the primary's level. 'mean' blends the two "
        "(each member's pull set by its normalised spread). 'min'/'max' are per-row REGION GATES, "
        "not a pessimistic/optimistic estimate: mean|V| normalises scale but not shift, so the "
        "member that is lower (higher) on a row wins it outright -- an MC-return V that sits near "
        "0 on failure-like states takes nearly all of them under min -- and dvd_value is biased "
        "low (high) against single-V arms. Default min. Needs --value-td-model",
    )
    parser.add_argument(
        "--actor-q-reduce",
        default="min",
        help="ensemble reduction the ACTOR ascends. 'min' (default) is the historical behaviour, "
        "but d/da of a min is the gradient of the argmin head ALONE, and the argmin switches on "
        "77%% of states between checkpoints -- measured cos(grad,grad) +0.165 for min vs +0.861 "
        "for mean, while the critics themselves correlate r=+0.966. 'lcb2' keeps the pessimism "
        "(mean - 2*std) while every head contributes to the gradient. The TD TARGET keeps min "
        "regardless: pessimism matters there and no gradient flows through it",
    )
    parser.add_argument(
        "--encoder-value-aux", type=float, default=0.0,
        help="weight on (V_head(enc(s)) - V_frozen(dino(s)))^2, added to the ACTOR loss (0 = off). "
             "Supervises the actor's encoder to predict the FROZEN value function, whose input "
             "pathway the actor cannot influence -- so unlike better_than_prior this target cannot "
             "be gamed by reshaping features. Costs one DINOv2 encode per batch.")
    parser.add_argument(
        "--encoder-drift", type=float, default=0.0,
        help="weight on ||enc(x) - target_enc(x)||^2 added to the ACTOR loss (0 = off). Only "
             "active with --actor-train-encoder, where the actor's gradient reaches the encoder "
             "and can inflate V by reshaping the representation rather than the action. Under "
             "--no-train-critic the target encoder never EMA-updates, so it is a fixed reference "
             "and this is a pure drift bound.")
    parser.add_argument(
        "--vtrace-sigma", type=float, default=0.0,
        help="V-trace-style trace cutting on the n-step target (0 = off, the uncorrected return). "
             "Width of the Gaussian that softens the DETERMINISTIC policy into a density, so an "
             "importance coefficient exists at all: c = exp(-||a_logged - a_now||^2/2 sigma^2). "
             "The n-step window stops extending once the running product of c falls below "
             "--vtrace-floor, bootstrapping there. Needs --nstep > 1 to do anything.")
    parser.add_argument(
        "--vtrace-clip", type=float, default=1.0,
        help="c_bar, the ceiling on each trace coefficient (IMPALA's clipping).")
    parser.add_argument(
        "--vtrace-floor", type=float, default=0.5,
        help="stop extending the n-step window once the running trace product drops below this. "
             "1.0 forces 1-step; 0.0 never truncates and reproduces the uncorrected return.")
    parser.add_argument(
        "--seg-terminate-patience", type=int, default=0,
        help="end a TRAINING episode once the needle->tripod offset has failed to approach "
             "the offline segmentation's endpoint reference for N primitive steps "
             "(0 = off). A CAUSAL APPROXIMATION of the offline rule, which is an argmin over the "
             "whole remaining trajectory and cannot be evaluated online. Calibrated offline on "
             "threading_d0: at 16 it fires a median 16 frames late "
             "and keeps 31.5%% of the episode vs the offline rule's 38.5%%. Stored as a "
             "TRUNCATION, never a terminal. Evaluation rollouts are unaffected.")
    parser.add_argument(
        "--encoder-value-aux-only", action="store_true",
        help="train the actor's pixel encoder with L_V ALONE. actor_state stays detached, so "
             "L_DVD and L_BC shape the residual head only; the encoder is moved into the actor "
             "optimiser and receives gradient exclusively from the value-prediction aux. "
             "Requires --encoder-value-aux > 0 and --no-train-critic (with a trained critic the "
             "critic's backward frees the encoder graph before the actor's), and is mutually "
             "exclusive with --actor-train-encoder, which would re-attach the DVD/BC path.")
    parser.add_argument(
        "--qdvd-normalise", choices=("meanabs", "zscore"), default="meanabs",
        help="how the two estimators are put on a common footing before --qdvd-combine compares "
             "them. 'meanabs' (default, the original) divides each by its own mean-abs, which "
             "equalises magnitude but NOT location -- V is non-negative so its normalised mean "
             "is pinned at +1.0 while Q's sits in [-1,+1], and the min/max then selects on that "
             "offset rather than on which action is better. 'zscore' centres and scales each "
             "side over the same rows, so win_frac sits near 0.5 and the arbitration rule is "
             "the only thing that differs between min/max/mean.")
    parser.add_argument(
        "--encoder-value-aux-head", choices=("mlp", "linear"), default="mlp",
        help="capacity of the --encoder-value-aux probe. 'mlp' (default, 265->256->1) can decode "
             "V from a tangled representation, so the loss collapses without the encoder having "
             "changed. 'linear' forces the encoder itself to make V linearly decodable.")
    parser.add_argument(
        "--qdvd-combine", choices=("off", "min", "max", "mean"), default="off",
        help="COMBINE the two value estimators PER ACTION CANDIDATE instead of mixing their "
             "batch means with --dvd-weight. Both are first put on their own normalised value "
             "scale (q_pi/q_scale and gamma^H V/|.|), then reduced elementwise over the K "
             "candidates of --multi-sample-k: 'min' is the conservative combination (the "
             "clipped-double-Q argument: two estimates of the SAME return, so the max of them "
             "is an overestimator), 'max' is its control, 'mean' the reference. When set, this "
             "REPLACES the (1-w)Q + w*DVD blend and --dvd-weight no longer selects the mixture. "
             "Logs qdvd_dvd_win_frac plus each side's per-row std, because a max/min over two "
             "quantities normalised only in aggregate selects on VARIANCE, not accuracy.")
    parser.add_argument(
        "--actor-k-reduce", choices=("mean", "max", "min"), default="mean",
        help="how the actor loss reduces over the K=--multi-sample-k action candidates per state "
             "BEFORE averaging over states. 'mean' (default) ascends the expected value of a "
             "prior draw; 'max' ascends the best of the K, the sampled-max improvement operator. "
             "'max' needs --multi-sample-k > 1 to differ, and is refused alongside --q-rows or "
             "--disable-q-loss-for-expert, both of which reindex q_pi and destroy the (B, K) "
             "layout the reshape depends on.")
    parser.add_argument(
        "--bon-reduce",
        choices=("min", "max", "mean"),
        default="min",
        help="ensemble reduction used ONLY to rank best-of-N candidates. min (default) ranks by "
        "worst-case, so a candidate one head dislikes is demoted even when the other nine agree "
        "it is best; max ranks optimistically, mean by ensemble expectation. The actor loss and "
        "the TD target keep the pessimistic min regardless -- this changes which of the N sampled "
        "candidates is EXECUTED, not what is learned",
    )
    parser.add_argument(
        "--replay-capacity", type=int, default=100_000,
        help="Online replay slots. The buffer is PREALLOCATED at this size, so under --pixel-obs "
             "it dominates the memory request (~602 KB/row = 60 GB at 100k). Short diagnostic "
             "runs can drop it and schedule on a fraction of the RAM.",
    )
    parser.add_argument(
        "--replay-episode-uniform",
        action="store_true",
        help="draw online replay with probability proportional to 1/episode_length, so every "
        "episode contributes equally instead of being weighted by how long it ran. Failures run "
        "the full horizon and successes terminate on the reward, so uniform-over-transitions "
        "under-represents successes: measured on square_d1, 48.5%% of EPISODES succeed but only "
        "30.1%% of ROWS come from one. Does not touch the expert buffer, which is sampled "
        "separately",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=1,
        help="n-step TD target. Each chunk already spans action_steps primitive steps, so n=5 "
        "propagates reward back ~40 primitive steps per update. Uncorrected (no importance "
        "weighting), so it is biased off-policy -- the standard DrQ-v2/RLPD trade",
    )
    parser.add_argument(
        "--env-dense-reward",
        action="store_true",
        help="build the env with robosuite reward_shaping=True and use ITS reward instead of the "
        "1.0-on-success value. The env reward was already being computed and discarded; this "
        "makes it the signal. EVALUATION keeps the sparse env, so success rates stay comparable",
    )
    parser.add_argument(
        "--reward-model",
        default=None,
        help="SARM/FARM checkpoint directory. The stored reward becomes "
        "sparse + --reward-model-weight * SARM(t), where SARM(t) is re-scored at episode end "
        "with the true bidirectional window (causal fill for the final 80 frames)",
    )
    parser.add_argument(
        "--reward-model-weight",
        type=float,
        default=0.02,
        help="weight on the learned reward. Small by default: SARM returns ~[0,1] per scored "
        "chunk and an episode has ~50 chunks, so an unweighted term is ~50x the sparse bonus",
    )
    parser.add_argument(
        "--terminate-on-reward-drop",
        type=float,
        default=0.0,
        help="end the episode when the causal SARM score falls by more than this in one chunk. "
        "Meaningful with a FARM checkpoint, whose hybrid failure-clip training makes the score "
        "decrease on a failing trajectory by construction. 0 disables",
    )
    parser.add_argument(
        "--value-model",
        default=None,
        help="directory of the offline value model (weights/, config.json, "
        "state_scaler.json), e.g. $CKPT_ROOT/value/<V dir>. Omit for the no-V control",
    )
    parser.add_argument(
        "--value-variant",
        default="v_relabel_hinge",
        choices=("v_td", "v_relabel_hinge"),
        help="which fit to use; the three seeds of it are always averaged",
    )
    parser.add_argument(
        "--value-intervention",
        default="none",
        choices=("none", "floor", "penalty", "shaping", "filter", "dueling"),
        help="floor: target = max(target, discount*V(s')). penalty: critic loss gains "
        "beta*relu(V(s) - Q(s, a_base)). shaping: potential-based reward shaping with Phi = V. "
        "filter: V(s) replaces mc_return in the soft BC filter. dueling: Q = V_frozen(s) + "
        "A(s,a), learning only the advantage",
    )
    parser.add_argument(
        "--value-penalty-beta", type=float, default=0.0,
        help="beta for --value-intervention penalty",
    )
    parser.add_argument(
        "--residual-max-norm",
        type=float,
        default=0.0,
        help="project the residual onto a ball of this chunk-norm radius (0 = off). 0.05 is the "
             "measured trust radius; runs to date sit at ~0.18.",
    )
    parser.add_argument(
        "--exploration-gate-mm",
        type=float,
        default=0.0,
        help="advance the exploration ramp only while the dynamics model's pod error (mm) is "
             "below this (0 = the usual step-based ramp)",
    )
    parser.add_argument(
        "--save-replay",
        dest="save_replay",
        action="store_true",
        default=True,
        help="write the online replay buffer to <output>/replay/<step>.npz at every checkpoint "
             "(DEFAULT ON) so the run's transitions survive the process -- offline critic/value/"
             "actor pretraining and off-policy re-analysis both need them",
    )
    parser.add_argument(
        "--no-save-replay",
        dest="save_replay",
        action="store_false",
        help="do not persist the replay buffer (~1.3 GB per 300k run)",
    )
    parser.add_argument(
        "--pose-from-seghead", action="store_true",
        help="replace the object slots of the observed state with SegHead+readout estimates "
             "(needle 9.0mm / tripod 1.55mm measured); proprio stays exact")
    parser.add_argument(
        "--state-noise-mm", type=float, default=0.0,
        help="Gaussian sigma (mm) added to the OBJECT slots of the observed state; proprio stays "
             "exact. Simulates a pose estimator of that accuracy (GeoHead measures 6.6mm needle).")
    parser.add_argument(
        "--state-noise-mode", choices=("iid", "episode"), default="iid",
        help="iid = redraw noise every timestep (jitter); episode = one held offset per episode (bias)")
    parser.add_argument(
        "--state-noise-slots", type=str, default="9:12,23:26",
        help="comma-separated a:b index ranges to corrupt (threading_d0: needle 9:12, tripod 23:26)")
    parser.add_argument(
        "--dump-episodes", action="store_true",
        help="write one npz per finished episode (agentview frames + success) to "
             "<output>/rollouts/ -- the data feed for the RoboMeter->V refresh loop",
    )
    parser.add_argument(
        "--value-reload", type=str, default="",
        help="published V directory to poll; when its STAMP mtime moves, hot-swap V in place "
             "(teacher-distilled refresh). Requires --value-model of the same architecture.",
    )
    parser.add_argument(
        "--tok-shift-pad", type=int, default=0,
        help="DrQ-style spatial shift on the CACHED 16x16 DINO token grid, in tokens (0 = off). "
             "The real latent-space analogue of an image shift: z is a flat vector with no spatial "
             "axes, but the grid the conv encoder reads does have them. Unlike --obs-shift-pad this "
             "needs no DINOv2 re-encode -- the tokens are cached, so only the small conv head reruns. "
             "One token is 14 px at 224, so pad 1 is a coarser jitter than --pixel-shift-pad 4.",
    )
    parser.add_argument(
        "--offline-pretrain-steps", type=int, default=0,
        help="Before any environment interaction, run N update steps on DEMONSTRATIONS ONLY, so "
             "the critic starts warm. DVD's early advantage is that it begins with a fitted V "
             "(pdvd_actor_frozf is at 0.722 before a single update) while Q starts random and "
             "gains only +0.10 in its first 25k steps. This tests whether that gap is a missing "
             "value function rather than a missing representation -- and unlike a frozen V, a "
             "pretrained Q keeps being corrected on-policy, so it should not inherit the collapse.",
    )
    parser.add_argument(
        "--obs-shift-pad", type=int, default=0,
        help="DrQ random shift on the FRAMES of a --latent-obs / --seg-obs arm (0 = off). These "
             "observations are normally cached post-encoder, which is why they had no "
             "augmentation at all; with this the arm stores frames and re-encodes shifted ones "
             "every update. The 'shifting breaks the patch grid' objection does NOT apply here: "
             "these features feed only the critic and actor, never a world model.",
    )
    parser.add_argument(
        "--seg-obs-finetune", action="store_true",
        help="--seg-obs with the SEGHEAD HEAD itself trained by the critic loss (DINOv3 stays "
             "frozen). The stored maps go stale the moment the head moves, so this arm stores "
             "IMAGES and recomputes DINOv3 tokens -> head -> readout every update. Caching the "
             "tokens instead would be 2.4 MB/row (240 GB at 100k); images are 602 KB.",
    )
    parser.add_argument(
        "--seg-obs-freeze", action="store_true",
        help="--seg-obs with the keypoint readout FROZEN at the weights fitted on this space "
             "(the ConvV trunk from --value-model), instead of trained by TD. The frozen/trained "
             "pair is what separates 'is the representation sufficient' from 'the readout needs "
             "to learn from reward' -- a randomly-initialised frozen trunk would test neither.",
    )
    parser.add_argument(
        "--seg-obs", action="store_true",
        help="actor and critic read the SegHead world model's (V,128,28,28) map through a "
             "TRAINABLE conv trunk + spatial-softmax keypoint readout, instead of an IMPALA "
             "encoding of raw pixels. The backbone stays frozen and the trunk is what the TD loss "
             "shapes -- the same division --pixel-obs has. Requires --dvd-dynamics seghead_wm.",
    )
    parser.add_argument(
        "--latent-obs-finetune", action="store_true",
        help="--latent-obs with the world model's CONV HEAD trained by the TD loss (DINOv2 stays "
             "frozen). Restores the learnable/augmentable pathway the frozen arm gives up, which "
             "our own ablation says is what drives pixel-Q's convergence. Stores DINOv2 tokens "
             "instead of z, because a cached z goes stale on the first optimiser step.",
    )
    parser.add_argument(
        "--latent-obs", action="store_true",
        help="actor and critic consume the latent world model's own input -- [z, normalised "
             "proprio] -- instead of the privileged 37-dim env state or an IMPALA encoding. "
             "Requires --dvd-dynamics latent56; incompatible with --pixel-obs/--dino-obs. The "
             "encoder is FROZEN, so unlike --pixel-obs nothing here learns from the TD loss.",
    )
    parser.add_argument(
        "--dino-obs", action="store_true",
        help="actor and critic consume V's own 777-dim input (pooled DINO-WM features + raw "
             "proprio) instead of the privileged 37-dim env state. Requires --dvd-dynamics "
             "dino_wm; incompatible with --pixel-obs.",
    )
    parser.add_argument(
        "--grad-probe-freq", type=int, default=0,
        help="every N env steps, log cosine of dV/d(image) against a converged pixel critic "
             "(thrpix2_q_s42@300k) on fixed frames. Feature-space V only. 0 disables. Reported "
             "as grad_ref_cos / grad_ref_null / grad_ref_excess; the ceiling between two "
             "independently converged critics is ~+0.030 signed excess, not 1.0.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help='resume a preempted run: "auto" for the newest checkpoints/*.pt in --output-dir, '
             "or a path to one .pt. Restores actor/critics/log_alpha/env_steps, plus the "
             "finetuned V, world model and IMPALA encoder when the checkpoint carries them, "
             "and the replay buffer if a matching replay/*.npz sits beside it.",
    )
    parser.add_argument(
        "--value-init", default=None,
        help="warm-start V from extra.value_fn of a prior train_online checkpoint, then train "
             "normally from env_steps=0. NOT --resume: no replay, actor, optimiser or step count "
             "is carried over. Pair with --dvd-dynamics-init to transplant f as well; use either "
             "one alone to test which component a prior run's result actually lives in.",
    )
    parser.add_argument(
        "--value-scratch",
        action="store_true",
        help="ignore the offline V weights and learn V online from random init; keeps the "
             "architecture, scaler and seed count from --value-model. Requires --value-finetune.",
    )
    parser.add_argument(
        "--encoder-value-aux-image-only",
        action="store_true",
        help="feed the value-aux head only the camera features, not the 9 proprio dims the pixel "
             "encoder appends. Without this the head can fit V from proprio alone and the aux "
             "never shapes the image encoder.",
    )
    parser.add_argument(
        "--value-random-frozen",
        action="store_true",
        help="CONTROL: replace V with a randomly initialised, never-trained V of the same "
             "architecture. Shows whether a well-trained value function matters.",
    )
    parser.add_argument(
        "--value-finetune",
        action="store_true",
        help="keep training V online by TD on its own bootstrap instead of freezing it. "
        "Rejected with --value-intervention shaping, which needs a fixed potential",
    )
    parser.add_argument("--value-finetune-lr", type=float, default=1e-4)


    parser.add_argument("--value-distill-robometer", default="",
                        help="spool dir shared with online_rl_mimicgen.robometer_server; "
                             "empty disables continuous distillation")
    parser.add_argument("--value-distill-every", type=int, default=500,
                        help="run one V regression step every N updates")
    parser.add_argument("--value-distill-batch", type=int, default=256)
    parser.add_argument("--value-distill-weight", type=float, default=1.0)
    parser.add_argument("--value-distill-capacity", type=int, default=20000,
                        help="FIFO cap on (V input, teacher progress) pairs; keeps the "
                             "distillation set on the CURRENT policy distribution")
    parser.add_argument("--value-distill-target", default="scaled",
                        choices=("progress", "success", "product", "scaled", "gated"),
                        help="which RoboMeter head V is regressed onto. 'progress' is the t/T ramp "
                             "and was trained on SUCCESSES ONLY, so it cannot separate a timeout "
                             "from a success; 'success' is the per-frame success probability, the "
                             "head that did train on failures but is ~0 over the first HALF of "
                             "every trajectory, so it gives the actor no shaping; 'product' is "
                             "per-frame progress*success, which zeroes the ramp for the same "
                             "reason; 'scaled' (default) is progress * a PER-EPISODE scalar from "
                             "the success head, keeping the dense ramp and putting the outcome in "
                             "the ceiling -- the offline partial_success convention.")
    def _min_pool_arg(v):


        t = str(v).strip().lower()
        if t in ("", "0", "off", "none", "auto"):
            return "0" if t in ("", "off", "none") else t
        try:
            n = int(t)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected 0 | auto | N >= 0, got {v!r}")
        if n < 0:
            raise argparse.ArgumentTypeError(f"expected 0 | auto | N >= 0, got {v!r}")
        return str(n)
    parser.add_argument("--value-distill-min-pool", type=_min_pool_arg, default="0",
                        help="0 (off) | auto | N. No V fit until every online stratum with a positive "
                             "--value-distill-mix share holds the rows one full declared-mix batch takes "
                             "from it (auto); N additionally requires >= N rows per stratum")
    parser.add_argument("--value-distill-aug-views", type=int, default=0,
                        help="augmented encodings stored per distillation row (DrQ on the V "
                             "regression); 0 = off. Fits draw an augmented view, probe the clean one.")
    parser.add_argument("--value-distill-shift-pad", type=int, default=0,
                        help="DrQ random-shift pad for the augmented distillation views")
    parser.add_argument("--value-distill-photometric", type=float, default=0.0,
                        help="contrast/brightness jitter strength for the augmented distillation views")
    parser.add_argument("--value-distill-mix", default="",
                        help="composition of every distillation batch, as pool=ratio pairs: "
                             "'demo=0.25,succ=0.45,fail=0.30'. demo is the offline expert "
                             "demonstrations, succ/fail are online episodes split by OUTCOME. "
                             "Empty uses that default. Ratios are renormalised over non-empty "
                             "pools, so the run is valid before any pool has data.")
    parser.add_argument("--value-distill-demo-episodes", type=int, default=0,
                        help="seed the demo pool with this many offline expert episodes, scored "
                             "ONCE at startup. 0 disables (fully on-policy distillation).")
    parser.add_argument("--value-distill-frames", type=int, default=64,
                        help="frames per clip handed to the teacher; 64 matches the RBM datasets "
                             "the adapters were trained on")
    parser.add_argument(
        "--zero-floor",
        action="store_true",
        help="floor the TD target at 0 with no model. r >= 0 and gamma > 0 imply Q >= 0, so this "
        "removes exactly the provably-impossible targets -- the model-free control for --value-"
        "intervention floor",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=0,
        help="append a noise latent z ~ N(0, I) of this width to the actor's input, so "
        "r = tanh(MLP(s, a_base, z)). Gives a deterministic-head actor its own source of "
        "diversity, independent of the prior's draws, that contraction cannot kill by pinning a "
        "single log_std bias to the floor. 0 disables",
    )
    parser.add_argument(
        "--bon-axis",
        choices=("residual", "prior"),
        help="what the 16 exploration candidates vary. 'residual': one prior draw, N residuals. "
        "'prior': N independent prior draws with a residual conditioned on each, as in dice-rl. "
        "Defaults to 'prior' with a deterministic residual, 'residual' otherwise",
    )
    parser.add_argument(
        "--adaptive-expert-ratio",
        action="store_true",
        help="anneal the offline/online mixing ratio instead of holding it at --expert-ratio; "
        "starts at 1.0 (pure offline critic warmup) and decays linearly to 0.1",
    )
    parser.add_argument("--adaptive-expert-ratio-start", type=float, default=1.0)
    parser.add_argument("--adaptive-expert-ratio-end", type=float, default=0.1)
    parser.add_argument(
        "--adaptive-expert-ratio-steps",
        type=int,
        default=40_000,
        help="env steps over which the anneal completes; dice-rl's own default",
    )
    parser.add_argument(
        "--expert-ratio",
        type=float,
        default=0.30,
        help="fixed offline fraction of each batch when --adaptive-expert-ratio is off",
    )

    critical = parser.add_argument_group("critical-state replay")
    critical.add_argument(
        "--critical-ladder",
        choices=("gated", "raw", "geometric"),
        default="gated",
        help="progress ladder behind the annotation; 'gated' credits `rim` only while the pod is "
        "still grasped, which is what makes dropped-pod failures visible at all",
    )
    critical.add_argument(
        "--truncate-after-catastrophe",
        action="store_true",
        help="drop transitions strictly after the catastrophe from online replay; the "
        "catastrophic transition itself is retained",
    )
    critical.add_argument(
        "--critical-sampling",
        action="store_true",
        help="oversample the critical window in the online half of each batch",
    )
    critical.add_argument("--critical-batch-fraction", type=float, default=0.25)
    critical.add_argument(
        "--critical-pre-window",
        type=int,
        default=2,
        help="transitions before the catastrophe that join it in the critical window",
    )
    critical.add_argument(
        "--catastrophic-as-terminal",
        action="store_true",
        help="SECONDARY ablation: also stop bootstrapping at the catastrophe. Independent of "
        "--truncate-after-catastrophe on purpose -- replay filtering and Bellman semantics are "
        "different claims",
    )
    critical.add_argument("--actor-gradient-source-normal", default="q", choices=("q", "dvd1"))
    dvd = parser.add_argument_group("DVD(1) actor gradient")
    dvd.add_argument("--dvd-weight", type=float, default=0.0,
                     help="mix -gamma^H V(f(s,a)) into the actor objective. 0 = Q baseline, "
                          "1 = replace dQ/da entirely. Needs --value-model.")
    dvd.add_argument("--dvd-latent-wm", default="",
                     help="--dvd-dynamics latent56: dir of the 56-d DINOv2 latent WM "
                          "(encoder/dynamics/config.json). Its action image SPANS the latent "
                          "(measured subspace overlap 1.0000 vs SegHead ConvV's 0.029).")
    dvd.add_argument("--dvd-value-aware-weight", type=float, default=0.0,
                     help="IterVAML: add w*||V(f(s,a)) - V(s')||^2 to f's loss, so f is trained to "
                          "be right about what DVD actually reads out of it. Additive to the "
                          "prediction loss, which anchors it against the level-set collapse.")
    dvd.add_argument("--encoder-init", default="",
                     help="initialise the pixel encoder from extra.encoder of this checkpoint and FREEZE it "
                          "(e.g. a finished pixel-Q run). Needs --pixel-obs; refuses --actor-train-encoder.")
    dvd.add_argument("--actor-encoder-finetune", action="store_true",
                     help="give the ACTOR its own copy of the --encoder-init encoder, trained by the actor "
                          "loss (DVD + BC). f, V and best-of-N keep reading the frozen original. Needs "
                          "--encoder-init and --no-train-critic.")
    dvd.add_argument("--dvd-encoder-learned", action="store_true",
                     help="with --dvd-on-encoder and no --encoder-init: f/V/BoN read a FRESH IMPALA encoder trained by "
                          "V's TD (DrQ) and RoboMeter distillation (rows re-encoded at fit time); f learns on its "
                          "detached features. Needs --no-train-critic --actor-encoder-finetune --actor-encoder-scratch.")
    dvd.add_argument("--actor-encoder-scratch", action="store_true",
                     help="with --actor-encoder-finetune: the actor's own encoder starts from a FRESH default init "
                          "instead of a copy of the --encoder-init encoder (f/V/BoN still read the frozen one)")
    dvd.add_argument("--dvd-on-encoder", action="store_true",
                     help="f and V consume the pixel encoder's features instead of the privileged "
                          "env state. Needs --pixel-obs and a V/f fitted on those features.")
    dvd.add_argument("--dvd-dynamics", default="model_residual",
                     help="the literal 'dino_wm' for the DINOv2 pixel world model (every campaign "
                          "cell), or the stem of a state-space dynamics checkpoint "
                          "$DVD_DYNAMICS_DIR/dynamics_<stem>.pt (not shipped here). dino_wm predicts "
                          "PATCH FEATURES, not a state: it needs a feature-V from "
                          "value/fit_v_mqattn.py, it emits dvd_feat_mse instead of dvd_pod_mm, and "
                          "--exploration-gate-mm / --dvd-mode replay / --mbpo-weight / "
                          "--bc-filter-source dvd are refused with it rather than reinterpreted.")
    dvd.add_argument("--dino-wm-ckpt", default="",
                     help="world-model checkpoint for --dvd-dynamics dino_wm; empty uses the "
                          "vendored dino_step_60000.pt in the HF snapshot")


    dvd.add_argument("--flow-steps", type=int, default=0,
                     help=f"Euler steps for the rectified-flow ODE and for its inverse (they MUST "
                          f"share the grid). 0 = dino_wm.FLOW_STEPS ({_FLOW_STEPS}), the operating "
                          f"point the checkpoint was published and evaluated at. Lowering it is a "
                          f"DIFFERENT map, so none of the measured round-trip numbers apply and "
                          f"the startup banner will say NOT MEASURED.")
    dvd.add_argument("--flow-inverse-iters", type=int, default=0,
                     help=f"fixed-point sweeps solving the implicit inverse step. 0 = "
                          f"dino_wm.FLOW_INVERSE_ITERS ({_FLOW_ITERS}), the only setting that "
                          f"clears the 10%% round-trip gate for BOTH published checkpoints. The "
                          f"ABSOLUTE checkpoint also clears it at 2 (5.70%%) and that is ~29%% off "
                          f"the anchored update; the RESIDUAL one does NOT -- 2 is 15.19%% (FAIL) "
                          f"and 3 is 8.75%% (straddles). Under-resolution is INVISIBLE in every "
                          f"logged metric, so this is set per checkpoint, never left to chance.")
    dvd.add_argument("--dvd-weight-late", type=float, default=-1.0)
    dvd.add_argument("--dvd-weight-ramp-start", type=int, default=0)
    dvd.add_argument("--dvd-weight-ramp-end", type=int, default=0,
                     help="linearly ramp dvd_weight from its base to --dvd-weight-late over "
                          "[ramp-start, ramp-end] env steps; Q-gradient early, DVD late")
    dvd.add_argument("--dvd-map-shift", type=int, default=0,
                     help="DrQ in FEATURE space: random +/-N cell shift of the 28x28 SegHead map "
                          "in f's fit batch (context and target shifted together, one offset per "
                          "row). The DVD pathway otherwise sees the same stored map every sample, "
                          "while pixel-Q's encoder gets a fresh shift each batch -- the asymmetry "
                          "the campaign reads as the source of pixq's convergence speed. 2 is the "
                          "analogue of DrQ's pad=4 at 224 (28/224*4 = 0.5, so 2 is generous).")
    dvd.add_argument("--dvd-finetune-seghead", action="store_true",
                     help="ALSO train the SegHead DPT head (3.5M params) under the dynamics loss. "
                          "The regression target is computed by an EMA copy of the head, because "
                          "unfreezing an encoder against a target from the SAME encoder collapses "
                          "it to constant features (measured: dinowm block sweep, copy_feat_mse "
                          "1.55 -> ~0 with a WORSE representation). A CollapseGuard aborts the run "
                          "if copy_feat_mse falls anyway. The DINOv3 trunk stays frozen -- "
                          "dino_tokens is @torch.no_grad().")
    dvd.add_argument("--dvd-seghead-lr", type=float, default=1e-5,
                     help="lr for the SegHead head; 10x below the predictor's by default")
    dvd.add_argument("--dvd-seghead-ema", type=float, default=0.999,
                     help="Polyak tau for the EMA target head")
    dvd.add_argument("--actor-train-encoder", action="store_true",
                     help="actor loss trains the pixel encoder (undetached); pairs with "
                          "--no-train-critic for a critic-free DVD method")
    dvd.add_argument("--dvd-finetune-dino-blocks", type=int, default=0,
                     help="unfreeze the last n DINOv2 blocks INSIDE the RL loop; they train under "
                          "the dynamics-fit loss at --dvd-dynamics-lr. Moves the actor obs, WM "
                          "input and V input together -- pair with VALUEFT so V can track.")
    dvd.add_argument("--dino-photometric", type=float, default=0.0,
                     help="photometric jitter strength on V's TD-finetune features; the grid-safe "
                          "augmentation for the DINO pathway (0=off)")
    dvd.add_argument("--actor-lr-drop-at", type=int, default=0,
                     help="env step after which actor lr is multiplied by --actor-lr-late-factor; "
                          "shrinks the mean-policy orbit once the climb is done")
    dvd.add_argument("--actor-lr-late-factor", type=float, default=0.3)
    dvd.add_argument("--dvd-rows-late", type=int, default=0)
    dvd.add_argument("--dvd-rows-switch", type=int, default=0,
                     help="env step at which dvd_rows becomes --dvd-rows-late (0 = never); "
                          "ignition wants few rows, the climb wants many")
    dvd.add_argument("--freeze-training-at", type=int, default=0,
                     help="stop ALL training at this env step but keep collecting and logging "
                          "episodes -- measures the frozen system's true level")
    dvd.add_argument("--expert-gate-success", type=float, default=0.0,
                     help="anneal the expert mix only while rolling success >= this fraction; "
                          "the schedule pauses wherever performance sags (performance-gated "
                          "annealing -- holds the frozen-V spike instead of withdrawing its support)")
    dvd.add_argument("--q-gap-weight", type=float, default=0.0,
                     help="reweight the actor's dQ/da rows by the stamped BoN score spread "
                          "(0=off, 1=full): the model curates WHERE Q improves, never how")
    dvd.add_argument("--rosmoq", action="store_true",
                     help="ROSMO for the CRITIC only: blend the TD bootstrap with the model's "
                          "one-step evaluation gamma^H V(f(s',a')) on a row subset; the actor "
                          "trains on plain dQ/da and never touches the model")
    dvd.add_argument("--rosmoq-weight", type=float, default=0.5)
    dvd.add_argument("--rosmoq-rows", type=int, default=64)
    dvd.add_argument("--rosmo", action="store_true",
                     help="replace the DVD gradient with ROSMO-style one-step improvement: "
                          "score K action candidates with gamma^H V(f(s,a)) under no_grad and "
                          "regress toward the exp-advantage-weighted target (no model gradients)")
    dvd.add_argument("--rosmo-k", type=int, default=4)
    dvd.add_argument("--rosmo-beta", type=float, default=30.0)
    dvd.add_argument("--rosmo-sigma", type=float, default=0.1,
                     help="candidate noise scale, as a fraction of residual_scale")
    dvd.add_argument("--q-rows", type=int, default=0,
                     help="subsample the ACTOR's dQ/da estimate to this many tiled rows "
                          "(dvd_rows semantics applied to the Q side; 0 = full batch, as always)")
    dvd.add_argument("--gated-training", action="store_true",
                     help="feedback-gated training: one of {f, V, pi} trains at a time, chosen "
                          "from model skill and TD residual with hysteresis + dwell; gates "
                          "multiply learning rates so diagnostics keep flowing")
    dvd.add_argument("--gate-skill-tau", type=float, default=0.55)
    dvd.add_argument("--gate-hyst", type=float, default=0.05)
    dvd.add_argument("--gate-td-tau", type=float, default=0.002)
    dvd.add_argument("--gate-td-hyst", type=float, default=0.0005)
    dvd.add_argument("--gate-dwell", type=int, default=200)
    dvd.add_argument("--gate-slow", type=float, default=0.0,
                     help="slow-lane multiplier for f and V during IMPROVE-POLICY (0 = hard freeze)")
    dvd.add_argument("--value-distill-pred", action="store_true",
                     help="distill V on the world model's PREDICTED inputs for each teacher-scored frame "
                          "as well as the actual ones (same target). Needs --dvd-proprio-source predicted")
    dvd.add_argument("--dino-wm-state-head", action="store_true",
                     help="give the DINO-WM a proprio head predicting the RESIDUAL s_{t+8}[:9] - s_t[:9] "
                          "(zero-init => predicts s_t until trained; trained by --dvd-finetune-dynamics)")
    dvd.add_argument("--dvd-proprio-source", choices=("current", "predicted"), default="current",
                     help="proprio V reads next to the PREDICTED features: s_t's (historical) or the "
                          "proprio head's s_t + delta (needs --dino-wm-state-head)")
    dvd.add_argument("--dvd-state-loss-weight", type=float, default=1.0,
                     help="weight of the proprio head's normalised residual MSE in f's finetune loss")
    dvd.add_argument("--dvd-residual-anchor", action="store_true",
                     help="anchor the residual MSE WM (anchor + delta) in the RL adapter; fixes "
                          "the delta-vs-absolute mismatch for NEW res arms (see dino_wm.forward)")
    dvd.add_argument("--dvd-rows-grouped", action="store_true",
                     help="sample dvd_rows as whole multi_sample_k blocks (states with ALL "
                          "their action draws): nested mean over draws-then-states")
    dvd.add_argument("--dvd-checkpoint", action="store_true",
                     help="per-layer gradient checkpointing on the WM predictor during DVD "
                          "backprop; exact gradients, needed for --dvd-rows 0")
    dvd.add_argument("--value-pool", default="",
                     choices=("", "mean", "meanmax", "flat", "sargmax", "xattn", "mqattn"),
                     help="pooling for V's input ONLY (empty = same as --dino-pool). 'flat' hands "
                          "V the unpooled (B,V*196*384) grid -- for the patch-transformer V; the "
                          "actor observation stays on --dino-pool.")
    dvd.add_argument("--dino-pool", default="mean",
                     choices=("mean", "meanmax", "attnpool", "xattn", "mqattn"),
                     help="how f's (B,2,196,384) output is pooled for V. MUST match the pooling "
                          "the value model was fit with; checked against its input_dim at startup. "
                          "attnpool is a LEARNED query (dino_wm.AttnPool) rather than a stateless "
                          "reduction: it needs attn_pool.pt beside --value-model, pools both views "
                          "jointly so its width is 384 not 768, and its query joins the value "
                          "optimiser under --value-finetune. "
                          "xattn is CROSS-ATTENTION POOLING (dino_wm.CrossAttnPool): "
                          "--xattn-queries learnable query tokens cross-attend over the same 392 "
                          "patches with --xattn-heads heads and learned key/value projections, "
                          "each query is projected to 768//K dims and the K outputs are "
                          "concatenated -- so its width is 768, the SAME as mean pooling, at every "
                          "K. It needs xattn_pool.pt beside --value-model. "
                          "mqattn is MULTI-QUERY ATTENTION POOLING "
                          "(pool_mqattn.MultiQueryAttnPool): AttnPool's RAW math -- the query "
                          "dots straight into the patch vector, the output is a convex "
                          "combination of the RAW patch vectors, no key/value projections and no "
                          "heads -- with K queries, optionally PER CAMERA so each softmax is "
                          "restricted to that camera's own 196 patches, concatenated and "
                          "projected by one flat Linear to 768, again the mean-pool width. Its "
                          "shape comes from mqattn_pool.pt beside --value-model, not from a flag.")
    dvd.add_argument("--xattn-queries", type=int, default=8,
                     help="--dino-pool xattn: number of learnable query tokens (4, 8 or 16 -- K "
                          "must divide the 768-dim output). Must MATCH the fitted xattn_pool.pt; "
                          "it is checked against the checkpoint at startup and exists so the cell's "
                          "config.json names the module it actually ran.")
    dvd.add_argument("--xattn-heads", type=int, default=6,
                     help="--dino-pool xattn: attention heads per query (must divide 384). Same "
                          "match-the-checkpoint rule as --xattn-queries.")


    dvd.add_argument("--mqattn-queries", type=int, default=0,
                     help="--dino-pool mqattn: assert the fitted module has K queries (PER CAMERA "
                          "when --mqattn-scope per_view). 0 (default) asserts nothing and takes "
                          "whatever mqattn_pool.pt holds; a disagreement is refused at startup.")
    dvd.add_argument("--mqattn-scope", default="", choices=("", "per_view", "joint"),
                     help="--dino-pool mqattn: assert the queries' softmax SCOPE. per_view = K "
                          "queries per camera, each softmax over that camera's 196 patches; joint "
                          "= K queries over all 392 (AttnPool's scope). '' (default) takes the "
                          "checkpoint's.")
    dvd.add_argument("--mqattn-out-dim", type=int, default=-1,
                     help="--dino-pool mqattn: assert the flat output projection's width (768 for "
                          "the fitted arms, = the mean-pool width). 0 asserts NO projection (the "
                          "raw concatenation); -1 (default) takes the checkpoint's.")
    dvd.add_argument("--dino-obs-pool-separate", action="store_true",
                     help="--dino-pool xattn --dino-obs: give the ACTOR and CRITIC their OWN "
                          "CrossAttnPool, initialised from the same offline fit as V's but trained "
                          "by the CRITIC loss, while V keeps its own. Without it (the default) one "
                          "module is shared by actor, critic and V. Also --dino-pool mqattn.")
    dvd.add_argument("--dino-obs-pool-scratch", action="store_true",
                     help="with --dino-obs-pool-separate --dino-pool mqattn: build the actor/critic "
                          "pool FRESH (the fitted module's architecture, its own default init) "
                          "instead of copying the offline fit, so the critic's TD loss learns the "
                          "readout from nothing -- pixel Q's scratch encoder on frozen DINOv2 "
                          "features. V keeps the fitted pool.")
    dvd.add_argument("--dino-obs-pool-ema-target", action="store_true",
                     help="with --dino-obs-pool-separate: pool s' for the TD target with a Polyak "
                          "copy of the actor/critic pool (tau = --tau), as pixel Q's target_encoder "
                          "encodes next_state. Without it the LIVE pool pools the target.")
    dvd.add_argument("--dvd-actor-pool", action="store_true",
                     help="give the ACTOR its OWN copy of the fitted pooling module (--dino-pool "
                          "attnpool|xattn|mqattn), deepcopied from the same offline fit V uses, "
                          "bound to dvd_term._pool so the DVD loss differentiates it, and owned by "
                          "the ACTOR optimiser at --actor-lr. V keeps its own instance "
                          "(dvd_term._value_pool) for its TD fit, its RoboMeter distillation rows "
                          "and the distiller's cache. Without it (the default) one module is "
                          "shared. Unlike --dino-obs-pool-separate this needs NO critic, which is "
                          "what makes it runnable on a --no-train-critic arm. --attn-pool-freeze "
                          "then freezes V'S OWN INSTANCE ONLY; the actor's copy is always "
                          "trainable, because a frozen copy is the whole cell being inert.")
    dvd.add_argument("--dvd-actor-pool-obs", default="actor", choices=("actor", "value"),
                     help="--dvd-actor-pool with --dino-obs: which module pools the OBSERVATION. "
                          "'actor' (default) = the actor's own copy, so what it sees and what it "
                          "differentiates are one lens. 'value' = V's instance, which makes the "
                          "cell a strict one-variable delta against the shared-pool sibling (only "
                          "dV/da's lens changes). The observation is pooled under no_grad either "
                          "way, so this changes what the actor SEES, never what trains the copy.")
    dvd.add_argument("--dvd-actor-pool-log-every", type=int, default=200,
                     help="--dvd-actor-pool: log pool_grad_actor_dvd (L_DVD on the actor's copy, "
                          "must be > 0), pool_grad_value_dvd (L_DVD on V's instance, must be 0.0), "
                          "pool_grad_value_fit (V's TD fit on V's instance in the same update), "
                          "pool_grad_actor_fit (V's TD fit on the copy, must be 0.0) and "
                          "pool_drift_actor_vs_value every N actor updates, each also printed as "
                          "one '[actor-pool] grad-diag' line, so 'the actor's copy is trained and "
                          "V's is not' is measured rather than assumed. Logging only: no RNG, no "
                          "training state. 0 disables it.")
    dvd.add_argument("--dvd-pool-actor-step", action="store_true",
                     help="SHARED unfrozen readout: ALSO put it in actor_optimizer so the DVD loss is "
                          "APPLIED to it. Without this flag the DVD gradient on a shared pool is wiped by "
                          "ValueEnsemble.fit_step's zero_grad before any step, i.e. the pool is trained "
                          "by V's losses ONLY. With it: a true L_DVD + V-loss mixture.")
    dvd.add_argument("--attn-pool-freeze", action="store_true",
                     help="with --dino-pool attnpool, xattn or mqattn: keep the pooling "
                          "module at its offline fit (out of the "
                          "value optimiser). A moving query co-adapts with V and leaves the "
                          "RoboMeter distiller's cached pooled rows stale.")
    dvd.add_argument("--attn-pool-scratch", action="store_true",
                     help="with --dino-pool mqattn --value-finetune: rebuild V's readout FRESH (the "
                          "fitted module's architecture, its own default init) instead of loading "
                          "the offline fit -- pair with --value-scratch for a V that carries nothing "
                          "from offline training.")
    dvd.add_argument("--no-dino-bf16", dest="dino_bf16", action="store_false",
                     help="run the world model in fp32. bf16 autocast is the default and is most "
                          "of what keeps this arm near the ~9-10 h an existing pixel arm takes")
    dvd.set_defaults(dino_bf16=True)
    dvd.add_argument("--dvd-rows", type=int, default=0,
                     help="rows entering the DVD term per update; 0 = all. The DPG path builds "
                          "batch_size * multi_sample_k = 1024 rows, and at the world model's "
                          "measured ~123 ms per batch of 32 that is ~41 h added over a 300k run "
                          "(~37,800 updates) -- against the ~9-10 h an existing pixel arm takes. "
                          "64 targets ~11-12 h total. Unbiased: the term is a .mean() and is "
                          "divided by its own magnitude before mixing, so this changes the "
                          "estimate's variance, not its expectation. dino_wm only.")
    dvd.add_argument("--dvd-finetune-dynamics", action="store_true",
                     help="adapt f(s,a) online on the agent's own transitions")
    dvd.add_argument("--dvd-dynamics-lr", type=float, default=1e-4)
    dvd.add_argument("--bc-filter-source", default="q", choices=("q", "dvd"),
                     help="which value releases the BC anchor. 'dvd' makes the filter agree with "
                          "what the actor is actually ascending.")
    dvd.add_argument("--no-train-critic", action="store_true",
                     help="skip the critic ensemble entirely. Only legal when nothing reads it: "
                          "--dvd-weight 1.0, no Q-sourced BC filter or BoN, no value intervention.")
    dvd.add_argument("--dvd-mode", default="dpg", choices=("dpg", "replay"),
                     help="'replay' is Heess et al. 2015 Alg. 2 -- replayed action, inferred "
                          "eta_k/xi_k, importance ratio. Requires --stochastic-residual.")
    dvd.add_argument("--dvd-dynamics-init", default=None,
                     help="warm-start f from extra.dvd_dynamics of a prior train_online "
                          "checkpoint (written only by --dvd-finetune-dynamics runs). The frozen "
                          "DINOv2 trunk is not in that payload and is rebuilt from the HF cache.")
    dvd.add_argument("--dvd-dynamics-scratch", action="store_true",
                     help="ignore the offline dynamics weights and learn f online from random "
                          "init; keeps the architecture. Requires --dvd-finetune-dynamics.")
    dvd.add_argument("--bon-source", default="q", choices=("q", "dvd"),
                     help="which value ranks best-of-N candidates")
    dvd.add_argument("--bon-dvd-until", type=int, default=0,
                     help="with --bon-source dvd: rank BoN candidates by V(f) until this env step, then by "
                          "the TD critic (min-Q). 0 = never hand off. Needs a trained critic.")
    dvd.add_argument("--bon-frozen-v", default="off", choices=("off", "min"),
                     help="with --bon-source dvd: also score the best-of-N candidates with a FROZEN copy of V as "
                          "loaded at startup (offline fit / --value-init), rank-normalise both scores within the "
                          "candidate set and execute the candidate with the highest MINIMUM rank (ties -> higher "
                          "rank sum). off = rank by the online V alone.")
    dvd.add_argument("--bc-target", default="base", choices=("base", "bon"),
                     help="what the BC anchor pulls the residual toward. base: 0, i.e. the row's own prior draw "
                          "(historical, bit-identical). bon: on the DVD rows only, the best of the state's K prior "
                          "draws by the online V(f) (centered mean over the ensemble's nets), as "
                          "(a* - base) / residual_scale projected onto what the residual can reach (the "
                          "--residual-max-norm ball, the tanh range); demo rows keep 0 and rows outside --dvd-rows "
                          "leave the BC term. Needs --method dice_rl, an MSE --dvd-dynamics dino_wm, a V that trains "
                          "online (--value-finetune or --value-mc-weight), 0 < --dvd-rows < batch x K, "
                          "--multi-sample-k >= 2, a plain V ensemble, --actor-k-reduce mean and no --bc-filter.")
    dvd.add_argument("--bc-target-after", type=int, default=50_000,
                     help="with --bc-target bon: env steps before the target switches on (the base anchor until "
                          "then); must be later than --value-finetune-after. The pretrained V's ranking of prior "
                          "draws is uninformative until it trains online.")
    dvd.add_argument("--bc-target-ramp", type=int, default=10_000,
                     help="with --bc-target bon: env steps after --bc-target-after over which the target scales "
                          "linearly from 0 to full (0 = at once). An abrupt switch multiplies the BC gradient while "
                          "Adam's second moment lags ~1000 updates, so the actor oversteps for hundreds of updates.")
    mb = parser.add_argument_group("mbpo")
    mb.add_argument("--mbpo-weight", type=float, default=0.0,
                    help="weight on the synthetic critic loss from model-generated chunks. "
                         "0 = baseline bit-for-bit. Uses the same --dvd-dynamics model as DVD but "
                         "to GENERATE data, not to differentiate. Needs --value-model.")
    mb.add_argument("--mbpo-chunks", type=int, default=1,
                    help="rollout length in chunks. Only 1 is implementable: the prior is "
                         "observation-conditioned and cannot be queried at a synthetic successor.")
    mb.add_argument("--mbpo-plain-sample", action="store_true",
                    help="roll out with a plain actor sample instead of best-of-N over the stored "
                         "prior chunks. Off-distribution relative to the deployed policy; this is "
                         "the ablation, not the default.")


    mb.add_argument("--mbpo-max-sigma-mm", type=float, default=0.0,
                    help="drop synthetic rows whose predicted object sigma exceeds this (mm). "
                         "0 disables. Requires a generative --dvd-dynamics.")
    mb.add_argument("--mbpo-success-margin-mm", type=float, default=0.0,
                    help="require predicted successes to clear the threshold by this margin (mm). "
                         "Guards against phantom reward; see TrainConfig for the measurement.")
    mb.add_argument("--mbpo-prior-surrogate", default="",
                    help="checkpoint from online_rl_mimicgen_mbpo.prior_surrogate. Required by "
                         "--mbpo-chunks > 1.")
    ins = parser.add_argument_group("insertion-phase-only task")
    ins.add_argument("--insertion-only", action="store_true",
                     help="start from a grasped demonstration state, terminate on insertion, "
                          "short horizon. The prior scores 36.7%% here vs 88%% on the full task.")
    ins.add_argument("--insertion-span", type=int, default=96,
                     help="frames before the annotated insertion to start from")
    ins.add_argument("--insertion-joint-sigma", type=float, default=0.0,
                     help="arm-joint jitter, radians. REJECTION-SAMPLED against the grasp: 0.02 "
                          "broke it in 88%% of draws, so raise this only with the probe's numbers")
    ins.add_argument("--insertion-holder-sigma", type=float, default=0.0,
                     help="holder jitter, metres -- cannot disturb the grasp")
    parser.add_argument("--dvd-chunks", type=int, default=1,
                        help="DVD(k): roll the dynamics model forward k action-chunks before "
                             "applying V, so the actor gradient is analytic over 8k primitive "
                             "steps instead of 8. k=1 is the original DVD(1). Chunks 2..k reuse "
                             "the recorded next_base, held fixed, so the gradient stays open-loop. "
                             "NOT the same as --nstep, which changes the value TARGET using real "
                             "rewards; this changes the gradient PATH using model predictions. "
                             "DINO WM (--dvd-dynamics dino_wm): only k=2 is defined, and it needs "
                             "--dino-wm-state-head --dvd-proprio-source predicted -- chunk 2 runs from "
                             "context [z_t, zhat_t+8] with the chunk-1 predicted proprio, V reads the "
                             "chunk-2 features and proprio, discount gamma^16; the executed follow chunk "
                             "is chosen by --dvd-follow and is detached.")
    parser.add_argument("--dvd-follow", choices=("recorded", "continuation"), default="recorded",
                        help="DVD(k=2): what chunk 2 executes. 'recorded' (default, historical) = the "
                             "stored next_base, the prior's fresh draw at the NEXT observation. "
                             "'continuation' = the row's OWN prior draw's next 8 steps (steps 8..15 "
                             "of the horizon-32 plan that produced its base, captured from the same "
                             "flow call -- no extra sampling); stored in replay as base_cont / "
                             "base_cont_samples, which are allocated only with this flag. Needs "
                             "--dvd-chunks 2. Demo rows use the demo's own next chunk.")
    parser.add_argument("--dvd-step2-action", choices=("prior", "actor"), default="prior",
                        help="DVD(k=2) on the DINO WM: chunk 2's action. 'prior' (default, historical) = "
                             "the --dvd-follow chunk, detached (open loop). 'actor' = CLOSED LOOP: the "
                             "continuation + residual_scale * actor(obs_hat_t+8, continuation) through "
                             "final_action, where obs_hat_t+8 = [pool(zhat_t+8) || chunk-1 predicted "
                             "proprio] is the actor's --dino-obs observation built from the IMAGINED "
                             "features (deterministic/mean residual). The gradient reaches the first "
                             "action through zhat_t+8 and the actor's parameters through the step-2 "
                             "residual; the continuation stays detached and the pool's parameters get no "
                             "gradient from the imagined observation. Needs --dvd-chunks 2 --dvd-follow "
                             "continuation --dino-obs --dino-wm-state-head --dvd-proprio-source predicted.")
    parser.add_argument("--dvd-row-chunks", type=int, default=1,
                        help="evaluate the actor's DVD term in N row chunks with EXACT gradient accumulation: the "
                             "rows --dvd-rows selected (all batch_size*multi_sample_k at 0) go through the world "
                             "model + V R/N rows at a time, each chunk's d(sum v)/d(action rows) is taken with "
                             "autograd.grad and freed, and the actor backprops a surrogate carrying those "
                             "gradients. Mathematically the same objective, dvd_value, dvd_scale (computed over ALL "
                             "rows) and actor gradient, at ~1/N of the WM+V activation memory (float32 rounding with "
                             "the WM in fp32; under the default bf16 autocast the WM's per-row outputs depend on the "
                             "batch size at bf16 rounding, as for any --dvd-rows change). 1 (default) = the "
                             "single-pass path, unchanged. "
                             "DINO-WM DPG path only (incl. --dvd-chunks 2 with the prior follow, --qdvd-combine, "
                             "--actor-k-reduce, --dvd-phase-gate, --rosmo); refused with --dvd-step2-action actor, "
                             "--value-td-model, --dvd-actor-pool/--dvd-pool-actor-step, a flow f and --dvd-mode "
                             "replay. The WM's own fit step (--dvd-fit-rows) is not affected.")
    parser.add_argument("--bon-lookahead", type=int, choices=(1, 2), default=1,
                        help="best-of-N scoring depth for --bon-source dvd. 1 (default, historical) = "
                             "gamma^8 V(f(ctx, s, a)). 2 = gamma^16 V after f(candidate) then f(the "
                             "candidate's own prior continuation), with the same chunk-2 logic as "
                             "--dvd-chunks 2 (context [z_t, zhat_t+8], predicted proprio). Needs "
                             "--dvd-dynamics dino_wm --dino-wm-state-head --dvd-proprio-source "
                             "predicted. Applies to collection; eval/eval_bon16.py scores depth 1 only "
                             "and refuses a depth-2 checkpoint. Cost: one extra WM forward over the N "
                             "candidates per decision.")
    px = parser.add_argument_group("pixel observations for the critic")
    px.add_argument("--pixel-obs", action="store_true",
                    help="critic and residual actor consume 9-dim proprioception plus both "
                         "camera images through an IMPALA encoder, instead of the privileged "
                         "66-dim object-state vector. Matches what the frozen DiT prior sees.")
    px.add_argument("--pixel-shift-pad", type=int, default=4,
                    help="DrQ random-shift padding in PIXELS. pad/IMG is the jitter FRACTION: "
                         "pad 4 at IMG 224 is 1.79%%, at IMG 112 is 3.57%%, and DrQ's design point "
                         "is 4/84 = 4.76%%. Halve it alongside IMG to hold the fraction fixed.")
    px.add_argument("--dino-shift-pad", type=int, default=0,
        help="DrQ random-shift padding for the images the DINO CRITIC observation is "
             "encoded from (0 = off). The world model is unaffected: it reads env_state, "
             "not these features.")
    px.add_argument("--no-pixel-augment", dest="pixel_augment", action="store_false",
                    help="disable DrQ random-shift on the critic's images. The pixel arm converges "
                         "~26 points faster than the privileged-state arm at 50k (5.7 sigma) with "
                         "lower seed variance, and it differs from that arm in TWO ways: the IMPALA "
                         "encoder starts near-random (an accidental warmup) and the images are "
                         "augmented. Removing the augmentation is the clean ablation -- it takes the "
                         "SAME intervention away rather than inventing a state-space analogue of a "
                         "random crop, which would not be the same manipulation.")
    px.add_argument(
        "--aug-target", choices=("both", "critic", "actor", "none"), default="both",
        help="WHICH consumer of the encoding gets the DrQ random shift. By default the actor and "
             "critic share ONE augmented tensor (actor_state = state.detach()), so the ~20-point "
             "DrQ gain cannot be attributed. 'critic' augments only the critic and the TD target; "
             "'actor' augments only the actor's input. Note the actor NEVER backprops into the "
             "encoder, so encoder shift-invariance can only arrive via 'critic' -- if 'actor' "
             "alone recovers the gain, the mechanism is policy smoothing, not representation.")
    px.set_defaults(pixel_augment=True)
    parser.add_argument("--probe-step-eps", type=float, default=0.0,
                        help="step-and-evaluate probe: after each gradient-SNR measurement, take a "
                             "step of this size along -g_Q and along -g_DVD (each unit-normalised, "
                             "so only DIRECTION is compared) and measure the actual greedy return "
                             "after each. cos(g_Q,g_DVD) says whether they agree; this says which "
                             "is right when they do not. 0 disables. Costs 2*(probe-episodes)+2 "
                             "rollouts per measurement, so pair it with a large --gradvar-every.")
    parser.add_argument("--probe-episodes", type=int, default=20,
                        help="greedy episodes per direction in the step-and-evaluate probe.")
    px.add_argument("--pixel-feature-dim", type=int, default=128,
                    help="IMPALA output width PER CAMERA; the critic input is 2*this + 9")
    fd = parser.add_argument_group("FlowDPG (arXiv 2606.22303) on a frozen prior")
    fd.add_argument("--flowdpg-mode", choices=("adapter", "mlp", "lora"), default="adapter",
                    help="shape of r_phi in v = v_frozen + r_phi. adapter: small TransformerBlocks "
                         "sharing the frozen time embedding (~15M). mlp: per-step MLP (~1.2M), "
                         "cannot attend across the chunk. lora: rank-r on the frozen blocks.")
    fd.add_argument("--flowdpg-hidden", type=int, default=256)
    fd.add_argument("--flowdpg-layers", type=int, default=2)
    fd.add_argument("--flowdpg-rank", type=int, default=16, help="lora mode only")
    fd.add_argument("--flowdpg-alpha", type=float, default=0.1,
                    help="critic step size in units of ||u_t||. This IS the trust region: the "
                         "shift satisfies ||delta||/||u_t|| == alpha exactly, so unlike the "
                         "action-space residual (which ran at 6.8x the measured 0.05 radius on "
                         "Square) it cannot leave the region by drifting.")
    fd.add_argument("--flowdpg-lambda-max", type=float, default=0.5,
                    help="ceiling of the critic/BC mix. 1.0 drops the BC anchor entirely.")
    fd.add_argument("--flowdpg-warmup", type=int, default=5000,
                    help="updates over which lambda ramps 0 -> max. An untrained critic's "
                         "gradient is noise; letting it lead early is how the actor leaves the "
                         "data manifold before the critic can object.")
    fd.add_argument("--flowdpg-cons-weight", type=float, default=1.0,
                    help="weight on ||a_hat - x1||^2. Load-bearing despite the exact projection "
                         "algebra: a_hat equals the clean action only where the velocity is "
                         "already right, and elsewhere the critic is queried off-manifold.")
    fd.add_argument("--flowdpg-expectile", type=float, default=0.7, help="IQL tau")
    ins.add_argument("--holder-sigma", type=float, default=0.0,
                     help="holder jitter on the FULL task, metres. The shipped task bolts the "
                          "coffee machine down (sd 0.0 over 1348 episodes), which leaves the "
                          "94%% prior only ~6 points of headroom against a 5.8-point eval noise "
                          "floor, and lets V key on absolute rather than holder-relative "
                          "position. 0.01 matches the insertion-phase task.")
    critical.add_argument(
        "--actor-gradient-source-critical",
        default="q",
        choices=("q", "dvd1"),
        help="hook for the follow-up experiment; 'dvd1' raises today rather than silently "
        "running as 'q'",
    )
    parser.add_argument(
        "--episode-horizon",
        type=int,
        help="episode length for TRAINING rollouts; defaults to the task's native horizon. "
        "Evaluation deliberately keeps the native horizon so success rates stay comparable "
        "across arms -- this ablates the training deadline, not the metric",
    )
    parser.add_argument("--actor-hidden", type=int, default=256)
    parser.add_argument("--actor-value-after", type=int, default=0,
                        help="env steps before the VALUE-driven half of the actor loss switches on. "
                             "Until then the actor optimises the BC anchor alone, so it tracks the "
                             "demonstration prior while the critic/V becomes informative instead of "
                             "being pushed by a near-random gradient from step 0. 0 = off.")
    parser.add_argument("--dvd-fixed-scale", type=float, default=0.0,
                        help="pin the DVD normaliser to this constant instead of the running "
                             "batch mean|V|. 0.0 (default) reproduces the historical behaviour "
                             "EXACTLY. The running value IS the logged dvd_value, which moves "
                             "with the STATE DISTRIBUTION rather than with V, so it silently "
                             "amplifies the DVD term against the fixed BC anchor as the policy "
                             "drifts into low-V states.")
    parser.add_argument("--dvd-fit-rows", type=int, default=0,
                        help="rows used for the DYNAMICS fit_step only, subsampled from the RL "
                             "batch. 0 (default) reuses --dvd-rows, which is the historical "
                             "behaviour. Separate because --dvd-rows also sets how many rows the "
                             "ACTOR's DVD gradient runs on, so raising that to enlarge f's "
                             "training batch would change two things at once (and cost ~41h/300k).")
    parser.add_argument("--dino-wm-arch", default="",
                        help="predictor shape of the dino-wm checkpoint as \"depth,heads,mlp_dim\". "
                             "Empty = the published 6,6,2048. Required for a non-baseline "
                             "checkpoint: depth and mlp_dim are inferred from the weights and "
                             "verified, but HEAD COUNT is not recoverable from any tensor shape, "
                             "so a wrong value silently regroups attention.")
    parser.add_argument("--actor-grad-clip", type=float, default=10.0,
                        help="max-norm for the ACTOR gradient. 10.0 (default) is the historical "
                             "value and is INERT -- the largest actor_grad ever logged across "
                             "4066 updates is 3.98, so it binds on 0.000%% of steps. Worth setting "
                             "because within an arm, the seed with the SMALLER early actor_grad "
                             "finishes better (r=-0.43, p=0.002, n=47). Note the norm is strongly "
                             "arm-dependent (dobs_froz ~0.31, pixdvd_ft ~3.0), so a value that "
                             "regularises one arm is a no-op for another.")
    parser.add_argument("--gradvar-every", type=int, default=0,
                        help="updates between actor-gradient SNR measurements (0 = off). Splits "
                             "the batch into microbatches and reports cos(g_halfA, g_halfB), "
                             "||mean g||/mean||g||, and McCandlish noise scale, separately for "
                             "the Q term, the DVD term and the BC anchor. Costs m+1 extra "
                             "backwards over the already-built graph -- no second forward and no "
                             "second world-model rollout -- so ~1.2%% of wall clock at 500.")
    parser.add_argument("--no-gradvar-by-phase", dest="gradvar_by_phase", action="store_false",
                        help="report gradient SNR pooled over the whole task instead of split by "
                             "REACH/GRASP/TRANSPORT/INSERT. Pooling hides the effect: the "
                             "whole-task DVD cosine of +0.006 is REACH -0.160 and INSERT +0.059 "
                             "cancelling out.")
    parser.set_defaults(gradvar_by_phase=True)
    parser.add_argument("--dvd-fit-online-only", action="store_true",
                        help="train the dynamics model f on the POLICY's transitions only, "
                             "excluding the expert rows of the batch. The actor's expert mix is "
                             "unaffected, so the actor can keep a high expert ratio as a "
                             "stabiliser while f sees only the distribution its gradient is "
                             "actually consumed at. Falls back to the full batch while "
                             "online_count < 8 (early on the batch is nearly all expert).")


    parser.add_argument("--dvd-fit-sampler", choices=("batch", "recent", "mix"), default="batch",
                        help="where the dynamics model f's fine-tuning rows come from. batch "
                             "(default): a subsample of the ACTOR's RL batch -- the historical "
                             "path, bit-identical, drawing nothing extra from any RNG. recent: f "
                             "draws its OWN --dvd-fit-rows (else --dvd-rows, else --batch-size) rows "
                             "uniformly from the newest --dvd-fit-recent-steps of the ONLINE buffer "
                             "(no expert rows). mix: round(--dvd-fit-recent-frac x rows) of them from "
                             "that window, the rest uniform over the whole online buffer. The actor, "
                             "V and the --actor-utd/--value-utd loops keep the RL batch either way. "
                             "recent/mix only take over once the online buffer holds that many rows; "
                             "before, f runs the batch path (logged as dvd_fit_own_draw 0/1). "
                             "Needs --dvd-dynamics dino_wm and --dvd-finetune-dynamics; refused "
                             "with --gated-training, whose skill gate reads f's fit rows.")
    parser.add_argument("--dvd-fit-recent-steps", type=int, default=25_000,
                        help="the recent window of --dvd-fit-sampler recent/mix, in ENV steps; "
                             "converted to rows as max(1, steps // action_steps), so 25000 is 3125 "
                             "rows. While the buffer holds fewer rows than that, it is every row.")
    parser.add_argument("--dvd-fit-recent-frac", type=float, default=0.5,
                        help="--dvd-fit-sampler mix only: the share of f's rows drawn from the recent "
                             "window, in (0, 1]; the rest are uniform over the whole online buffer. "
                             "1.0 is the same draw as --dvd-fit-sampler recent.")
    parser.add_argument("--dvd-fit-probe-every", type=int, default=0,
                        help="updates between probes of f on the NEWEST rows of the online buffer: "
                             "a deterministic tail slice of --dvd-fit-rows (else --dvd-rows, else "
                             "--batch-size) rows, no gradient, no optimiser step, torch RNG restored "
                             "afterwards. Logged as dvd_feat_mse_recent (plus the _recent copy "
                             "baseline and skill). It measures f where the CURRENT policy is, "
                             "whichever rows f trains on, so batch/recent/mix arms and a frozen f "
                             "share one yardstick. 0 = off. --dvd-dynamics dino_wm only.")
    parser.add_argument("--value-fit-online-only", action="store_true",
                        help="train V on the POLICY's transitions only, excluding the expert rows. "
                             "Independent of --dvd-fit-online-only, so the dynamics model and the "
                             "value function can see different expert mixes while the actor keeps "
                             "the full batch.")
    parser.add_argument("--jitter-goal-mm", type=float, default=0.0,
                        help="widen the TRIPOD's placement bounds to +-this many mm. The shipped "
                             "task pins it (goal_pos range 0.0 mm over 100 demo episodes) so a "
                             "policy can memorise it; this makes it something that must be seen.")
    parser.add_argument("--value-distill-after", type=int, default=0,
                        help="env steps to wait before the RoboMeter distiller starts FITTING V. Episodes are "
                             "still scored and pooled from the start; only the regression steps wait. Pair with "
                             "--value-finetune-after at the same value to hold V completely fixed (TD, its Polyak "
                             "target and distillation) for the first N env steps. 0 = from the beginning.")
    parser.add_argument("--value-finetune-after", type=int, default=0,
                        help="env steps to wait before --value-finetune starts updating V. 0 = "
                             "from the beginning (the historical behaviour). Until then V stays at "
                             "its distilled weights, so the actor warms up on the DENSE progress "
                             "signal instead of on a collapsing sparse-return target.")
    parser.add_argument("--value-lcb-lambda", type=float, default=2.0,
                        help="lambda in mean - lambda*sd for --value-aggregate lcb. NOT clamped "
                             "to [0,1] afterwards: clamping was measured to move the bias by "
                             "+0.0004 while zeroing dV/ds on 5-19%% of states, which DVD needs.")
    parser.add_argument("--state-augment", type=float, default=0.0,
                        help="sigma of Gaussian noise added to the CRITIC's scaled state, the "
                             "state-space analogue of the pixel arm's DrQ random shift. 0 "
                             "disables. The pixel arm is +26 points at 50k over the privileged "
                             "state arm and differs from it only in observation and this "
                             "regulariser; 0.05 matches DrQ's 4-of-84-pixel magnitude.")
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--fixed-goal", action="store_true",
                        help="pin the goal object (square's peg / hammer's drawer) at the demo "
                             "distribution's mode via degenerate placement bounds, keeping the "
                             "task key -- and so the prior's text conditioning -- unchanged. "
                             "Composes with --state-repr. NOTE: every eval seed now shares one "
                             "goal, so the 5.8-point eval noise floor measured across goals no "
                             "longer applies.")
    parser.add_argument("--full-fixed", action="store_true",
                        help="pin the MANIPULATED OBJECT as well as the goal, so every episode "
                             "starts from one scene: square nut at (0,0) with the peg at "
                             "(0.10,0.00); hammer at (0,-0.19) with the drawer at (0.10,0.25). "
                             "The upper-bound probe -- with every placement degree of freedom "
                             "removed, a policy that still fails to improve is not limited by "
                             "placement variation.")
    parser.add_argument("--dvd-static-grad", choices=("none", "degenerate", "degenerate_wide", "goal"), default="none",
                        help="zero the actor gradient on state dims the action cannot move. "
                             "degenerate: dims whose scaler span hit the 1e-5 floor -- on square "
                             "the peg's z alone reconstructs the deployed DVD gradient at cos "
                             "0.99997 because 2/span amplifies it 2e6x. goal: also the goal "
                             "block, which holds 80%% of threading's gradient though the tripod "
                             "never moves. V's input VALUE is unchanged either way.")
    parser.add_argument("--dvd-gate-source", choices=("eef_obj", "obj_goal"), default="eef_obj",
                        help="geometry that opens --dvd-phase-gate. eef_obj is the coffee-era "
                             "||eef-obj|| < --dvd-held-mm, which fires on only 3-6%% of threading "
                             "frames. obj_goal uses ||obj-goal|| < --dvd-gate-mm, the measured "
                             "sign change in the DVD gradient (REACH -0.160, INSERT +0.059 "
                             "against a critic reference).")
    parser.add_argument("--dvd-gate-target", choices=("dvd", "q"), default="dvd",
                        help="which estimator the OPEN gate routes to. dvd (default): DVD on open "
                             "rows, Q on closed ones. q: the critic on open rows and NOTHING on "
                             "closed ones -- the control that separates 'DVD's early gradient was "
                             "harmful' from 'concentrating updates on the reward phase helps'.")
    parser.add_argument("--dvd-gate-mm", type=float, default=150.0,
                        help="object-to-goal threshold for --dvd-gate-source obj_goal. 150 mm is "
                             "threading's measured REACH/INSERT boundary.")
    parser.add_argument("--prior-temperature", type=float, default=1.0,
                        help="scale the frozen prior's initial flow-noise draw, widening the gap "
                             "between the candidate chunks best-of-N ranks. DVD must separate "
                             "candidates THROUGH the dynamics model, and the measured ratio of "
                             "model error to candidate spread is 2.2x on threading (its vf arm "
                             "gains) vs 3.2-3.8x on square (its vf arm loses); this raises the "
                             "denominator. 1.0 leaves the sampler bit-identical.")
    parser.add_argument("--fixed-goal-rot", action="store_true",
                        help="pin ONLY the goal's z-rotation, leaving its POSITION random. "
                             "--fixed-goal removes both; measurement on hammer D1 says they are "
                             "not equally important -- drawer position is uncorrelated with "
                             "success (|r| <= 0.03) while rotation magnitude is not (53.4%% at "
                             "|yaw| < 9deg vs 31.2%% at > 20deg, r = -0.214). This arm isolates "
                             "rotation so a --fixed-goal result can be attributed.")
    parser.add_argument("--state-repr", choices=("absolute", "goal_relative"), default="absolute",
                        help="what the RL stack's state encodes. goal_relative replaces the "
                             "manipulated object's absolute position with (obj_pos - goal_pos) "
                             "per GOAL_RELATIVE_MAPS and fits scaler stats on transformed demo "
                             "states, so the critic sees the mm-scale pose that decides success "
                             "instead of world coordinates normalized by the placement range. "
                             "The frozen prior's own input pipeline is untouched. Refuses any "
                             "absolute-fit consumer (--value-model, dvd dynamics, --init-*-from)")
    parser.add_argument("--value-aggregate", choices=("mean", "min", "max", "lcb"), default="mean",
                        help="how the V seeds are combined. 'min' is pessimism from machinery "
                             "already present: measured on the policy's own states, the V that "
                             "works sits at bias -0.161 and the one that does not at +0.080, and "
                             "min halves that optimism (+0.080 -> +0.038) while the "
                             "matched-progress gap moves -0.0006.")


    parser.add_argument("--critic-layers", type=int, default=3,
                        help="hidden layers in each critic MLP")
    parser.add_argument("--critic-ensemble-size", type=int, default=10,
                        help="number of Q heads (dice_rl only; policy_decorator uses 2)")
    parser.add_argument(
        "--utd",
        type=int,
        default=1,
        help="gradient steps per collected chunk (update-to-data ratio); RLPD uses 20",
    )
    parser.add_argument(
        "--value-nets", type=int, default=0,
        help="keep only the first K seeds of the loaded V ensemble (0 = all, unchanged). Saves about 1%% of a "
             "chunk; the cost is that best-of-N loses the ensemble spread it ranks and gates with.",
    )
    parser.add_argument(
        "--dvd-finetune-last-k", type=int, default=0,
        help="online world-model fine-tuning updates only the last K predictor layers (plus the state head "
             "and the action encoder); 0 = every layer, unchanged. The actor gradient still flows through "
             "all layers, so this saves the fit step's weight gradients, not the actor backward.",
    )
    parser.add_argument(
        "--compile-parts", default="",
        help="torch.compile these forwards, comma separated: encoder (the frozen DINOv2 trunk), wm (the "
             "world-model predictor), v (the value ensemble and the readout), or all. Compiles the bound "
             "forward, so checkpoints are unchanged. Off by default.",
    )
    parser.add_argument(
        "--time-backward", action="store_true",
        help="with --time-components, also time Tensor.backward and the optimiser steps class-wide, which "
             "splits the update's rest bucket into forward / backward / optimiser. Adds a sync per call, so "
             "never A/B a run that has it against one that does not.",
    )
    parser.add_argument(
        "--time-components", action="store_true",
        help="wrap the collection and update stages in CUDA-synced timers and log t_<stage> seconds per train "
             "row (env_step, prior_sample, bon_select, replay_sample, dino_encode, dvd_obj_fwd, v_fit, wm_fit, "
             "distill). Costs a few percent; for profiling runs, not production.",
    )
    parser.add_argument(
        "--update-on-episode-end", action="store_true",
        help="hold every update to the episode boundary and run them in one burst there: one update per chunk "
             "collected in that episode, times --utd. Same number of gradient steps as the default; the actor, V "
             "and f (and so the best-of-N scorer) stay frozen for the whole episode.",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=1,
        help="collected action chunks per update round. 1 = update after every chunk (default, unchanged); "
             "N > 1 = run the --utd loop once every N chunks, i.e. N times fewer gradient steps per env step. "
             "Schedules counted in updates (--value-distill-every, Polyak targets) stretch N-fold in env steps.",
    )
    parser.add_argument(
        "--value-mc-weight", type=float, default=0.0,
        help="weight of an online regression of V onto mc_return, the REALISED discounted "
             "return-to-go stamped at episode flush. Runs in the same V fit block as the TD step "
             "and the RoboMeter distillation, on the same rows, and needs neither a bootstrap nor "
             "the teacher. 0 = off.",
    )
    parser.add_argument(
        "--actor-utd",
        type=int,
        default=1,
        help="ACTOR-ONLY update-to-data ratio: the actor (update_dice, which carries the DVD term and the "
             "BC anchor) takes this many gradient steps per collected chunk, each on a FRESH replay batch; "
             "V, f and the RoboMeter distiller stay at one step. The mirror of --value-utd. 1 = unchanged.",
    )
    dvd.add_argument(
        "--value-utd",
        type=int,
        default=1,
        help="V-ONLY update-to-data ratio: V's TD step (--value-finetune, with its target update) and the "
             "RoboMeter distillation fits run this many times per actor update, each extra step on a FRESH "
             "replay batch; the actor and f stay at --utd. 1 = unchanged.",
    )


    parser.add_argument("--batch-size", type=int, default=256,
                        help="transitions per gradient step (default 256); raise to average the "
                             "actor gradient harder WITHOUT adding optimiser steps, unlike --utd")
    parser.add_argument("--actor-sample-k", type=int,
                        help="K for the actor loss; dice-rl uses 16 (target uses 4)")
    parser.add_argument("--disable-q-loss-for-expert", action="store_true")
    parser.add_argument("--eval-strategy", choices=("greedy", "max_q_min"), default="greedy")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument(
        "--bon-select", choices=("max", "median", "mean", "quantile"), default="max",
        help="which ranked prior candidate to EXECUTE during exploration. 'max' (default) is "
             "best-of-N. 'median' takes the middle-ranked candidate and 'mean' the one scoring "
             "nearest the mean -- both still execute a real prior draw, but stop best-of-N from "
             "concentrating the collected data on the candidate the value function is most "
             "optimistic (and off-distribution, most wrong) about.")
    parser.add_argument(
        "--bon-quantile", type=float, default=0.75,
        help="rank quantile used by --bon-select quantile (1.0 = argmax, 0.5 = median).")
    parser.add_argument(
        "--bon-samples",
        type=int,
        default=0,
        help="prior candidates ranked during exploration (0 = historical default of 16). Kept "
        "separate from --actor-sample-k/--multi-sample-k, which also alter the actor loss",
    )
    parser.add_argument(
        "--multi-sample-k",
        type=int,
        default=1,
        help="K latent draws per state for multi-sample expectation training: the actor loss and "
        "the TD target are both averaged over the K resulting candidates. 1 keeps the "
        "single-sample behaviour. Requires --bon-axis prior",
    )
    parser.add_argument("--dvd-phase-gate", action="store_true",
                        help="route the DVD term per-sample: model gradient while the pod is held, "
                             "critic gradient after release (see TrainConfig.dvd_phase_gate)")
    parser.add_argument("--dvd-held-mm", type=float, default=50.0)
    parser.set_defaults(deterministic_residual=None)
    parser.add_argument("--wandb-name")
    args = parser.parse_args()

    deterministic_residual = (
        args.method == "dice_rl"
        if args.deterministic_residual is None
        else args.deterministic_residual
    )
    bon_axis = args.bon_axis or ("prior" if deterministic_residual else "residual")
    if args.encoder_value_aux_only:
        if args.encoder_value_aux <= 0.0:
            raise SystemExit("--encoder-value-aux-only needs --encoder-value-aux > 0: it names "
                             "L_V as the encoder's only trainer, so at weight 0 nothing would "
                             "train the encoder at all.")
        if args.actor_train_encoder:
            raise SystemExit("--encoder-value-aux-only is mutually exclusive with "
                             "--actor-train-encoder: that one attaches actor_state, so L_DVD and "
                             "L_BC would reach the encoder too and the isolation is lost.")


    if args.qdvd_combine != "off":


        if args.q_rows or args.disable_q_loss_for_expert:
            raise SystemExit(
                "--qdvd-combine pairs q_pi with the DVD term row by row; --q-rows subsamples "
                "q_pi into a new random order and --disable-q-loss-for-expert masks it, so "
                "neither leaves the two sides aligned.")
        if args.dvd_rows and not args.dvd_rows_grouped:
            raise SystemExit(
                "--qdvd-combine with --dvd-rows needs --dvd-rows-grouped: ungrouped selection "
                "takes a FLAT random subset of the B*K tiled rows, so the surviving candidates "
                "are ragged per state and no (states, K) rectangle exists to reduce over.")
        if args.dvd_mode == "replay" or args.dvd_phase_gate:
            raise SystemExit(
                "--qdvd-combine is defined on the DPG path; --dvd-mode replay and "
                "--dvd-phase-gate each build the DVD term on a different branch.")
        if args.no_train_critic:
            raise SystemExit(
                "--qdvd-combine needs a TRAINED critic: with --no-train-critic and "
                "--dvd-weight >= 1 the actor path zeroes q_pi outright, so the combination "
                "would arbitrate against a constant 0.")
    if args.actor_k_reduce in ("max", "min"):
        if args.dvd_weight > 0.0 and args.dvd_rows and not args.dvd_rows_grouped:
            raise SystemExit(
                "--actor-k-reduce max/min now also reduces the DVD term over the K candidates, "
                "which needs the (states, K) layout; ungrouped --dvd-rows takes a FLAT random "
                "subset, so pass --dvd-rows-grouped (ROWSGROUP=1) or --dvd-rows 0.")
        if args.q_rows or args.disable_q_loss_for_expert:
            raise SystemExit(
                "--actor-k-reduce max/min reshapes q_pi to (batch, K); --q-rows subsamples those rows "
                "and --disable-q-loss-for-expert masks them, so neither leaves the (B, K) layout "
                "intact and the max would be taken over the wrong axis.")
        if max(args.multi_sample_k, args.actor_sample_k or 1) <= 1:
            raise SystemExit(
                "--actor-k-reduce max/min needs more than one action candidate per state; pass "
                "--multi-sample-k K (the running arms use 4).")
    if max(args.multi_sample_k, args.actor_sample_k or 1) > 1 and bon_axis != "prior":

        parser.error("--multi-sample-k > 1 requires --bon-axis prior")
    if deterministic_residual and bon_axis != "prior" and not args.latent_dim:


        parser.error(
            "a deterministic residual requires --bon-axis prior (or --latent-dim > 0)"
        )
    if args.value_distill_min_pool != "0":

        if not args.value_distill_robometer:
            parser.error("--value-distill-min-pool needs --value-distill-robometer; without a "
                         "distiller it would be silently ignored")
        if args.value_distill_mix.strip() == "natural":
            parser.error("--value-distill-min-pool gates the DECLARED mix; --value-distill-mix "
                         "natural has no per-stratum share to wait for")
    if args.value_dual_combine is not None and not args.value_td_model:
        parser.error("--value-dual-combine needs --value-td-model; without a second V it would "
                     "be silently ignored")
    args.value_dual_combine = args.value_dual_combine or "min"
    if args.value_td_model:
        _check_value_td_args(parser, args, deterministic_residual)
    if args.bon_dvd_until:
        if args.bon_dvd_until < 0:
            parser.error("--bon-dvd-until must be >= 0")
        if args.bon_source != "dvd":
            parser.error("--bon-dvd-until hands BoN off FROM V(f): it needs --bon-source dvd")
        if args.no_train_critic:
            parser.error("--bon-dvd-until hands BoN off TO the critic: it cannot be combined with "
                         "--no-train-critic (an untrained critic would rank the candidates)")
    if args.bon_frozen_v != "off":
        if args.bon_source != "dvd":
            parser.error("--bon-frozen-v ranks best-of-N candidates by V(f): it needs --bon-source dvd")
        if args.value_td_model:
            parser.error("--bon-frozen-v copies a single ValueEnsemble; --value-td-model makes V a DualValue")
        if args.dvd_dynamics in ("latent56", "seghead_wm"):
            parser.error("--bon-frozen-v copies a plain ValueEnsemble; latent56 / seghead_wm wrap V differently")
        if args.dino_pool in ("attnpool", "xattn", "mqattn") and not args.attn_pool_freeze:
            parser.error("--bon-frozen-v with a TRAINABLE attention query would feed the frozen V features pooled by "
                         "the moving query: add --attn-pool-freeze")


    if args.attn_pool_scratch and (args.dino_pool != "mqattn" or args.attn_pool_freeze or not args.value_finetune):
        parser.error("--attn-pool-scratch needs --dino-pool mqattn --value-finetune and no --attn-pool-freeze")
    if (args.dino_obs_pool_scratch or args.dino_obs_pool_ema_target) and not args.dino_obs_pool_separate:
        parser.error("--dino-obs-pool-scratch / --dino-obs-pool-ema-target modify variant 2's "
                     "actor/critic pool: they need --dino-obs-pool-separate")
    if args.dino_obs_pool_separate:
        if args.dino_pool not in ("xattn", "mqattn"):
            parser.error(f"--dino-obs-pool-separate builds a second trainable pool: it needs "
                         f"--dino-pool xattn or mqattn (got {args.dino_pool!r})")
        if args.dino_obs_pool_scratch and args.dino_pool != "mqattn":
            parser.error(f"--dino-obs-pool-scratch is implemented for --dino-pool mqattn (got "
                         f"{args.dino_pool!r})")
        if not args.dino_obs:
            parser.error("--dino-obs-pool-separate splits the ACTOR/CRITIC observation pool from "
                         "V's: it needs --dino-obs")
        if args.attn_pool_freeze:
            parser.error("--dino-obs-pool-separate with --attn-pool-freeze: variant 3 is ONE "
                         "shared pool frozen at the offline fit, variant 2 is TWO pools of which "
                         "the actor/critic's trains. Pick one.")
        if args.no_train_critic:
            parser.error("--dino-obs-pool-separate needs a trained critic: the second pool's "
                         "parameters live in the critic optimiser and the actor reads a detached "
                         "copy, so --no-train-critic would leave it at its offline fit forever")
        if args.freeze_init_critic:
            parser.error("--dino-obs-pool-separate with --freeze-init-critic: that arm skips "
                         "critic_loss.backward(), so the second pool would sit at its offline fit "
                         "forever -- the same silently-inert cell --no-train-critic is refused for")
        if args.method != "dice_rl":
            parser.error(f"--dino-obs-pool-separate is implemented for --method dice_rl (got "
                         f"{args.method!r}): only update_dice applies the --dino-obs observation "
                         f"rewrite and reads the actor's state through the detached copy")


    if args.dvd_pool_actor_step:
        if args.dino_pool not in ("attnpool", "xattn", "mqattn"):
            parser.error("--dvd-pool-actor-step trains the FITTED shared readout: it needs --dino-pool "
                         f"attnpool, xattn or mqattn (got {args.dino_pool!r})")
        if args.attn_pool_freeze:
            parser.error("--dvd-pool-actor-step with --attn-pool-freeze: a frozen pool cannot be "
                         "trained by the DVD loss -- the cell would be inert. Drop one.")
        if not args.value_finetune:
            parser.error("--dvd-pool-actor-step is the DVD + V MIXTURE: it needs --value-finetune so "
                         "V's losses also train the pool (otherwise use --dvd-actor-pool)")
        if args.dvd_actor_pool:
            parser.error("--dvd-pool-actor-step (ONE shared pool) and --dvd-actor-pool (a separate "
                         "actor COPY) are different designs. Pick one.")
        if args.method != "dice_rl":
            parser.error("--dvd-pool-actor-step is implemented in update_dice only (--method dice_rl)")
    if args.dvd_actor_pool:
        if args.dino_pool not in ("attnpool", "xattn", "mqattn"):
            parser.error(f"--dvd-actor-pool copies the FITTED pooling module: it needs --dino-pool "
                         f"attnpool, xattn or mqattn (got {args.dino_pool!r}). Mean pooling is "
                         f"parameter-free -- there is nothing for the actor to own")
        if not args.value_model:
            parser.error("--dvd-actor-pool copies the readout fitted beside V: it needs "
                         "--value-model")
        if args.dvd_dynamics != "dino_wm":
            parser.error(f"--dvd-actor-pool is implemented for --dvd-dynamics dino_wm (got "
                         f"{args.dvd_dynamics!r}): DvdTerm.value_input only pools in feature space")
        if args.dvd_weight <= 0.0:
            parser.error("--dvd-actor-pool with --dvd-weight 0: L_DVD is the only loss that "
                         "reaches the actor's copy, so nothing would train it and the run would be "
                         "byte-identical to the shared-pool arm while being labelled a two-module "
                         "cell")
        if args.method != "dice_rl":
            parser.error(f"--dvd-actor-pool is implemented for --method dice_rl (got "
                         f"{args.method!r}): only update_dice evaluates the DVD objective")
        if args.dino_obs_pool_separate:
            parser.error("--dvd-actor-pool with --dino-obs-pool-separate: this one gives the ACTOR "
                         "a copy trained by L_DVD at --actor-lr, that one gives the OBSERVATION a "
                         "copy trained by the critic's TD loss. Pick one")
        if args.value_pool == "sargmax":
            parser.error("--dvd-actor-pool with --value-pool sargmax: that path rebinds "
                         "dvd_term._pool after the copy is taken, so the actor would "
                         "differentiate a module nothing reads. Pick one")
        if args.bon_frozen_v != "off":
            parser.error("--dvd-actor-pool with --bon-frozen-v: best-of-N ranks through "
                         "value_input, i.e. through the actor's MOVING copy, so the frozen-V "
                         "fixed-lens guarantee --attn-pool-freeze buys is void")
        if args.dino_obs and args.value_intervention in ("floor", "penalty"):
            parser.error(f"--dvd-actor-pool --dino-obs with --value-intervention "
                         f"{args.value_intervention}: that intervention evaluates V on "
                         f"batch['state'], which under --dvd-actor-pool-obs actor is pooled by the "
                         f"ACTOR'S copy -- V would be read through a lens it was never fitted in")
        if args.dvd_actor_pool_obs == "value" and not args.dino_obs:
            parser.error("--dvd-actor-pool-obs value picks the module that pools the --dino-obs "
                         "OBSERVATION, but this arm has no feature observation to pool")
        if (args.dino_pool == "xattn" and args.dino_obs and not args.attn_pool_freeze
                and not args.no_train_critic and not args.freeze_init_critic):
            parser.error("--dvd-actor-pool on CAP variant 1 (--dino-pool xattn --dino-obs, trainable "
                         "pool, a critic that trains): the critic's TD gradient would reach the "
                         "actor's copy, which the critic optimiser does not own, so the critic would "
                         "train no pool while the arm is labelled variant 1. Freeze V's instance "
                         "(--attn-pool-freeze) or run it with --no-train-critic")
        if args.dvd_actor_pool_log_every < 0:
            parser.error("--dvd-actor-pool-log-every must be >= 0 (0 disables the diagnostic)")
    elif args.dvd_actor_pool_obs != "actor":
        parser.error("--dvd-actor-pool-obs only has a meaning with --dvd-actor-pool (there is one "
                     "module otherwise): it would be recorded in config.json and change nothing")
    elif args.dvd_actor_pool_log_every != 200:
        parser.error("--dvd-actor-pool-log-every only has a meaning with --dvd-actor-pool (there is "
                     "no diagnostic otherwise): it would be recorded in config.json and change "
                     "nothing")
    if (args.dino_pool == "xattn" and args.dino_obs and not args.attn_pool_freeze
            and not args.dino_obs_pool_separate):

        _ct = (args.method == "dice_rl" and not args.no_train_critic
               and not args.freeze_init_critic)
        if not _ct and not args.value_finetune:
            parser.error(
                "--dino-pool xattn --dino-obs (CAP variant 1) with nothing that can train the "
                f"shared pool: method={args.method!r} no_train_critic={args.no_train_critic} "
                f"freeze_init_critic={args.freeze_init_critic} value_finetune=False. No optimiser "
                "would hold the module, so the run would be byte-identical to variant 3 while "
                "being labelled variant 1. Add --value-finetune, run it on a critic-training "
                "dice_rl arm, or say what you mean with --attn-pool-freeze")
    if args.dino_pool != "xattn" and (args.xattn_queries != 8 or args.xattn_heads != 6):
        parser.error(f"--xattn-queries/--xattn-heads configure dino_wm.CrossAttnPool: they need "
                     f"--dino-pool xattn (got {args.dino_pool!r})")


    if args.dino_pool == "mqattn":
        if args.value_pool not in ("", "mqattn"):
            parser.error(f"--dino-pool mqattn with --value-pool {args.value_pool!r}: "
                         "MultiQueryAttnPool is the pooling for BOTH the observation and V's "
                         "input and ignores the mode argument, so the --value-pool would have no "
                         "effect while naming a different reduction. Drop it")
        try:
            _check_mqattn_flag_ranges(args)
        except ValueError as _e:
            parser.error(str(_e))
    else:
        if args.value_pool == "mqattn":
            parser.error(f"--value-pool mqattn needs --dino-pool mqattn (got "
                         f"{args.dino_pool!r}): the MultiQueryAttnPool module is loaded on that "
                         "path only, so V's input would be built by dino_wm.pool_features, which "
                         "has no 'mqattn' mode")
        if _mqattn_flag_spec(args):
            parser.error(f"--mqattn-queries/--mqattn-scope/--mqattn-out-dim assert the shape of "
                         f"the fitted pool_mqattn.MultiQueryAttnPool: they need --dino-pool "
                         f"mqattn (got {args.dino_pool!r})")
    try:

        _check_mqattn_value_dir(args.value_model, args.dino_pool, args.value_pool,
                                _mqattn_flag_spec(args))
    except ValueError as _e:
        parser.error(str(_e))
    if args.encoder_init:
        if not args.pixel_obs:
            parser.error("--encoder-init loads a PIXEL encoder: it needs --pixel-obs")
        if args.actor_train_encoder:
            parser.error("--encoder-init freezes the encoder; --actor-train-encoder would train it")
        if not Path(args.encoder_init).is_file():
            parser.error(f"--encoder-init {args.encoder_init}: no such file")
    if args.actor_encoder_finetune:
        if not args.encoder_init and not args.dvd_encoder_learned:
            parser.error("--actor-encoder-finetune copies the --encoder-init encoder: it needs --encoder-init")
        if not args.no_train_critic:


            parser.error("--actor-encoder-finetune needs --no-train-critic")
        if args.encoder_value_aux or args.encoder_value_aux_only:
            parser.error("--actor-encoder-finetune does not combine with --encoder-value-aux*")
    if args.value_utd < 1:
        parser.error("--value-utd must be >= 1")
    if args.update_every < 1:
        parser.error("--update-every must be >= 1")
    if args.update_on_episode_end and args.update_every > 1:
        parser.error("--update-on-episode-end and --update-every are two schedules for the same thing; pick one")
    if args.update_on_episode_end and (args.value_mc_weight or args.reward_model):
        parser.error("--update-on-episode-end needs transitions in the buffer before the boundary burst; --value-mc-weight / a reward model flush them after it")
    if args.value_utd > 1 and not (args.value_finetune or args.value_distill_robometer):
        parser.error("--value-utd > 1 needs a V learner: --value-finetune and/or --value-distill-robometer")
    if args.value_utd > 1 and args.utd > 1:
        parser.error("--value-utd > 1 with --utd > 1: pick one (--value-utd multiplies V steps per actor update)")
    if args.actor_utd < 1:
        parser.error("--actor-utd must be >= 1")
    if args.value_distill_after < 0 or args.value_finetune_after < 0:
        parser.error("--value-distill-after and --value-finetune-after must be >= 0")
    if args.value_distill_after > 0 and not args.value_distill_robometer:
        parser.error("--value-distill-after has nothing to delay without --value-distill-robometer")
    if args.actor_utd > 1 and args.utd > 1:
        parser.error("--actor-utd > 1 with --utd > 1: pick one (--actor-utd multiplies ACTOR steps per "
                     "collected chunk; --utd multiplies every learner)")
    if args.actor_utd > 1 and args.value_utd > 1:
        parser.error("--actor-utd > 1 with --value-utd > 1: the point of each is to move ONE learner's "
                     "update rate while the other is held fixed")
    if args.actor_utd > 1 and args.method != "dice_rl":
        parser.error("--actor-utd only reaches the dice_rl actor loop")


    if args.dvd_encoder_learned:
        _need = [f for f, ok in (("--dvd-on-encoder", args.dvd_on_encoder), ("--pixel-obs", args.pixel_obs),
                                 ("--no-train-critic", args.no_train_critic),
                                 ("--actor-encoder-finetune", args.actor_encoder_finetune),
                                 ("--actor-encoder-scratch", args.actor_encoder_scratch),
                                 ("--value-finetune", args.value_finetune)) if not ok]
        if _need:
            parser.error(f"--dvd-encoder-learned needs {' '.join(_need)}")
        if args.encoder_init:
            parser.error("--dvd-encoder-learned trains V's encoder from scratch: drop --encoder-init")
        if args.value_distill_aug_views or args.value_utd > 1 or getattr(args, "value_td_model", ""):
            parser.error("--dvd-encoder-learned does not combine with --value-distill-aug-views, --value-utd > 1 "
                         "or --value-td-model")
    if args.actor_encoder_scratch and not args.actor_encoder_finetune:
        parser.error("--actor-encoder-scratch re-initialises the actor's own encoder: it needs --actor-encoder-finetune")
    if args.nstep > 1 and args.value_finetune and args.dvd_dynamics in ("latent56", "seghead_wm"):
        parser.error("--nstep > 1 with --value-finetune on latent56/seghead_wm: those V fit batches "
                     "encode next_image (the 1-step frame); only the dino_wm branch reads "
                     "next_image_boot")
    if args.dvd_on_encoder and args.value_distill_robometer and args.value_distill_aug_views:
        parser.error("--value-distill-aug-views is not implemented for --dvd-on-encoder")
    base_output = args.base_output or TASKS[args.task]["il_output"]
    checkpoint = find_latest_checkpoint(base_output)
    config = TrainConfig(
        method=args.method,
        task=args.task,
        output_dir=str(args.output_dir),
        base_checkpoint=str(checkpoint),
        wandb_project=args.wandb_project,
        holder_sigma=args.holder_sigma,
        fixed_goal=args.fixed_goal,
        prior_temperature=args.prior_temperature,
        dvd_static_grad=args.dvd_static_grad,
        dvd_gate_source=args.dvd_gate_source,
        dvd_gate_target=args.dvd_gate_target,
        dvd_gate_mm=args.dvd_gate_mm,
        fixed_goal_rot=args.fixed_goal_rot,
        full_fixed=args.full_fixed,
        dvd_chunks=args.dvd_chunks,
        dvd_follow=args.dvd_follow,
        dvd_step2_action=args.dvd_step2_action,
        dvd_row_chunks=args.dvd_row_chunks,
        bon_lookahead=args.bon_lookahead,
        pixel_obs=args.pixel_obs,
        pixel_feature_dim=args.pixel_feature_dim,
        pixel_augment=args.pixel_augment,
        aug_target=args.aug_target,
        probe_step_eps=args.probe_step_eps,
        probe_episodes=args.probe_episodes,
        pixel_shift_pad=args.pixel_shift_pad,
        dino_shift_pad=args.dino_shift_pad,
        wandb_name=args.wandb_name,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq or args.eval_freq,
        progressive_exploration_steps=(
            args.progressive_exploration_steps
            if args.progressive_exploration_steps is not None
            else args.total_env_steps // 4
        ),
        step_penalty=args.step_penalty,
        eval_video_episodes=args.eval_video_episodes,
        residual_scale=args.residual_scale,
        bc_loss_weight=args.bc_loss_weight,
        gamma=args.gamma,
        initial_log_std=args.initial_log_std,
        log_std_min=args.log_std_min,
        bc_filter=args.bc_filter,
        bc_filter_warmup_updates=args.bc_filter_warmup_updates,
        bc_filter_soft=args.bc_filter_soft,
        q_underestimation_threshold=args.q_underestimation_threshold,
        retain_bc_for_expert=args.retain_bc_for_expert,
        deterministic_residual=deterministic_residual,
        latent_dim=args.latent_dim,
        nstep=args.nstep,
        replay_capacity=args.replay_capacity,
        replay_episode_uniform=args.replay_episode_uniform,
        bon_reduce=args.bon_reduce,
        actor_q_reduce=args.actor_q_reduce,
        actor_k_reduce=args.actor_k_reduce,
        qdvd_combine=args.qdvd_combine,
        encoder_value_aux_head=args.encoder_value_aux_head,
        qdvd_normalise=args.qdvd_normalise,
        encoder_value_aux_only=args.encoder_value_aux_only,
        seg_terminate_patience=args.seg_terminate_patience,
        encoder_drift=args.encoder_drift,
        encoder_value_aux=args.encoder_value_aux,
        vtrace_sigma=args.vtrace_sigma,
        vtrace_clip=args.vtrace_clip,
        vtrace_floor=args.vtrace_floor,
        target_actor=args.target_actor,
        value_target_tau=args.value_target_tau,
        value_loss=args.value_loss,
        bootstrap_timeouts=args.bootstrap_timeouts,
        value_residual=args.value_residual or args.value_residual_unfrozen or args.value_residual_cond,
        value_residual_cond=args.value_residual_cond,
        value_residual_unfrozen=args.value_residual_unfrozen,
        env_dense_reward=args.env_dense_reward,
        reward_model=args.reward_model,
        reward_model_weight=args.reward_model_weight,
        terminate_on_reward_drop=args.terminate_on_reward_drop,
        value_model=args.value_model,
        value_variant=args.value_variant,
        value_intervention=args.value_intervention,
        value_penalty_beta=args.value_penalty_beta,
        value_finetune=args.value_finetune,
        value_init=args.value_init,
        value_scratch=args.value_scratch,
        value_random_frozen=args.value_random_frozen,
        encoder_value_aux_image_only=args.encoder_value_aux_image_only,
        save_replay=args.save_replay,
        state_noise_mm=args.state_noise_mm,
        state_noise_slots=args.state_noise_slots,
        state_noise_mode=args.state_noise_mode,
        dvd_on_encoder=args.dvd_on_encoder,
        encoder_init=args.encoder_init,
        actor_encoder_finetune=args.actor_encoder_finetune,
        actor_encoder_scratch=args.actor_encoder_scratch,
        dvd_encoder_learned=args.dvd_encoder_learned,
        dvd_value_aware_weight=args.dvd_value_aware_weight,
        dvd_latent_wm=args.dvd_latent_wm,
        pose_from_seghead=args.pose_from_seghead,
        dump_episodes=args.dump_episodes,
        value_reload=args.value_reload,
        resume=args.resume,
        dino_obs=args.dino_obs,
        latent_obs=args.latent_obs,
        seg_obs=args.seg_obs,
        seg_obs_freeze=args.seg_obs_freeze,
        seg_obs_finetune=args.seg_obs_finetune,
        obs_shift_pad=args.obs_shift_pad,
        offline_pretrain_steps=args.offline_pretrain_steps,
        tok_shift_pad=args.tok_shift_pad,
        latent_obs_finetune=args.latent_obs_finetune,
        grad_probe_freq=args.grad_probe_freq,
        flowdpg_mode=args.flowdpg_mode,
        flowdpg_hidden=args.flowdpg_hidden,
        flowdpg_layers=args.flowdpg_layers,
        flowdpg_rank=args.flowdpg_rank,
        flowdpg_alpha=args.flowdpg_alpha,
        flowdpg_lambda_max=args.flowdpg_lambda_max,
        flowdpg_warmup=args.flowdpg_warmup,
        flowdpg_cons_weight=args.flowdpg_cons_weight,
        flowdpg_expectile=args.flowdpg_expectile,
        residual_max_norm=args.residual_max_norm,
        exploration_gate_mm=args.exploration_gate_mm,
        dvd_dynamics_init=args.dvd_dynamics_init,
        dvd_dynamics_scratch=args.dvd_dynamics_scratch,
        value_finetune_lr=args.value_finetune_lr,
        zero_floor=args.zero_floor,
        bon_axis=bon_axis,
        episode_horizon=args.episode_horizon,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        critic_layers=args.critic_layers,
        state_augment=args.state_augment,
        state_repr=args.state_repr,
        value_aggregate=args.value_aggregate,
        actor_value_after=args.actor_value_after,
        dvd_fixed_scale=args.dvd_fixed_scale,
        dvd_fit_rows=args.dvd_fit_rows,
        dino_wm_arch=args.dino_wm_arch,
        actor_grad_clip=args.actor_grad_clip,
        gradvar_every=args.gradvar_every,
        gradvar_by_phase=args.gradvar_by_phase,
        dvd_fit_online_only=args.dvd_fit_online_only,
        dvd_fit_sampler=args.dvd_fit_sampler,
        dvd_fit_recent_steps=args.dvd_fit_recent_steps,
        dvd_fit_recent_frac=args.dvd_fit_recent_frac,
        dvd_fit_probe_every=args.dvd_fit_probe_every,
        value_fit_online_only=args.value_fit_online_only,
        jitter_goal_mm=args.jitter_goal_mm,
        value_finetune_after=args.value_finetune_after,
        value_distill_after=args.value_distill_after,
        value_lcb_lambda=args.value_lcb_lambda,
        critic_ensemble_size=args.critic_ensemble_size,
        utd=args.utd,
        actor_utd=args.actor_utd,
        value_utd=args.value_utd,
        update_every=args.update_every,
        update_on_episode_end=args.update_on_episode_end,
        time_components=args.time_components or args.time_backward,
        time_backward=args.time_backward,
        compile_parts=args.compile_parts,
        dvd_finetune_last_k=args.dvd_finetune_last_k,
        value_nets=args.value_nets,
        value_mc_weight=args.value_mc_weight,
        batch_size=args.batch_size,
        multi_sample_k=args.multi_sample_k,
        actor_sample_k=args.actor_sample_k,
        bon_samples=args.bon_samples,
        bon_select=args.bon_select,
        bon_quantile=args.bon_quantile,
        disable_q_loss_for_expert=args.disable_q_loss_for_expert,
        eval_strategy=args.eval_strategy,
        eval_samples=args.eval_samples,
        expert_ratio=args.expert_ratio,
        adaptive_expert_ratio=args.adaptive_expert_ratio,
        adaptive_expert_ratio_start=args.adaptive_expert_ratio_start,
        adaptive_expert_ratio_end=args.adaptive_expert_ratio_end,
        adaptive_expert_ratio_steps=args.adaptive_expert_ratio_steps,
        critical_ladder=args.critical_ladder,
        truncate_after_catastrophe=args.truncate_after_catastrophe,
        critical_sampling=args.critical_sampling,
        critical_batch_fraction=args.critical_batch_fraction,
        critical_pre_window=args.critical_pre_window,
        catastrophic_as_terminal=args.catastrophic_as_terminal,
        actor_gradient_source_normal=args.actor_gradient_source_normal,
        mbpo_weight=args.mbpo_weight,
        mbpo_chunks=args.mbpo_chunks,

        mbpo_best_of_n=not args.mbpo_plain_sample,
        mbpo_prior_surrogate=args.mbpo_prior_surrogate,
        mbpo_max_sigma_mm=args.mbpo_max_sigma_mm,
        mbpo_success_margin_mm=args.mbpo_success_margin_mm,
        dvd_weight=args.dvd_weight,
        dvd_dynamics=args.dvd_dynamics,
        dino_wm_ckpt=args.dino_wm_ckpt,
        dino_pool=args.dino_pool,
        attn_pool_freeze=args.attn_pool_freeze,
        attn_pool_scratch=args.attn_pool_scratch,
        xattn_queries=args.xattn_queries,
        xattn_heads=args.xattn_heads,
        mqattn_queries=args.mqattn_queries,
        mqattn_scope=args.mqattn_scope,
        mqattn_out_dim=args.mqattn_out_dim,
        dino_obs_pool_separate=args.dino_obs_pool_separate,
        dino_obs_pool_scratch=args.dino_obs_pool_scratch,
        dino_obs_pool_ema_target=args.dino_obs_pool_ema_target,
        dvd_actor_pool=args.dvd_actor_pool,
        dvd_pool_actor_step=args.dvd_pool_actor_step,
        dvd_actor_pool_obs=args.dvd_actor_pool_obs,
        dvd_actor_pool_log_every=args.dvd_actor_pool_log_every,
        value_pool=args.value_pool,
        dvd_checkpoint=args.dvd_checkpoint,
        dvd_rows_late=args.dvd_rows_late,
        dvd_rows_switch=args.dvd_rows_switch,
        dvd_rows_grouped=args.dvd_rows_grouped,
        dvd_residual_anchor=args.dvd_residual_anchor,
        dino_wm_state_head=args.dino_wm_state_head,
        dvd_proprio_source=args.dvd_proprio_source,
        dvd_state_loss_weight=args.dvd_state_loss_weight,
        value_distill_pred=args.value_distill_pred,
        freeze_training_at=args.freeze_training_at,
        expert_gate_success=args.expert_gate_success,
        q_gap_weight=args.q_gap_weight,
        rosmoq=args.rosmoq, rosmoq_weight=args.rosmoq_weight, rosmoq_rows=args.rosmoq_rows,
        rosmo=args.rosmo, rosmo_k=args.rosmo_k, rosmo_beta=args.rosmo_beta,
        rosmo_sigma=args.rosmo_sigma,
        q_rows=args.q_rows,
        gated_training=args.gated_training,
        gate_skill_tau=args.gate_skill_tau, gate_hyst=args.gate_hyst,
        gate_td_tau=args.gate_td_tau, gate_td_hyst=args.gate_td_hyst,
        gate_dwell=args.gate_dwell, gate_slow=args.gate_slow,
        dino_photometric=args.dino_photometric,
        dvd_finetune_dino_blocks=args.dvd_finetune_dino_blocks,
        actor_train_encoder=args.actor_train_encoder,
        dvd_weight_late=args.dvd_weight_late,
        dvd_weight_ramp_start=args.dvd_weight_ramp_start,
        dvd_weight_ramp_end=args.dvd_weight_ramp_end,
        actor_lr_drop_at=args.actor_lr_drop_at,
        actor_lr_late_factor=args.actor_lr_late_factor,
        dino_bf16=args.dino_bf16,
        flow_steps=args.flow_steps,
        flow_inverse_iters=args.flow_inverse_iters,
        dvd_rows=args.dvd_rows,
        dvd_finetune_dynamics=args.dvd_finetune_dynamics,
        dvd_finetune_seghead=args.dvd_finetune_seghead,
        dvd_map_shift=args.dvd_map_shift,
        dvd_seghead_lr=args.dvd_seghead_lr,
        dvd_seghead_ema=args.dvd_seghead_ema,
        dvd_dynamics_lr=args.dvd_dynamics_lr,
        dvd_mode=args.dvd_mode,
        dvd_phase_gate=args.dvd_phase_gate,
        dvd_held_mm=args.dvd_held_mm,
        train_critic=not args.no_train_critic,
        bc_filter_source=args.bc_filter_source,
        bon_source=args.bon_source,
        bon_dvd_until=args.bon_dvd_until,
        bon_frozen_v=args.bon_frozen_v,
        bc_target=args.bc_target,
        bc_target_after=args.bc_target_after,
        bc_target_ramp=args.bc_target_ramp,
        insertion_only=args.insertion_only,
        insertion_span=args.insertion_span,
        insertion_joint_sigma=args.insertion_joint_sigma,
        insertion_holder_sigma=args.insertion_holder_sigma,
        actor_gradient_source_critical=args.actor_gradient_source_critical,
        inference_steps=args.inference_steps,
        bc_filter_uncertainty=args.bc_filter_uncertainty,
        reward_model_causal=args.reward_model_causal,
        reward_shaping_form=args.reward_shaping_form,
        sparse_reward_weight=args.sparse_reward_weight,
        residual_head=args.residual_head,
        residual_conv_kernel=args.residual_conv_kernel,
        residual_conv_channels=args.residual_conv_channels,
        residual_conv_broadcast=args.residual_conv_broadcast,
        init_actor_from=args.init_actor_from,
        init_critic_from=args.init_critic_from,
        freeze_init_critic=args.freeze_init_critic,
        value_distill_robometer=args.value_distill_robometer,
        value_distill_every=args.value_distill_every,
        value_distill_batch=args.value_distill_batch,
        value_distill_weight=args.value_distill_weight,
        value_distill_capacity=args.value_distill_capacity,
        value_distill_frames=args.value_distill_frames,
        value_distill_mix=args.value_distill_mix,
        value_distill_target=args.value_distill_target,
        value_distill_aug_views=args.value_distill_aug_views,
        value_distill_min_pool=args.value_distill_min_pool,
        value_distill_shift_pad=args.value_distill_shift_pad,
        value_distill_photometric=args.value_distill_photometric,
        value_distill_demo_episodes=args.value_distill_demo_episodes,
        value_td_model=args.value_td_model,
        value_dual_combine=args.value_dual_combine,
    )
    if config.dvd_dynamics in ("seghead_wm", "latent56") or config.seg_obs or config.latent_obs:
        parser.error("the seghead_wm / latent56 world models are not part of this release")
    _assert_flags_reached_config(parser, args, config)
    try:
        check_dvd_fit_sampler(config)
        check_bc_target(config)
        check_dvd_k2(config)
        check_dvd_row_chunks(config)
    except ValueError as error:
        parser.error(str(error))
    return config


def _assert_flags_reached_config(parser, args, config: TrainConfig) -> None:
    fields = {f.name for f in dataclasses.fields(config)}
    dropped = []
    for action in parser._actions:
        if action.dest not in fields or action.dest in _COMPUTED_FIELDS:
            continue
        supplied = getattr(args, action.dest)
        if supplied != getattr(config, action.dest):
            dropped.append(f"--{action.dest.replace('_', '-')} "
                           f"(you passed {supplied!r}, config has {getattr(config, action.dest)!r})")
    if dropped:
        raise SystemExit(
            "these flags never reached TrainConfig -- add them to the constructor above:\n  "
            + "\n  ".join(dropped)
        )


_COMPUTED_FIELDS = {
    "output_dir", "base_checkpoint", "save_freq", "progressive_exploration_steps",
    "deterministic_residual", "bon_axis",
    "value_residual",
}


def main() -> None:
    _cfg = parse_args()


    if os.environ.get("TRAIN_ONLINE_DRYRUN"):
        print("DRYRUN_CONFIG " + json.dumps(dataclasses.asdict(_cfg), default=str), flush=True)
        return
    _cfg._dvd_weight_base = _cfg.dvd_weight
    train(_cfg)


if __name__ == "__main__":
    main()
