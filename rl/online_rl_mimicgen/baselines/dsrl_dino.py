"""DSRL + DVD through the DINO world model and the feature-space V of a vf-arm run.
usage: bash scripts/run.sh configs/dsrl/<arm>.sh TASK SEED [STEPS], e.g. configs/dsrl/ours.sh square_d0 43
(arms: ours, oursdist, oursmc, randv, q; configs/README.md). No --resume: the replay is not checkpointed."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import torch

from .dsrl_mimicgen import (DSRLMimicGenEnv, DSRLNA, CampaignLog, FlowNoisePrior, StateSlice, build_parser,
                            check_equivalence, check_grad_path)
from stable_baselines3.common.buffers import ReplayBuffer  # noqa: E402  (fork on sys.path via dsrl_mimicgen)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from .. import dino_dvd
from ..common import CACHE_ROOT, TASKS, FrozenPolicyPrior, StateScaler, find_latest_checkpoint, set_seed
import os  # noqa: E402  (--expert-ratio reads PIXEL_CACHE_DIR)
from ..pixel import PROPRIO as PIXEL_PROPRIO, PixelEncoder, preprocess as pixel_preprocess
from ..train_online import _check_mqattn_value_dir, rollout_chunk


ACTIVATIONS = {"mish": torch.nn.Mish, "tanh": torch.nn.Tanh}


from stable_baselines3.sac import policies as sac_policies  # noqa: E402
_SAC_LOG_STD_MIN_DEFAULT = float(sac_policies.LOG_STD_MIN)


class PixelProprioSlice(BaseFeaturesExtractor):
    """Image-only actor input, the vf arm's own:"""

    def __init__(self, observation_space, out_dim: int = 128, shift_pad: int = 4):
        encoder = PixelEncoder(out_dim=out_dim, shift_pad=shift_pad)
        super().__init__(observation_space, features_dim=encoder.out_dim)
        self.encoder = encoder
        self._images: torch.Tensor | None = None
        self._augment = False

    def set_images(self, images: torch.Tensor, augment: bool) -> None:
        self._images, self._augment = images, bool(augment)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        images, self._images = self._images, None
        if images is None or images.shape[0] != observations.shape[0]:
            raise RuntimeError("PixelProprioSlice: set_images() must precede every actor call with the same rows "
                               f"(got {None if images is None else images.shape[0]} images for {observations.shape[0]} "
                               "observations)")
        images = images.to(observations.device)
        return self.encoder(images, observations[:, :PIXEL_PROPRIO], augment=self._augment)


class ExecutedNoisePrior:
    """--executed-noise-only:"""

    def __init__(self, base: FlowNoisePrior, fill: str):
        if fill not in ("zeros", "resample"):
            raise ValueError(f"unknown unexecuted-noise fill {fill!r}")
        self.base, self.fill = base, fill
        self.prior, self.policy, self.objective = base.prior, base.policy, base.objective
        self.state_dim, self.device, self.dtype = base.state_dim, base.device, base.dtype
        self.action_dim = base.action_dim
        self.horizon = base.executed.stop - base.executed.start
        self.executed = slice(0, self.horizon)

    def full(self, noise: torch.Tensor) -> torch.Tensor:
        """(B, 8, A) executed-position noise -> (B, H, A) DiT noise, the other positions filled."""
        noise = noise.to(self.device)
        b = noise.shape[0]
        out = (torch.zeros if self.fill == "zeros" else torch.randn)(
            b, self.base.horizon, self.action_dim, device=self.device, dtype=noise.dtype)

        ex = self.base.executed
        return torch.cat([out[:, :ex.start], noise.reshape(b, self.horizon, self.action_dim), out[:, ex.stop:]], dim=1)

    def cond(self, observation: dict) -> torch.Tensor:
        return self.base.cond(observation)

    def chunk(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.base.chunk(cond, self.full(noise))

    def chunk_with_grad(self, cond: torch.Tensor, noise: torch.Tensor, checkpoint_steps: bool = False) -> torch.Tensor:
        return self.base.chunk_with_grad(cond, self.full(noise), checkpoint_steps)

    def pad(self, chunk: torch.Tensor) -> torch.Tensor:
        return chunk.reshape(chunk.shape[0], -1)

    def __call__(self, obs, noise, return_numpy: bool = True):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        noise = torch.as_tensor(noise, device=self.device).reshape(-1, self.horizon, self.action_dim)
        out = self.pad(self.chunk(obs[:, self.state_dim:], noise))
        return out.cpu().numpy() if return_numpy else out


class DinoFrameEnv(DSRLMimicGenEnv):
    """DSRLMimicGenEnv that also hands the world model's frames to the replay (through `info`) and whole episodes to the RoboMeter distiller."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variant != "na":
            raise ValueError("DinoFrameEnv supports --variant na only")
        self.cur_frame = self.prev_frame = self.cur_pos = None
        self.primitive_steps = 0
        self.ep_frames: list[tuple[np.ndarray, np.ndarray]] = []
        self.ep_actions: list[np.ndarray] = []
        self.episode_hook = None

    def _track(self, observation: dict) -> None:
        self.cur_frame = np.ascontiguousarray(pixel_preprocess(observation["pixels"]), dtype=np.uint8)
        self.cur_pos = np.asarray(observation["agent_pos"], dtype=np.float32).copy()

    def reset(self, *, seed=None, options=None):
        observation, _ = self.env.reset(seed=self.seed + self.episode_index)
        self.episode_index += 1
        self._track(observation)
        self.prev_frame = None
        self.ep_frames, self.ep_actions = [], []
        return self._observe(observation), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        chunk = action.reshape(self.flow.horizon, self.flow.action_dim)[self.flow.executed]
        image_t, pos_t = self.cur_frame, self.cur_pos
        prev_t = self.prev_frame if self.prev_frame is not None else image_t
        observation, reward, done, success, steps = rollout_chunk(
            self.env, chunk, step_penalty=self.step_penalty, use_env_reward=self.dense_reward)
        self._track(observation)
        self.primitive_steps += int(steps)
        self.prev_frame = image_t
        self.ep_frames.append((image_t, pos_t))
        self.ep_actions.append(np.asarray(chunk, dtype=np.float32).reshape(-1).copy())
        terminated, truncated = bool(success), bool(done and not success)
        info = {"is_success": bool(success), "primitive_steps": int(steps),
                "image": image_t, "prev_image": prev_t, "next_image": self.cur_frame}
        if terminated or truncated:
            if self.episode_hook is not None:

                self.episode_hook(self.ep_frames + [(self.cur_frame, self.cur_pos)], list(self.ep_actions), bool(success))
            self.ep_frames, self.ep_actions = [], []
        return self._observe(observation), float(reward), terminated, truncated, info


class DinoFrameReplayBuffer(ReplayBuffer):
    """SB3 ReplayBuffer plus uint8 frame columns indexed like the transition arrays."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1,
                 optimize_memory_usage=False, handle_timeout_termination=True, store_prev=True):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        if n_envs != 1 or optimize_memory_usage:
            raise ValueError("DinoFrameReplayBuffer needs n_envs=1 and optimize_memory_usage=False")
        shape = (self.buffer_size, dino_dvd.N_VIEWS, dino_dvd.IMAGE_SIZE, dino_dvd.IMAGE_SIZE, 3)

        self.images = np.zeros(shape, dtype=np.uint8)


        self.prev_images = np.zeros(shape, dtype=np.uint8) if store_prev else None
        self.next_images = np.zeros(shape, dtype=np.uint8)

    def add(self, obs, next_obs, action, reward, done, infos, noise_action=None):
        i, info = self.pos, infos[0]
        self.images[i] = info["image"]
        if self.prev_images is not None:
            self.prev_images[i] = info["prev_image"]
        self.next_images[i] = info["next_image"]
        super().add(obs, next_obs, action, reward, done, infos, noise_action)


    demo = None
    demo_ratio = 0.0

    def _split(self, inds):
        inds = np.asarray(inds)
        return inds >= 0, (-1 - inds)

    def sample_indexed(self, batch_size: int, env=None):
        upper = self.buffer_size if self.full else self.pos
        k = 0 if self.demo is None else min(int(round(batch_size * self.demo_ratio)), batch_size)
        inds = np.random.randint(0, upper, size=batch_size)
        data = self._get_samples(inds, env=env)
        if k <= 0:
            return data, inds
        rows = np.random.randint(0, len(self.demo), size=k)


        at = np.random.choice(batch_size, size=k, replace=False)
        dev = data.observations.device
        t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float32)
        sel = torch.as_tensor(at, device=dev, dtype=torch.long)
        obs = data.observations.clone();       obs[sel] = t(self.demo.obs[rows])
        nobs = data.next_observations.clone(); nobs[sel] = t(self.demo.next_obs[rows])
        act = data.actions.clone();            act[sel] = t(self.demo.action[rows])
        rew = data.rewards.clone();            rew[sel] = t(self.demo.reward[rows])
        don = data.dones.clone();              don[sel] = t(self.demo.done[rows])
        inds = inds.copy(); inds[at] = -1 - rows
        return type(data)(obs, act, nobs, don, rew), inds

    def frames(self, key: str, inds, device) -> torch.Tensor:
        column = getattr(self, key)
        if column is None:
            raise RuntimeError(f"DinoFrameReplayBuffer: column {key!r} was not stored (store_prev=False)")
        online, demo_rows = self._split(inds)
        if online.all():
            return torch.as_tensor(column[inds], device=device)
        out = np.empty((len(online),) + column.shape[1:], dtype=column.dtype)
        out[online] = column[np.asarray(inds)[online]]
        out[~online] = self.demo.frames(key, demo_rows[~online])
        return torch.as_tensor(out, device=device)

    def terminal(self, inds, device) -> torch.Tensor:
        """done = terminated OR truncated, i.e."""
        online, demo_rows = self._split(inds)
        d = np.empty((len(online), 1), dtype=np.float32)
        d[online] = self.dones[np.asarray(inds)[online], 0].reshape(-1, 1)
        if not online.all():
            d[~online] = self.demo.done[demo_rows[~online]]
        return torch.as_tensor(d, device=device, dtype=torch.float32)


class DSRLDVDDino(DSRLNA):
    """DSRL-NA whose noise actor climbs gamma^8 V(f(ctx, s, prior(s, w))) through the DINO world model."""

    dvd_term = value_fn = value_optimizer = adapter = distiller = attn_pool = None
    dvd_weight = 1.0
    value_tau = 0.005
    dvd_rows = 64
    dvd_fit_rows = 64
    dvd_checkpoint = False
    dvd_fit_utd = 1.0
    dvd_actor_steps = 5
    state_dim = 0
    fits = 0
    image_actor = False
    actor_grad_clip = 0.0
    entropy_per_dim = False
    noise_dim = 1
    frame_env = None
    deterministic_actor = False
    explore_noise = 0.1
    eval_mode = False
    ent_coef_lr = 0.0
    explore_ramp = 0
    ramp_draws = ramp_actor = 0
    q_critic = False
    critic_grad_clip = 0.0
    critic_lr = 0.0
    q_target_entropy = False
    dvd_diag_every = 8
    dvd_diag_draws = 8
    _diag_calls = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):


        if self.explore_ramp > 0 and self.num_timesteps >= learning_starts and not self.eval_mode:
            steps = int(getattr(self.frame_env, "primitive_steps", 0))
            p_actor = min(1.0, steps / float(self.explore_ramp))
            self.ramp_draws += 1
            if np.random.random() >= p_actor:
                noise = torch.randn(n_envs, self.diffusion_act_chunk, self.diffusion_act_dim, device=self.device)
                action = self.diffusion_policy(self._last_obs, noise, return_numpy=True)
                return action, action
            self.ramp_actor += 1

        if self.image_actor and self.num_timesteps >= learning_starts:
            self.actor.features_extractor.set_images(
                torch.as_tensor(self.frame_env.cur_frame[None], device=self.device), augment=False)
        if self.deterministic_actor and self.num_timesteps >= learning_starts:


            obs = torch.as_tensor(self._last_obs, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                self.actor.set_training_mode(False)
                w = self.actor(obs, deterministic=True)
                if not self.eval_mode and self.explore_noise > 0:
                    w = (w + self.explore_noise * torch.randn_like(w)).clamp(-1.0, 1.0)
                w = w * float(self.action_space.high[0])
                action = self.diffusion_policy(obs, w.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
                                               return_numpy=False)
            action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim).cpu().numpy()
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def _refresh_demo_ratio(self) -> float:
        """--expert-ratio:"""
        buf = self.replay_buffer
        if getattr(buf, "demo", None) is None:
            return 0.0
        from .dsrl_demos import expert_ratio_at
        s0, s1, st = self.expert_sched
        buf.demo_ratio = expert_ratio_at(int(getattr(self.frame_env, "primitive_steps", 0)),
                                         start=s0, end=s1, steps=st,
                                         fixed=self.expert_ratio, adaptive=self.adaptive_expert)
        return buf.demo_ratio

    def _q_forward(self, critic, obs, actions, images, augment: bool):
        """One critic forward with ITS OWN images."""
        if self.image_actor:
            critic.features_extractor.set_images(images, augment)
        return critic(obs, actions)

    def q_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """--q-critic:"""
        import torch.nn.functional as F
        from stable_baselines3.common.utils import polyak_update

        flow, buf, dev = self.diffusion_policy, self.replay_buffer, self.device
        magnitude = float(self.action_space.high[0])
        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
        if self.ent_coef_optimizer is not None and not self.ent_coef_lr:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        if self.ent_coef_lr and self.ent_coef_optimizer is not None:
            for g in self.ent_coef_optimizer.param_groups:
                g["lr"] = self.ent_coef_lr
        if self.critic_lr:
            for opt in (self.critic.optimizer, self.critic_noise.optimizer):
                for g in opt.param_groups:
                    g["lr"] = self.critic_lr

        n_actor = gradient_steps if self.dvd_actor_steps <= 0 else min(self.dvd_actor_steps, gradient_steps)
        actor_steps = set(np.linspace(gradient_steps - 1, 0, n_actor, dtype=int).tolist())
        a_clip = self.actor_grad_clip if self.actor_grad_clip > 0 else float("inf")
        c_clip = self.critic_grad_clip if self.critic_grad_clip > 0 else float("inf")
        ent_div = float(self.noise_dim if self.entropy_per_dim else 1)
        logs: dict[str, list[float]] = defaultdict(list)

        demo_frac = self._refresh_demo_ratio()
        for step in range(gradient_steps):
            data, inds = buf.sample_indexed(batch_size, env=self._vec_normalize_env)
            obs, next_obs = data.observations, data.next_observations
            images = buf.frames("images", inds, dev)
            next_images = buf.frames("next_images", inds, dev)

            if self.image_actor:
                self.actor.features_extractor.set_images(images, augment=True)
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)


            self._q_updates = getattr(self, "_q_updates", 0) + 1
            policy_turn = (self._q_updates % max(1, int(getattr(self, "policy_delay", 1))) == 0)

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                if policy_turn:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()
                    logs["ent_coef_loss"].append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            logs["ent_coef"].append(ent_coef.item())


            with torch.no_grad():
                if self.image_actor:
                    self.actor.features_extractor.set_images(next_images, augment=True)
                next_pi, next_log_prob = self.actor.action_log_prob(next_obs)
                next_actions = flow(next_obs, next_pi * magnitude, return_numpy=False)
                next_q = torch.cat(self._q_forward(self.critic_target, next_obs, next_actions, next_images, True),
                                   dim=1)
                next_q = (next_q.min(dim=1, keepdim=True).values if self.critic_backup_combine_type == "min"
                          else next_q.mean(dim=1, keepdim=True))
                if self.q_target_entropy:
                    next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1) / ent_div
                target_q = data.rewards + (1 - buf.terminal(inds, dev)) * self.gamma * next_q
            current_q = self._q_forward(self.critic, obs, data.actions, images, True)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            logs["critic_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c_clip)))
            self.critic.optimizer.step()
            logs["critic_loss"].append(critic_loss.item())
            with torch.no_grad():
                logs["q_mean"].append(float(torch.cat(current_q, dim=1).min(dim=1).values.mean()))
                logs["q_target_mean"].append(float(target_q.mean()))
                logs["q_reward_mean"].append(float(data.rewards.mean()))
                logs["q_done_mean"].append(float(buf.terminal(inds, dev).mean()))


            if step in actor_steps and policy_turn:
                q_w = torch.cat(self._q_forward(self.critic_noise, obs, actions_pi, images, True), dim=1)
                q_w = (q_w.min(dim=1, keepdim=True).values if self.critic_backup_combine_type == "min"
                       else q_w.mean(dim=1, keepdim=True))


                scale = q_w.detach().abs().mean().clamp_min(1e-3)
                q_term = (-q_w / scale).mean()
                ent_term = (ent_coef * log_prob).mean() / ent_div
                actor_loss = ent_term + q_term
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                logs["actor_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), a_clip)))
                self.actor.optimizer.step()
                logs["actor_loss"].append(actor_loss.item())
                logs["actor_entropy_term"].append(float(ent_term.detach()))
                logs["actor_q_term"].append(float(q_term.detach()))
                logs["q_w_mean"].append(float(q_w.detach().mean()))
                logs["q_w_scale"].append(float(scale))
                logs["noise_abs_mean"].append(float((actions_pi.detach() * magnitude).abs().mean()))

            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)


        for _ in range(self.noise_critic_grad_steps):
            data, inds = buf.sample_indexed(batch_size, env=self._vec_normalize_env)
            obs, images = data.observations, buf.frames("images", inds, dev)
            with torch.no_grad():
                noise = torch.randn(obs.shape[0], flow.horizon, flow.action_dim, device=dev)
                diffused = flow(obs, noise, return_numpy=False)
                q_a = self._q_forward(self.critic, obs, diffused, images, True)

            q_n = self._q_forward(self.critic_noise, obs, noise.reshape(obs.shape[0], -1) / magnitude, images, True)
            distill_loss = sum(0.5 * F.mse_loss(n, a.detach()) for a, n in zip(q_a, q_n))
            self.critic_noise.optimizer.zero_grad()
            distill_loss.backward()


            logs["noise_critic_grad_norm"].append(
                float(torch.nn.utils.clip_grad_norm_(self.critic_noise.parameters(), c_clip)))
            self.critic_noise.optimizer.step()
            logs["noise_critic_loss"].append(distill_loss.item())
        self.critic_noise.set_training_mode(False)

        if self.explore_ramp > 0:
            self.logger.record("train/actor_fraction", self.ramp_actor / max(1, self.ramp_draws))
            self.logger.record("train/explore_p_actor",
                               min(1.0, int(getattr(self.frame_env, "primitive_steps", 0)) / float(self.explore_ramp)))
            self.ramp_draws = self.ramp_actor = 0
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in logs.items():
            self.logger.record(f"train/{key}", float(np.mean(values)))


    @torch.no_grad()
    def _dvd_scores(self, scaled: torch.Tensor, action: torch.Tensor, ctx: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Per-row gamma^8 V(f(ctx, s, a)) -- exactly DvdTerm.objective's value before it is averaged and normalised (dvd_term.py:436-441), with model_proprio(record=False) so dvd_v_proprio_shift_mm keeps meaning the actor's rows."""
        term, adapter = self.dvd_term, self.adapter
        saved, was = getattr(adapter, "ctx_feats", None), adapter.training
        adapter.eval()
        out = []
        try:
            for s0 in range(0, len(scaled), chunk):
                sl = slice(s0, s0 + chunk)
                adapter.ctx_feats = ctx[sl]
                pred = term._rollout(scaled[sl], action[sl].reshape(len(scaled[sl]), -1), None, None)
                x = term.value_input(pred, scaled[sl], term.model_proprio(pred))
                out.append((term.discount * self.value_fn(x)).reshape(-1).float())
        finally:
            adapter.ctx_feats = saved
            adapter.train(was)
        return torch.cat(out)

    @torch.no_grad()
    def _decode_full(self, cond: torch.Tensor, w: torch.Tensor, fill: torch.Tensor | None) -> torch.Tensor:
        """Decode actor-shaped noise w (n, horizon, A) with an EXPLICIT fill for the unexecuted DiT positions, so the actor and the prior draws it is compared with share one fill (ExecutedNoisePrior.full redraws it on every call)."""
        flow = self.diffusion_policy
        if isinstance(flow, ExecutedNoisePrior):
            ex = flow.base.executed
            full = torch.cat([fill[:, :ex.start], w, fill[:, ex.stop:]], dim=1)
            return flow.base.chunk(cond, full)
        return flow.chunk(cond, w)

    @torch.no_grad()
    def _dvd_diagnostics(self, obs: torch.Tensor, w_actor: torch.Tensor, ctx: torch.Tensor, logs) -> None:
        """Paired actor-vs-prior read on the actor step's own rows, under the V and f the actor just climbed."""
        flow, sd = self.diffusion_policy, self.state_dim
        n, K = len(obs), max(1, int(self.dvd_diag_draws))
        scaled, cond = obs[:, :sd], obs[:, sd:]
        executed = isinstance(flow, ExecutedNoisePrior)
        H, A = (flow.base.horizon if executed else flow.horizon), flow.action_dim

        def fills(m):
            if not executed:
                return None
            return (torch.zeros if flow.fill == "zeros" else torch.randn)(m, H, A, device=obs.device)

        fill = fills(n)
        j_act = self._dvd_scores(scaled, self._decode_full(cond, w_actor, fill), ctx)
        w_k = torch.randn(n * K, *w_actor.shape[1:], device=obs.device)
        rep_ = lambda t: t.repeat_interleave(K, dim=0)
        j_pri = self._dvd_scores(rep_(scaled), self._decode_full(rep_(cond), w_k, None if fill is None else rep_(fill)),
                                 rep_(ctx)).view(n, K)
        spread = j_pri.std(dim=1)
        adv = j_act - j_pri.mean(dim=1)
        logs["dvd_advantage"].append(float(adv.mean()))
        logs["better_than_prior"].append(float((j_act[:, None] > j_pri).float().mean()))
        logs["dvd_prior_spread"].append(float(spread.mean()))
        logs["dvd_adv_over_spread"].append(float(adv.mean() / spread.mean().clamp_min(1e-8)))
        logs["dvd_bestk_gap"].append(float((j_pri.max(dim=1).values - j_act).mean()))
        logs["dvd_actor_value_paired"].append(float(j_act.mean()))
        if executed and flow.fill == "resample":
            F = 4
            j_fill = self._dvd_scores(scaled.repeat_interleave(F, 0),
                                      self._decode_full(cond.repeat_interleave(F, 0), w_actor.repeat_interleave(F, 0),
                                                        fills(n * F)),
                                      ctx.repeat_interleave(F, 0)).view(n, F)
            logs["dvd_fill_spread"].append(float(j_fill.std(dim=1).mean()))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.q_critic:
            return self.q_train(gradient_steps, batch_size)
        flow, sd, term, adapter = self.diffusion_policy, self.state_dim, self.dvd_term, self.adapter
        buf, dev = self.replay_buffer, self.device
        magnitude = float(self.action_space.high[0])
        self.policy.set_training_mode(True)


        optimizers = [self.actor.optimizer] + ([self.ent_coef_optimizer]
                                               if self.ent_coef_optimizer is not None and not self.ent_coef_lr else [])
        self._update_learning_rate(optimizers)
        if self.ent_coef_lr and self.ent_coef_optimizer is not None:
            for g in self.ent_coef_optimizer.param_groups:
                g["lr"] = self.ent_coef_lr
        n_actor = gradient_steps if self.dvd_actor_steps <= 0 else min(self.dvd_actor_steps, gradient_steps)
        actor_steps = set(np.linspace(gradient_steps - 1, 0, n_actor, dtype=int).tolist())
        n_fit = max(1, int(round(self.train_freq.frequency * self.dvd_fit_utd)))
        fit_steps = set(np.linspace(0, gradient_steps - 1, min(n_fit, gradient_steps), dtype=int).tolist())
        logs: dict[str, list[float]] = defaultdict(list)

        demo_frac = self._refresh_demo_ratio()
        self._diag_calls += 1
        diag_turn = self.dvd_diag_every > 0 and self._diag_calls % self.dvd_diag_every == 0
        diag_actor_done = diag_fit_done = False
        for step in range(gradient_steps):
            data, inds = buf.sample_indexed(batch_size, env=self._vec_normalize_env)
            obs = data.observations
            if self.image_actor:

                self.actor.features_extractor.set_images(buf.frames("images", inds, dev), augment=True)
            if self.deterministic_actor:

                actions_pi, log_prob = self.actor(obs, deterministic=True), None
                ent_coef = torch.zeros((), device=dev)
            else:
                actions_pi, log_prob = self.actor.action_log_prob(obs)
                log_prob = log_prob.reshape(-1, 1)
            if self.deterministic_actor:
                pass
            elif self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                logs["ent_coef_loss"].append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            logs["ent_coef"].append(ent_coef.item())

            if step in actor_steps:

                rows = batch_size if self.dvd_rows <= 0 else min(self.dvd_rows, batch_size)
                ri = inds[:rows]
                noise = (actions_pi[:rows] * magnitude).view(rows, flow.horizon, flow.action_dim)
                action = flow.chunk_with_grad(obs[:rows, sd:], noise, self.dvd_checkpoint)
                dino_dvd.set_context(adapter, dino_dvd.encode_frames(adapter, buf.frames("prev_images", ri, dev)),
                                     dino_dvd.encode_frames(adapter, buf.frames("images", ri, dev)))
                dvd_objective, dvd_value = term.objective(obs[:rows, :sd], action.reshape(rows, -1), None, None)


                ent_term = (torch.zeros((), device=dev) if log_prob is None
                            else (ent_coef * log_prob).mean() / (self.noise_dim if self.entropy_per_dim else 1))
                actor_loss = ent_term + self.dvd_weight * dvd_objective
                logs["actor_entropy_term"].append(float(ent_term.detach()))
                logs["actor_dvd_term"].append(float(dvd_objective.detach()) if torch.is_tensor(dvd_objective) else float(dvd_objective))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()


                clip = self.actor_grad_clip if self.actor_grad_clip > 0 else float("inf")
                logs["actor_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), clip)))
                self.actor.optimizer.step()
                logs["actor_loss"].append(actor_loss.item())
                logs["dvd_value"].append(dvd_value)
                logs["dvd_scale"].append(float(term.last_scale))
                logs["dvd_v_proprio_shift_mm"].append(float(getattr(term, "last_proprio_shift_mm", float("nan"))))
                logs["noise_abs_mean"].append(float(noise.detach().abs().mean()))
                if demo_frac:
                    logs["expert_ratio"].append(float(demo_frac))
                if diag_turn and not diag_actor_done:
                    diag_actor_done = True


                    self._dvd_diagnostics(obs[:rows].detach(), noise.detach(), adapter.ctx_feats.detach(), logs)

            if step in fit_steps:
                self.fits += 1
                with torch.no_grad():
                    raw_now = term.raw_state(obs[:, :sd])
                    raw_next = term.raw_state(data.next_observations[:, :sd])
                    enc_now = dino_dvd.encode_frames(adapter, buf.frames("images", inds, dev))
                    feat_next = torch.cat([term._pool(dino_dvd.encode_frames(adapter, buf.frames("next_images", inds, dev)),
                                                      term.value_pool_mode), raw_next[:, :term.proprio_dim]], dim=-1)
                    done = buf.terminal(inds, dev)


                feat_now = torch.cat([term._pool(enc_now, term.value_pool_mode), raw_now[:, :term.proprio_dim]], dim=-1)
                if diag_turn and not diag_fit_done:
                    diag_fit_done = True
                    with torch.no_grad():


                        fr_ = batch_size if self.dvd_fit_rows <= 0 else min(self.dvd_fit_rows, batch_size)
                        live = (done[:fr_].reshape(-1) < 0.5)
                        if int(live.sum()) >= 8:
                            a_ex = data.actions[:fr_].view(fr_, flow.horizon, flow.action_dim)[:, flow.executed].reshape(fr_, -1)
                            ctx_fit = torch.stack([dino_dvd.encode_frames(adapter, buf.frames("prev_images", inds[:fr_], dev)).float(),
                                                   enc_now[:fr_].float()], dim=1)
                            v_pred = self._dvd_scores(obs[:fr_, :sd], a_ex, ctx_fit)[live]
                            v_real = (term.discount * self.value_fn(feat_next[:fr_])).reshape(-1).float()[live]

                            v_now = self.value_fn(feat_now[:fr_].detach()).reshape(-1).float()[live]
                            logs["dvd_v_real_now"].append(float(v_now.mean()))
                            logs["dvd_v_real_next"].append(float(v_real.mean()))
                            logs["dvd_v_pred_next"].append(float(v_pred.mean()))
                            logs["dvd_wm_v_gap"].append(float((v_pred - v_real).mean()))
                            if float(v_pred.std()) > 1e-8 and float(v_real.std()) > 1e-8:
                                logs["dvd_wm_v_corr"].append(float(torch.corrcoef(torch.stack([v_pred, v_real]))[0, 1]))
                logs["dvd_v_td_loss"].append(self.value_fn.fit_step({
                    "state": feat_now, "next_state": feat_next, "reward": data.rewards, "done": done,
                    "discount": torch.full_like(data.rewards, self.gamma)}, self.value_optimizer))
                self.value_fn.soft_update_target(self.value_tau)
                loss = self.distiller.maybe_fit(self.fits) if self.distiller is not None else float("nan")
                if loss == loss:
                    logs["rm_distill_loss"].append(loss)
                fr = batch_size if self.dvd_fit_rows <= 0 else min(self.dvd_fit_rows, batch_size)
                fi = inds[:fr]
                fit = {"state": obs[:fr, :sd], "next_state": data.next_observations[:fr, :sd],
                       "action": data.actions[:fr].view(fr, flow.horizon, flow.action_dim)[:, flow.executed].reshape(fr, -1),
                       "image": buf.frames("images", fi, dev), "next_image": buf.frames("next_images", fi, dev),
                       "prev_feat": dino_dvd.encode_frames(adapter, buf.frames("prev_images", fi, dev))}
                for key, value in term.fit_step(dino_dvd.fit_batch(adapter, fit, fr)).items():
                    logs[key].append(value)

        if self.explore_ramp > 0:
            self.logger.record("train/actor_fraction", self.ramp_actor / max(1, self.ramp_draws))
            self.logger.record("train/explore_p_actor",
                               min(1.0, int(getattr(self.frame_env, "primitive_steps", 0)) / float(self.explore_ramp)))
            self.ramp_draws = self.ramp_actor = 0
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in logs.items():
            self.logger.record(f"train/{key}", float(np.mean(values)))
        for key, value in (getattr(self, "distill_stats", None) or {}).items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.logger.record(f"train/{key}", float(value))
        if diag_turn and self.distiller is not None and getattr(self.distiller, "size", 0):


            for key, value in self.distiller.probe(512).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    self.logger.record(f"train/{key}", float(value))

    def _excluded_save_params(self) -> list[str]:

        return super()._excluded_save_params() + ["dvd_term", "value_fn", "value_optimizer", "adapter", "distiller",
                                                  "frame_env", "attn_pool"]

    def dvd_state_dict(self) -> dict:
        if self.q_critic:


            return {"q_critic": True, "critic_noise_optimizer": self.critic_noise.optimizer.state_dict()}
        state = {"value_fn": self.value_fn.state_dict(),

                 "dynamics": {k: v for k, v in self.adapter.state_dict().items() if not k.startswith("wm.dino.")},
                 "value_optimizer": self.value_optimizer.state_dict(),
                 "dynamics_optimizer": self.dvd_term.optimiser.state_dict()}
        target = getattr(self.value_fn, "_target", None)
        if target is not None:
            state["value_target"] = target.state_dict()
        if self.attn_pool is not None:
            state["attn_pool"] = self.attn_pool.state_dict()
        return state


class DinoCampaignLog(CampaignLog):
    """CampaignLog plus train_online-style episode rows, and checkpoints WITHOUT the replay (see module docstring)."""

    eval_episodes, eval_workers, eval_seed_start = 0, 2, 10000

    def configure_async_eval(self, episodes: int, workers: int, seed_start: int) -> None:
        """--async-eval-episodes: each checkpoint is scored by eval_dsrl_ckpt in a subprocess while training continues."""
        self.eval_episodes, self.eval_workers, self.eval_seed_start = episodes, max(1, workers), seed_start
        self._eval_queue, self._eval_running, self._eval_seen = [], [], set()

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                with self.metrics.open("a") as handle:
                    handle.write(json.dumps({"kind": "episode", "env_steps": self.env_steps + int(info["primitive_steps"]),
                                             "episode": self.episodes, "length": self.episode_steps + int(info["primitive_steps"]),
                                             "success": bool(info["is_success"])}) + "\n")
        if self.eval_episodes:
            self._pump_evals()
        return CampaignLog._on_step(self)

    def _save(self) -> None:
        path = self.output_dir / "checkpoints" / f"{self.env_steps:07d}"
        self.model.save(path)
        torch.save(self.model.dvd_state_dict(), f"{path}_dvd.pt")
        self.last_save = self.env_steps
        print(f"[dsrl-dino] checkpoint {path}.zip + {path.name}_dvd.pt (replay not saved; no --resume)", flush=True)
        if self.eval_episodes and self.env_steps not in self._eval_seen:     # _on_training_end re-saves the last step
            self._eval_seen.add(self.env_steps)
            self._eval_queue.append((self.env_steps, Path(f"{path}.zip"), 0))
            self._pump_evals()

    def _pump_evals(self) -> None:
        for job in list(self._eval_running):
            proc, steps, ckpt, attempt, out, log = job
            if proc.poll() is None:
                continue
            self._eval_running.remove(job)
            log.close()
            try:
                res = json.loads(out.read_text())
            except (OSError, ValueError):
                res = None
            if proc.returncode != 0 or res is None:
                print(f"[dsrl-dino] async eval of {ckpt.name} failed (exit {proc.returncode}, attempt {attempt + 1}); "
                      f"see {out.with_suffix('.log')}", flush=True)
                if attempt == 0:
                    self._eval_queue.append((steps, ckpt, 1))
                continue
            row = {"env_steps": steps, "success_rate": float(res["success_rate"]), "episodes": int(res["episodes"])}
            for k in ("avg_length", "avg_return"):
                if k in res:
                    row[k] = float(res[k])
            with self.metrics.open("a") as handle:
                handle.write(json.dumps({"kind": "eval", **row}) + "\n")
            if self.wandb_run is not None:
                self.wandb_run.log({f"eval/{k}": v for k, v in row.items()}, step=self.env_steps)
            print(f"[dsrl-dino] eval {ckpt.name}: {row['success_rate']:.1f}% over {row['episodes']} episodes", flush=True)
        while self._eval_queue and len(self._eval_running) < self.eval_workers:
            steps, ckpt, attempt = self._eval_queue.pop(0)
            out = self.output_dir / "evals" / f"{steps:07d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            log = out.with_suffix(".log").open("a")
            cmd = [sys.executable, "-m", "online_rl_mimicgen.baselines.eval_dsrl_ckpt", "--ckpt", str(ckpt),
                   "--out", str(out), "--episodes", str(self.eval_episodes), "--seed-start", str(self.eval_seed_start)]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[2]),
                                    env={**os.environ, "WANDB_MODE": "disabled"})
            self._eval_running.append((proc, steps, ckpt, attempt, out, log))

    def drain_evals(self) -> None:
        """Block until every checkpoint's eval has been logged (called after learn(), before wandb finishes)."""
        while self.eval_episodes and (self._eval_queue or self._eval_running):
            self._pump_evals()
            time.sleep(10)


def episode_hook(model: DSRLDVDDino, device):
    """Episode -> RoboMeter distiller, with V's actual inputs and the world model's predicted ones (--value-distill-pred)."""
    term, adapter, distiller = model.dvd_term, model.adapter, model.distiller

    def hook(frames, actions, success):
        class _Agentview:
            def __init__(self, src): self.src = src
            def __len__(self): return len(self.src)
            def __getitem__(self, i): return self.src[i][0][0]

        def encode(_clip, props, idx):
            both = torch.as_tensor(np.stack([frames[i][0] for i in idx]), device=device)
            pooled = term._pool(dino_dvd.encode_frames(adapter, both), term.value_pool_mode)
            proprio = torch.as_tensor(np.stack(props)[:, :term.proprio_dim], device=device, dtype=pooled.dtype)
            return torch.cat([pooled, proprio], dim=-1)

        def encode_pred(idx):
            pos = [k for k, i in enumerate(idx) if 1 <= i <= len(actions)]
            if not pos:
                return np.zeros((0, 1), np.float32), []
            ci_all, out = [idx[k] - 1 for k in pos], []
            with torch.no_grad():
                for s0 in range(0, len(ci_all), 32):
                    ci = ci_all[s0:s0 + 32]
                    pi = [max(i - 1, 0) for i in ci]
                    cur = dino_dvd.encode_frames(adapter, torch.as_tensor(np.stack([frames[i][0] for i in ci]), device=device))
                    prv = dino_dvd.encode_frames(adapter, torch.as_tensor(np.stack([frames[i][0] for i in pi]), device=device))
                    dino_dvd.set_context(adapter, prv, cur)
                    st = torch.as_tensor(np.stack([frames[i][1][:9] for i in ci]), device=device)
                    ac = torch.as_tensor(np.stack([actions[i] for i in ci]), device=device)
                    pf = adapter(st, ac)
                    out.append(torch.cat([term._pool(pf, term.value_pool_mode),
                                          adapter.last_state_pred[:, :term.proprio_dim].float()], dim=-1))
            return torch.cat(out), pos

        distiller.add_episode(_Agentview(frames), [p for _, p in frames], encode, success=success, pred_fn=encode_pred)
        model.distill_stats = distiller.stats(probe_n=0)

    return hook


def build_dino_parser():
    p = build_parser()
    g = p.add_argument_group("dino", "the vf arm's DINO world model and feature-space V")
    g.add_argument("--dino-wm-ckpt", default=None, help="required unless --q-critic")
    g.add_argument("--dino-wm-arch", default="")
    g.add_argument("--dino-pool", default="mean")
    g.add_argument("--dvd-proprio-source", choices=("current", "predicted"), default="predicted")
    g.add_argument("--dvd-state-loss-weight", type=float, default=1.0)
    g.add_argument("--dvd-fit-rows", type=int, default=64)
    g.add_argument("--value-distill-robometer", default=None, help="RoboMeter sidecar spool dir (required unless --q-critic)")
    g.add_argument("--value-distill-every", type=int, default=20)
    g.add_argument("--value-distill-batch", type=int, default=256)
    g.add_argument("--value-distill-frames", type=int, default=64)
    g.add_argument("--value-distill-capacity", type=int, default=20000)
    g.add_argument("--value-distill-mix", default="succ=0.6,fail=0.4")
    g.add_argument("--value-distill-target", default="progress")
    g.add_argument("--value-distill-pred", action="store_true")

    g.add_argument("--value-scratch", action="store_true",
                   help="random V head (train_online --value-scratch, the 'randhead' cells); the readout stays pretrained")
    g.add_argument("--image-actor", action="store_true",
                   help="actor reads PixelProprioSlice: IMPALA over both cameras + 9-D proprio (the vf arm's --pixel-obs)")
    g.add_argument("--pixel-feature-dim", type=int, default=128)
    g.add_argument("--pixel-shift-pad", type=int, default=4)
    g.add_argument("--ent-coef-lr", type=float, default=0.0,
                   help="Adam lr of the SAC temperature alone (0 = the network lr, SB3's default)")
    g.add_argument("--deterministic-actor", action="store_true",
                   help="TD3-style noise actor: w = mu(s), no entropy term, Gaussian exploration noise at collection")
    g.add_argument("--explore-noise", type=float, default=0.1, help="TD3 collection noise std (actor output units)")
    g.add_argument("--entropy-per-dim", action="store_true",
                   help="divide alpha*log pi in the actor loss by the noise dimension (H*A) so it matches the normalised DVD term")
    g.add_argument("--actor-grad-clip", type=float, default=0.0,
                   help="clip the actor's grad norm (the vf cell's actor_grad_clip is 10); 0 = off")
    g.add_argument("--progressive-exploration-steps", type=int, default=0,
                   help="env steps over which the actor linearly takes over collection (the Q baseline's 75000); "
                        "0 = the actor drives every chunk right after the warm-up")

    g.add_argument("--q-critic", action="store_true",
                   help="train a TD pixel Q instead of the DVD term: Q^A(s, decoded action) against "
                        "min_i Q^A_target(s', prior(s', w')), Q^W(s, w) distilled from it, actor climbs Q^W. "
                        "The world model, V, the mqattn readout and the RoboMeter distiller are NOT built")
    g.add_argument("--critic-lr", type=float, default=0.0, help="Adam lr of Q^A and Q^W (0 = the network lr, --lr)")
    g.add_argument("--critic-grad-clip", type=float, default=0.0,
                   help="clip the critic grad norm (the Q baseline's 10); 0 = off")
    g.add_argument("--q-target-entropy-bonus", action="store_true",
                   help="put the fork's `- alpha*log pi(w'|s')` back inside the TD target (per-dim scaled when "
                        "--entropy-per-dim). Off by default: at alpha ~ 1 and 224-D noise it dominates the sparse reward")
    g.add_argument("--executed-noise-only", choices=("off", "zeros", "resample"), default="off",
                   help="actor outputs the noise of the 8 EXECUTED positions only (8x7 = 56-D instead of the DiT's 32x7); "
                        "the other positions are zeros or a fresh N(0, I) draw per call (ExecutedNoisePrior)")
    g.add_argument("--value-pool-scratch", action="store_true",
                   help="re-initialise the mqattn readout instead of loading <value-model>/mqattn_pool.pt (shape kept); "
                        "with --value-scratch V is fully random")
    g.add_argument("--dvd-diag-every", type=int, default=8,
                   help="train() calls between no-grad DVD diagnostic passes (paired actor-vs-prior advantage, noise "
                        "spread, V(real) vs V(f), distiller V probe); 0 disables them")
    g.add_argument("--dvd-diag-draws", type=int, default=8, help="prior noise draws per actor row in the paired advantage")
    g.add_argument("--policy-delay", type=int, default=1,
                   help="--q-critic only: update the actor AND the temperature once every N critic updates "
                        "(TD3-style delay; the other machine's DSRL-NA uses 2). The counter spans calls, so "
                        "it works at --train-freq 1 --utd 1 where each call is one gradient step")
    g.add_argument("--async-eval-episodes", type=int, default=0,
                   help="evaluate every saved checkpoint with eval_dsrl_ckpt (single noise draw) in a background "
                        "subprocess and log eval/success_rate against eval/env_steps; 0 = off (evals left to a daemon)")
    g.add_argument("--async-eval-workers", type=int, default=2, help="concurrent eval subprocesses per run")
    g.add_argument("--async-eval-seed-start", type=int, default=10000, help="eval env seeds seed_start .. + episodes - 1")
    g.add_argument("--no-value-distill", action="store_true",
                   help="ablation: V is trained by TD ONLY. The RoboMeter distiller and the episode hook that "
                        "feeds the reward server are never built, so the shared spool sees no traffic from this run")
    g.add_argument("--expert-ratio", type=float, default=0.0,
                   help="fraction of every batch drawn from the task's MimicGen DEMO set (the dice_rl/pixq "
                        "campaign uses 0.3 with --adaptive-expert-ratio). 0 = rollouts only, the DSRL default "
                        "and what the other machine's ag_dsrl_na_* runs do. NO BC loss is added: the actor "
                        "samples its own noise, so demo rows contribute states/frames/actions, never w")
    g.add_argument("--adaptive-expert-ratio", action="store_true",
                   help="anneal the demo share start->end over --adaptive-expert-ratio-steps "
                        "(common.py TrainConfig.current_expert_ratio, non-gated branch)")
    g.add_argument("--adaptive-expert-ratio-start", type=float, default=1.0)
    g.add_argument("--adaptive-expert-ratio-end", type=float, default=0.1)
    g.add_argument("--adaptive-expert-ratio-steps", type=int, default=40000)
    g.add_argument("--expert-capacity", type=int, default=0,
                   help="cap on demo chunks kept (0 = all). obs+next_obs cost ~33 KiB/row because each "
                        "carries the prior's ~4.1k-float conditioning; frames stay on the mmap")
    g.add_argument("--log-std-min", type=float, default=-20.0,
                   help="floor on the squashed-Gaussian actor's log sigma. SB3's default is -20 (sigma 2e-9, i.e. the "
                        "policy may become deterministic); the other machine's NoiseActor clamps at -5 (sigma 0.0067). "
                        "Applied by setting stable_baselines3.sac.policies.LOG_STD_MIN, which policies.py:177 reads "
                        "at call time. Recorded in config.json so evals reproduce the same clamp")
    g.add_argument("--activation", choices=("mish", "tanh"), default="mish",
                   help="hidden activation of the actor AND the critics (SB3 takes one activation_fn for the whole "
                        "policy). mish matches the residual arm's common.MLP and the other machine's DSRL-NA; tanh is "
                        "the original default. Recorded in config.json so evals rebuild it correctly")
    return p


def main() -> None:
    args = build_dino_parser().parse_args()
    if args.q_critic:

        if args.variant != "na":
            raise SystemExit("dsrl_dino --q-critic runs DSRL-NA: --variant na")
        if args.deterministic_actor:
            raise SystemExit("--q-critic implements the SAC noise actor only (not --deterministic-actor)")
    elif args.variant != "na" or args.dvd_weight != 1.0 or args.value_model is None:
        raise SystemExit("dsrl_dino runs DSRL-NA + DVD critic-free: --variant na --dvd-weight 1.0 --value-model <V dir>")
    if not args.q_critic and (args.dino_wm_ckpt is None
                              or (args.value_distill_robometer is None and not args.no_value_distill)):
        raise SystemExit("the DVD path needs --dino-wm-ckpt and --value-distill-robometer (unless --no-value-distill)")
    if args.resume:
        raise SystemExit("dsrl_dino does not checkpoint its replay, so --resume is not supported")
    if args.value_distill_pred and args.dvd_proprio_source != "predicted":
        raise SystemExit("--value-distill-pred needs --dvd-proprio-source predicted")
    if args.action_magnitude is None:
        args.action_magnitude = 1.0
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is populated; refusing to overwrite")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                                 "method": "dsrl_na_q_dino" if args.q_critic
                                                 else "dsrl_na_dvd_dino"}, indent=2) + "\n")
    set_seed(args.seed)

    task_cfg = TASKS[args.task]
    state_dim = int(task_cfg["state_dim"])
    checkpoint = find_latest_checkpoint(args.base_output)
    prior = FrozenPolicyPrior(args.task, checkpoint, args.action_steps, args.inference_steps)
    full_flow = FlowNoisePrior(prior, state_dim)


    flow = full_flow if args.executed_noise_only == "off" else ExecutedNoisePrior(full_flow, args.executed_noise_only)
    scaler = StateScaler(task_cfg["dataset_root"])
    env = DinoFrameEnv(args.task, flow, scaler, seed=args.seed, variant="na", magnitude=args.action_magnitude,
                       step_penalty=args.step_penalty, dense_reward=args.env_dense_reward, horizon=args.episode_horizon)
    gap = check_equivalence(full_flow, env.probe_observation)
    print(f"[dsrl-dino] prior {checkpoint}  noise {full_flow.horizon}x{full_flow.action_dim}  cond {env.cond_dim}  "
          f"equivalence gap {gap:.2e}", flush=True)
    if gap > 1e-4:
        raise SystemExit(f"flow(cond, w) differs from prior.action_chunk by {gap:.3e}")
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    grad_gap, grad_norm = check_grad_path(full_flow, env.probe_observation)
    print(f"[dsrl-dino] differentiable prior: gap {grad_gap:.2e}  |da/dw| {grad_norm:.3e}", flush=True)
    if grad_gap > 1e-3 or not grad_norm > 0.0:
        raise SystemExit("the differentiable prior path does not match the sampler or carries no gradient")
    if args.executed_noise_only != "off":
        probe = torch.randn(2, flow.horizon, flow.action_dim, device=flow.device, requires_grad=True)
        flow.chunk_with_grad(flow.cond(env.probe_observation).repeat(2, 1), probe).sum().backward()
        print(f"[dsrl-dino] executed-noise-only: actor noise {flow.horizon}x{flow.action_dim} = "
              f"{flow.horizon * flow.action_dim}-D (DiT noise {full_flow.horizon}x{flow.action_dim}, other positions "
              f"{args.executed_noise_only}); env action space {env.action_space.shape}; |da/dw_56| {float(probe.grad.norm()):.3e}",
              flush=True)
        if env.action_space.shape != (flow.horizon * flow.action_dim,) or not float(probe.grad.norm()) > 0.0:
            raise SystemExit("executed-noise-only wiring is wrong (action space or gradient)")

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or out.name, config=vars(args), dir=str(out))
        if args.async_eval_episodes:
            wandb_run.define_metric("eval/env_steps")
            wandb_run.define_metric("eval/*", step_metric="eval/env_steps")

    device = prior.device
    dvd_term = value_fn = value_optimizer = adapter = distiller = attn_pool = None
    if not args.q_critic:
        from ..dvd_term import DvdTerm
        from ..robometer_distill import RoboMeterDistiller
        from ..value_model import ValueEnsemble

        adapter = dino_dvd.build(args.dino_wm_ckpt, device=device, bf16=True, action_dim=7, arch=args.dino_wm_arch,
                                 state_head=True)
        wm = adapter.wm
        if not getattr(wm, "has_state_head", False) or bool((wm.state_delta_scale == 1.0).all()):
            raise SystemExit(f"{args.dino_wm_ckpt} has no trained proprio head (state_delta_scale unset)")

        _check_mqattn_value_dir(args.value_model, args.dino_pool, "", {})
        value_fn = ValueEnsemble(args.value_model, args.value_variant, scaler, state_dim=state_dim, device=device,
                                 finetune=True, scratch=args.value_scratch)
        if args.value_target_tau > 0.0:
            value_fn.enable_target(device)
        attn_pool = None
        if args.dino_pool == "mqattn":


            from ..pool_mqattn import MultiQueryAttnPool
            pool_path = Path(args.value_model) / "mqattn_pool.pt"
            if not pool_path.exists():
                raise SystemExit(f"--dino-pool mqattn needs {pool_path}")
            attn_pool = MultiQueryAttnPool.from_blob(torch.load(pool_path, map_location=device, weights_only=False),
                                                     device=device)
            if args.value_pool_scratch:


                _blob = torch.load(pool_path, map_location="cpu", weights_only=False)
                _sd = _blob["state_dict"] if isinstance(_blob, dict) and "state_dict" in _blob else _blob
                attn_pool = MultiQueryAttnPool(**MultiQueryAttnPool.spec_from_state_dict(_sd)).to(device)
                print(f"[dsrl-dino] --value-pool-scratch: mqattn readout re-initialised (spec from {pool_path.name})",
                      flush=True)
        value_params = list(value_fn.parameters()) + ([] if attn_pool is None else list(attn_pool.parameters()))
        value_optimizer = torch.optim.Adam(value_params, lr=args.value_lr)
        proprio_dim = dino_dvd.resolve_proprio_dim(value_fn, args.dino_pool)
        dvd_term = DvdTerm(value_fn, scaler, args.action_steps * flow.action_dim, device, discount=args.gamma,
                           horizon=args.action_steps, chunks=1, model_tag="dino_wm", finetune_dynamics=True,
                           dynamics_lr=args.dvd_dynamics_lr, dynamics=adapter, feature_space=True, pool_mode=args.dino_pool,
                           value_pool_mode=None, proprio_dim=proprio_dim, fixed_scale=args.dvd_fixed_scale,
                           proprio_source=args.dvd_proprio_source, state_loss_weight=args.dvd_state_loss_weight)
        if attn_pool is not None:


            dvd_term._pool = attn_pool
        mix = {k: float(v) for k, v in (kv.split("=") for kv in args.value_distill_mix.split(","))}
        distiller = None if args.no_value_distill else RoboMeterDistiller(value_fn, value_optimizer, spool=args.value_distill_robometer, device=device,
                                       frames_per_clip=args.value_distill_frames, every=args.value_distill_every,
                                       batch_size=args.value_distill_batch, capacity=args.value_distill_capacity,
                                       loss_weight=1.0, mix=mix, target=args.value_distill_target, aug_views=0, min_pool="0")
        print(f"[dsrl-dino] V {args.value_model} input_dim={value_fn.input_dim} proprio_dim={proprio_dim} "
              f"scratch={args.value_scratch} pool={args.dino_pool}"
              f"{'' if attn_pool is None else f' (mqattn K={attn_pool.k} per_view={attn_pool.per_view}, trained with V)'} "
              f"target_tau={args.value_target_tau}; distill spool={args.value_distill_robometer} mix={mix} "
              f"target={args.value_distill_target} pred={args.value_distill_pred}", flush=True)
    else:
        print("[dsrl-dino] --q-critic: NO world model, NO value model, NO mqattn readout, NO RoboMeter distiller.\n"
              f"[dsrl-dino] trained here: Q^A(s, decoded {env.action_space.shape[0]}-D action) by TD, "
              f"Q^W(s, w) distilled from Q^A ({args.noise_critic_steps} step(s)/update), the SAC noise actor and its "
              f"own IMPALA encoder; {args.n_critics} critics reduced by 'min', tau {args.tau}, "
              f"critic lr {args.critic_lr or args.lr}, critic grad clip {args.critic_grad_clip}, "
              f"TD-target entropy bonus {'ON (fork)' if args.q_target_entropy_bonus else 'OFF'}, "
              f"done = terminated-or-timeout, gamma^{args.action_steps} = {args.gamma ** args.action_steps:.4f}",
              flush=True)

    if args.log_std_min != _SAC_LOG_STD_MIN_DEFAULT:
        sac_policies.LOG_STD_MIN = float(args.log_std_min)
        print(f"[dsrl-dino] actor log_std floor {args.log_std_min} (SB3 default {_SAC_LOG_STD_MIN_DEFAULT}); "
              f"min sigma {math.exp(args.log_std_min):.2e}", flush=True)

    net = [args.layer_size] * args.num_layers
    if args.image_actor:
        extractor = dict(features_extractor_class=PixelProprioSlice,
                         features_extractor_kwargs=dict(out_dim=args.pixel_feature_dim, shift_pad=args.pixel_shift_pad))
    else:
        extractor = dict(features_extractor_class=StateSlice, features_extractor_kwargs=dict(state_dim=state_dim))
    model = DSRLDVDDino(
        "MlpPolicy", DummyVecEnv([lambda: env]), diffusion_policy=flow, diffusion_act_dim=(flow.horizon, flow.action_dim),
        noise_critic_grad_steps=args.noise_critic_steps, critic_backup_combine_type="min",
        learning_rate=args.lr, buffer_size=args.buffer_size, learning_starts=args.init_rollout_chunks,
        batch_size=args.batch_size, tau=args.tau, gamma=args.gamma ** args.action_steps, train_freq=args.train_freq,
        gradient_steps=args.utd, ent_coef="auto", target_update_interval=1, target_entropy=args.target_entropy,
        seed=args.seed, device=device, verbose=0, replay_buffer_class=DinoFrameReplayBuffer,

        replay_buffer_kwargs=(dict(store_prev=False) if args.q_critic else None),
        policy_kwargs=dict(net_arch=dict(pi=net, qf=net), activation_fn=ACTIVATIONS[args.activation],
                           log_std_init=0.0,
                           post_linear_modules=[torch.nn.LayerNorm], n_critics=args.n_critics,
                           share_features_extractor=False, **extractor))
    model.dvd_term, model.value_fn, model.value_optimizer = dvd_term, value_fn, value_optimizer
    model.adapter, model.distiller, model.attn_pool = adapter, distiller, attn_pool
    model.image_actor, model.frame_env = bool(args.image_actor), env
    model.actor_grad_clip = float(args.actor_grad_clip)
    model.entropy_per_dim, model.noise_dim = bool(args.entropy_per_dim), int(flow.horizon * flow.action_dim)
    model.deterministic_actor, model.explore_noise = bool(args.deterministic_actor), float(args.explore_noise)
    model.ent_coef_lr = float(args.ent_coef_lr)
    model.q_critic, model.critic_lr = bool(args.q_critic), float(args.critic_lr)
    model.critic_grad_clip, model.q_target_entropy = float(args.critic_grad_clip), bool(args.q_target_entropy_bonus)
    model.expert_ratio = float(args.expert_ratio)
    model.adaptive_expert = bool(args.adaptive_expert_ratio)
    model.expert_sched = (args.adaptive_expert_ratio_start, args.adaptive_expert_ratio_end,
                          int(args.adaptive_expert_ratio_steps))
    if args.expert_ratio > 0 or args.adaptive_expert_ratio:
        from .dsrl_demos import DemoStore
        cache = Path(os.environ.get("PIXEL_CACHE_DIR", str(CACHE_ROOT / "pixel_cache")))
        model.replay_buffer.demo = DemoStore.build(
            dataset_root=task_cfg["dataset_root"],
            pixel_cache_path=cache / f"pixel_cache_{Path(task_cfg['dataset_root']).name}_{dino_dvd.IMAGE_SIZE}.npy",


            flow=flow, scaler=scaler, action_steps=args.action_steps,
            step_penalty=args.step_penalty, device=device, capacity=int(args.expert_capacity),


            cache_dir=Path(os.environ.get("DSRL_DEMO_CACHE", str(CACHE_ROOT / "dsrl_demo_store"))),
            prior_id=str(checkpoint))
        s0, s1, st = model.expert_sched
        print(f"[dsrl-dino] expert demos in the replay: "
              f"{'adaptive ' + str(s0) + ' -> ' + str(s1) + ' over ' + str(st) + ' env steps' if args.adaptive_expert_ratio else 'fixed ' + str(args.expert_ratio)}"
              f"; NO BC loss (the actor samples its own noise, so demo w is never needed)", flush=True)

    model.policy_delay = int(args.policy_delay)
    model.dvd_diag_every = int(args.dvd_diag_every)
    model.dvd_diag_draws = int(args.dvd_diag_draws)
    model.explore_ramp = int(args.progressive_exploration_steps)
    if model.explore_ramp > 0:
        print(f"[dsrl-dino] progressive exploration: the actor's share of chunks rises 0 -> 1 over the first "
              f"{model.explore_ramp} env steps after the warm-up; the rest are N(0, I) prior draws", flush=True)
    if args.ent_coef_lr and not args.deterministic_actor:
        print(f"[dsrl-dino] SAC temperature lr {args.ent_coef_lr} (networks {args.lr}); target entropy {args.target_entropy}", flush=True)
    if args.deterministic_actor:
        print(f"[dsrl-dino] deterministic (TD3-style) noise actor: w = mu(s) + N(0, {args.explore_noise}^2) at collection, "
              "mu(s) at eval, no entropy term", flush=True)
    fx = model.actor.features_extractor
    print(f"[dsrl-dino] actor input: {type(fx).__name__} -> {fx.features_dim} "
          f"({'IMPALA ' + str(args.pixel_feature_dim) + 'x2 cams @224 + 9 scaled proprio, DrQ pad ' + str(args.pixel_shift_pad) + ', encoder in actor optimiser' if args.image_actor else 'scaled state prefix (privileged object poses included)'}); "
          f"actor MLP {net}, lr {args.lr}; batch {args.batch_size}, "
          f"{'value signal: TD pixel Q (Q^W)' if args.q_critic else ('dvd_rows ' + ('all' if args.dvd_rows <= 0 else str(args.dvd_rows)))}, "
          f"train every {args.train_freq} chunks x {args.utd} grad steps, {args.dvd_actor_steps} actor step(s)"
          f"{'' if args.q_critic else f', dvd_fit_utd {args.dvd_fit_utd}'}", flush=True)
    if args.q_critic:
        cx = model.critic.features_extractor
        print(f"[dsrl-dino] critic input: {type(cx).__name__} -> {cx.features_dim} + {env.action_space.shape[0]} action "
              f"dims; share_features_extractor=False, so Q^A, its Polyak target and Q^W each own an IMPALA encoder "
              f"trained by their own loss (DrQ pad {args.pixel_shift_pad} on the critic input AND the TD target)"
              if args.image_actor else
              f"[dsrl-dino] critic input: {type(cx).__name__} -> {cx.features_dim} (state prefix)", flush=True)
    model.dvd_weight, model.value_tau = args.dvd_weight, args.value_target_tau
    model.dvd_rows, model.dvd_fit_rows, model.dvd_checkpoint = args.dvd_rows, args.dvd_fit_rows, args.dvd_grad_checkpoint
    model.dvd_fit_utd, model.dvd_actor_steps, model.state_dim = args.dvd_fit_utd, args.dvd_actor_steps, state_dim
    if not args.q_critic and distiller is not None:
        env.episode_hook = episode_hook(model, device)
    if args.no_value_distill:
        print("[dsrl-dino] --no-value-distill: V trained by TD only; NO RoboMeter distiller, NO episode hook "
              "(the reward server is never contacted by this run)", flush=True)

    logger = DinoCampaignLog(out, args.total_env_steps, args.save_freq, env, wandb_run)
    logger.configure_async_eval(args.async_eval_episodes, args.async_eval_workers, args.async_eval_seed_start)
    started = time.time()
    model.learn(total_timesteps=10 ** 9, callback=logger, log_interval=None)
    logger.drain_evals()
    print(f"[dsrl-dino] done in {(time.time() - started) / 3600:.2f} h", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
