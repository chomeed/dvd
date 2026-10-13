"""DSRL (Wagenmaker et al., CoRL 2025) on MimicGen, steering the campaign's frozen DiT flow prior."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.utils.checkpoint
from gymnasium import spaces

_SB3 = Path(os.environ.get("DSRL_ROOT", Path(__file__).resolve().parent / "dsrl")) / "stable-baselines3"
if not (_SB3 / "stable_baselines3" / "dsrl").is_dir():
    raise SystemExit(
        f"DSRL's stable-baselines3 fork is missing at {_SB3}. Clone the DSRL repository with its "
        "stable-baselines3 submodule and set DSRL_ROOT to it (environment.md, section 3).")
sys.path.insert(0, str(_SB3))

from stable_baselines3 import DSRL, SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from ..common import TASKS, FrozenPolicyPrior, StateScaler, find_latest_checkpoint, set_seed  # noqa: E402
from ..train_online import make_env, rollout_chunk  # noqa: E402


class FlowNoisePrior:
    """prior(s, w):"""

    def __init__(self, prior: FrozenPolicyPrior, state_dim: int):
        self.prior = prior
        self.policy = prior.policy
        self.objective = self.policy.objective
        self.horizon = int(self.objective.horizon)
        self.action_dim = int(self.objective.action_dim)
        start = int(self.policy.config.n_obs_steps) - 1
        self.executed = slice(start, start + prior.action_steps)
        self.state_dim = state_dim
        self.device = prior.device
        self.dtype = next(self.policy.noise_predictor.parameters()).dtype

    @torch.no_grad()
    def cond(self, observation: dict) -> torch.Tensor:
        """The (1, C) conditioning vector, built on exactly select_action's path."""
        from lerobot.policies.utils import populate_queues

        policy = self.policy
        batch = dict(self.prior._prepare(observation))


        batch.pop("action", None)
        batch = policy._prepare_batch(batch)
        policy.reset()
        policy._queues = populate_queues(policy._queues, batch)
        batch = {k: torch.stack(list(policy._queues[k]), dim=1) if k in policy._queues else v
                 for k, v in batch.items()}
        return policy.observation_encoder.encode(batch)

    @torch.no_grad()
    def chunk(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """(B, C) cond + (B, H, A) noise -> (B, action_steps, A) env actions, clamped like the prior's."""
        obj = self.objective
        integrate = {"euler": obj._euler_integrate, "rk4": obj._rk4_integrate}[
            obj.config.integration_method]
        grid = torch.linspace(0, 1, obj.config.num_integration_steps + 1, device=self.device)
        x = integrate(self.policy.noise_predictor, noise.to(self.device, self.dtype), grid,
                      cond.to(self.device, self.dtype))
        actions = self.prior.postprocessor(x[:, self.executed])
        return actions.to(self.device, dtype=torch.float32).clamp(-1.0, 1.0)

    def chunk_with_grad(self, cond: torch.Tensor, noise: torch.Tensor,
                        checkpoint_steps: bool = False) -> torch.Tensor:
        """chunk() with the integration left differentiable w.r.t."""
        obj = self.objective
        if obj.config.integration_method != "euler":
            raise NotImplementedError("the differentiable path mirrors _euler_integrate only")
        steps = int(obj.config.num_integration_steps)
        grid = torch.linspace(0, 1, steps + 1, device=self.device)
        cond = cond.to(self.device, self.dtype)
        x = noise.to(self.device, self.dtype)
        for i in range(steps):
            t, dt = grid[i].item(), (grid[i + 1] - grid[i]).item()
            t_batch = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
            if checkpoint_steps:
                velocity = torch.utils.checkpoint.checkpoint(
                    self._velocity, x, t_batch, cond, use_reentrant=False)
            else:
                velocity = self._velocity(x, t_batch, cond)
            x = x + dt * velocity
        actions = self.prior.postprocessor(x[:, self.executed])
        return actions.to(self.device, dtype=torch.float32).clamp(-1.0, 1.0)

    def _velocity(self, x, t, cond):
        return self.policy.noise_predictor(x, t, conditioning_vec=cond)

    def pad(self, chunk: torch.Tensor) -> torch.Tensor:
        """(B, action_steps, A) -> (B, H * A), executed slice filled, the rest zero (note 3 above)."""
        out = chunk.new_zeros(chunk.shape[0], self.horizon, self.action_dim)
        out[:, self.executed] = chunk
        return out.reshape(chunk.shape[0], -1)

    def __call__(self, obs, noise, return_numpy: bool = True):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        noise = torch.as_tensor(noise, device=self.device).reshape(-1, self.horizon, self.action_dim)
        out = self.pad(self.chunk(obs[:, self.state_dim:], noise))
        return out.cpu().numpy() if return_numpy else out


def check_equivalence(flow: FlowNoisePrior, observation: dict, seed: int = 0) -> float:
    """Max |prior.action_chunk - flow.chunk(cond, same N(0, I) draw)|."""
    torch.manual_seed(seed)
    reference = flow.prior.action_chunk(observation)
    torch.manual_seed(seed)
    noise = torch.randn((1, flow.horizon, flow.action_dim), dtype=flow.dtype, device=flow.device)
    ours = flow.chunk(flow.cond(observation), noise)
    return float((reference - ours).abs().max())


def check_grad_path(flow: FlowNoisePrior, observation: dict) -> tuple[float, float]:
    """(max |chunk_with_grad - chunk|, ||d sum(a) / d w||)."""
    cond = flow.cond(observation).repeat(2, 1)
    noise = torch.randn((2, flow.horizon, flow.action_dim), device=flow.device, requires_grad=True)
    with torch.no_grad():
        reference = flow.chunk(cond, noise)
    ours = flow.chunk_with_grad(cond, noise)
    ours.sum().backward()
    grad = 0.0 if noise.grad is None else float(noise.grad.norm())
    return float((ours.detach() - reference).abs().max()), grad


class StateSlice(BaseFeaturesExtractor):
    """Actor and critics read the scaled-state prefix only; the cached DiT vector is for the prior."""

    def __init__(self, observation_space: spaces.Box, state_dim: int):
        super().__init__(observation_space, features_dim=state_dim)
        self.state_dim = state_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, : self.state_dim]


class DSRLMimicGenEnv(gymnasium.Env):
    """One transition = one action chunk, rewarded exactly as train_online's rollout_chunk."""

    def __init__(self, task: str, flow: FlowNoisePrior, scaler: StateScaler, *, seed: int,
                 variant: str, magnitude: float, step_penalty: float, dense_reward: bool,
                 horizon: int | None):
        self.env = make_env(task, horizon, dense_reward)
        self.flow, self.scaler, self.variant = flow, scaler, variant
        self.seed, self.episode_index = seed, 0
        self.step_penalty, self.dense_reward = step_penalty, dense_reward
        self.state_dim = int(TASKS[task]["state_dim"])
        probe, _ = self.env.reset(seed=seed)
        self.cond_dim = int(flow.cond(probe).shape[1])
        self.probe_observation = probe
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim + self.cond_dim,),
                                            np.float32)
        self.action_space = spaces.Box(-magnitude, magnitude, (flow.horizon * flow.action_dim,),
                                       np.float32)
        self._cond: torch.Tensor | None = None

    def _observe(self, observation: dict) -> np.ndarray:
        self._cond = self.flow.cond(observation)
        state = np.asarray(self.scaler(observation["agent_pos"]), dtype=np.float32)
        return np.concatenate((state, self._cond[0].float().cpu().numpy())).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, _ = self.env.reset(seed=self.seed + self.episode_index)
        self.episode_index += 1
        return self._observe(observation), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.variant == "sac":
            noise = torch.as_tensor(action, device=self.flow.device).view(
                1, self.flow.horizon, self.flow.action_dim)
            chunk = self.flow.chunk(self._cond, noise)[0].cpu().numpy()
        else:
            chunk = action.reshape(self.flow.horizon, self.flow.action_dim)[self.flow.executed]
        observation, reward, done, success, steps = rollout_chunk(
            self.env, chunk, step_penalty=self.step_penalty, use_env_reward=self.dense_reward)
        info = {"is_success": bool(success), "primitive_steps": int(steps)}
        return self._observe(observation), float(reward), bool(success), bool(done and not success), info


class DSRLNA(DSRL):
    """DSRL-NA whose warm-up draws w ~ N(0, I) through the prior, as the reference does."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps >= learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        noise = torch.randn(n_envs, self.diffusion_act_chunk, self.diffusion_act_dim,
                            device=self.device)
        action = self.diffusion_policy(self._last_obs, noise, return_numpy=True)
        return action, action

    def _excluded_save_params(self) -> list[str]:


        return super()._excluded_save_params() + ["diffusion_policy"]


class DSRLDVD(DSRLNA):
    """DSRL-NA whose noise actor climbs the DVD(1) value gamma^8 V(f(s, prior(s, w)))."""

    dvd_term = None
    value_fn = None
    value_optimizer = None
    dvd_weight = 1.0
    value_tau = 0.0
    dvd_rows = 256
    dvd_checkpoint = False
    dvd_fit_utd = 1
    dvd_actor_steps = -1
    state_dim = 0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        from stable_baselines3.common.utils import polyak_update

        lam = float(self.dvd_weight)
        use_q = lam < 1.0
        flow = self.diffusion_policy
        sd = self.state_dim
        magnitude = float(self.action_space.high[0])
        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        optimizers = [self.actor.optimizer]
        if use_q:
            optimizers += [self.critic.optimizer, self.critic_noise.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        n_actor = gradient_steps if self.dvd_actor_steps <= 0 else min(self.dvd_actor_steps, gradient_steps)
        actor_steps = set(np.linspace(gradient_steps - 1, 0, n_actor, dtype=int).tolist())
        n_fit = max(1, int(round(self.train_freq.frequency * self.dvd_fit_utd)))
        fit_steps = set(np.linspace(0, gradient_steps - 1, min(n_fit, gradient_steps), dtype=int).tolist())
        logs: dict[str, list[float]] = defaultdict(list)

        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = data.observations
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)


            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                logs["ent_coef_loss"].append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            logs["ent_coef"].append(ent_coef.item())

            if use_q:

                with torch.no_grad():
                    next_pi, next_log_prob = self.actor.action_log_prob(data.next_observations)
                    next_actions = flow(data.next_observations, next_pi * magnitude, return_numpy=False)
                    next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                    next_q = next_q.min(dim=1, keepdim=True).values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q = data.rewards + (1 - data.dones) * self.gamma * next_q
                current_q = self.critic(obs, data.actions)
                critic_loss = 0.5 * sum(torch.nn.functional.mse_loss(q, target_q) for q in current_q)
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                logs["critic_loss"].append(critic_loss.item())

            if step in actor_steps:
                actor_loss = (ent_coef * log_prob).mean()
                if use_q:
                    q_w = torch.cat(self.critic_noise(obs, actions_pi), dim=1).min(dim=1, keepdim=True).values
                    actor_loss = actor_loss + (1 - lam) * (
                        -q_w / q_w.detach().abs().mean().clamp_min(1e-3)).mean()
                rows = min(self.dvd_rows, batch_size)
                noise = (actions_pi[:rows] * magnitude).view(rows, flow.horizon, flow.action_dim)
                action = flow.chunk_with_grad(obs[:rows, sd:], noise, self.dvd_checkpoint)
                anchor = None
                if self.dvd_term.anchored:


                    anchor = (data.actions[:rows].view(rows, flow.horizon, flow.action_dim)
                              [:, flow.executed].reshape(rows, -1),
                              data.next_observations[:rows, :sd])
                dvd_objective, dvd_value = self.dvd_term.objective(
                    obs[:rows, :sd], action.reshape(rows, -1), None, anchor)
                actor_loss = actor_loss + lam * dvd_objective
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                logs["actor_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float("inf"))))
                self.actor.optimizer.step()
                logs["actor_loss"].append(actor_loss.item())
                logs["dvd_value"].append(dvd_value)
                logs["dvd_scale"].append(float(self.dvd_term.last_scale))
                logs["noise_abs_mean"].append(float(noise.detach().abs().mean()))

            if use_q and step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            if step in fit_steps:
                state, next_state = obs[:, :sd], data.next_observations[:, :sd]
                if self.value_optimizer is not None:
                    logs["dvd_v_td_loss"].append(self.value_fn.fit_step({
                        "state": state, "next_state": next_state, "reward": data.rewards,
                        "done": data.dones, "discount": torch.full_like(data.rewards, self.gamma),
                    }, self.value_optimizer))
                    self.value_fn.soft_update_target(self.value_tau)
                executed = data.actions.view(-1, flow.horizon, flow.action_dim)[:, flow.executed]
                for key, value in self.dvd_term.fit_step({
                        "state": state, "next_state": next_state,
                        "action": executed.reshape(len(state), -1)}).items():
                    logs[key].append(value)

        if use_q:
            for _ in range(self.noise_critic_grad_steps):
                data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                distill_loss = self.update_noise_critic(data)
                self.critic_noise.optimizer.zero_grad()
                distill_loss.backward()
                self.critic_noise.optimizer.step()
                logs["noise_critic_loss"].append(distill_loss.item())
        self.critic_noise.set_training_mode(False)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in logs.items():
            self.logger.record(f"train/{key}", float(np.mean(values)))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["dvd_term", "value_fn", "value_optimizer"]

    def dvd_state_dict(self) -> dict:
        state = {"value_fn": self.value_fn.state_dict(),
                 "dynamics": self.dvd_term.dynamics.state_dict()}
        if self.value_optimizer is not None:
            state["value_optimizer"] = self.value_optimizer.state_dict()
        target = getattr(self.value_fn, "_target", None)
        if target is not None:
            state["value_target"] = target.state_dict()
        if self.dvd_term.optimiser is not None:
            state["dynamics_optimizer"] = self.dvd_term.optimiser.state_dict()
        return state

    def load_dvd_state_dict(self, state: dict) -> None:
        self.value_fn.load_state_dict(state["value_fn"])
        self.dvd_term.dynamics.load_state_dict(state["dynamics"])
        if self.value_optimizer is not None:
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        if "value_target" in state:
            self.value_fn._target.load_state_dict(state["value_target"])
        if self.dvd_term.optimiser is not None:
            self.dvd_term.optimiser.load_state_dict(state["dynamics_optimizer"])


class DSRLSAC(SAC):
    """DSRL-SAC with the reference's prior warm-up:"""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps >= learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        high = float(self.action_space.high[0])
        noise = torch.randn(n_envs, *self.action_space.shape).clamp_(-high, high).numpy()
        return noise, self.policy.scale_action(noise)


class CampaignLog(BaseCallback):
    """train_online's metrics.jsonl rows (kind=train every 500 env steps) and checkpoint cadence."""

    def __init__(self, output_dir: Path, total_env_steps: int, save_freq: int, env, wandb_run=None,
                 log_every: int = 500):
        super().__init__()
        self.output_dir, self.total, self.save_freq = output_dir, total_env_steps, save_freq
        self.env, self.wandb_run, self.log_every = env, wandb_run, log_every
        self.metrics = output_dir / "metrics.jsonl"
        self.env_steps = self.episode_steps = self.episodes = self.last_save = 0
        self.window: dict[str, list[float]] = defaultdict(list)
        self.trailing: deque[float] = deque(maxlen=150)
        self.started = time.perf_counter()

    def restore(self, state: dict) -> None:
        self.env_steps, self.episodes = int(state["env_steps"]), int(state["episodes"])
        self.last_save = int(state["env_steps"])
        self.trailing.extend(state["trailing"])
        self.started = time.perf_counter() - float(state["elapsed_s"])

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            before = self.env_steps
            self.env_steps += int(info["primitive_steps"])
            self.episode_steps += int(info["primitive_steps"])
            if done:
                success = float(info["is_success"])
                self.window["episode_success"].append(success)
                self.window["episode_length"].append(float(self.episode_steps))
                self.trailing.append(success)
                self.episode_steps = 0
                self.episodes += 1
            if self.env_steps // self.log_every > before // self.log_every:
                self._log()
        if self.env_steps - self.last_save >= self.save_freq:
            self._save()
        return self.env_steps < self.total

    def _log(self) -> None:
        row = {key: float(np.mean(v)) for key, v in self.window.items() if v}
        row.update({
            "env_steps": self.env_steps,
            "updates": int(self.model._n_updates),
            "transitions": int(self.num_timesteps),
            "buffer_size": int(self.model.replay_buffer.size()),
            "episodes": self.episodes,
            "episode_success_trailing150": float(np.mean(self.trailing)) if self.trailing else None,
            "elapsed_s": time.perf_counter() - self.started,
        })
        row.update({key.split("/", 1)[1]: float(v)
                    for key, v in self.model.logger.name_to_value.items()
                    if key.startswith("train/") and np.isscalar(v)})
        with self.metrics.open("a") as handle:
            handle.write(json.dumps({"kind": "train", **row}) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log({f"train/{k}": v for k, v in row.items() if v is not None},
                               step=self.env_steps)
        print(json.dumps(row), flush=True)
        self.window.clear()

    def _save(self) -> None:
        path = self.output_dir / "checkpoints" / f"{self.env_steps:07d}"
        self.model.save(path)
        self.last_save = self.env_steps


        dvd_path = None
        if hasattr(self.model, "dvd_state_dict"):


            dvd_path = f"{path}_dvd.pt"
            torch.save(self.model.dvd_state_dict(), dvd_path)
        bundle = self.output_dir / "resume"
        bundle.mkdir(exist_ok=True)
        self.model.save_replay_buffer(bundle / "replay_buffer.tmp.pkl")
        (bundle / "replay_buffer.tmp.pkl").replace(bundle / "replay_buffer.pkl")
        state = {
            "checkpoint": f"{path}.zip", "dvd_checkpoint": dvd_path,
            "env_steps": self.env_steps, "episodes": self.episodes,
            "trailing": list(self.trailing), "elapsed_s": time.perf_counter() - self.started,
            "num_timesteps": int(self.model.num_timesteps), "n_updates": int(self.model._n_updates),
            "episode_index": int(self.env.episode_index),
            "wandb_id": None if self.wandb_run is None else self.wandb_run.id,
        }
        (bundle / "state.tmp.json").write_text(json.dumps(state))
        (bundle / "state.tmp.json").replace(bundle / "state.json")
        print(f"[dsrl] checkpoint {path}.zip (+ resume bundle)", flush=True)

    def _on_training_end(self) -> None:
        self._log()
        self._save()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="threading_d0", choices=sorted(TASKS))
    p.add_argument("--base-output", type=Path, required=True,
                   help="prior training dir; the latest checkpoint below it is used")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=("na", "sac"), default="na")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-env-steps", type=int, default=300_000)

    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--inference-steps", type=int, default=10)
    p.add_argument("--episode-horizon", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--env-dense-reward", action="store_true",
                   help="robosuite staged shaping; threading_d0 implements none (its branch is "
                        "`pass`), so there this is identical to the sparse reward")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="PER PRIMITIVE STEP, as in train_online; SB3 receives gamma**action_steps")

    p.add_argument("--action-magnitude", type=float, default=None,
                   help="noise box. Default 1.0 for na (the reference NA actor is squashed into "
                        "the env's [-1, 1] action box) and 1.5 for sac (the DSRL repository's README)")
    p.add_argument("--utd", type=int, default=20, help="gradient steps per update")
    p.add_argument("--train-freq", type=int, default=4,
                   help="transitions per update. The reference steps 4 envs per update, so 4 with "
                        "one env keeps its 20 gradient steps per 4 transitions")
    p.add_argument("--noise-critic-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--layer-size", type=int, default=2048)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--n-critics", type=int, default=2)
    p.add_argument("--target-entropy", type=float, default=0.0)
    p.add_argument("--init-rollout-chunks", type=int, default=4000,
                   help="prior-only warm-up transitions. The reference's 2001 x 4 envs x 4 steps "
                        "= 32k primitive steps; 4000 x 8 matches that. Counted in the budget")
    p.add_argument("--buffer-size", type=int, default=60_000,
                   help="~4.2k float32 per obs x 2 per row: 60k rows ~ 2 GB")
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--save-replay", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from --output-dir/resume (written at every checkpoint)")
    p.add_argument("--no-tf32", action="store_true",
                   help="keep fp32 matmuls. TF32 is 2.8x faster on the 100M-param DiT at batch "
                        "256 (254 -> 92 ms per integration, A100) for a 2.7e-4 max action gap")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-name", default=None)

    g = p.add_argument_group("dvd", "noise actor climbs gamma^8 V(f(s, prior(s, w))) (variant na only)")
    g.add_argument("--dvd-weight", type=float, default=0.0,
                   help="0 = DSRL-NA; 1 = critic-free DVD (the vf arm's objective); between = mix")
    g.add_argument("--value-model", type=Path, default=None,
                   help="V directory (weights/, config.json, state_scaler.json): a state-space V here, "
                        "the feature-space $CKPT_ROOT/value/<V dir> in dsrl_dino, which shares this parser")
    g.add_argument("--value-variant", default="v_td")
    g.add_argument("--dvd-dynamics", default="model_residual_threadfull")
    g.add_argument("--no-value-finetune", action="store_true")
    g.add_argument("--no-dvd-finetune-dynamics", action="store_true")
    g.add_argument("--value-lr", type=float, default=1e-4)
    g.add_argument("--dvd-dynamics-lr", type=float, default=1e-4)
    g.add_argument("--value-target-tau", type=float, default=0.005,
                   help="Polyak target for V's TD bootstrap; 0 = bootstrap from V itself")
    g.add_argument("--dvd-fit-utd", type=float, default=1.0,
                   help="V and f steps per collected transition (the vf arm: 1)")
    g.add_argument("--dvd-actor-steps", type=int, default=5,
                   help="actor steps per update (of --utd). Each backprops through 10 DiT calls: "
                        "467 ms at 256 rows on a 3090, so all 20 would be ~24 h per run")
    g.add_argument("--dvd-rows", type=int, default=256, help="rows per actor step taken through f")
    g.add_argument("--dvd-grad-checkpoint", action="store_true",
                   help="recompute DiT activations in backward: 12.6 -> 2.3 GiB at 256 rows, 1.4x slower")
    g.add_argument("--dvd-fixed-scale", type=float, default=0.0)
    g.add_argument("--dvd-static-grad", default="none",
                   choices=("none", "degenerate", "degenerate_wide", "goal"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.dvd_weight > 0.0 and (args.variant != "na" or args.value_model is None):
        raise SystemExit("--dvd-weight > 0 needs --variant na and --value-model")
    if args.action_magnitude is None:
        args.action_magnitude = 1.0 if args.variant == "na" else 1.5
    out = args.output_dir
    state = None
    if args.resume:
        state_path = out / "resume" / "state.json"
        if not state_path.exists():
            raise SystemExit(f"--resume: {state_path} does not exist (no checkpoint was completed)")
        state = json.loads(state_path.read_text())
        from ..legacy import refuse_unconverted_resume_state
        refuse_unconverted_resume_state(state, state_path)
        if state["env_steps"] >= args.total_env_steps:
            print(f"[dsrl] {out} already reached {state['env_steps']} env steps; nothing to do")
            return
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is populated; refusing to overwrite (pass --resume to continue)")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if state is None:
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "method": f"dsrl_{args.variant}"}, indent=2) + "\n")

    set_seed(args.seed + (0 if state is None else int(state["env_steps"])))

    task_cfg = TASKS[args.task]
    state_dim = int(task_cfg["state_dim"])
    checkpoint = find_latest_checkpoint(args.base_output)
    prior = FrozenPolicyPrior(args.task, checkpoint, args.action_steps, args.inference_steps)
    flow = FlowNoisePrior(prior, state_dim)
    scaler = StateScaler(task_cfg["dataset_root"])
    env = DSRLMimicGenEnv(args.task, flow, scaler, seed=args.seed, variant=args.variant,
                          magnitude=args.action_magnitude, step_penalty=args.step_penalty,
                          dense_reward=args.env_dense_reward, horizon=args.episode_horizon)
    gap = check_equivalence(flow, env.probe_observation)
    print(f"[dsrl] prior {checkpoint}\n[dsrl] noise {flow.horizon}x{flow.action_dim} "
          f"executed {flow.executed}  cond {env.cond_dim}  equivalence gap {gap:.2e}", flush=True)
    if gap > 1e-4:
        raise SystemExit(f"flow(cond, w) differs from prior.action_chunk by {gap:.3e}; "
                         "the steered policy would not start from the prior")
    if not args.no_tf32:

        cond = flow.cond(env.probe_observation)
        noise = torch.randn((1, flow.horizon, flow.action_dim), device=flow.device)
        exact = flow.chunk(cond, noise)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tf_gap = float((flow.chunk(flow.cond(env.probe_observation), noise) - exact).abs().max())
        print(f"[dsrl] TF32 on: action gap vs fp32 {tf_gap:.2e}", flush=True)
    if args.dvd_weight > 0.0:
        grad_gap, grad_norm = check_grad_path(flow, env.probe_observation)
        print(f"[dsrl-dvd] differentiable prior: gap vs sampler {grad_gap:.2e}  "
              f"|d a / d w| {grad_norm:.3e}", flush=True)
        if grad_gap > 1e-5 or not grad_norm > 0.0:
            raise SystemExit("the differentiable prior path does not match the sampler or carries "
                             "no gradient; DVD would silently train on the entropy term alone")

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or out.name,
                               config=vars(args), dir=str(out),
                               id=None if state is None else state.get("wandb_id"),
                               resume="allow")

    net = [args.layer_size] * args.num_layers
    common = dict(
        learning_rate=args.lr, buffer_size=args.buffer_size,
        learning_starts=args.init_rollout_chunks, batch_size=args.batch_size, tau=args.tau,
        gamma=args.gamma ** args.action_steps, train_freq=args.train_freq,
        gradient_steps=args.utd, ent_coef="auto", target_update_interval=1,
        target_entropy=args.target_entropy, seed=args.seed, device=prior.device, verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=net, qf=net), activation_fn=torch.nn.Tanh, log_std_init=0.0,
            post_linear_modules=[torch.nn.LayerNorm], n_critics=args.n_critics,
            features_extractor_class=StateSlice,
            features_extractor_kwargs=dict(state_dim=state_dim),
            share_features_extractor=False),
    )
    vec_env = DummyVecEnv([lambda: env])
    if args.dvd_weight > 0.0:
        from ..dvd_term import DvdTerm

        from ..value_model import ValueEnsemble

        value_finetune = not args.no_value_finetune
        value_fn = ValueEnsemble(args.value_model, args.value_variant, scaler, state_dim=state_dim,
                                 device=prior.device, finetune=value_finetune)
        value_optimizer = None
        if value_finetune:
            if args.value_target_tau > 0.0:
                value_fn.enable_target(prior.device)
            value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=args.value_lr)
        dvd_term = DvdTerm(
            value_fn, scaler, args.action_steps * flow.action_dim, prior.device,
            discount=args.gamma, horizon=args.action_steps, chunks=1,
            model_tag=args.dvd_dynamics, finetune_dynamics=not args.no_dvd_finetune_dynamics,
            dynamics_lr=args.dvd_dynamics_lr, static_grad_block=args.dvd_static_grad,
            goal_slice=(23, 26), fixed_scale=args.dvd_fixed_scale)
        if dvd_term.generative and not dvd_term.anchored:
            raise SystemExit(f"{args.dvd_dynamics} is a Gaussian f (mu_scale); DSRLDVD only "
                             "anchors flow models. Use model_residual_* or model_flow_*")
        model = DSRLDVD("MlpPolicy", vec_env, diffusion_policy=flow,
                        diffusion_act_dim=(flow.horizon, flow.action_dim),
                        noise_critic_grad_steps=args.noise_critic_steps,
                        critic_backup_combine_type="min", **common)
        model.dvd_term, model.value_fn, model.value_optimizer = dvd_term, value_fn, value_optimizer
        model.dvd_weight, model.value_tau = args.dvd_weight, args.value_target_tau
        model.dvd_rows, model.dvd_checkpoint = args.dvd_rows, args.dvd_grad_checkpoint
        model.dvd_fit_utd, model.dvd_actor_steps = args.dvd_fit_utd, args.dvd_actor_steps
        model.state_dim = state_dim
    elif args.variant == "na":
        model = DSRLNA("MlpPolicy", vec_env, diffusion_policy=flow,
                       diffusion_act_dim=(flow.horizon, flow.action_dim),
                       noise_critic_grad_steps=args.noise_critic_steps,
                       critic_backup_combine_type="min", **common)
    else:
        model = DSRLSAC("MlpPolicy", vec_env, **common)

    logger = CampaignLog(out, args.total_env_steps, args.save_freq, env, wandb_run)
    if state is not None:
        from stable_baselines3.common.save_util import load_from_zip_file


        _, params, tensors = load_from_zip_file(state["checkpoint"], device=prior.device)
        model.set_parameters(params, exact_match=True, device=prior.device)
        for name, value in (tensors or {}).items():
            getattr(model, name).data.copy_(value.data)
        model.load_replay_buffer(out / "resume" / "replay_buffer.pkl")
        if state.get("dvd_checkpoint"):
            model.load_dvd_state_dict(torch.load(state["dvd_checkpoint"], map_location=prior.device))
        model.num_timesteps, model._n_updates = int(state["num_timesteps"]), int(state["n_updates"])
        env.episode_index = int(state["episode_index"])
        logger.restore(state)
        print(f"[dsrl] resumed at env_steps={logger.env_steps} updates={model._n_updates} "
              f"buffer={model.replay_buffer.size()}", flush=True)
    model.learn(total_timesteps=10 ** 9, callback=logger, log_interval=None,
                reset_num_timesteps=state is None)
    if args.save_replay:
        model.save_replay_buffer(out / "replay_buffer.pkl")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
