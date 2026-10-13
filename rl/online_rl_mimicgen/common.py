from __future__ import annotations
import os

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


_ASSETS = Path(os.environ.get("ASSET_ROOT", Path(__file__).resolve().parents[2] / "assets"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", _ASSETS / "datasets"))
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", _ASSETS / "checkpoints"))
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", _ASSETS / "cache"))


TASKS = {

    "hammer_cleanup_d1": {
        "dataset_root": DATA_ROOT / "mimicgen_hammer_cleanup_d1_224x224",
        "il_output": CKPT_ROOT / "priors/hammer_cleanup_d1_35k_statefix",
        "state_dim": 37,
        "horizon": 500,
    },

    "threading_d0": {
        "dataset_root": DATA_ROOT / "mimicgen_threading_d0_224x224",
        "il_output": CKPT_ROOT / "priors/threading_d0_55k_statefix",
        "state_dim": 37,
        "horizon": 400,
    },

    "square_d1": {
        "dataset_root": DATA_ROOT / "mimicgen_square_d1_224x224",
        "il_output": CKPT_ROOT / "priors/square_d1_45k_statefix",
        "state_dim": 26,
        "horizon": 400,
    },

    "square_d0": {
        "dataset_root": DATA_ROOT / "mimicgen_square_d0_224x224",
        "il_output": CKPT_ROOT / "priors/square_d0_40k_rs9",
        "state_dim": 23,
        "horizon": 400,
    },
}


PRIOR_POLICY_TYPES = ("diffusion", "multi_task_dit", "multi_task_dit_patch")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(output_dir: Path) -> Path:
    last = output_dir / "checkpoints/last/pretrained_model"
    if (last / "config.json").exists():
        return last.resolve()
    candidates = sorted(
        path / "pretrained_model"
        for path in (output_dir / "checkpoints").glob("[0-9]*")
        if (path / "pretrained_model/config.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No LeRobot checkpoint found below {output_dir}")
    return candidates[-1].resolve()


class StateScaler:
    """q01/q99 min-max scaling to [-1, 1], clipped at +/-5."""

    FLOOR = 1e-5

    def __init__(self, dataset_root: Path):
        stats = json.loads((dataset_root / "meta/stats.json").read_text())["observation.state"]
        self._fit(stats, dataset_root.name)

    def _fit(self, stats: dict, name: str) -> None:
        self.low = np.asarray(stats["q01"], dtype=np.float32)
        self.high = np.asarray(stats["q99"], dtype=np.float32)
        span = self.high - self.low
        collapsed = span <= self.FLOOR
        if collapsed.any() and "min" in stats and "max" in stats:
            low, high = (np.asarray(stats[k], dtype=np.float32) for k in ("min", "max"))
            usable = collapsed & ((high - low) > self.FLOOR)
            if usable.any():
                self.low = np.where(usable, low, self.low)
                self.high = np.where(usable, high, self.high)
                self.degenerate_dims = np.flatnonzero(usable).tolist()


                print(f"[StateScaler] {name}: quantiles collapsed on dims "
                      f"{self.degenerate_dims}; using min/max there. Consider repairing "
                      f"meta/stats.json.", flush=True)
            else:
                self.degenerate_dims = []
        else:
            self.degenerate_dims = []
        self.scale = np.maximum(self.high - self.low, 1e-6)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self._scale(state)

    def _scale(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return np.clip(2.0 * (state - self.low) / self.scale - 1.0, -5.0, 5.0).astype(np.float32)

    def audit(self, states: np.ndarray) -> dict:
        """Fraction of each dim that saturates at the clip."""
        states = np.asarray(states, dtype=np.float32)
        z = self._scale(states)
        frac = (np.abs(z) >= 4.999).mean(axis=0)
        raw_sd = states.std(axis=0)
        bad = np.flatnonzero((frac > 0.5) & (raw_sd > 1e-3))
        return {"clip_fraction": frac, "saturated_dims": bad.tolist()}


GOAL_RELATIVE_MAPS = {


    "threading_d0": {"state_dim": 37, "obj_pos": (9, 12), "goal_pos": (23, 26)},

    "square_d1": {"state_dim": 26, "obj_pos": (9, 12), "goal_pos": (23, 26)},


    "hammer_cleanup_d1": {"state_dim": 37, "obj_pos": (9, 12), "goal_pos": (23, 26)},
}


class GoalRelativeScaler(StateScaler):
    """StateScaler over the goal-relative representation (--state-repr goal_relative)."""

    def __init__(self, dataset_root: Path, task: str):
        if task not in GOAL_RELATIVE_MAPS:
            raise ValueError(
                f"--state-repr goal_relative has no dim map for task {task!r}; add it to "
                f"GOAL_RELATIVE_MAPS (known: {sorted(GOAL_RELATIVE_MAPS)})"
            )
        spec = GOAL_RELATIVE_MAPS[task]
        if int(spec["state_dim"]) != int(TASKS[task]["state_dim"]):
            raise RuntimeError(
                f"GOAL_RELATIVE_MAPS[{task!r}] says state_dim {spec['state_dim']} but TASKS "
                f"says {TASKS[task]['state_dim']}; fix whichever is wrong before training"
            )
        self.task = task
        self.obj_slice = slice(*spec["obj_pos"])
        self.goal_slice = slice(*spec["goal_pos"])


        super().__init__(dataset_root)
        import pyarrow.dataset as pads

        table = pads.dataset(dataset_root / "data", format="parquet").to_table(
            columns=["observation.state"]
        )
        states = np.asarray(
            table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32
        )
        if states.shape[1] != int(spec["state_dim"]):
            raise RuntimeError(
                f"{dataset_root} parquet states are {states.shape[1]}-dim but the "
                f"{task} dim map expects {spec['state_dim']}"
            )
        rel = states[:, self.obj_slice] - states[:, self.goal_slice]
        fresh_low = np.quantile(rel, 0.01, axis=0).astype(np.float32)
        fresh_high = np.quantile(rel, 0.99, axis=0).astype(np.float32)
        if (fresh_high - fresh_low <= self.FLOOR).any():
            raise RuntimeError(
                f"goal_relative on {task}: fresh quantiles collapsed on the replaced dims "
                f"(span {fresh_high - fresh_low}); the relative pose is degenerate in the demos"
            )
        self.low[self.obj_slice] = fresh_low
        self.high[self.obj_slice] = fresh_high
        self.scale = np.maximum(self.high - self.low, 1e-6)
        start, stop = spec["obj_pos"]
        self.degenerate_dims = [d for d in self.degenerate_dims if not start <= d < stop]
        print(
            f"[state-repr] goal_relative on {task}: dims {spec['obj_pos']} <- obj_pos - "
            f"goal_pos (goal dims {spec['goal_pos']} kept absolute); fresh quantiles on the "
            f"replaced dims only, fit on {len(states)} demo frames; all other dims scale "
            f"bit-identically to the absolute arm", flush=True,
        )

    def transform(self, state: np.ndarray) -> np.ndarray:
        """RAW absolute state -> RAW goal-relative state."""
        state = np.asarray(state, dtype=np.float32)
        out = state.copy()
        out[..., self.obj_slice] = state[..., self.obj_slice] - state[..., self.goal_slice]
        return out

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self._scale(self.transform(state))

    def audit(self, states: np.ndarray) -> dict:


        return super().audit(self.transform(states))


def check_state_repr(config: "TrainConfig") -> None:
    """Startup guard for --state-repr."""
    if config.state_repr == "absolute":
        return
    if config.state_repr != "goal_relative":
        raise ValueError(f"unknown --state-repr {config.state_repr!r}")
    if config.task not in GOAL_RELATIVE_MAPS:
        raise ValueError(
            f"--state-repr goal_relative has no dim map for task {config.task!r}; add it to "
            f"GOAL_RELATIVE_MAPS (known: {sorted(GOAL_RELATIVE_MAPS)})"
        )
    absolute_fit = []
    if config.value_model:
        absolute_fit.append(f"--value-model {config.value_model} (offline V)")
    if config.dvd_weight > 0.0 or config.bon_source == "dvd" or config.mbpo_weight > 0.0:
        absolute_fit.append(
            f"--dvd-dynamics {config.dvd_dynamics} (offline dynamics f, via dvd_weight="
            f"{config.dvd_weight} / bon_source={config.bon_source!r} / mbpo_weight="
            f"{config.mbpo_weight})"
        )
    if config.init_actor_from:
        absolute_fit.append(f"--init-actor-from {config.init_actor_from}")
    if config.init_critic_from:
        absolute_fit.append(f"--init-critic-from {config.init_critic_from}")
    if absolute_fit:
        raise ValueError(
            "--state-repr goal_relative changes the state every network consumes, but these "
            "were fit on ABSOLUTE states and would silently read the wrong input space: "
            + "; ".join(absolute_fit)
            + ". Refit them on the goal-relative representation (there is no escape flag)."
        )


def _set_sampling_steps(cfg, steps: int) -> None:
    """Point the generative policy at `steps` denoising/integration steps."""
    if getattr(cfg, "is_flow_matching", False):
        cfg.num_integration_steps = steps
    else:
        cfg.num_inference_steps = steps


class FrozenPolicyPrior:
    """Frozen LeRobot generative policy with a shorter online replan chunk."""

    def __init__(
        self,
        task: str,
        checkpoint: Path,
        action_steps: int,
        inference_steps: int,
        device: str = "cuda",
        prior_temperature: float = 1.0,
    ):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.import_utils import register_third_party_plugins
        from lerobot_env_mimicgen import MimicGenEnvConfig

        register_third_party_plugins()
        self.task = task
        self.device = torch.device(device)
        self.env_cfg = MimicGenEnvConfig(task=task)
        self.cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if self.cfg.type not in PRIOR_POLICY_TYPES:
            raise ValueError(
                f"Online adapters support {PRIOR_POLICY_TYPES} priors, got {self.cfg.type}"
            )
        self.cfg.pretrained_path = checkpoint
        self.cfg.device = str(self.device)
        self.cfg.n_action_steps = int(action_steps)
        _set_sampling_steps(self.cfg, int(inference_steps))
        self.policy = make_policy(cfg=self.cfg, env_cfg=self.env_cfg)
        self.policy.eval().requires_grad_(False)


        _set_sampling_steps(self.policy.config, int(inference_steps))
        if hasattr(self.policy, "diffusion"):
            self.policy.diffusion.num_inference_steps = int(inference_steps)
        self.prior_temperature = float(prior_temperature)
        if self.prior_temperature != 1.0:
            self._scale_sampling_noise(self.prior_temperature)
        if hasattr(self.policy, "objective"):
            _set_sampling_steps(self.policy.objective.config, int(inference_steps))
            if hasattr(self.policy.objective, "num_inference_steps"):
                self.policy.objective.num_inference_steps = int(inference_steps)
        self.inference_steps = int(inference_steps)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.action_steps = int(action_steps)
        self.horizon = int(self.cfg.horizon)
        self.action_dim = int(self.cfg.output_features["action"].shape[0])
        self.checkpoint = checkpoint
        self.prior_state_dim = self._resolve_state_width()

    def _resolve_state_width(self) -> int | None:
        """How many leading state dims this prior actually consumes, or None if it takes them all."""
        want = self.cfg.input_features.get("observation.state")
        if want is None:
            return None
        want = int(want.shape[0])
        have = int(TASKS[self.task]["state_dim"])
        if want == have:
            return None
        if want > have:
            raise ValueError(
                f"prior at {self.checkpoint} wants a {want}-dim observation.state but "
                f"{self.task} only has {have} dims; wrong prior for this task"
            )
        if want != 9:
            raise ValueError(
                f"prior at {self.checkpoint} wants {want} of {self.task}'s {have} state dims. "
                "Only the 9-dim proprio prefix is a slice this code can justify -- see "
                "FrozenPolicyPrior._resolve_state_width."
            )
        print(f"[prior] narrowing observation.state {have} -> {want} (proprio prefix) for "
              f"{self.checkpoint.name}; critic/V/dynamics keep all {have}", flush=True)
        return want

    def _narrow(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Trim observation.state to what the prior was trained on."""
        if self.prior_state_dim is not None and "observation.state" in batch:
            batch["observation.state"] = batch["observation.state"][..., : self.prior_state_dim]
        return batch

    def _prepare(self, observation: dict[str, Any]) -> dict[str, Any]:
        from lerobot.envs.utils import preprocess_observation

        batch = self._narrow(preprocess_observation(observation))
        batch["task"] = [self.env_cfg.task]
        return self.preprocessor(batch)

    @torch.inference_mode()
    def action_chunk(self, observation: dict[str, Any], continuation: bool = False):
        """Return unnormalized environment actions as a CUDA tensor (1, T, A)."""
        batch = self._prepare(observation)
        self.policy.reset()
        with self._plan_capture(continuation) as plan:
            first = self.policy.select_action(batch)
        remaining = list(self.policy._queues["action"])
        normalized = torch.stack([first, *remaining], dim=1)
        if normalized.shape[1] != self.action_steps:
            raise RuntimeError(f"Expected {self.action_steps} prior actions, got {normalized.shape}")
        actions = self.postprocessor(normalized)
        actions = actions.to(self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        if not continuation:
            return actions
        return actions, self._continuation(plan, normalized)


    def _plan_owner(self):
        """The module whose conditional_sample() returns the FULL horizon plan:"""
        for name in ("objective", "diffusion"):
            owner = getattr(self.policy, name, None)
            if owner is not None and hasattr(owner, "conditional_sample"):
                return owner
        raise RuntimeError(f"prior continuation: {self.cfg.type} exposes no conditional_sample to read "
                           "the full-horizon plan from")

    def _plan_capture(self, enabled: bool):
        """Context manager recording the full (B, horizon, A) NORMALISED plan select_action samples."""
        import contextlib

        if not enabled:
            return contextlib.nullcontext(None)

        @contextlib.contextmanager
        def _capture():
            owner = self._plan_owner()
            had_instance_attr = "conditional_sample" in owner.__dict__
            inner = owner.conditional_sample
            plan: list[torch.Tensor] = []

            def recording(*args, **kwargs):
                out = inner(*args, **kwargs)
                plan.append(out)
                return out

            owner.conditional_sample = recording
            try:
                yield plan
            finally:
                if had_instance_attr:
                    owner.conditional_sample = inner
                else:
                    del owner.conditional_sample
        return _capture()

    def _continuation(self, plan: list, normalized: torch.Tensor) -> torch.Tensor:
        """Steps [A, 2A) after the executed chunk of the SAME draw, unnormalised and clamped like it."""
        if len(plan) != 1:
            raise RuntimeError(f"prior continuation: expected ONE conditional_sample call per chunk, saw {len(plan)}")
        full = plan[0]
        a = self.action_steps
        start = int(getattr(self.policy.config, "n_obs_steps", 1)) - 1
        if full.shape[1] < start + 2 * a:
            raise RuntimeError(f"prior continuation: plan horizon {full.shape[1]} < {start + 2 * a} "
                               f"(n_obs_steps-1 + 2 x {a} action steps)")
        if not torch.equal(full[:, start:start + a].to(normalized.dtype), normalized):
            raise RuntimeError("prior continuation: the captured plan's slice [n_obs_steps-1, +action_steps) does "
                               "not reproduce the executed chunk -- the policy's slicing is not what this assumes")
        cont = self.postprocessor(full[:, start + a:start + 2 * a].to(normalized.dtype))
        return cont.to(self.device, dtype=torch.float32).clamp(-1.0, 1.0)

    @torch.inference_mode()

    def _scale_sampling_noise(self, sigma: float) -> None:
        """Widen the flow's initial noise draw so the K candidate chunks spread further apart."""
        objective = getattr(self.policy, "objective", None)
        if objective is None or not hasattr(objective, "conditional_sample"):
            raise RuntimeError(
                "--prior-temperature needs policy.objective.conditional_sample; this prior "
                f"({self.cfg.type}) does not expose it"
            )
        original = objective.conditional_sample

        def scaled(model, batch_size, conditioning_vec, _orig=original, _s=sigma):
            real_randn = torch.randn

            def wider(*args, **kwargs):
                return real_randn(*args, **kwargs) * _s

            torch.randn = wider
            try:
                return _orig(model, batch_size, conditioning_vec)
            finally:
                torch.randn = real_randn

        objective.conditional_sample = scaled
        print(f"[prior] sampling temperature {sigma}: the flow's initial noise is scaled, "
              f"weights untouched", flush=True)

    def action_chunk_samples(self, observation: dict[str, Any], samples: int,
                             continuation: bool = False):
        """Return `samples` independent prior chunks for one observation, (S, T, A)."""
        from lerobot.envs.utils import preprocess_observation

        batch = self._narrow(preprocess_observation(observation))
        batch = {
            key: value.repeat(samples, *([1] * (value.ndim - 1))) for key, value in batch.items()
        }
        batch["task"] = [self.env_cfg.task] * samples
        batch = self.preprocessor(batch)
        self.policy.reset()
        with self._plan_capture(continuation) as plan:
            first = self.policy.select_action(batch)
        remaining = list(self.policy._queues["action"])
        normalized = torch.stack([first, *remaining], dim=1)
        if normalized.shape[:2] != (samples, self.action_steps):
            raise RuntimeError(
                f"Expected {(samples, self.action_steps)} prior actions, got {normalized.shape}"
            )
        actions = self.postprocessor(normalized)
        actions = actions.to(self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        if not continuation:
            return actions
        return actions, self._continuation(plan, normalized)

    def close(self) -> None:
        del self.policy
        torch.cuda.empty_cache()


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: tuple[int, ...], layernorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for size in hidden:
            layers.append(nn.Linear(current, size))
            if layernorm:
                layers.append(nn.LayerNorm(size))
            layers.append(nn.Mish())
            current = size
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ChunkConvHead(nn.Module):
    """Emit the action chunk as a 1D signal over time instead of 56 independent scalars."""

    def __init__(self, feature_dim: int, action_steps: int, action_dim: int,
                 kernel: int, channels: int, broadcast: bool = False):
        super().__init__()
        self.action_steps = action_steps
        self.action_dim = action_dim


        self.broadcast = broadcast
        self.project = nn.Linear(feature_dim, channels if broadcast else action_steps * channels)


        self.position = nn.Parameter(torch.zeros(1, channels, action_steps))
        padding = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=padding),
            nn.SiLU(),
            nn.Conv1d(channels, action_dim, kernel, padding=padding),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.broadcast:
            x = self.project(feature).unsqueeze(-1).expand(-1, -1, self.action_steps)
        else:
            x = self.project(feature).view(-1, self.position.shape[1], self.action_steps)
        x = self.conv(x + self.position)
        return x.transpose(1, 2).reshape(feature.shape[0], -1)


class ResidualActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden: tuple[int, ...] = (256, 256, 256),
        layernorm: bool = False,
        initial_log_std: float = -3.0,
        log_std_min: float = -5.0,
        stochastic: bool = True,
        latent_dim: int = 0,
        head: str = "mlp",
        action_steps: int = 8,
        conv_kernel: int = 3,
        conv_channels: int = 16,
        conv_broadcast: bool = False,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.log_std_min = log_std_min
        self.stochastic = stochastic


        self.latent_dim = latent_dim
        self.backbone = MLP(
            state_dim + chunk_dim + latent_dim, hidden[-1], hidden[:-1], layernorm
        )


        self.head = head
        self.action_steps = action_steps
        self.action_dim = chunk_dim // action_steps
        if head == "conv":
            self.mean = ChunkConvHead(
                hidden[-1], action_steps, self.action_dim, conv_kernel, conv_channels,
                broadcast=conv_broadcast,
            )
        else:
            self.mean = nn.Linear(hidden[-1], chunk_dim)
            nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.mean.bias)


        if stochastic:


            self.log_std = nn.Linear(hidden[-1], chunk_dim)
            nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
            nn.init.constant_(self.log_std.bias, initial_log_std)

    def forward(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        parts = [state, base_action]
        if self.latent_dim:


            parts.append(
                torch.zeros(state.shape[0], self.latent_dim, device=state.device,
                            dtype=state.dtype)
                if latent is None else latent
            )
        feature = self.backbone(torch.cat(parts, dim=-1))
        mean = self.mean(feature)
        if not self.stochastic:
            return mean, None
        log_std = self.log_std(feature).clamp(self.log_std_min, 1.0)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = None
        if self.latent_dim and not deterministic:
            latent = torch.randn(
                state.shape[0], self.latent_dim, device=state.device, dtype=state.dtype
            )
        if not self.stochastic:
            mean, _ = self(state, base_action, latent)
            residual = torch.tanh(mean)


            return residual, torch.zeros((state.shape[0], 1), device=state.device)
        mean, log_std = self(state, base_action, latent)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        residual = torch.tanh(raw)
        if deterministic:
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
        else:
            log_prob = normal.log_prob(raw) - torch.log(1.0 - residual.square() + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return residual, log_prob

    def log_prob_of(self, state, base_action, residual, latent=None):
        """log p(residual | state, base_action) under the CURRENT parameters."""
        mean, log_std = self(state, base_action, latent)
        normal = torch.distributions.Normal(mean, log_std.exp())
        safe = residual.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(safe)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - safe.square() + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class CriticEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        size: int,
        hidden: tuple[int, ...] = (256, 256, 256),
        layernorm: bool = False,
    ):
        super().__init__()
        self.q = nn.ModuleList(
            MLP(state_dim + chunk_dim, 1, hidden, layernorm) for _ in range(size)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action), dim=-1)
        return torch.stack([critic(value) for critic in self.q], dim=0)

    def conservative(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self(state, action).min(dim=0).values

    def reduce(self, state: torch.Tensor, action: torch.Tensor, mode: str) -> torch.Tensor:
        """Ensemble reduction chosen at the CALL SITE, for best-of-N ranking only."""
        values = self(state, action)
        if mode.startswith("lcb"):


            lam = float(mode[3:]) if len(mode) > 3 else 2.0
            return values.mean(dim=0) - lam * values.std(dim=0)
        if mode == "max":
            return values.max(dim=0).values
        if mode == "mean":
            return values.mean(dim=0)
        if mode == "min":
            return values.min(dim=0).values
        raise ValueError(f"reduction must be min, max, mean or lcb[lam]; got {mode!r}")


class ReplayBuffer:


    FIELDS = (
        "state", "base", "action", "next_state", "next_base", "reward", "done", "discount",
        "mc_return", "data_source", "base_samples", "next_base_samples",
        "next_state_1step",
    )


    FIELD_FALLBACK = {"next_state_1step": "next_state"}


    META_DEFAULTS = {
        "is_critical": 0.0,
        "dvd_gap": 0.0,
        "is_catastrophic": 0.0,
        "critical_type": 0.0,
        "steps_to_catastrophe": float("nan"),
        "trajectory_id": -1.0,
        "timestep": -1.0,
        "catastrophic_timestep": float("nan"),


        "behaviour_logp": float("nan"),


        "episode_len": 0.0,
    }

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        chunk_dim: int,
        samples: int = 1,
        next_samples: int | None = None,
        image_shape: tuple[int, ...] | None = None,
        cond_dim: int | None = None,
        feat_shape: tuple[int, ...] | None = None,
        seg_feat: bool = False,
        tok_shape=None,
        boot_image: bool = False,
        continuation: bool = False,
    ):


        next_samples = samples if next_samples is None else next_samples
        self.capacity = capacity
        self.state_dim = state_dim
        self.chunk_dim = chunk_dim
        self.samples = samples
        self.next_samples = next_samples
        self.arrays = {
            "base_samples": np.empty((capacity, samples * chunk_dim), dtype=np.float32),
            "next_base_samples": np.empty((capacity, next_samples * chunk_dim), dtype=np.float32),
            "state": np.empty((capacity, state_dim), dtype=np.float32),
            "base": np.empty((capacity, chunk_dim), dtype=np.float32),
            "action": np.empty((capacity, chunk_dim), dtype=np.float32),
            "next_state": np.empty((capacity, state_dim), dtype=np.float32),
            "next_state_1step": np.empty((capacity, state_dim), dtype=np.float32),
            "next_base": np.empty((capacity, chunk_dim), dtype=np.float32),
            "reward": np.empty((capacity, 1), dtype=np.float32),
            "done": np.empty((capacity, 1), dtype=np.float32),
            "discount": np.empty((capacity, 1), dtype=np.float32),
            "mc_return": np.empty((capacity, 1), dtype=np.float32),
            "data_source": np.empty((capacity, 1), dtype=np.float32),
        }
        for key, default in self.META_DEFAULTS.items():
            self.arrays[key] = np.full((capacity, 1), default, dtype=np.float32)


        self.image_shape = tuple(image_shape) if image_shape else None
        if self.image_shape:
            for key in ("image", "next_image"):
                self.arrays[key] = np.zeros((capacity, *self.image_shape), dtype=np.uint8)


        self.boot_image = bool(boot_image) and self.image_shape is not None
        if self.boot_image:
            self.arrays["next_image_boot"] = np.zeros((capacity, *self.image_shape), dtype=np.uint8)


        self.cond_dim = int(cond_dim) if cond_dim else None
        if self.cond_dim:
            self.arrays["cond"] = np.zeros((capacity, self.cond_dim), dtype=np.float32)


        self.feat_shape = tuple(feat_shape) if feat_shape else None
        if self.feat_shape:
            self.arrays["prev_feat"] = np.zeros((capacity, *self.feat_shape), dtype=np.float16)


        self.seg_feat = bool(seg_feat) and self.feat_shape is not None
        if self.seg_feat:
            self.arrays["seg_feat"] = np.zeros((capacity, *self.feat_shape), dtype=np.float16)


            self.arrays["next_feat"] = np.zeros((capacity, *self.feat_shape), dtype=np.float16)


        self.tok_shape = tuple(tok_shape) if tok_shape else None
        if self.tok_shape:
            for key in ("tok", "next_tok"):
                self.arrays[key] = np.zeros((capacity, *self.tok_shape), dtype=np.float16)


        self.continuation = bool(continuation)
        if self.continuation:
            self.arrays["base_cont"] = np.empty((capacity, chunk_dim), dtype=np.float32)
            self.arrays["base_cont_samples"] = np.empty((capacity, samples * chunk_dim), dtype=np.float32)
        self.position = 0
        self.size = 0


        self._critical_slots: set[int] = set()

    def save(self, path) -> None:
        """Persist the buffer so a run's collected experience outlives the process."""
        import numpy as _np
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v[: self.size] for k, v in self.arrays.items()}
        payload["__meta__"] = _np.array(
            [self.size, self.position, self.capacity, self.state_dim, self.chunk_dim,
             self.samples, self.next_samples], dtype=_np.int64)
        if self.image_shape:
            payload["__image_shape__"] = _np.array(self.image_shape, dtype=_np.int64)
        if self.cond_dim:
            payload["__cond_dim__"] = _np.array([self.cond_dim], dtype=_np.int64)
        if self.feat_shape:
            payload["__feat_shape__"] = _np.array(self.feat_shape, dtype=_np.int64)


        tmp = path.with_name(path.name + ".tmp.npz")
        _np.savez(tmp, **payload)
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "ReplayBuffer":
        import numpy as _np
        z = _np.load(Path(path), allow_pickle=False)
        size, position, capacity, state_dim, chunk_dim, samples, next_samples = z["__meta__"]
        shape = tuple(int(v) for v in z["__image_shape__"]) if "__image_shape__" in z else None
        cond_dim = int(z["__cond_dim__"][0]) if "__cond_dim__" in z else None
        feat = tuple(int(v) for v in z["__feat_shape__"]) if "__feat_shape__" in z else None
        buf = cls(int(capacity), int(state_dim), int(chunk_dim),
                  samples=int(samples), next_samples=int(next_samples), image_shape=shape,
                  cond_dim=cond_dim, feat_shape=feat, seg_feat="seg_feat" in z,
                  tok_shape=(z["tok"].shape[1:] if "tok" in z else None),
                  boot_image="next_image_boot" in z,
                  continuation="base_cont_samples" in z)
        for k in buf.arrays:
            if k in z:
                buf.arrays[k][: int(size)] = z[k]
        buf.size, buf.position = int(size), int(position)
        buf._critical_slots = {
            i for i in range(buf.size) if float(buf.arrays["is_critical"][i][0]) > 0.5
        }
        return buf

    def add(self, **transition: np.ndarray | float) -> None:
        for key in self.FIELDS:
            if key not in transition and key in self.FIELD_FALLBACK:
                self.arrays[key][self.position] = transition[self.FIELD_FALLBACK[key]]
            else:
                self.arrays[key][self.position] = transition[key]
        for key, default in self.META_DEFAULTS.items():
            self.arrays[key][self.position] = transition.get(key, default)
        if self.image_shape:
            for key in ("image", "next_image"):
                if key in transition:
                    self.arrays[key][self.position] = transition[key]
        if self.boot_image and ("next_image_boot" in transition or "next_image" in transition):
            self.arrays["next_image_boot"][self.position] = transition.get(
                "next_image_boot", transition.get("next_image"))
        if self.cond_dim and "cond" in transition:
            self.arrays["cond"][self.position] = transition["cond"]
        if self.continuation:
            nb = transition["next_base"]
            self.arrays["base_cont"][self.position] = transition.get("base_cont", nb)
            self.arrays["base_cont_samples"][self.position] = (
                transition["base_cont_samples"] if "base_cont_samples" in transition
                else np.tile(np.asarray(nb, dtype=np.float32).reshape(-1), self.samples))
        if self.feat_shape and "prev_feat" in transition:
            self.arrays["prev_feat"][self.position] = transition["prev_feat"]


        if self.seg_feat and "seg_feat" in transition:
            self.arrays["seg_feat"][self.position] = transition["seg_feat"]
        if self.seg_feat and "next_feat" in transition:
            self.arrays["next_feat"][self.position] = transition["next_feat"]
        if self.tok_shape:
            for key in ("tok", "next_tok"):
                if key in transition:
                    self.arrays[key][self.position] = transition[key]


        if float(self.arrays["is_critical"][self.position][0]) > 0.5:
            self._critical_slots.add(self.position)
        else:
            self._critical_slots.discard(self.position)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @property
    def critical_size(self) -> int:
        return len(self._critical_slots)

    def _episode_uniform_indices(self, batch_size: int) -> np.ndarray:
        """Draw slots with probability proportional to 1 / episode_len."""
        lengths = self.arrays["episode_len"][: self.size, 0].astype(np.float64)
        known = lengths > 0.0
        if not known.any():
            return np.random.randint(0, self.size, size=batch_size)
        weights = np.empty_like(lengths)
        weights[known] = 1.0 / lengths[known]
        weights[~known] = float(np.median(weights[known]))
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.random.randint(0, self.size, size=batch_size)
        return np.random.choice(self.size, size=batch_size, replace=True, p=weights / total)

    def sample(
        self, batch_size: int, device: torch.device, critical_fraction: float = 0.0,
        episode_uniform: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Uniform over the buffer, or stratified when `critical_fraction > 0`."""
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        if episode_uniform and critical_fraction <= 0.0:
            indices = self._episode_uniform_indices(batch_size)
        elif critical_fraction <= 0.0 or not self._critical_slots:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:

            available = np.fromiter(
                (slot for slot in self._critical_slots if slot < self.size), dtype=np.int64
            )
            wanted = min(int(round(critical_fraction * batch_size)), batch_size)


            take = min(wanted, available.size)
            critical = np.random.choice(available, size=take, replace=False) if take else \
                np.empty(0, dtype=np.int64)


            ordinary = np.random.randint(0, self.size, size=batch_size - critical.size)
            for _ in range(8):
                rejected = self.arrays["is_critical"][ordinary, 0] > 0.5
                if not rejected.any():
                    break
                ordinary[rejected] = np.random.randint(0, self.size, size=int(rejected.sum()))
            indices = np.concatenate((critical, ordinary))
            np.random.shuffle(indices)
        return self.gather(indices, device)

    def gather(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        """Every column at `indices`, as tensors on `device`."""
        return {
            key: torch.as_tensor(value[indices], device=device)
            for key, value in self.arrays.items()
        }

    def ensure_continuation(self) -> None:
        """Add the continuation columns to a buffer built without them (a resumed pre-flag replay)."""
        if self.continuation:
            return
        self.continuation = True
        nb = self.arrays["next_base"]
        self.arrays["base_cont"] = nb.copy()
        self.arrays["base_cont_samples"] = np.tile(nb, (1, self.samples))
        print(f"[replay] continuation columns added to a buffer that lacked them: {self.size} existing "
              f"rows fall back to next_base (the recorded follow)", flush=True)

    def recent_indices(self, window: int) -> np.ndarray:
        """Slots of the newest `window` rows, OLDEST FIRST in write order."""
        if self.size == 0:
            raise RuntimeError("Cannot take recent rows of an empty replay buffer")
        n = min(max(int(window), 1), self.size)
        return (self.position - n + np.arange(n, dtype=np.int64)) % self.capacity

    def row_age(self, indices: np.ndarray) -> np.ndarray:
        """Age of each slot in ROWS:"""
        return (self.position - 1 - np.asarray(indices, dtype=np.int64)) % self.capacity

    def recent_mix_indices(
        self, batch_size: int, window: int, recent_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`recent_count` slots uniform over the newest `window` rows, the rest uniform over the buffer."""
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        batch_size = int(batch_size)
        recent_count = min(max(int(recent_count), 0), batch_size)
        pool = self.recent_indices(window)
        recent = pool[np.random.randint(0, pool.size, size=recent_count)]
        rest = batch_size - recent_count
        if rest == 0:
            return recent, np.ones(recent_count, dtype=bool)
        whole = np.random.randint(0, self.size, size=rest)
        indices = np.concatenate((recent, whole))
        from_window = np.concatenate((np.ones(recent_count, dtype=bool), np.zeros(rest, dtype=bool)))
        order = np.random.permutation(batch_size)
        return indices[order], from_window[order]


def fill_mc_return(episode: list[dict]) -> list[dict]:
    """Stamp each chunk transition with its realized discounted return-to-go."""
    running = 0.0
    for transition in reversed(episode):
        running = float(transition["reward"]) + float(transition["discount"]) * running
        transition["mc_return"] = running
    return episode


def fill_episode_len(episode: list[dict]) -> list[dict]:
    """Stamp every transition with the number of chunks in its episode."""
    n = float(len(episode))
    for transition in episode:
        transition["episode_len"] = n
    return episode


def fill_nstep(episode: list[dict], nstep: int, traces: list[float] | None = None,
               floor: float = 0.5) -> list[dict]:
    """Rewrite each transition's bootstrap target to an n-step one."""
    if nstep <= 1:
        return episode
    original = [
        (
            float(step["reward"]),
            float(step["discount"]),
            float(step["done"]),
            step["next_state"],
            step["next_base"],
            step["next_base_samples"],
        )
        for step in episode
    ]
    horizon = len(episode)
    for index, transition in enumerate(episode):
        accumulated = 0.0
        compounded = 1.0
        trace_prod = 1.0
        last = index
        for offset in range(min(nstep, horizon - index)):
            reward, discount, done, *_ = original[index + offset]
            if offset > 0 and traces is not None:

                trace_prod *= traces[index + offset]
                if trace_prod < floor:
                    break
            accumulated += compounded * reward
            compounded *= discount
            last = index + offset
            if done > 0.5:
                break
        _, _, done, next_state, next_base, next_base_samples = original[last]
        if "next_image" in episode[last]:


            transition["next_image_boot"] = episode[last]["next_image"]

        transition["next_state_1step"] = original[index][3]
        transition["reward"] = accumulated
        transition["discount"] = compounded
        transition["done"] = done
        transition["next_state"] = next_state
        transition["next_base"] = next_base
        transition["next_base_samples"] = next_base_samples
    return episode


def save_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)


def concat_batches(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:


    keys = ReplayBuffer.FIELDS + tuple(ReplayBuffer.META_DEFAULTS)


    present = [set(batch) for batch in batches]
    shared = set.intersection(*present) if present else set()
    partial = set.union(*present) - shared if present else set()
    if partial:
        raise KeyError(
            f"batches disagree on columns {sorted(partial)}; every buffer feeding concat_batches "
            "must be built with the same optional fields (image_shape / cond_dim)")
    return {key: torch.cat([batch[key] for batch in batches], dim=0)
            for key in tuple(keys) + tuple(sorted(shared - set(keys)))}


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
            target_parameter.lerp_(parameter, tau)


@dataclass
class TrainConfig:
    method: str
    task: str
    output_dir: str
    base_checkpoint: str
    wandb_project: str = "mimicgen_online_rl"
    wandb_name: str | None = None
    seed: int = 42
    total_env_steps: int = 20_000
    action_steps: int = 8

    inference_steps: int = 10
    batch_size: int = 256


    bon_samples: int = 0
    replay_capacity: int = 100_000

    flowdpg_mode: str = "adapter"
    flowdpg_hidden: int = 256
    flowdpg_layers: int = 2
    flowdpg_rank: int = 16
    flowdpg_alpha: float = 0.1
    flowdpg_lambda_max: float = 0.5
    flowdpg_warmup: int = 5000
    flowdpg_cons_weight: float = 1.0
    flowdpg_expectile: float = 0.7
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    residual_scale: float = 0.10


    bc_filter_uncertainty: float = 0.0


    reward_model_causal: bool = False
    reward_shaping_form: str = "additive"
    sparse_reward_weight: float = 1.0
    replay_episode_uniform: bool = False
    bon_reduce: str = "min"
    actor_q_reduce: str = "min"
    actor_k_reduce: str = "mean"
    qdvd_combine: str = "off"
    encoder_value_aux_head: str = "mlp"
    qdvd_normalise: str = "meanabs"
    bon_select: str = "max"
    bon_quantile: float = 0.75
    encoder_value_aux_only: bool = False
    seg_terminate_patience: int = 0
    encoder_drift: float = 0.0
    encoder_value_aux: float = 0.0
    vtrace_sigma: float = 0.0
    vtrace_clip: float = 1.0
    vtrace_floor: float = 0.5
    target_actor: bool = False
    value_target_tau: float = 0.0
    bootstrap_timeouts: bool = False
    value_loss: str = "smooth_l1"
    value_residual: bool = False
    value_residual_unfrozen: bool = False
    value_residual_cond: bool = False
    residual_head: str = "mlp"
    residual_conv_kernel: int = 3
    residual_conv_channels: int = 16

    residual_conv_broadcast: bool = False


    init_actor_from: str = ""
    init_critic_from: str = ""


    freeze_init_critic: bool = False


    step_penalty: float = -0.01
    learning_starts: int = 8_000
    progressive_exploration_steps: int = 5_000
    eval_freq: int = 5_000
    eval_episodes: int = 5
    save_freq: int = 5_000
    expert_ratio: float = 0.30
    bc_loss_weight: float = 100.0
    critic_ensemble_size: int = 10
    eval_video_episodes: int = 1


    initial_log_std: float | None = None
    log_std_min: float = -5.0


    bc_filter: bool = False
    bc_filter_warmup_updates: int = 5_000


    bc_filter_soft: bool = False
    q_underestimation_threshold: float = 0.0

    retain_bc_for_expert: bool = True


    deterministic_residual: bool = False


    latent_dim: int = 0


    nstep: int = 1
    env_dense_reward: bool = False
    reward_model: str | None = None
    reward_model_weight: float = 0.02


    terminate_on_reward_drop: float = 0.0


    value_model: str | None = None
    value_variant: str = "v_relabel_hinge"


    value_intervention: str = "none"
    value_penalty_beta: float = 0.0


    value_finetune: bool = False
    value_aggregate: str = "mean"
    value_finetune_after: int = 0
    value_distill_after: int = 0


    state_noise_mm: float = 0.0
    state_noise_slots: str = "9:12,23:26"


    state_noise_mode: str = "iid"

    dvd_on_encoder: bool = False


    dvd_value_aware_weight: float = 0.0


    dvd_latent_wm: str = ""


    pose_from_seghead: bool = False
    jitter_goal_mm: float = 0.0
    actor_value_after: int = 0
    dvd_fixed_scale: float = 0.0
    dvd_fit_rows: int = 0
    dino_wm_arch: str = ""
    actor_grad_clip: float = 10.0


    aug_target: str = "both"


    probe_step_eps: float = 0.0
    probe_episodes: int = 20


    gradvar_every: int = 0


    gradvar_by_phase: bool = True
    dvd_fit_online_only: bool = False


    dvd_fit_sampler: str = "batch"
    dvd_fit_recent_steps: int = 25_000
    dvd_fit_recent_frac: float = 0.5
    dvd_fit_probe_every: int = 0
    value_fit_online_only: bool = False
    value_lcb_lambda: float = 2.0


    value_init: str | None = None
    value_scratch: bool = False


    save_replay: bool = True


    dump_episodes: bool = False
    value_reload: str = ""


    resume: str = ""


    dino_obs: bool = False
    latent_obs: bool = False
    latent_obs_finetune: bool = False
    seg_obs: bool = False
    seg_obs_freeze: bool = False
    seg_obs_finetune: bool = False
    obs_shift_pad: int = 0
    offline_pretrain_steps: int = 0
    tok_shift_pad: int = 0


    grad_probe_freq: int = 0


    residual_max_norm: float = 0.0


    exploration_gate_mm: float = 0.0

    dvd_dynamics_init: str | None = None
    dvd_dynamics_scratch: bool = False
    value_finetune_lr: float = 1e-4


    value_random_frozen: bool = False
    encoder_value_aux_image_only: bool = False
    value_distill_robometer: str = ""
    value_distill_every: int = 500
    value_distill_batch: int = 256
    value_distill_weight: float = 1.0
    value_distill_capacity: int = 20000
    value_distill_frames: int = 64
    value_distill_target: str = "scaled"
    value_distill_aug_views: int = 0
    value_distill_min_pool: str = "0"
    value_distill_shift_pad: int = 0
    value_distill_photometric: float = 0.0
    value_distill_mix: str = ""
    value_distill_demo_episodes: int = 0


    value_td_model: str = ""
    value_dual_combine: str = "min"


    zero_floor: bool = False


    bon_axis: str = "residual"


    multi_sample_k: int = 1


    actor_sample_k: int | None = None


    disable_q_loss_for_expert: bool = False


    eval_strategy: str = "greedy"
    eval_samples: int = 16


    utd: int = 1
    value_mc_weight: float = 0.0
    actor_utd: int = 1
    value_utd: int = 1
    update_every: int = 1
    update_on_episode_end: bool = False
    time_components: bool = False
    time_backward: bool = False
    compile_parts: str = ""
    dvd_finetune_last_k: int = 0
    value_nets: int = 0


    episode_horizon: int | None = None
    actor_hidden: int = 256
    critic_hidden: int = 256
    critic_layers: int = 3
    state_augment: float = 0.0


    state_repr: str = "absolute"


    adaptive_expert_ratio: bool = False
    adaptive_expert_ratio_start: float = 1.0
    adaptive_expert_ratio_end: float = 0.1
    adaptive_expert_ratio_steps: int = 40_000

    freeze_training_at: int = 0
    expert_gate_success: float = 0.0


    critical_ladder: str = "gated"


    truncate_after_catastrophe: bool = False

    critical_sampling: bool = False
    critical_batch_fraction: float = 0.25


    critical_pre_window: int = 2


    catastrophic_as_terminal: bool = False


    actor_gradient_source_normal: str = "q"
    actor_gradient_source_critical: str = "q"


    dvd_weight: float = 0.0


    dvd_dynamics: str = "model_residual"

    dino_wm_ckpt: str = ""


    dino_pool: str = "mean"


    attn_pool_freeze: bool = False


    xattn_queries: int = 8
    xattn_heads: int = 6


    mqattn_queries: int = 0
    mqattn_scope: str = ""
    mqattn_out_dim: int = -1


    dino_obs_pool_separate: bool = False


    dino_obs_pool_scratch: bool = False


    attn_pool_scratch: bool = False


    dino_obs_pool_ema_target: bool = False


    dvd_actor_pool: bool = False


    dvd_actor_pool_obs: str = "actor"


    dvd_actor_pool_log_every: int = 200


    dvd_pool_actor_step: bool = False


    dino_bf16: bool = True


    flow_steps: int = 0
    flow_inverse_iters: int = 0


    dvd_rows: int = 0


    dvd_finetune_dynamics: bool = False
    dvd_finetune_seghead: bool = False
    dvd_map_shift: int = 0
    dvd_seghead_lr: float = 1e-5
    dvd_seghead_ema: float = 0.999
    dvd_dynamics_lr: float = 1e-4


    value_pool: str = ""
    dvd_finetune_dino_blocks: int = 0


    dvd_checkpoint: bool = False


    dino_photometric: float = 0.0


    actor_train_encoder: bool = False


    dvd_weight_late: float = -1.0
    dvd_weight_ramp_start: int = 0
    dvd_weight_ramp_end: int = 0
    actor_lr_drop_at: int = 0
    actor_lr_late_factor: float = 0.3
    dvd_rows_late: int = 0
    dvd_rows_switch: int = 0


    dvd_rows_grouped: bool = False


    dvd_residual_anchor: bool = False


    dino_wm_state_head: bool = False
    dvd_proprio_source: str = "current"
    dvd_state_loss_weight: float = 1.0


    value_distill_pred: bool = False


    q_gap_weight: float = 0.0
    rosmoq: bool = False
    rosmoq_weight: float = 0.5
    rosmoq_rows: int = 64
    rosmo: bool = False
    rosmo_k: int = 4
    rosmo_beta: float = 30.0
    rosmo_sigma: float = 0.1
    q_rows: int = 0
    gated_training: bool = False
    gate_skill_tau: float = 0.55
    gate_hyst: float = 0.05
    gate_td_tau: float = 0.002
    gate_td_hyst: float = 0.0005
    gate_dwell: int = 200
    gate_slow: float = 0.0
    dvd_dino_lr: float = 1e-5


    holder_sigma: float = 0.0


    fixed_goal: bool = False


    prior_temperature: float = 1.0


    fixed_goal_rot: bool = False


    full_fixed: bool = False
    dvd_chunks: int = 1


    dvd_follow: str = "recorded"
    bon_lookahead: int = 1


    dvd_step2_action: str = "prior"


    dvd_row_chunks: int = 1
    pixel_obs: bool = False
    pixel_feature_dim: int = 128
    pixel_augment: bool = True
    pixel_shift_pad: int = 4
    encoder_init: str = ""
    actor_encoder_finetune: bool = False
    actor_encoder_scratch: bool = False
    dvd_encoder_learned: bool = False


    dino_shift_pad: int = 0
    insertion_only: bool = False
    insertion_span: int = 96
    insertion_joint_sigma: float = 0.0
    insertion_holder_sigma: float = 0.0


    bc_filter_source: str = "q"
    bon_source: str = "q"
    bon_dvd_until: int = 0

    bon_frozen_v: str = "off"


    bc_target: str = "base"
    bc_target_after: int = 50_000
    bc_target_ramp: int = 10_000


    dvd_mode: str = "dpg"


    dvd_phase_gate: bool = False
    dvd_held_mm: float = 50.0


    dvd_static_grad: str = "none"
    dvd_gate_source: str = "eef_obj"
    dvd_gate_target: str = "dvd"
    dvd_gate_mm: float = 150.0


    mbpo_weight: float = 0.0


    mbpo_chunks: int = 1


    mbpo_best_of_n: bool = True


    mbpo_prior_surrogate: str = ""


    mbpo_max_sigma_mm: float = 0.0


    mbpo_success_margin_mm: float = 0.0


    train_critic: bool = True

    def critical_enabled(self) -> bool:
        """Whether anything in this run needs annotations computed at all."""
        return (
            self.truncate_after_catastrophe
            or self.critical_sampling
            or self.catastrophic_as_terminal
            or self.actor_gradient_source_critical != "q"
        )

    def note_success(self, rolling_success: float) -> None:
        """Feed the annealing gate its evidence:"""
        self._gate_recent = rolling_success

    def current_expert_ratio(self, env_steps: int) -> float:


        if getattr(self, "expert_gate_success", 0.0) and self.adaptive_expert_ratio:
            prog = getattr(self, "_gate_progress", 0.0)
            last = getattr(self, "_gate_last_steps", 0)
            recent = getattr(self, "_gate_recent", None)
            if env_steps > last:
                if recent is None or recent >= self.expert_gate_success:
                    prog = min(prog + (env_steps - last), self.adaptive_expert_ratio_steps)
                self._gate_progress, self._gate_last_steps = prog, env_steps
            frac = prog / max(self.adaptive_expert_ratio_steps, 1)
            span = self.adaptive_expert_ratio_start - self.adaptive_expert_ratio_end
            return self.adaptive_expert_ratio_start - span * frac

        if not self.adaptive_expert_ratio:
            return self.expert_ratio
        if env_steps >= self.adaptive_expert_ratio_steps:
            return self.adaptive_expert_ratio_end
        progress = env_steps / self.adaptive_expert_ratio_steps
        span = self.adaptive_expert_ratio_start - self.adaptive_expert_ratio_end
        return self.adaptive_expert_ratio_start - span * progress

    def resolved_actor_k(self) -> int:
        return self.multi_sample_k if self.actor_sample_k is None else self.actor_sample_k

    def store_prior_continuation(self) -> bool:
        """Replay carries base_cont / base_cont_samples (only --dvd-follow continuation reads them)."""
        return getattr(self, "dvd_follow", "recorded") == "continuation"

    def need_prior_continuation(self) -> bool:
        """Collection asks the prior for each draw's continuation (storage, or --bon-lookahead 2)."""
        return self.store_prior_continuation() or int(getattr(self, "bon_lookahead", 1)) > 1

    def dvd_fit_recent_rows(self) -> int:
        """--dvd-fit-recent-steps as a ROW count W."""
        return max(1, int(self.dvd_fit_recent_steps) // int(self.action_steps))

    def dvd_fit_batch_rows(self) -> int:
        """R, the row count of f's OWN draw (--dvd-fit-sampler recent/mix) and of the recent probe."""
        return int(self.dvd_fit_rows or self.dvd_rows or self.batch_size)

    def dvd_fit_recent_count(self) -> int:
        """How many of f's R own-draw rows come from the recent window."""
        rows = self.dvd_fit_batch_rows()
        if self.dvd_fit_sampler == "recent":
            return rows
        if self.dvd_fit_sampler == "mix":
            return int(round(self.dvd_fit_recent_frac * rows))
        return 0

    def dvd_fit_min_online_rows(self) -> int:
        """The online-buffer size at which f's OWN draw (--dvd-fit-sampler recent/mix) takes over."""
        return self.dvd_fit_batch_rows()

    def dvd_fit_own_draw(self, online_size: int) -> bool:
        """Whether this update fits f on its own draw (True) or on the batch path (False)."""
        return self.dvd_fit_sampler != "batch" and int(online_size) >= self.dvd_fit_min_online_rows()

    def needs_pending(self) -> bool:
        """Whether transitions must be held back until the episode ends before being stored."""
        if self.reward_model is not None or self.nstep > 1:
            return True
        if self.bc_filter and not (
            self.bc_filter_soft and self.value_intervention == "filter"
        ) and not (self.bc_filter_source == "dvd" and self.dvd_dynamics == "dino_wm"):

            return True


        if self.critical_enabled():
            return True
        return False

    def resolved_initial_log_std(self) -> float:
        if self.initial_log_std is not None:
            return self.initial_log_std
        return -3.0 if self.method == "dice_rl" else -1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(
    path: Path,
    config: TrainConfig,
    actor: ResidualActor,
    critics: CriticEnsemble,
    target_critics: CriticEnsemble,
    env_steps: int,
    updates: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "actor": actor.state_dict(),
            "critics": critics.state_dict(),
            "target_critics": target_critics.state_dict(),
            "env_steps": env_steps,
            "updates": updates,
            "extra": extra or {},
        },
        path,
    )


def load_agent_checkpoint(path: Path, device: torch.device):
    from .legacy import upgrade_checkpoint
    payload = upgrade_checkpoint(torch.load(path, map_location=device, weights_only=False))
    config = TrainConfig(**payload["config"])
    state_dim = int(TASKS[config.task]["state_dim"])
    chunk_dim = config.action_steps * 7
    layernorm = config.method == "dice_rl"
    ensemble = config.critic_ensemble_size if config.method == "dice_rl" else 2
    actor = ResidualActor(
        state_dim,
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
    ).to(device)
    critics = CriticEnsemble(
        state_dim, chunk_dim, ensemble,


        hidden=(config.critic_hidden,) * getattr(config, "critic_layers", 3), layernorm=layernorm,
    ).to(device)
    actor.load_state_dict(payload["actor"])
    critics.load_state_dict(payload["critics"])
    actor.eval()
    critics.eval()
    return config, actor, critics, payload
