#!/usr/bin/env python3
"""Record MimicGen policy rollouts straight into LeRobot v3.0 datasets."""

from __future__ import annotations
import os

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import TASKS, FrozenPolicyPrior, StateScaler, load_agent_checkpoint, set_seed


PIXEL_FEATURES = {
    "observation.images.agentview": "agentview",
    "observation.images.eye_in_hand": "eye_in_hand",
}

DERIVED_FEATURES = {
    "action",
    "observation.state",
    "observation.sim_state",
    "reward",
    "done",
    "success",
    "is_first",
    "is_last",
    "is_terminal",
    "source_demo_index",
    *PIXEL_FEATURES,
}


def source_key_for(feature_key: str) -> str | None:
    """Raw robosuite observation key backing a LeRobot feature, or None if derived."""
    if feature_key in DERIVED_FEATURES or not feature_key.startswith("observation."):
        return None
    name = feature_key.removeprefix("observation.")

    return "object-state" if name == "object" else name


EXTRA_OBSERVABLES = ("robot0_joint_pos", "robot0_eef_vel_lin", "robot0_eef_vel_ang")


class RecordingMimicGenEnv:
    """`MimicGenGymEnv` that also exposes the raw robosuite observation and simulator state."""

    def __init__(self, holder_sigma: float = 0.0, fixed_goal: bool = False, **kwargs: Any) -> None:
        if fixed_goal and holder_sigma:
            raise ValueError("--fixed-goal names its own env variant; it cannot combine with "
                             "--holder-sigma")
        if fixed_goal:


            from .fixed_goal_env import FixedGoalEnv as base_env
        else:
            from lerobot_env_mimicgen.environment import MimicGenGymEnv as base_env

        if holder_sigma:
            raise ValueError("--holder-sigma (coffee holder jitter) is not part of this release")

        class _Env(base_env):
            raw_obs: dict[str, np.ndarray]

            def _ensure_env(self):  # type: ignore[override]
                created = self._env is None
                super()._ensure_env()
                if created:
                    for name in EXTRA_OBSERVABLES:
                        self._env.modify_observable(name, "enabled", True)
                        self._env.modify_observable(name, "active", True)

            def _format_observation(self, raw):  # type: ignore[override]
                self.raw_obs = raw
                return super()._format_observation(raw)

        self._env = _Env(**kwargs)

    def __getattr__(self, name: str) -> Any:

        return getattr(object.__getattribute__(self, "_env"), name)

    def sim_state(self) -> np.ndarray:
        """Full MuJoCo state, the same vector the source HDF5 kept under ``states``."""
        return self._env._env.sim.get_state().flatten().astype(np.float32, copy=False)


def build_features(reference_root: Path) -> dict[str, dict]:
    """Reference feature spec minus the columns LeRobot fills in itself."""
    from lerobot.utils.constants import DEFAULT_FEATURES

    features = json.loads((reference_root / "meta/info.json").read_text())["features"]
    result = {}
    for key, spec in features.items():
        if key in DEFAULT_FEATURES:
            continue
        spec = dict(spec)


        spec.pop("info", None)
        result[key] = spec
    return result


def native_settings(checkpoint: Path) -> tuple[int, int]:
    """The checkpoint's own chunk length and flow/diffusion step count."""
    config = json.loads((checkpoint / "config.json").read_text())
    steps = config.get("num_integration_steps") or config.get("num_inference_steps")
    if steps is None:
        raise KeyError(f"{checkpoint}/config.json declares neither integration nor inference steps")
    return int(config["n_action_steps"]), int(steps)


def reference_task_instruction(reference_root: Path) -> str:
    import pyarrow.parquet as pq

    tasks = pq.read_table(reference_root / "meta/tasks.parquet").to_pydict()["task"]
    if len(tasks) != 1:
        raise RuntimeError(f"Expected exactly one task in the reference dataset, found {tasks}")
    return str(tasks[0])


class BasePolicy:
    """The frozen policy alone, stepped the way ``lerobot_eval.rollout`` steps it."""

    description = "frozen base policy, select_action per primitive step"

    def __init__(self, prior: FrozenPolicyPrior):
        self.prior = prior

    def reset(self) -> None:
        self.prior.policy.reset()

    @torch.inference_mode()
    def act(self, observation: dict[str, Any]) -> np.ndarray:
        batch = self.prior._prepare(observation)
        action = self.prior.postprocessor(self.prior.policy.select_action(batch))
        return action.to("cpu").numpy()[0].clip(-1.0, 1.0).astype(np.float32)


class ResidualPolicy:
    """Base chunk plus the deterministic residual, one prior draw per decision (no best-of-N)."""

    description = "residual actor over frozen base policy, deterministic, best_of_n=1"

    def __init__(self, prior: FrozenPolicyPrior, actor, scaler: StateScaler, residual_scale: float,
                 max_norm: float = 0.0):
        self.prior = prior
        self.actor = actor
        self.scaler = scaler
        self.residual_scale = residual_scale
        self.max_norm = max_norm
        self._queue: list[np.ndarray] = []

    def reset(self) -> None:
        self._queue = []

    @torch.inference_mode()
    def act(self, observation: dict[str, Any]) -> np.ndarray:
        from .train_online import final_action, flatten_chunk

        if not self._queue:
            state = torch.as_tensor(
                self.scaler(observation["agent_pos"]), device=self.prior.device
            ).unsqueeze(0)
            base = flatten_chunk(self.prior.action_chunk(observation))
            residual, _ = self.actor.sample(state, base, deterministic=True)
            action = final_action(base, residual, self.residual_scale, self.max_norm)
            chunk = action.view(self.prior.action_steps, self.prior.action_dim).cpu().numpy()
            self._queue = [row.astype(np.float32) for row in chunk]
        return self._queue.pop(0)


def rollout(
    env: RecordingMimicGenEnv,
    policy: BasePolicy | ResidualPolicy,
    seed: int,
    raw_keys: dict[str, str],
    task_instruction: str,
) -> tuple[list[dict], bool]:
    """One episode, returned as ready-to-write LeRobot frames plus its success flag."""
    observation, _ = env.reset(seed=seed)
    policy.reset()
    frames: list[dict] = []
    terminated_flags: list[bool] = []
    success = False
    done = False

    while not done:
        raw = env.raw_obs
        action = policy.act(observation)
        frame: dict[str, Any] = {
            "task": task_instruction,
            "action": action,
            "observation.state": observation["agent_pos"].astype(np.float32),
            "observation.sim_state": env.sim_state(),
            "source_demo_index": np.asarray([seed], dtype=np.int64),
        }
        for feature_key, pixel_key in PIXEL_FEATURES.items():
            frame[feature_key] = observation["pixels"][pixel_key]
        for feature_key, source in raw_keys.items():
            frame[feature_key] = np.asarray(raw[source], dtype=np.float32).reshape(-1)

        observation, reward, terminated, truncated, info = env.step(action)
        step_success = bool(info["is_success"])
        success = success or step_success
        frame["reward"] = np.asarray([reward], dtype=np.float32)
        frame["success"] = np.asarray([step_success], dtype=bool)
        frames.append(frame)
        terminated_flags.append(bool(terminated))
        done = bool(terminated or truncated)

    last = len(frames) - 1
    for index, frame in enumerate(frames):
        frame["is_first"] = np.asarray([index == 0], dtype=bool)
        frame["is_last"] = np.asarray([index == last], dtype=bool)
        frame["done"] = np.asarray([index == last], dtype=bool)


        frame["is_terminal"] = np.asarray([terminated_flags[index]], dtype=bool)
    return frames, success


def make_dataset(repo_id: str, root: Path, features: dict, fps: int, overwrite: bool):
    from lerobot.datasets import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="Panda",
        use_videos=True,
        image_writer_processes=0,

        image_writer_threads=0,
        streaming_encoding=True,
        encoder_threads=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path, help="frozen LeRobot policy; required without --checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="residual RL checkpoint; omit to record the base policy alone"
    )
    parser.add_argument("--task", help="required when recording the base policy alone")
    parser.add_argument(
        "--action-steps", type=int, help="default: the checkpoint's native n_action_steps"
    )
    parser.add_argument(
        "--inference-steps", type=int, help="default: the checkpoint's native integration steps"
    )
    parser.add_argument("--output-root", type=Path, required=True, help="parent dir for the two datasets")
    parser.add_argument("--repo-prefix", required=True, help="e.g. $HF_NAMESPACE/mimicgen_square_d0_224x224_mtdit_flow_40k")
    parser.add_argument("--successes", type=int, default=100)
    parser.add_argument("--failures", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--max-episodes", type=int, default=2_000, help="hard cap on attempted episodes")
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="the task's demo dataset (feature schema, fps and task string are copied from it)")
    parser.add_argument("--seed", type=int, default=0, help="torch/numpy seed for the policy itself")
    parser.add_argument("--holder-sigma", type=float, default=0.0,
                        help="displace the coffee machine by N(0, sigma) metres in x and y at every "
                             "reset, matching --holder-sigma in train_online.py. Recording with 0 "
                             "while the consuming run uses a jittered holder would fit the dynamics "
                             "model to a distribution the policy never visits.")
    parser.add_argument("--fixed-goal", action="store_true",
                        help="pin the goal object at the demo mode, matching --fixed-goal in "
                             "train_online.py. Recording the baseline goal distribution while the "
                             "consuming run pinned it would fit the dynamics model to goal poses "
                             "the policy never visits.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    device = torch.device("cuda")
    set_seed(args.seed)

    if args.checkpoint is not None:
        config, actor, critics, _ = load_agent_checkpoint(args.checkpoint, device)
        del critics
        task = config.task
        base_checkpoint = args.base_checkpoint or Path(config.base_checkpoint)

        action_steps = args.action_steps or config.action_steps
        inference_steps = args.inference_steps or config.inference_steps
        residual_scale = config.residual_scale
    else:
        if args.base_checkpoint is None or args.task is None:
            parser.error("recording the base policy alone needs --base-checkpoint and --task")
        actor = None
        task = args.task
        base_checkpoint = args.base_checkpoint
        native_steps, native_inference = native_settings(base_checkpoint)
        action_steps = args.action_steps or native_steps
        inference_steps = args.inference_steps or native_inference
        residual_scale = None

    prior = FrozenPolicyPrior(task, base_checkpoint, action_steps, inference_steps)
    if actor is None:
        policy = BasePolicy(prior)
    else:
        policy = ResidualPolicy(prior, actor, StateScaler(TASKS[task]["dataset_root"]), residual_scale,
                                float(getattr(config, "residual_max_norm", 0.0) or 0.0))
    logging.info(
        "Policy: %s | action_steps=%d inference_steps=%d", policy.description, action_steps, inference_steps
    )

    features = build_features(args.reference_root)
    task_instruction = reference_task_instruction(args.reference_root)
    raw_keys = {key: source for key in features if (source := source_key_for(key)) is not None}
    fps = json.loads((args.reference_root / "meta/info.json").read_text())["fps"]

    from lerobot_env_mimicgen.environment import normalize_task

    spec = normalize_task(task)
    env = RecordingMimicGenEnv(
        task=task,
        holder_sigma=args.holder_sigma,
        fixed_goal=args.fixed_goal,
        episode_length=spec.horizon,
        observation_height=224,
        observation_width=224,
        render_gpu_device_id=0,
    )

    quotas = {"success": args.successes, "failure": args.failures}
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = {
        outcome: make_dataset(
            f"{args.repo_prefix}_{outcome}",
            args.output_root / f"{args.repo_prefix.split('/')[-1]}_{outcome}",
            features,
            fps,
            args.overwrite,
        )
        for outcome in quotas
        if quotas[outcome] > 0
    }
    counts = dict.fromkeys(datasets, 0)
    episodes_meta: dict[str, list[dict]] = {outcome: [] for outcome in datasets}

    started = time.monotonic()
    attempted = 0
    try:
        for offset in range(args.max_episodes):
            if all(counts[outcome] >= quotas[outcome] for outcome in datasets):
                break
            seed = args.seed_start + offset
            frames, success = rollout(env, policy, seed, raw_keys, task_instruction)
            attempted += 1
            outcome = "success" if success else "failure"
            kept = outcome in datasets and counts[outcome] < quotas[outcome]
            if kept:
                dataset = datasets[outcome]
                for frame in frames:
                    dataset.add_frame(frame)
                episodes_meta[outcome].append(
                    {"episode_index": counts[outcome], "seed": seed, "length": len(frames)}
                )
                dataset.save_episode(parallel_encoding=True)
                counts[outcome] += 1
            logging.info(
                "seed=%d %s steps=%d %s | success %d/%d failure %d/%d | attempted=%d elapsed=%.1fm",
                seed,
                outcome,
                len(frames),
                "kept" if kept else "dropped",
                counts.get("success", 0),
                quotas["success"],
                counts.get("failure", 0),
                quotas["failure"],
                attempted,
                (time.monotonic() - started) / 60,
            )
    finally:
        env.close()
        prior.close()

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "base_checkpoint": str(base_checkpoint),
        "task": task,
        "policy": policy.description,
        "action_steps": action_steps,
        "inference_steps": inference_steps,
        "residual_scale": residual_scale,
        "seed_start": args.seed_start,
        "episodes_attempted": attempted,
        "counts": counts,
        "quotas": quotas,
    }
    for outcome, dataset in datasets.items():
        dataset.finalize()
        info = dict(summary, outcome=outcome, episodes=episodes_meta[outcome])
        (Path(dataset.root) / "meta/rollout_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    for outcome in datasets:
        if counts[outcome] < quotas[outcome]:
            logging.error(
                "Shard ended with %d/%d %s episodes after %d attempts",
                counts[outcome],
                quotas[outcome],
                outcome,
                attempted,
            )
            sys.exit(2)


if __name__ == "__main__":
    main()
