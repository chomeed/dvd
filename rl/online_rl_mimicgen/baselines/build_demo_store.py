"""Build (and cache) the DSRL expert-demo store for a task, without starting a run.
usage (after `source scripts/env.sh`):"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .. import dino_dvd
from ..common import CACHE_ROOT, TASKS, FrozenPolicyPrior, StateScaler, find_latest_checkpoint
from .dsrl_demos import DemoStore
from .dsrl_mimicgen import FlowNoisePrior
from .dsrl_dino import ExecutedNoisePrior


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--base-output", required=True)
    p.add_argument("--executed-noise-only", choices=("off", "zeros", "resample"), default="off")
    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--inference-steps", type=int, default=10)


    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--capacity", type=int, default=0)
    p.add_argument("--cache-dir", default=os.environ.get("DSRL_DEMO_CACHE", str(CACHE_ROOT / "dsrl_demo_store")))
    p.add_argument("--pixel-cache-dir", default=os.environ.get("PIXEL_CACHE_DIR", str(CACHE_ROOT / "pixel_cache")))
    a = p.parse_args()

    task_cfg = TASKS[a.task]
    checkpoint = find_latest_checkpoint(Path(a.base_output))


    prior = FrozenPolicyPrior(a.task, checkpoint, a.action_steps, a.inference_steps)
    full_flow = FlowNoisePrior(prior, int(task_cfg["state_dim"]))
    flow = (full_flow if a.executed_noise_only == "off"
            else ExecutedNoisePrior(full_flow, a.executed_noise_only))
    scaler = StateScaler(task_cfg["dataset_root"])

    cache_name = f"pixel_cache_{Path(task_cfg['dataset_root']).name}_{dino_dvd.IMAGE_SIZE}.npy"
    print(f"[build-demo-store] task {a.task}  prior {checkpoint}  noise {a.executed_noise_only}  "
          f"horizon {flow.horizon}", flush=True)
    store = DemoStore.build(
        dataset_root=task_cfg["dataset_root"],
        pixel_cache_path=Path(a.pixel_cache_dir) / cache_name,
        flow=flow, scaler=scaler, action_steps=a.action_steps, step_penalty=a.step_penalty,
        device=full_flow.device, capacity=int(a.capacity), cache_dir=Path(a.cache_dir),
        prior_id=str(checkpoint))
    print(f"[build-demo-store] done: {len(store)} rows", flush=True)


if __name__ == "__main__":
    main()
