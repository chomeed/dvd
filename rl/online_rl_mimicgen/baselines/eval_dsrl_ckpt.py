"""Single-noise-draw evaluation of a dsrl_dino checkpoint.
usage: python -m online_rl_mimicgen.baselines.eval_dsrl_ckpt --ckpt <run>/checkpoints/0025000.zip --out <json>
       [--episodes 100] [--seed-start 10000] [--base-output <prior dir>] [--dataset-root <LeRobot dataset>]
       [--config <config.json>]
Run it from the repository root (python -m puts the working directory first on sys.path).
Runs trained by the pre-rename code (old method tag, absolute paths of another machine) are accepted through
..legacy; only the SB3 weights are read from the .zip, never its pickled class attributes.
The policy is rebuilt from the run's config.json, found at <run>/config.json where <run> is two levels above the REAL
checkpoint file: --ckpt is resolved through symlinks, as in the research code. A checkpoint must therefore be a real
file inside <run>/checkpoints/ (fetch a run with local_dir, not out of the Hugging Face hub cache, whose snapshot files
are links into blobs/), or --config must name the config explicitly."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .dsrl_dino import DinoFrameEnv, DSRLDVDDino, ExecutedNoisePrior, PixelProprioSlice
from .dsrl_mimicgen import FlowNoisePrior, StateSlice
from stable_baselines3.common.save_util import load_from_zip_file  # noqa: E402  (fork on sys.path via dsrl_mimicgen)
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from .. import legacy
from ..common import TASKS, FrozenPolicyPrior, StateScaler, find_latest_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--base-output", type=Path, default=None,
                    help="prior run directory, overriding base_output in the run's config.json (an absolute path of "
                         "the training machine). Default: that path if it exists, else $CKPT_ROOT/priors/<its name>")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="the task's LeRobot dataset (meta/stats.json feeds the state scaler); "
                         "default $DATA_ROOT/mimicgen_<task>_224x224")
    ap.add_argument("--config", type=Path, default=None,
                    help="the run's config.json. Default: <run>/config.json with <run> two levels above the real "
                         "checkpoint file (--ckpt is resolved through symlinks). With --config, the run is the "
                         "directory holding that file")
    a = ap.parse_args()
    run = a.config.absolute().parent if a.config else a.ckpt.resolve().parents[1]
    config_path = a.config or run / "config.json"
    if not config_path.is_file():
        raise SystemExit(
            f"[eval-dsrl] no config.json at {config_path}. --ckpt {a.ckpt} resolves to {a.ckpt.resolve()} and the run "
            f"directory is taken to be two levels above that real file. Checkpoints must be real files inside "
            f"<run>/checkpoints/ (not links into another store such as the Hugging Face hub cache), or pass "
            f"--config <the run's config.json>")
    cfg = legacy.upgrade_config(json.loads(config_path.read_text()))


    method = cfg.get("method")
    if method not in ("dsrl_na_dvd_dino", "dsrl_na_q_dino"):
        raise SystemExit(f"{run} is not a dsrl_dino run (method={method!r})")
    if bool(cfg.get("q_critic")) != (method == "dsrl_na_q_dino"):
        raise SystemExit(f"{run}: config.json disagrees with itself (method={method!r}, "
                         f"q_critic={cfg.get('q_critic')!r}); refusing to guess which policy this is")

    task_cfg = TASKS[cfg["task"]]
    state_dim = int(task_cfg["state_dim"])
    base_output = legacy.remap_asset_path(cfg["base_output"], "prior", a.base_output)
    prior = FrozenPolicyPrior(cfg["task"], find_latest_checkpoint(base_output), cfg["action_steps"],
                              cfg["inference_steps"])
    flow = FlowNoisePrior(prior, state_dim)
    if cfg.get("executed_noise_only", "off") != "off":

        flow = ExecutedNoisePrior(flow, cfg["executed_noise_only"])
    env = DinoFrameEnv(cfg["task"], flow, StateScaler(a.dataset_root or task_cfg["dataset_root"]), seed=a.seed_start,
                       variant="na", magnitude=cfg["action_magnitude"], step_penalty=cfg["step_penalty"],
                       dense_reward=cfg["env_dense_reward"], horizon=cfg["episode_horizon"])
    net = [cfg["layer_size"]] * cfg["num_layers"]
    if cfg.get("image_actor"):
        extractor = dict(features_extractor_class=PixelProprioSlice,
                         features_extractor_kwargs=dict(out_dim=cfg["pixel_feature_dim"], shift_pad=cfg["pixel_shift_pad"]))
    else:
        extractor = dict(features_extractor_class=StateSlice, features_extractor_kwargs=dict(state_dim=state_dim))
    device = prior.device


    if float(cfg.get("log_std_min", -20.0)) != -20.0:
        from stable_baselines3.sac import policies as _sacp
        _sacp.LOG_STD_MIN = float(cfg["log_std_min"])
    _act = {"mish": torch.nn.Mish, "tanh": torch.nn.Tanh}[cfg.get("activation", "tanh")]
    model = DSRLDVDDino(
        "MlpPolicy", DummyVecEnv([lambda: env]), diffusion_policy=flow, diffusion_act_dim=(flow.horizon, flow.action_dim),
        noise_critic_grad_steps=cfg["noise_critic_steps"], critic_backup_combine_type="min", learning_rate=cfg["lr"],
        buffer_size=1, learning_starts=0, batch_size=cfg["batch_size"], tau=cfg["tau"],
        gamma=cfg["gamma"] ** cfg["action_steps"], train_freq=cfg["train_freq"], gradient_steps=cfg["utd"],
        ent_coef="auto", target_update_interval=1, target_entropy=cfg["target_entropy"], seed=a.seed_start,
        device=device, verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=net, qf=net), activation_fn=_act, log_std_init=0.0,
                           post_linear_modules=[torch.nn.LayerNorm], n_critics=cfg["n_critics"],
                           share_features_extractor=False, **extractor))
    _, params, _ = load_from_zip_file(str(a.ckpt), load_data=False, device=device)
    model.set_parameters(params, exact_match=True, device=device)
    model.image_actor, model.frame_env = bool(cfg.get("image_actor")), env
    model.deterministic_actor, model.eval_mode = bool(cfg.get("deterministic_actor")), True
    model.policy.set_training_mode(False)
    model.num_timesteps = 10 ** 12

    successes, lengths, started = [], [], time.time()
    for i in range(a.episodes):
        seed = a.seed_start + i
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        env.episode_index = i
        obs, _ = env.reset()
        success, steps = False, 0
        while True:
            model._last_obs = obs[None]
            with torch.no_grad():
                action, _ = model._sample_action(learning_starts=0)
            obs, _, terminated, truncated, info = env.step(action[0])
            steps += int(info["primitive_steps"])
            success = success or bool(info["is_success"])
            if terminated or truncated:
                break
        successes.append(bool(success)); lengths.append(steps)
        print(f"[eval-dsrl] episode {i} seed {seed}: {'success' if success else 'fail'} ({steps} steps)  "
              f"running {100 * np.mean(successes):.1f}%", flush=True)

    out = {"ckpt": str(a.ckpt), "run": run.name, "method": method, "episodes": len(successes),
           "success_rate": 100.0 * float(np.mean(successes)), "successes": successes, "lengths": lengths,
           "seed_start": a.seed_start, "protocol": ("dsrl deterministic noise mu(s) per chunk (TD3-style actor, no exploration noise, no best-of-N)"
                         if model.deterministic_actor else "dsrl single noise draw per chunk (stochastic pi_W sample, no best-of-N)"),
           "minutes": (time.time() - started) / 60}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[eval-dsrl] {run.name} {a.ckpt.stem}: {out['success_rate']:.1f}% over {len(successes)} episodes "
          f"(actor trained by {'a TD pixel Q (--q-critic)' if method == 'dsrl_na_q_dino' else 'the DVD term'})",
          flush=True)


if __name__ == "__main__":
    main()
