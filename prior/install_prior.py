#!/usr/bin/env python3
"""Install the release step of a prior-training run where every RL cell of the task looks for its prior.

    python prior/install_prior.py --task TASK --run DIR --step NNNNNN --out DIR [--dataset DIR] [--weights copy|hardlink]
    python prior/install_prior.py --task TASK --verify DIR [--dataset DIR]     # check an installed prior; write nothing
    TASK=<task> STAGE=install bash prior/train_prior.sh                        # the first line with the paths below

  --run      the LeRobot run directory prior/train_prior.sh trains into ($RUNS_ROOT/prior_train/<run>)
  --step     the release step, six digits: 040000 / 045000 / 055000 / 035000 for square_d0 / square_d1 / threading_d0 /
             hammer_cleanup_d1. Never `last`: that is the final step of the run
  --out      the prior directory configs/tasks.sh names ($CKPT_ROOT/priors/<prior>; for square_d1, threading_d0 and
             hammer_cleanup_d1 the name still ends in _statefix although a retrained prior needs no state fix)
  --dataset  the 9-D training set ($DATA_ROOT/mimicgen_<task>_224x224_robotstate9) or the demo dataset; its observation.state
             and action statistics (the state's first 9 entries) must equal the installed normaliser
  --weights  how model.safetensors (1.0 GB) gets there: copy [default] or hardlink (same file system, no extra disk)

WHAT IT DOES. Copies <run>/checkpoints/<step>/pretrained_model (every file, tokenizer/ included; training_state/ never)
to <out>/checkpoints/<step>/pretrained_model: the layout of the released priors, one step and no `last`. That matters:
online_rl_mimicgen.common.find_latest_checkpoint, which both trainers and the evaluators apply to --base-output,
prefers checkpoints/last over every numbered step, so a run directory used as the prior would load its final step.
It REFUSES when <out> exists at all, so a released prior is never replaced: move it away first, or install under
another CKPT_ROOT. The copy is assembled as <out>.partial, checked, and renamed.

CHECKS (exit 1 if one fails; --verify runs them on an installed prior, including the four released ones):
  - config.json: type multi_task_dit, observation.state shape [9], no n_state_dims (the field of the unavailable fork)
  - model.safetensors: the conditioning width of the first DiT block equals
        timestep_embed_dim + n_obs_steps x (vision width x cameras + 9 + hidden_dim)      (4370 for these priors)
    read from the safetensors header only: the network really takes a 9-D state
  - train_config.json: trained on this task's data (dataset.repo_id names mimicgen_<task>_224x224); how the state was
    cut is reported (the 9-D copy, or --policy.n_state_dims of the fork for three released priors)
  - with --dataset: the normaliser's observation.state.* and action.* tensors equal the dataset statistics exactly
  - find_latest_checkpoint(<out>) resolves to the installed directory
Then it prints the command that scores the installed prior.

KNOWN CONSEQUENCE. With a retrained square_d1 prior installed, scripts/fetch_assets.py --check reports the derived
square_d1 item as failing scripts/make_square_d1_statefix.py --verify: its weights are not the released ones. That is
correct, not a fault of the installation.
Needs torch and safetensors (the main environment); CPU only; imports online_rl_mimicgen from this checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = "observation.state"
KEEP = 9
TASKS = ("square_d0", "square_d1", "threading_d0", "hammer_cleanup_d1")
REQUIRED = ("config.json", "model.safetensors", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def helpers():
    """safetensors_header of scripts/make_square_d1_statefix.py and the statistics helpers of prior/state9_dataset.py."""
    for sub in ("scripts", "prior"):
        if str(REPO / sub) not in sys.path:
            sys.path.insert(0, str(REPO / sub))
    import make_square_d1_statefix
    import state9_dataset
    return make_square_d1_statefix, state9_dataset


def find_latest(prior_dir: Path) -> Path:
    """online_rl_mimicgen.common.find_latest_checkpoint of THIS checkout (what the RL code calls on --base-output)."""
    rl = str(REPO / "rl")
    if rl in sys.path:
        sys.path.remove(rl)
    sys.path.insert(0, rl)
    from online_rl_mimicgen.common import find_latest_checkpoint
    return find_latest_checkpoint(prior_dir)


def check(pretrained: Path, task: str, dataset: Path | None) -> list[str]:
    """The content checks of the module docstring on one pretrained_model directory; prints one line per check."""
    statefix, state9 = helpers()
    problems: list[str] = []

    def report(passed: bool, text: str) -> None:
        print(f"  {'ok  ' if passed else 'FAIL'} {text}")
        if not passed:
            problems.append(text)

    missing = [name for name in REQUIRED if not (pretrained / name).is_file()]
    normalisers = sorted(pretrained.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    report(not missing and len(normalisers) == 1, f"files: {', '.join(missing) or 'all of ' + ', '.join(REQUIRED)}"
           f"{' missing' if missing else ' present'}; {len(normalisers)} normaliser file(s)")
    if missing:
        return problems
    config = json.loads((pretrained / "config.json").read_text())
    shape = config.get("input_features", {}).get(STATE, {}).get("shape")
    report(config.get("type") == "multi_task_dit" and shape == [KEEP] and "n_state_dims" not in config,
           f"config.json: type {config.get('type')}, {STATE} shape {shape}, n_state_dims "
           f"{config['n_state_dims'] if 'n_state_dims' in config else 'absent'}")

    header = statefix.safetensors_header(pretrained / "model.safetensors")
    cond = header.get("noise_predictor.transformer_blocks.0.adaLN_modulation.1.weight", {}).get("shape")
    vision = header.get("observation_encoder.vision_encoder.model.vision_model.post_layernorm.weight", {}).get("shape")
    cameras = sum(1 for f in config.get("input_features", {}).values() if f.get("type") == "VISUAL")
    if cond is None or vision is None:
        report(False, "model.safetensors: conditioning or vision tensor not in the header")
    else:
        want = config["timestep_embed_dim"] + config["n_obs_steps"] * (vision[0] * cameras + KEEP + config["hidden_dim"])
        report(cond[1] == want, f"model.safetensors: conditioning width {cond[1]} = {config['timestep_embed_dim']} + "
               f"{config['n_obs_steps']} x ({vision[0]} x {cameras} cameras + {KEEP} state + {config['hidden_dim']}) = {want}")

    train = json.loads((pretrained / "train_config.json").read_text())
    repo_id = train.get("dataset", {}).get("repo_id") or ""
    name = repo_id.split("/")[-1]
    how = ("the 9-D copy of the dataset" if name == f"mimicgen_{task}_224x224_robotstate9" else
           f"--policy.n_state_dims={train['policy']['n_state_dims']} of the fork" if
           name == f"mimicgen_{task}_224x224" and train.get("policy", {}).get("n_state_dims") == KEEP else None)
    report(how is not None, f"train_config.json: dataset {repo_id}, {train.get('steps')} steps; state cut by {how or 'NOTHING KNOWN'}"
           + ("" if how else f" (not a {task} prior?)"))

    if dataset is not None:
        stats_path = dataset / "meta" / "stats.json"
        if not stats_path.is_file():
            report(False, f"--dataset {dataset}: no meta/stats.json")
        else:
            stats = json.loads(stats_path.read_text())
            width = len(stats[STATE]["min"])
            if width > KEEP:
                stats = state9.cut_stats(stats, width)
            found = state9.normaliser_problems(stats, pretrained)
            report(not found, f"normaliser: {STATE}.* and action.* {'equal' if not found else 'DIFFER from'} the statistics "
                   f"of {dataset.name}" + (f" (state: its first {KEEP} of {width})" if width > KEEP else "")
                   + (f": {found[0]}" if found else ""))
    return problems


def installed_step(prior_dir: Path) -> Path:
    steps = sorted(p for p in (prior_dir / "checkpoints").glob("*") if p.is_dir())
    if [p.name for p in steps if not re.fullmatch(r"\d{6,}", p.name)] or len(steps) != 1:
        raise SystemExit(f"{prior_dir}/checkpoints holds {[p.name for p in steps]}; an installed prior holds exactly one "
                         f"numbered step and no `last`")
    return steps[0] / "pretrained_model"


def resolves_to(prior_dir: Path, pretrained: Path, shown: Path) -> list[str]:
    got = find_latest(prior_dir)
    good = got == pretrained.resolve()
    print(f"  {'ok  ' if good else 'FAIL'} find_latest_checkpoint({shown}) -> "
          f"{Path(shown) / got.relative_to(prior_dir.resolve()) if good else got}")
    return [] if good else [f"find_latest_checkpoint resolves to {got}"]


def install(task: str, run: Path, step: str, out: Path, dataset: Path | None, weights: str) -> None:
    if not re.fullmatch(r"\d{6,}", step):
        raise SystemExit(f"--step must be a numbered checkpoint such as 045000 (got {step!r}); `last` is the final step")
    if out.exists() or out.is_symlink():
        raise SystemExit(f"{out} exists: this script never replaces a prior (it may be the RELEASED one). Move it away "
                         f"first, or install under another CKPT_ROOT")
    src = run / "checkpoints" / step / "pretrained_model"
    if not (src / "config.json").is_file():
        have = sorted(p.name for p in (run / "checkpoints").glob("*")) if (run / "checkpoints").is_dir() else []
        raise SystemExit(f"no checkpoint {src} (the run holds {have or 'no checkpoints'}); train first: "
                         f"TASK={task} STAGE=train bash prior/train_prior.sh")
    if (run / "checkpoints" / step).is_symlink():
        raise SystemExit(f"{run}/checkpoints/{step} is a link; install from the real step directory")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)                                 # left by an interrupted call of this script
    target = partial / "checkpoints" / step / "pretrained_model"
    target.mkdir(parents=True)
    copied = 0
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        if path.is_dir():
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "model.safetensors" and weights == "hardlink":
            try:
                os.link(os.path.realpath(path), target / relative)
            except OSError as error:
                raise SystemExit(f"--weights hardlink: {error}; use --weights copy") from None
        else:
            shutil.copyfile(path, target / relative)
        copied += 1
    print(f"checking {target}")
    problems = check(target, task, dataset)
    problems += resolves_to(partial, target, partial)
    if problems:
        raise SystemExit(f"FAILED ({len(problems)} check(s)); nothing installed, the copy is kept as {partial}")
    partial.rename(out)
    installed = out / "checkpoints" / step / "pretrained_model"
    problems = resolves_to(out, installed, out)
    if problems:
        raise SystemExit("FAILED after the rename: " + problems[0])
    print(f"installed {installed}\n  {copied} files from {src}\n"
          f"score it (one GPU, 100 episodes; the released priors: 74 / 38 / 64 / 40 % on square_d0 / square_d1 / "
          f"threading_d0 / hammer_cleanup_d1):\n  python eval/eval_prior.py --task {task} --out <json>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--run", type=Path, help="the LeRobot run directory")
    parser.add_argument("--step", help="the release step, e.g. 045000")
    parser.add_argument("--out", type=Path, help="$CKPT_ROOT/priors/<prior>")
    parser.add_argument("--dataset", type=Path, default=None, help="dataset whose statistics the normaliser must equal")
    parser.add_argument("--weights", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--verify", type=Path, default=None, metavar="DIR", help="check an installed prior directory")
    args = parser.parse_args()
    if args.verify is not None:
        if args.run or args.step or args.out:
            parser.error("--verify takes no --run / --step / --out")
        pretrained = installed_step(args.verify)
        print(f"checking {pretrained}")
        problems = check(pretrained, args.task, args.dataset) + resolves_to(args.verify, pretrained, args.verify)
        if problems:
            raise SystemExit(f"FAILED ({len(problems)} check(s))")
        print("all checks passed")
        return
    if not (args.run and args.step and args.out):
        parser.error("--run, --step and --out are required (or --verify DIR)")
    install(args.task, args.run, args.step, args.out, args.dataset, args.weights)


if __name__ == "__main__":
    sys.exit(main())
