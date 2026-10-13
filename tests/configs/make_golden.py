"""Regenerate tests/configs/golden/ from the reference research code. PROVENANCE TOOL: it only runs on the machine
that holds the pre-rename research tree, its assets and the campaign launchers; the committed golden files are its output.

    python tests/configs/make_golden.py --launchers <dir holding launch_dicerl_ab.sh and launch_hammer_dsrl.sh>
                                        [--live-tree <research tree>] [--out tests/configs/golden]

For every cell of tests/configs/_cells.py, with SEED=99 and STEPS=600000 and 1000000:

  DICE-RL  the launcher itself is run with TRAIN_ONLINE_DRYRUN=1, so the research trainer resolves its own argv into
           the TrainConfig it would train with, prints it and exits before building anything;
  DSRL     the research trainer has no dry run, so a temporary copy of the launcher whose LAST command (`exec python -m
           ...dsrl_dino`) is replaced by a recorder is run instead: every line that chooses assets, flags, names and
           groups is the launcher's own, the environment is the launcher's own, and the recorder parses the argv with
           the research module's own build_dino_parser(). The same copy trick records the DICE-RL argv, run name,
           wandb group and environment.

What is written (one JSON per cell) is that output with exactly these normalisations, so that nothing in it names a
machine and the names are the ones this repository uses:

  keys / flags    renamed with online_rl_mimicgen/legacy.py (svg_* -> dvd_*, and the option values svg/svg1)
  asset paths     <CKPT_ROOT>/priors/<prior>[/checkpoints/<step>/pretrained_model], <CKPT_ROOT>/value/<V dir>,
                  <CKPT_ROOT>/wm/mimicgen_<task>_224x224_dinowm_<tag>/wm_dinov2s14_<tag>.pt (the reference machine keeps
                  one task's file outside any per-task directory: recorded as source.wm_layout = "flat"),
                  <SPOOL_ROOT>/<task> (the reference spool directory name is recorded as source.spool),
                  <RUNS_ROOT>/<run name>
  run name        "svg" -> "dvd" (the reference name is recorded as source.name)

Nothing is trained, no run directory is created, no process is signalled; bytecode writing is disabled so the research
tree is not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "rl"))

from _cells import CELLS, GOLDEN_SEED, GOLDEN_STEPS, Cell  # noqa: E402
from online_rl_mimicgen import legacy  # noqa: E402

RECORDER = r'''
import json, os, sys
out, rest = sys.argv[1], sys.argv[2:]
cut = rest.index("--")
record = {"meta": dict(token[1:].split("=", 1) for token in rest[:cut]), "module": rest[cut + 1], "argv": rest[cut + 2:],
          "env": {key: os.environ.get(key) for key in ("PYTORCH_CUDA_ALLOC_CONF", "PIXEL_IMG", "WANDB_MODE", "HF_HUB_OFFLINE")}}
if record["module"].endswith(".dsrl_dino"):
    from pathlib import Path
    import online_rl_mimicgen
    from online_rl_mimicgen.baselines import dsrl_dino
    args = dsrl_dino.build_dino_parser().parse_args(record["argv"])
    record["config"] = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    record["package"] = str(Path(online_rl_mimicgen.__file__).resolve().parent)
json.dump(record, open(out, "w"))
'''

# the launcher's last command -> the recorder (same environment, same working directory, same argv)
EXEC = {
    "launch_dicerl_ab.sh": 'exec $CONDA_PREFIX/bin/python -m online_rl_mimicgen.train_online "${ARGS[@]}" \\',
    "launch_hammer_dsrl.sh": 'exec $CONDA_PREFIX/bin/python -m online_rl_mimicgen.baselines.dsrl_dino "${COMMON[@]}" "${ARGS[@]}" \\',
}
RECORD = ('exec $CONDA_PREFIX/bin/python "$GOLDEN_RECORDER" "$GOLDEN_OUT" "@GROUP=${WANDB_RUN_GROUP:-}" "@NAME=$NAME" -- ')
INERT = {"launch_dicerl_ab.sh": {"mkdir -p $P/dicerl": ":"}, "launch_hammer_dsrl.sh": {}}
SCRUB = ("PYTORCH_CUDA_ALLOC_CONF", "WANDB_RUN_GROUP", "WANDB_MODE", "NAME", "OUT", "PROJECT", "SAVE", "EVAL_EPS", "WARMUP",
         "DSRL_DEMO_CACHE", "NAME_ONLY", "PYTHONPATH", "TASK", "ARM", "SEED", "STEPS", "G", "DISTILL")


def recording_copy(launcher: Path, workdir: Path) -> Path:
    text = launcher.read_text()
    line = EXEC[launcher.name]
    if text.count(line) != 1:
        raise SystemExit(f"{launcher}: expected its launch command exactly once:\n  {line}")
    module = line.split(" -m ")[1].split(" ")[0]
    text = text.replace(line, RECORD + module + line.split(module, 1)[1])
    for old, new in INERT[launcher.name].items():
        if text.count(old) != 1:
            raise SystemExit(f"{launcher}: expected `{old}` exactly once")
        text = text.replace(old, new)
    copy = workdir / f"record_{launcher.name}"
    copy.write_text(text)
    return copy


def run(script: Path, cell: Cell, steps: int, extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(TASK=cell.task, ARM=cell.live_arm, SEED=str(GOLDEN_SEED), STEPS=str(steps), G="0",
               TRAIN_ONLINE_DRYRUN="1", PYTHONDONTWRITEBYTECODE="1", **extra)
    done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{script.name} {cell.id} steps={steps} exited {done.returncode}:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done


def reference(cell: Cell, steps: int, launchers: Path, workdir: Path) -> dict:
    """What the reference launcher + research code resolve for this cell: argv, name, group, environment, config."""
    launcher = launchers / cell.launcher
    out = workdir / f"{cell.method}_{cell.id}_{steps}.json"
    run(recording_copy(launcher, workdir), cell, steps, {"GOLDEN_RECORDER": str(workdir / "recorder.py"), "GOLDEN_OUT": str(out)})
    record = json.loads(out.read_text())
    if cell.method == "dice_rl":
        lines = [line for line in run(launcher, cell, steps, {}).stdout.splitlines() if line.startswith("DRYRUN_CONFIG ")]
        if len(lines) != 1:
            raise SystemExit(f"{launcher.name} {cell.id}: expected one DRYRUN_CONFIG line, got {len(lines)}")
        record["config"] = json.loads(lines[0][len("DRYRUN_CONFIG "):])
    return record


def normalise_path(key: str, value: str, task: str, names: tuple[str, str], notes: dict) -> str:
    path = PurePosixPath(value)
    if key == "base_output":
        if path.parent.name != "priors":
            raise SystemExit(f"base_output {value}: not below a priors/ directory")
        return f"<CKPT_ROOT>/priors/{path.name}"
    if key == "base_checkpoint":
        cut = len(path.parts) - 1 - path.parts[::-1].index("priors")
        return "<CKPT_ROOT>/priors/" + "/".join(path.parts[cut + 1:])
    if key == "value_model":
        return f"<CKPT_ROOT>/value/{path.name}"
    if key == "dino_wm_ckpt":
        tag = re.fullmatch(r"wm_dinov2s14_(.+)\.pt", path.name).group(1)
        folder = f"mimicgen_{task}_224x224_dinowm_{tag}"
        notes["wm_layout"] = "per-task" if path.parent.name == folder else "flat"
        return f"<CKPT_ROOT>/wm/{folder}/{path.name}"
    if key == "value_distill_robometer":
        notes["spool"] = path.name
        return f"<SPOOL_ROOT>/{task}"
    if key == "output_dir":
        if path.name != names[0]:
            raise SystemExit(f"output_dir {value} does not end in the run name {names[0]}")
        return f"<RUNS_ROOT>/{names[1]}"
    raise KeyError(key)


PATH_KEYS = ("base_output", "base_checkpoint", "value_model", "dino_wm_ckpt", "value_distill_robometer", "output_dir")


def normalise(cell: Cell, record: dict) -> dict:
    live_name = record["meta"]["NAME"]
    names = (live_name, live_name.replace(legacy.OLD, legacy.NEW))
    notes: dict = {"spool": None, "wm_layout": None}

    config = legacy.upgrade_config(record["config"])
    for key in PATH_KEYS:
        if config.get(key):
            config[key] = normalise_path(key, config[key], cell.task, names, notes)
    if config.get("wandb_name") != live_name:
        raise SystemExit(f"{cell.id}: wandb_name {config.get('wandb_name')!r} is not the run name {live_name!r}")
    config["wandb_name"] = names[1]

    argv, dest = [], None
    for token in record["argv"]:
        if token.startswith("--"):
            dest = legacy.KEY_RENAMES.get(token[2:].replace("-", "_"), token[2:].replace("-", "_"))
            argv.append("--" + dest.replace("_", "-"))
            continue
        if dest in PATH_KEYS:
            token = normalise_path(dest, token, cell.task, names, {})
        elif dest == "wandb_name":
            token = names[1]
        else:
            token = legacy.VALUE_RENAMES.get(dest, {}).get(token, token)
        argv.append(token)

    for where, values in (("config", config.values()), ("argv", argv)):
        for value in values:
            if isinstance(value, str) and (value.startswith("/") or legacy.OLD in value.lower()):
                raise SystemExit(f"{cell.id}: {where} still holds a machine path or an old name: {value!r}")
    return {"name": names[1], "group": record["meta"]["GROUP"], "module": record["module"],
            "env": {"PYTORCH_CUDA_ALLOC_CONF": record["env"]["PYTORCH_CUDA_ALLOC_CONF"]},
            "argv": argv, "config": config, "_live": {"name": live_name, **notes, "package": record.get("package")}}


def commit_of(tree: Path | None) -> str | None:
    """HEAD commit of the research tree, read from .git without running git."""
    if tree is None:
        return None
    head = (tree / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = tree / ".git" / head[5:]
    if ref.exists():
        return ref.read_text().strip()
    for line in (tree / ".git" / "packed-refs").read_text().splitlines():
        if line.endswith(" " + head[5:]):
            return line.split()[0]
    raise SystemExit(f"cannot resolve {head} in {tree}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launchers", required=True, type=Path)
    ap.add_argument("--live-tree", type=Path, default=None, help="research tree: records its HEAD commit and checks that "
                                                                 "the DSRL parser was imported from it")
    ap.add_argument("--out", type=Path, default=HERE / "golden")
    ap.add_argument("--only", default="", help="substring of <method>/<cell id>: regenerate matching cells only")
    args = ap.parse_args()
    commit = commit_of(args.live_tree)
    base, other = GOLDEN_STEPS
    with tempfile.TemporaryDirectory(prefix="make_golden_") as tmp:
        workdir = Path(tmp)
        (workdir / "recorder.py").write_text(RECORDER)
        for cell in CELLS:
            if args.only not in f"{cell.method}/{cell.id}":
                continue
            launcher = args.launchers / cell.launcher
            first = normalise(cell, reference(cell, base, args.launchers, workdir))
            second = normalise(cell, reference(cell, other, args.launchers, workdir))
            live = first.pop("_live")
            if args.live_tree is not None and live["package"] is not None:
                if Path(live["package"]).parent != args.live_tree.resolve():
                    raise SystemExit(f"{cell.id}: the DSRL parser came from {live['package']}, not from {args.live_tree}")
            golden = {
                "_about": "Resolved configuration of one campaign cell, taken from the reference research code by "
                          "tests/configs/make_golden.py (see its docstring for the normalisations). Do not edit.",
                "cell": {"method": cell.method, "config": cell.config_path, "task": cell.task, "variant": cell.variant,
                         "campaign": cell.campaign},
                "source": {"launcher": cell.launcher, "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
                           "arm": cell.live_arm, "seed": GOLDEN_SEED, "steps": base, "commit": commit,
                           "name": live["name"], "spool": live["spool"], "wm_layout": live["wm_layout"]},
                **first,
                f"steps_{other}": {
                    "name": second["name"],
                    "config": {key: value for key, value in second["config"].items() if first["config"].get(key) != value},
                    "argv": [[index, token] for index, (token, was) in enumerate(zip(second["argv"], first["argv"])) if token != was],
                },
            }
            if set(second["config"]) != set(first["config"]) or len(second["argv"]) != len(first["argv"]):
                raise SystemExit(f"{cell.id}: the {other}-step resolution has other keys or another argv length")
            text = json.dumps(golden, indent=1, sort_keys=False) + "\n"
            for leak in {str(Path.home()), str(args.launchers.resolve().parent), os.environ.get("USER", "\0")}:
                if leak and leak in text:
                    raise SystemExit(f"{cell.id}: the golden text would contain {leak!r}")
            path = args.out / cell.method / f"{cell.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{cell.method}/{cell.id}: {len(first['config'])} keys, {len(first['argv'])} argv tokens, "
                  f"group {first['group']}, {other}: {sorted(golden[f'steps_{other}']['config'])}", flush=True)


if __name__ == "__main__":
    main()
