"""The configuration a trainer resolves from an argv, without training.

    python scripts/resolve_config.py MODULE [TRAINER ARGUMENT ...]        # prints "DRYRUN_CONFIG <json>"
    python scripts/resolve_config.py --help

Used by scripts/run.sh and scripts/smoke_train.sh under TRAIN_ONLINE_DRYRUN=1 for the DSRL trainer, which has no dry
run of its own, and by tests/configs/test_configs.py for both trainers. It calls the trainers' own argument code and
nothing else:

  online_rl_mimicgen.train_online         parse_args() -> TrainConfig, serialised as the trainer's own dry run does
  online_rl_mimicgen.baselines.dsrl_dino  build_dino_parser().parse_args(), serialised as main() writes config.json
                                          (paths as strings, plus the "method" tag). main()'s argument checks run
                                          only in a real launch.

No environment, model, run directory or wandb run is created. The DICE-RL trainer resolves the newest checkpoint
directory below --base-output, so that prior must exist; nothing else is read from the asset roots. This checkout's
rl/ is put first on sys.path, so the result never comes from another copy of the code. Needs the main environment
(torch) and, for DSRL, DSRL_ROOT (scripts/env.sh sets it).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

RL = Path(__file__).resolve().parents[1] / "rl"
DICE_RL = "online_rl_mimicgen.train_online"
DSRL = "online_rl_mimicgen.baselines.dsrl_dino"
EPILOG = f"""modules:
  {DICE_RL:<40}DICE-RL (configs/dice_rl/*.sh)
  {DSRL:<40}DSRL (configs/dsrl/*.sh)

Everything after MODULE is handed to that trainer's parser unchanged (also --help, which then prints the trainer's
own help). The usual way to get a cell's argv is scripts/run.sh:

  RUN_PRINT_ARGV=1 bash scripts/run.sh configs/dsrl/ours.sh square_d0 43        # the argv, one token per line
  TRAIN_ONLINE_DRYRUN=1 bash scripts/run.sh configs/dsrl/ours.sh square_d0 43   # this script on that argv

exit status: 0 and one "DRYRUN_CONFIG <json>" line on stdout; 2 on a usage error or an argv the trainer rejects.
"""


def _use_this_tree() -> None:
    """Import online_rl_mimicgen from this checkout's rl/, whatever PYTHONPATH says, and prove it."""
    if sys.path[:1] != [str(RL)]:
        sys.path.insert(0, str(RL))
    import online_rl_mimicgen
    origin = Path(online_rl_mimicgen.__file__).resolve()
    if RL not in origin.parents:
        raise SystemExit(f"online_rl_mimicgen was already imported from {origin}, not from {RL}")


def resolve(module: str, argv: list[str]) -> dict:
    _use_this_tree()
    if module == DICE_RL:
        from online_rl_mimicgen import train_online
        saved = sys.argv
        sys.argv = [saved[0], *argv]
        try:
            config = train_online.parse_args()
        finally:
            sys.argv = saved
        return json.loads(json.dumps(dataclasses.asdict(config), default=str))
    if module == DSRL:
        from online_rl_mimicgen.baselines import dsrl_dino
        args = dsrl_dino.build_dino_parser().parse_args(argv)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        return {**config, "method": "dsrl_na_q_dino" if args.q_critic else "dsrl_na_dvd_dino"}
    raise SystemExit(f"unknown trainer module {module!r}; expected {DICE_RL} or {DSRL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/resolve_config.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0] + " Prints one line, DRYRUN_CONFIG <json>: every resolved argument of "
                    "the trainer (301 TrainConfig fields for DICE-RL, 92 keys for DSRL).",
        epilog=EPILOG)
    parser.add_argument("module", metavar="MODULE", choices=(DICE_RL, DSRL), help="the trainer whose argument code is called")
    parser.add_argument("argv", metavar="TRAINER ARGUMENT", nargs=argparse.REMAINDER,
                        help="the trainer's argv, exactly as scripts/run.sh would pass it")
    args = parser.parse_args()
    print("DRYRUN_CONFIG " + json.dumps(resolve(args.module, args.argv), default=str), flush=True)


if __name__ == "__main__":
    main()
