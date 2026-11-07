#!/usr/bin/env python3
"""LeRobot's trainer (python -m lerobot.scripts.lerobot_train) with one compatibility shim for the pinned wandb.

    python prior/lerobot_train.py <lerobot_train arguments>       # what prior/train_prior.sh STAGE=train runs

WHY. LeRobot 0.6.2 @ 4aaff99b prints the run's address with wandb.run.get_url() when its wandb logger starts
(WandBLogger.__init__ in lerobot/common/wandb_utils.py). wandb 0.29.0, the version of requirements.txt, no longer has
Run.get_url (its replacement is the Run.url property), so every lerobot_train run with --wandb.enable=true stops with
AttributeError before its first step, in online, offline and disabled mode alike. The released prior runs had wandb
0.27.2, which still had it. The RL code never uses LeRobot's wandb logger, so the campaign never met this.

WHAT IT DOES. If wandb's Run class (the parent of the run object of every mode) has no get_url, it gets one that
returns Run.url; nothing else is changed. Then lerobot_train's own main() runs as `python -m` would run it: the same
argument parsing, plugin registration, training loop and checkpoints. The address is only printed, so the training
is unaffected. Where wandb still has get_url, or is not installed, this is exactly python -m lerobot.scripts.lerobot_train.
DataLoader workers start with "spawn" and re-import this file as __mp_main__, which only defines functions.
"""
from __future__ import annotations


def add_get_url() -> bool:
    """Give wandb's Run a get_url() returning its url where it has none; True if it was added."""
    try:
        from wandb.sdk.wandb_run import Run
    except ImportError:
        return False
    if hasattr(Run, "get_url"):
        return False
    Run.get_url = lambda self: self.url
    return True


def main() -> None:
    add_get_url()
    from lerobot.scripts import lerobot_train
    lerobot_train.main()


if __name__ == "__main__":
    main()
