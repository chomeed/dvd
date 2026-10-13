"""Shared helpers of tests/rl. The tests are plain scripts (python tests/rl/test_x.py) that print what they measured
and assert; they also collect under pytest. Nothing here touches a GPU or writes outside a temporary directory.

Optional inputs, all through environment variables (a test whose input is missing prints SKIP and passes):

  LIVE_TREE     root of the pre-rename research tree (the directory holding online_rl_mimicgen/ and 0811/)
  LEGACY_RUNS   os.pathsep-separated directories holding run directories written by that tree
                (<dir>/<run>/config.json, checkpoints/); a run directory itself is accepted too
  CKPT_ROOT     scripts/env.sh checkpoint root: value/<V dir>/ and wm/<dir>/<file>.pt are the production weights
  DSRL_ROOT     DSRL repository with its stable-baselines3 fork (needed to import the DSRL modules)
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "rl"
PACKAGE = RL / "online_rl_mimicgen"


class Skip(unittest.SkipTest):
    """Raised by a test whose optional input is absent; pytest reports unittest.SkipTest as a skip too."""


def skip(reason: str):
    raise Skip(reason)


def use_new_tree():
    """Put rl/ first on sys.path and prove that online_rl_mimicgen is THIS repository's package."""
    if str(RL) not in sys.path[:1]:
        sys.path.insert(0, str(RL))
    import online_rl_mimicgen
    origin = Path(online_rl_mimicgen.__file__).resolve()
    assert PACKAGE.resolve() in origin.parents, f"online_rl_mimicgen resolved to {origin}, not to {PACKAGE}"
    return online_rl_mimicgen


def live_tree() -> Path:
    root = os.environ.get("LIVE_TREE", "")
    if not root:
        skip("LIVE_TREE is not set (root of the pre-rename research tree)")
    root = Path(root)
    if not (root / "online_rl_mimicgen" / "train_online.py").exists():
        skip(f"LIVE_TREE={root} holds no online_rl_mimicgen/train_online.py")
    return root


def legacy_run_dirs() -> list[Path]:
    """Every run directory (has a config.json) named by LEGACY_RUNS, sorted; [] when the variable is unset."""
    runs = []
    for entry in filter(None, os.environ.get("LEGACY_RUNS", "").split(os.pathsep)):
        entry = Path(entry)
        if (entry / "config.json").exists():
            runs.append(entry)
        elif entry.is_dir():
            runs += [p.parent for p in entry.glob("*/config.json")]
    return sorted(set(runs))


def read_config(run: Path) -> dict | None:
    try:
        config = json.loads((run / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def main(namespace: dict) -> None:
    """Run every test_* function of a module in definition order; exit 1 if any failed."""
    failed = 0
    for name, fn in list(namespace.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Skip as reason:
            print(f"SKIP {name}: {reason}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"FAIL {name}", flush=True)
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
