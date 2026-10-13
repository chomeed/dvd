"""Shared helpers of tests/pipeline. The tests are plain scripts (python tests/pipeline/test_x.py) that print what
they measured and assert; each check is a test_* function, so a test runner can collect them too. CPU only unless a
test says otherwise, and nothing is written outside a temporary directory.

Environment: source scripts/env.sh first (PIXEL_IMG=224 and the asset roots). Tests that need assets look for them
under those roots and print SKIP when they are absent, so the suite passes on a fresh clone:

  CKPT_ROOT    wm/<dir>/wm_dinov2s14_<tag>.pt, value/<V dir>/, rm_scores/<name>/predictions.parquet
  CACHE_ROOT   wm_frames/<task>_meta.npz
  DATA_ROOT    mimicgen_<task>_224x224/meta/stats.json
  LEGACY_RUNS  os.pathsep-separated directories of DICE-RL run directories written by the pre-rename research code
               (<dir>/<run>/checkpoints/*.pt); a run directory itself is accepted too
"""
from __future__ import annotations

import os
import sys
import traceback
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "rl"
TASKS = ("square_d0", "square_d1", "threading_d0", "hammer_cleanup_d1")
BASE_TAG = {"square_d0": "ph_conv_150k", "square_d1": "ph_conv_150k", "threading_d0": "ph_conv_168k",
            "hammer_cleanup_d1": "ph_conv_217k"}

os.environ.setdefault("PIXEL_IMG", "224")           # scripts/env.sh sets it; the offline modules assert it


class Skip(unittest.SkipTest):
    """Raised by a test whose optional input is absent; pytest reports unittest.SkipTest as a skip too."""


def skip(reason: str):
    raise Skip(reason)


def use_repo(*subdirs: str) -> None:
    """Put this repository's rl/, its env package and the given sibling directories first on sys.path and prove that
    online_rl_mimicgen is THIS repository's package, not another checkout that happens to be on PYTHONPATH."""
    for sub in (*subdirs, "envs/lerobot_env_mimicgen/src", "rl"):
        p = str(REPO / sub)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import online_rl_mimicgen
    origin = Path(online_rl_mimicgen.__file__).resolve()
    assert RL.resolve() in origin.parents, f"online_rl_mimicgen resolved to {origin}, not below {RL}"


def root(name: str) -> Path:
    """An asset root as scripts/env.sh defines it (default: <repo>/assets/...)."""
    assets = Path(os.environ.get("ASSET_ROOT", REPO / "assets"))
    default = {"DATA_ROOT": assets / "datasets", "CKPT_ROOT": assets / "checkpoints", "CACHE_ROOT": assets / "cache"}[name]
    return Path(os.environ.get(name, default))


def wm_file(task: str, tag: str | None = None) -> Path:
    tag = tag or BASE_TAG[task]
    return root("CKPT_ROOT") / "wm" / f"mimicgen_{task}_224x224_dinowm_{tag}" / f"wm_dinov2s14_{tag}.pt"


def legacy_checkpoints() -> list[Path]:
    """The last checkpoint of every run directory named by LEGACY_RUNS; [] when the variable is unset. A run that
    has only its step-0 checkpoint is left out: that file is written before training and holds no trained module."""
    out = []
    for entry in filter(None, os.environ.get("LEGACY_RUNS", "").split(os.pathsep)):
        entry = Path(entry)
        runs = [entry] if (entry / "checkpoints").is_dir() else sorted(p.parent for p in entry.glob("*/checkpoints"))
        for run in runs:
            ckpts = [c for c in sorted(run.glob("checkpoints/[0-9]*.pt")) if c.stem.isdigit() and int(c.stem) > 0]
            if ckpts:
                out.append(ckpts[-1])
    return out


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
    print("ALL OK", flush=True)
