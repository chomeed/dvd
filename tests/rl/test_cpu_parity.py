"""Numeric parity with the research tree on production weights, CPU only.

    LIVE_TREE=<research tree> CKPT_ROOT=<checkpoint root> python tests/rl/test_cpu_parity.py

For every V directory $CKPT_ROOT/value/*/ (mqattn_pool.pt + weights/) and every world model $CKPT_ROOT/wm/*/*.pt,
_parity_worker.py computes SHA-256 hashes of: the mqattn readout output, each V net's output and the ensemble's
output on seeded features; the DINOv2 encoding of seeded frames, the world model's predicted next-frame features
and its proprio-head prediction. It runs once with LIVE_TREE's modules and once with this repository's, in separate
processes (both packages are called online_rl_mimicgen). Equal hashes = bit-identical fp32 outputs.

Needs HF_HOME to hold facebook/dinov2-small (or network access); no GPU, no dataset, no simulator. Skips when
LIVE_TREE or CKPT_ROOT is unset or holds no weights. This does not cover GPU / bf16 / torch.compile paths.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import _util

WORKER = Path(__file__).with_name("_parity_worker.py")
_cache = {}


def outputs() -> tuple[dict, dict]:
    if "pair" in _cache:
        return _cache["pair"]
    live = _util.live_tree()
    root = os.environ.get("CKPT_ROOT", "")
    if not root:
        _util.skip("CKPT_ROOT is not set (production V directories and world models)")
    root = Path(root)
    n_value, n_wm = len(list((root / "value").glob("*/mqattn_pool.pt"))), len(list((root / "wm").glob("*/*.pt")))
    if not n_value and not n_wm:
        _util.skip(f"no value/*/mqattn_pool.pt or wm/*/*.pt below CKPT_ROOT={root}")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    pair = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, tree in (("live", live), ("new", _util.RL)):
            out = Path(tmp) / f"{name}.json"
            done = subprocess.run([sys.executable, str(WORKER), str(tree), str(out), str(root)], env=env, cwd=tmp,
                                  capture_output=True, text=True, timeout=3600)
            if done.returncode != 0:
                raise AssertionError(f"parity worker failed on the {name} tree ({tree}):\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
            pair.append(json.loads(out.read_text()))
    live_out, new_out = pair
    assert Path(live_out["package"]).parent != Path(new_out["package"]).parent, "both workers imported the same package"
    assert _util.PACKAGE.resolve() == Path(new_out["package"]).resolve().parent, new_out["package"]
    print(f"[parity] live {live_out['package']} ({live_out['wiring']})\n[parity] new  {new_out['package']} ({new_out['wiring']})"
          f"\n[parity] torch {new_out['torch']}, {n_value} V directories, {n_wm} world models under {root}")
    _cache["pair"] = (live_out, new_out)
    return _cache["pair"]


def compare(kind: str, fields: tuple[str, ...]) -> None:
    live, new = outputs()
    if not new[kind]:
        _util.skip(f"no {kind} weights below CKPT_ROOT")
    assert sorted(live[kind]) == sorted(new[kind]), "the two trees saw different weights"
    bad = []
    for name in sorted(new[kind]):
        a, b = live[kind][name], new[kind][name]
        equal = all(a[f] == b[f] for f in fields)
        shown = "  ".join(f"{f}={str(b[f])[:12]}" for f in fields)
        print(f"  {'==' if equal else '!='} {name}: {shown}" + ("" if equal else f"   LIVE: " + "  ".join(f"{f}={str(a[f])[:12]}" for f in fields)))
        if not equal:
            bad.append(name)
    assert not bad, f"{kind} outputs differ between the trees for: {bad}"


def test_mqattn_readout_parity():
    """MultiQueryAttnPool.from_blob(<V dir>/mqattn_pool.pt)(features): same parameters, same output bytes."""
    compare("value", ("pool_layout", "pool_out"))


def test_value_ensemble_parity():
    """ValueEnsemble(<V dir>): every member net and the ensemble mean, same output bytes."""
    compare("value", ("nets", "nets_layout", "nets_out", "ensemble_out", "ensemble_mean"))


def test_world_model_parity():
    """dino build + encode_frames + forward: DINOv2 features, predicted features and proprio head, same output bytes."""
    compare("wm", ("layout", "encoded", "predicted", "state_pred", "predicted_mean"))


if __name__ == "__main__":
    _util.main(globals())
