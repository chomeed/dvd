"""Output hashes of the V ensemble, the mqattn readout and the DINO world model under ONE source tree, on CPU.

    python _parity_worker.py <tree root that holds online_rl_mimicgen/> <out.json> <CKPT_ROOT>

test_cpu_parity.py runs this once on the research tree and once on this repository and compares the two files. The
script is deliberately tree-agnostic: the only name it has to try twice is the world-model wiring module, which was
renamed. Inputs are seeded random tensors; everything runs in fp32 on one thread, so equal code gives equal bytes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

tree, out_path, ckpt_root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
os.environ["PIXEL_IMG"] = "224"
sys.path.insert(0, str(tree))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import online_rl_mimicgen  # noqa: E402

origin = Path(online_rl_mimicgen.__file__).resolve()
assert tree in origin.parents, f"online_rl_mimicgen resolved to {origin}, not below {tree}"

from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool  # noqa: E402
from online_rl_mimicgen.value_model import ValueEnsemble  # noqa: E402

for wiring, flag in (("dino_dvd", "dvd_checkpoint"), ("dino_svg", "svg_checkpoint")):
    if (origin.parent / f"{wiring}.py").exists():
        dino = importlib.import_module(f"online_rl_mimicgen.{wiring}")
        break
else:
    raise SystemExit(f"no world-model wiring module below {origin.parent}")


def sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(tensor.detach().float().cpu().numpy()).tobytes()).hexdigest()


def layout(state: dict) -> str:
    return hashlib.sha256("|".join(f"{k}:{tuple(v.shape)}" for k, v in state.items()).encode()).hexdigest()


result = {"package": str(origin), "wiring": wiring, "torch": torch.__version__, "value": {}, "wm": {}}
gen = torch.Generator().manual_seed(0)
feats = torch.randn(4, 2, 196, 384, generator=gen)
proprio = torch.randn(4, 64, generator=gen)

for pool_file in sorted((ckpt_root / "value").glob("*/mqattn_pool.pt")):
    root = pool_file.parent
    config = json.loads((root / "config.json").read_text())
    stats = json.loads((root / "state_scaler.json").read_text())
    scaler = SimpleNamespace(low=np.asarray(stats["q01"], dtype=np.float32), high=np.asarray(stats["q99"], dtype=np.float32))
    variant = sorted({p.name.rsplit("_seed", 1)[0] for p in (root / "weights").glob("*_seed*.pt")})[0]
    with torch.no_grad():
        pool = MultiQueryAttnPool.from_blob(torch.load(pool_file, map_location="cpu", weights_only=False), device="cpu")
        pooled = pool(feats)
        ensemble = ValueEnsemble(root, variant, scaler, state_dim=int(config["state_dim"]), device="cpu")
        x = torch.cat([pooled, proprio[:, :ensemble.input_dim - pooled.shape[-1]]], dim=-1)
        members = torch.stack([net(x) for net in ensemble.nets])
        value = ensemble(x)
    result["value"][root.name] = {
        "variant": variant, "nets": len(ensemble.nets), "input_dim": int(ensemble.input_dim),
        "pool_layout": layout(pool.state_dict()), "pool_out": sha(pooled),
        "nets_layout": layout(ensemble.state_dict()), "nets_out": sha(members), "ensemble_out": sha(value),
        "ensemble_mean": float(value.mean())}

for ckpt in sorted((ckpt_root / "wm").glob("*/*.pt")):
    meta_file = ckpt.with_suffix(".json")
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    adapter = dino.build(str(ckpt), device="cpu", bf16=False, action_dim=7, arch="", state_head=bool(meta.get("state_head", True)),
                         **{flag: False})
    frames = torch.randint(0, 256, (2, 2, 224, 224, 3), generator=gen, dtype=torch.uint8)
    raw_state = torch.randn(2, 23, generator=gen)
    action = torch.randn(2, 56, generator=gen) * 0.3
    with torch.no_grad():
        previous, current = dino.encode_frames(adapter, frames), dino.encode_frames(adapter, frames.flip(0))
        dino.set_context(adapter, previous, current)
        predicted = adapter(raw_state, action)
    state_pred = adapter.last_state_pred
    result["wm"][f"{ckpt.parent.name}/{ckpt.name}"] = {
        "layout": layout(adapter.state_dict()), "encoded": sha(current), "predicted": sha(predicted),
        "predicted_mean": float(predicted.mean()), "state_pred": sha(state_pred) if state_pred is not None else None}

out_path.write_text(json.dumps(result, indent=1) + "\n")
print(f"[parity-worker] {origin.parent}: {len(result['value'])} V directories, {len(result['wm'])} world models", flush=True)
