#!/usr/bin/env python3
"""Build the square_d1 prior every square_d1 cell loads ("statefix") from the published, un-fixed checkpoint.

    python scripts/make_square_d1_statefix.py                      # default --src and --out under $CKPT_ROOT/priors
    python scripts/make_square_d1_statefix.py --src DIR --out DIR [--weights copy|hardlink|symlink] [--force]
    python scripts/make_square_d1_statefix.py --verify DIR         # check an existing fixed prior, write nothing

WHY. The published square_d1 prior (Hub repository mimicgen_square_d1_flow_matching_prior_45k) was trained with a
patched policy that reads only the first 9 dimensions of the 26-D dataset state (eef position 3, quaternion 4, gripper
2): its config.json says observation.state shape [26] together with "n_state_dims": 9. The in-tree LeRobot
multi_task_dit policy this repository runs has no such field; it would build the network for a 26-D state, which does
not fit the weights. The reference runs used a copy with three files edited and the weights untouched; that copy is
on no Hub repository, so this script reproduces it. (The other three priors were published already fixed.)

WHAT IS EDITED (the recipe of the reference machine, step for step):
  config.json                              input_features["observation.state"].shape [26] -> [9]; "n_state_dims" removed
  policy_preprocessor.json                 every observation.state feature of type STATE: shape [26] -> [9]
  policy_preprocessor_step_*_normalizer_processor.safetensors
                                           every observation.state.<stat> tensor of length 26 cut to its first 9 entries
  everything else                          copied unchanged (model.safetensors, the postprocessor, train_config.json, ...)
Both JSON files are rewritten with indent 4 and no trailing newline, as the reference copy was.

  --src      the downloaded un-fixed prior's pretrained_model directory
             [$CKPT_ROOT/priors/square_d1_45k/checkpoints/045000/pretrained_model, where scripts/fetch_assets.py puts it]
  --out      [$CKPT_ROOT/priors/square_d1_45k_statefix/checkpoints/045000/pretrained_model, the prior configs/tasks.sh names]
  --weights  how model.safetensors (996 MB, unchanged) gets into --out: copy [default], hardlink (same file system; no
             extra disk), symlink (absolute link to the source file)
  --force    write into an --out that already holds files (they are replaced)
  --verify   only run the checks below on an existing directory

CHECKS, run on what is on disk at the end (exit 1 if one fails):
  - config.json: observation.state shape [9], no n_state_dims; policy_preprocessor.json: no observation.state of another size
  - normalizer: every observation.state.<stat> has 9 entries (count: 1) and equals the first 9 of the source's
  - weights: the conditioning width of the first transformer block equals
        timestep_embed_dim + n_obs_steps * (vision width * cameras + 9 + hidden_dim)
    read from the safetensors header (no weight is loaded): the checkpoint really was trained on a 9-D state
  - SHA-256 of the three edited files and of the unchanged weights against the reference machine's copy. A mismatch of
    an edited file with all other checks passing is reported as a warning (a different safetensors or json version
    may serialise the same content differently); --strict makes it an error.

Needs only the standard library, torch and safetensors (the main environment); no GPU, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_KEY = "observation.state"
FULL_DIM, STATE_DIM = 26, 9
WEIGHTS = "model.safetensors"
# SHA-256 of the fixed prior on the reference machine (2026-10-12). The first three are the edited files; the weights
# are byte-identical to the published checkpoint.
REFERENCE_SHA256 = {
    "config.json": "5a2542ad81d57f666217cd71fb4d27fddf90bceed645837189777f34291876cc",
    "policy_preprocessor.json": "8051f76ed475aa7f003a27b6422fb3f5c13c5941a9f9752b17b226f5fd63efb9",
    "policy_preprocessor_step_4_normalizer_processor.safetensors":
        "b0ae38902ed7e463eec8a40b090a6e7f12f0c96078482fc024bddd45699d5a7b",
    WEIGHTS: "28143356f81cb125d5a15887a2123ca89e02ee01ba0d04f972b52510fea3f129",
}
EDITED = tuple(name for name in REFERENCE_SHA256 if name != WEIGHTS)


def ckpt_root() -> Path:
    assets = Path(os.environ.get("ASSET_ROOT", REPO / "assets"))
    return Path(os.environ.get("CKPT_ROOT", assets / "checkpoints"))


def default_src() -> Path:
    return ckpt_root() / "priors" / "square_d1_45k" / "checkpoints" / "045000" / "pretrained_model"


def default_out() -> Path:
    return ckpt_root() / "priors" / "square_d1_45k_statefix" / "checkpoints" / "045000" / "pretrained_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict:
    """The JSON header of a safetensors file: {tensor name: {dtype, shape, data_offsets}}; no tensor data is read."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def state_features(node, found: list) -> list:
    """Every {"type": "STATE", "shape": [...]} dict stored under the key observation.state, anywhere in a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == STATE_KEY and isinstance(value, dict) and value.get("type") == "STATE" and "shape" in value:
                found.append(value)
            else:
                state_features(value, found)
    elif isinstance(node, list):
        for value in node:
            state_features(value, found)
    return found


def normalizer_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))


def source_kind(src: Path) -> str:
    """'unfixed' (shape [26] + n_state_dims 9), 'fixed' (shape [9], no n_state_dims) or an error."""
    config = json.loads((src / "config.json").read_text())
    shape = config["input_features"][STATE_KEY]["shape"]
    if shape == [FULL_DIM] and config.get("n_state_dims") == STATE_DIM:
        return "unfixed"
    if shape == [STATE_DIM] and "n_state_dims" not in config:
        return "fixed"
    raise SystemExit(f"{src}/config.json: observation.state shape {shape} with n_state_dims={config.get('n_state_dims')!r} is "
                     f"neither the published square_d1 prior ([{FULL_DIM}] with n_state_dims {STATE_DIM}) nor a fixed one")


def place_weights(src: Path, dst: Path, how: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if how == "symlink":
        dst.symlink_to(src.resolve())
    elif how == "hardlink":
        try:
            os.link(src.resolve(), dst)
        except OSError as error:
            raise SystemExit(f"--weights hardlink: cannot link {src} to {dst} ({error}); use --weights copy")
    else:
        shutil.copyfile(src, dst)


def build(src: Path, out: Path, weights: str, force: bool) -> None:
    from safetensors.torch import load_file, save_file

    if not (src / "config.json").is_file() or not (src / WEIGHTS).exists():
        raise SystemExit(f"no prior at {src} (need config.json and {WEIGHTS}). Download the published checkpoint first:\n"
                         f"  python scripts/fetch_assets.py --tier eval --task square_d1\n"
                         f"or pass --src <its pretrained_model directory>")
    if src.resolve() == out.resolve():
        raise SystemExit("--src and --out are the same directory; the source is never edited in place")
    kind = source_kind(src)
    if kind == "fixed":
        raise SystemExit(f"{src} is already a fixed prior (observation.state [{STATE_DIM}], no n_state_dims); nothing to do. "
                         f"--verify {src} checks it")
    if not normalizer_files(src):
        raise SystemExit(f"{src} holds no policy_preprocessor_step_*_normalizer_processor.safetensors")
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"{out} already holds files; --verify {out} checks them, --force replaces them")
    out.mkdir(parents=True, exist_ok=True)

    place_weights(src / WEIGHTS, out / WEIGHTS, weights)            # first: the only step that can fail for lack of space
    copied = 1
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        if relative.parts[0] == ".cache" or path.is_dir() or relative.as_posix() == WEIGHTS:
            continue                                                # .cache: download bookkeeping of huggingface_hub
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copyfile(path, target)                               # follows links: the Hub cache stores files as links
        copied += 1

    config = json.loads((out / "config.json").read_text())
    config["input_features"][STATE_KEY]["shape"] = [STATE_DIM]
    config.pop("n_state_dims")
    (out / "config.json").write_text(json.dumps(config, indent=4))

    preprocessor = json.loads((out / "policy_preprocessor.json").read_text())
    features = [f for f in state_features(preprocessor, []) if f["shape"] == [FULL_DIM]]
    if not features:
        raise SystemExit(f"{src}/policy_preprocessor.json names no observation.state feature of shape [{FULL_DIM}]")
    for feature in features:
        feature["shape"] = [STATE_DIM]
    (out / "policy_preprocessor.json").write_text(json.dumps(preprocessor, indent=4))

    cut = 0
    for path in normalizer_files(out):
        tensors = load_file(path)
        for key in list(tensors):
            if key.startswith(STATE_KEY + ".") and tensors[key].shape[-1] == FULL_DIM:
                tensors[key] = tensors[key][..., :STATE_DIM].contiguous()
                cut += 1
        save_file(tensors, path)
    if not cut:
        raise SystemExit(f"no {STATE_KEY}.* statistic of length {FULL_DIM} in the normalizer of {src}")
    print(f"wrote {out}\n  {copied} files from {src} (weights: {weights}); config.json and policy_preprocessor.json "
          f"({len(features)} feature entr{'y' if len(features) == 1 else 'ies'}) set to a {STATE_DIM}-D state; "
          f"{cut} normalizer statistics cut to {STATE_DIM}")


def verify(out: Path, src: Path | None, strict: bool) -> bool:
    """The checks of the module docstring on what is on disk. Prints one line per check; True when all pass."""
    from safetensors.torch import load_file

    ok = True

    def check(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  {'ok  ' if passed else 'FAIL'} {text}")

    config = json.loads((out / "config.json").read_text())
    shape = config["input_features"][STATE_KEY]["shape"]
    check(shape == [STATE_DIM] and "n_state_dims" not in config,
          f"config.json: observation.state shape {shape}, n_state_dims {'absent' if 'n_state_dims' not in config else config['n_state_dims']}")
    shapes = [f["shape"] for f in state_features(json.loads((out / "policy_preprocessor.json").read_text()), [])]
    check(bool(shapes) and all(s == [STATE_DIM] for s in shapes), f"policy_preprocessor.json: observation.state shapes {shapes}")

    files = normalizer_files(out)
    check(bool(files), f"{len(files)} normalizer file(s)")
    for path in files:
        tensors = load_file(path)
        stats = {key: tuple(value.shape) for key, value in tensors.items() if key.startswith(STATE_KEY + ".")}
        good = bool(stats) and all(s == ((1,) if key.endswith(".count") else (STATE_DIM,)) for key, s in stats.items())
        check(good, f"{path.name}: {len(stats)} observation.state statistics, lengths {sorted(set(s[-1] for s in stats.values()))}")
        if src is not None and (src / path.name).exists():
            original = load_file(src / path.name)
            same = set(original) == set(tensors) and all(
                bool((tensors[key] == (original[key][..., :STATE_DIM] if key.startswith(STATE_KEY + ".") and
                                       original[key].shape[-1] == FULL_DIM else original[key])).all()) for key in tensors)
            check(same, f"{path.name}: all {len(tensors)} tensors equal the source's (observation.state: its first {STATE_DIM} entries)")

    header = safetensors_header(out / WEIGHTS)
    cond = header.get("noise_predictor.transformer_blocks.0.adaLN_modulation.1.weight", {}).get("shape")
    vision = header.get("observation_encoder.vision_encoder.model.vision_model.post_layernorm.weight", {}).get("shape")
    cameras = sum(1 for feature in config["input_features"].values() if feature.get("type") == "VISUAL")
    if cond is None or vision is None:
        check(False, f"{WEIGHTS}: conditioning or vision tensor not found in the header; cannot confirm the state width")
    else:
        expected = config["timestep_embed_dim"] + config["n_obs_steps"] * (vision[0] * cameras + STATE_DIM + config["hidden_dim"])
        check(cond[1] == expected,
              f"{WEIGHTS}: conditioning width {cond[1]} = {config['timestep_embed_dim']} + {config['n_obs_steps']} x "
              f"({vision[0]} x {cameras} cameras + {STATE_DIM} state + {config['hidden_dim']} text) = {expected}")

    for name in (*EDITED, WEIGHTS):
        path = out / name
        if not path.exists():
            check(False, f"{name}: missing")
            continue
        same = sha256(path) == REFERENCE_SHA256[name]
        if same or name == WEIGHTS or strict or not ok:
            check(same, f"{name}: SHA-256 {'equals' if same else 'DIFFERS from'} the reference machine's file")
        else:       # every content check above passed, so only the serialisation of the same content differs
            print(f"  warn {name}: SHA-256 differs from the reference machine's file although the content checks pass "
                  f"(another safetensors / json version may write the same content differently; --strict makes "
                  f"this an error)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=None, help="un-fixed prior: its pretrained_model directory")
    parser.add_argument("--out", type=Path, default=None, help="fixed prior to write: a pretrained_model directory")
    parser.add_argument("--weights", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--force", action="store_true", help="replace files in an existing --out")
    parser.add_argument("--strict", action="store_true", help="a SHA-256 mismatch of an edited file is an error")
    parser.add_argument("--verify", type=Path, default=None, metavar="DIR", help="check an existing fixed prior; write nothing")
    args = parser.parse_args()

    if args.verify is not None:
        if not (args.verify / "config.json").is_file():
            raise SystemExit(f"no config.json in {args.verify}")
        print(f"checking {args.verify}")
        good = verify(args.verify, args.src if args.src and (args.src / "config.json").is_file() else None, args.strict)
    else:
        src, out = args.src or default_src(), args.out or default_out()
        build(src, out, args.weights, args.force)
        good = verify(out, src, args.strict)
    if not good:
        raise SystemExit("FAILED: this directory is not the fixed square_d1 prior of the reference runs")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
