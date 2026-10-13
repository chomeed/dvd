"""Compatibility with artifacts written by the pre-rename research code. THE ONE PLACE that knows the old names.

The method is called DVD in this repository and was called SVG in the research tree that trained the reference
runs. That tree is this code with `svg` spelled where `dvd` is spelled here, so everything it wrote to disk carries
the old spelling:

  config.json / checkpoint["config"]   53 TrainConfig fields (svg_weight, bon_svg_until, qsvg_combine, ...), 15 DSRL
                                       argument names, and the option VALUE of five fields (bon_source "svg",
                                       bc_filter_source "svg", actor_gradient_source_normal/_critical "svg1",
                                       svg_gate_target "svg")
  DICE-RL checkpoint <step>.pt         extra["svg_dynamics"] (fine-tuned world model), extra["opt"]["svg"]
  DSRL run                             method tag "dsrl_na_svg_dino", sidecar <step>_svg.pt next to <step>.zip
  any config.json                      absolute paths of the training machine (prior, V directory, world model)

No module parameter is named after the method, so state dicts load unchanged once their containers are renamed.
Not handled here because nothing loads them: metric names in metrics.jsonl / wandb (svg_value, svg_scale, ... are
dvd_value, dvd_scale, ... in new runs), and the class attributes pickled into a DSRL .zip (they reference a module
name that no longer exists, which is why eval_dsrl_ckpt reads the weights with load_data=False).

The tables below are the names that differ between the two trees (TrainConfig fields, argparse dests and choices,
checkpoint keys). tests/rl/test_legacy.py re-derives them from both source trees and checks them against every real
config it is pointed at; do not extend them by hand. Readers call

  upgrade_config(d)           old or new config dict -> new names; idempotent; raises on an old name it does not know
  upgrade_checkpoint(p)       the same for a DICE-RL checkpoint payload; load_checkpoint(path) = torch.load + upgrade
  sidecar(zip)                <step>_dvd.pt, or the legacy <step>_svg.pt
  remap_asset_path(p, kind)   a stored absolute asset path -> the scripts/env.sh roots when it does not exist here
  remap_config_assets(d, ..)  the same for the path fields of a config
  refuse_unconverted_checkpoint(payload, path, flag)   raise if a DICE-RL payload still has the old container names
  refuse_unconverted_resume_state(state, path)         the same for the resume bundle of baselines/dsrl_mimicgen

The readers of finished runs upgrade on the fly: common.load_agent_checkpoint, eval/eval_bon16.py and
baselines/eval_dsrl_ckpt. The TRAINERS do not: they look for the new names only, and a container they do not find is a
container they do not load. An unconverted old DICE-RL checkpoint handed to train_online --resume would therefore
continue from the PRETRAINED world model with a fresh optimiser and say nothing, so the trainer's loaders that read a
renamed container (--resume, --dvd-dynamics-init, and --resume of baselines/dsrl_mimicgen) call the two refuse_*
functions and stop with the command to run. Convert first:

  python -m online_rl_mimicgen.legacy checkpoint <old.pt> <new.pt>
  python -m online_rl_mimicgen.legacy config <config.json>            # prints the upgraded config

--value-init, --encoder-init, --init-actor-from and --init-critic-from read containers whose names never changed
(extra.value_fn, extra.encoder, actor, critics), so they accept old and converted checkpoints alike.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

OLD, NEW = "svg", "dvd"

_TRAIN_CONFIG_KEYS = (
    "bon_svg_until", "qsvg_combine", "qsvg_normalise", "svg_actor_pool", "svg_actor_pool_log_every",
    "svg_actor_pool_obs", "svg_checkpoint", "svg_chunks", "svg_dino_lr", "svg_dynamics", "svg_dynamics_init",
    "svg_dynamics_lr", "svg_dynamics_scratch", "svg_encoder_learned", "svg_finetune_dino_blocks",
    "svg_finetune_dynamics", "svg_finetune_last_k", "svg_finetune_seghead", "svg_fit_online_only",
    "svg_fit_probe_every", "svg_fit_recent_frac", "svg_fit_recent_steps", "svg_fit_rows", "svg_fit_sampler",
    "svg_fixed_scale", "svg_follow", "svg_gate_mm", "svg_gate_source", "svg_gate_target", "svg_held_mm",
    "svg_latent_wm", "svg_map_shift", "svg_mode", "svg_on_encoder", "svg_phase_gate", "svg_pool_actor_step",
    "svg_proprio_source", "svg_residual_anchor", "svg_row_chunks", "svg_rows", "svg_rows_grouped",
    "svg_rows_late", "svg_rows_switch", "svg_seghead_ema", "svg_seghead_lr", "svg_state_loss_weight",
    "svg_static_grad", "svg_step2_action", "svg_value_aware_weight", "svg_weight", "svg_weight_late",
    "svg_weight_ramp_end", "svg_weight_ramp_start",
)
_DSRL_ARG_KEYS = (
    "no_svg_finetune_dynamics", "svg_actor_steps", "svg_diag_draws", "svg_diag_every", "svg_dynamics",
    "svg_dynamics_lr", "svg_fit_rows", "svg_fit_utd", "svg_fixed_scale", "svg_grad_checkpoint",
    "svg_proprio_source", "svg_rows", "svg_state_loss_weight", "svg_static_grad", "svg_weight",
)

TRAIN_CONFIG_RENAMES = {key: key.replace(OLD, NEW) for key in _TRAIN_CONFIG_KEYS}
DSRL_ARG_RENAMES = {key: key.replace(OLD, NEW) for key in _DSRL_ARG_KEYS}
KEY_RENAMES = {**TRAIN_CONFIG_RENAMES, **DSRL_ARG_RENAMES}

VALUE_RENAMES = {
    "bon_source": {"svg": "dvd"},
    "bc_filter_source": {"svg": "dvd"},
    "actor_gradient_source_normal": {"svg1": "dvd1"},
    "actor_gradient_source_critical": {"svg1": "dvd1"},
    "dvd_gate_target": {"svg": "dvd"},
    "method": {"dsrl_na_svg_dino": "dsrl_na_dvd_dino"},
}
_OLD_VALUES = frozenset(old for table in VALUE_RENAMES.values() for old in table)

EXTRA_RENAMES = {"svg_dynamics": "dvd_dynamics"}
OPTIMIZER_RENAMES = {"svg": "dvd"}

SIDECAR_SUFFIX, LEGACY_SIDECAR_SUFFIX = "_dvd.pt", "_svg.pt"
# <run>/resume/state.json of baselines/dsrl_mimicgen (state-space DSRL; not a campaign arm)
RESUME_STATE_RENAMES = {"svg_checkpoint": "dvd_checkpoint"}

CONVERT_CHECKPOINT = "python -m online_rl_mimicgen.legacy checkpoint <old.pt> <new.pt>"


def _rename_keys(mapping: Mapping[str, Any], table: Mapping[str, str], what: str) -> dict[str, Any]:
    out = {}
    for key, value in mapping.items():
        new = table.get(key, key)
        if new != key and new in mapping:
            raise ValueError(f"{what} holds both the legacy key {key!r} and its new name {new!r}; refusing to pick one")
        out[new] = value
    return out


def _refuse_old_keys(mapping: Mapping[str, Any], what: str) -> None:
    left = sorted(key for key in mapping if OLD in str(key).lower())
    if left:
        raise ValueError(f"{what}: unknown legacy key(s) {left}. They are not in online_rl_mimicgen/legacy.py, so their "
                         f"meaning in this code is not established; refusing to drop or guess them")


def is_legacy(config: Mapping[str, Any]) -> bool:
    """True if the config dict was written by the pre-rename code (any old key or old option value)."""
    return any(key in KEY_RENAMES for key in config) or any(
        isinstance(value, str) and value in VALUE_RENAMES.get(KEY_RENAMES.get(key, key), ())
        for key, value in config.items())


def upgrade_config(config: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """A run's config (DICE-RL config.json / checkpoint["config"], or a DSRL config.json) with this repository's names.

    Keys and the five option values listed in the module docstring are renamed; every other value, including paths
    and run names that merely contain "svg", is returned untouched. A config that is already new comes back equal, so
    readers call this unconditionally. strict (default) raises if a key still contains "svg" or any value is still
    one of the old option values: such a leftover is a field this table does not know, and silently dropping or
    passing it on would change what the run is taken to be. strict=False returns those leftovers unchanged.
    """
    out = _rename_keys(config, KEY_RENAMES, "config")
    for key, table in VALUE_RENAMES.items():
        value = out.get(key)
        if isinstance(value, str) and value in table:
            out[key] = table[value]
    if strict:
        _refuse_old_keys(out, "config")
        left = sorted(f"{key}={value!r}" for key, value in out.items() if isinstance(value, str) and value in _OLD_VALUES)
        if left:
            raise ValueError(f"config: legacy option value(s) in field(s) this table does not know: {left}")
    return out


def upgrade_checkpoint(payload: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """A DICE-RL checkpoint payload (train_online's <step>.pt) with this repository's names.

    Renames payload["config"] (upgrade_config), extra["svg_dynamics"] and extra["opt"]["svg"]. The containers are
    copied shallowly and the tensors are shared; idempotent.
    """
    out = dict(payload)
    if isinstance(out.get("config"), Mapping):
        out["config"] = upgrade_config(out["config"], strict=strict)
    if isinstance(out.get("extra"), Mapping):
        extra = out["extra"] = _rename_keys(out["extra"], EXTRA_RENAMES, "checkpoint extra")
        if isinstance(extra.get("opt"), Mapping):
            extra["opt"] = _rename_keys(extra["opt"], OPTIMIZER_RENAMES, "checkpoint extra['opt']")
        if strict:
            _refuse_old_keys(extra, "checkpoint extra")
            _refuse_old_keys(extra.get("opt") or {}, "checkpoint extra['opt']")
    if strict:
        _refuse_old_keys(out, "checkpoint")
    return out


def legacy_checkpoint_keys(payload: Mapping[str, Any]) -> list[str]:
    """The old container names a DICE-RL checkpoint payload still carries, e.g. ['extra.svg_dynamics', 'extra.opt.svg']."""
    extra = payload.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    opt = extra.get("opt")
    opt = opt if isinstance(opt, Mapping) else {}
    return ([f"extra.{key}" for key in EXTRA_RENAMES if key in extra]
            + [f"extra.opt.{key}" for key in OPTIMIZER_RENAMES if key in opt])


def refuse_unconverted_checkpoint(payload: Mapping[str, Any], path: str | os.PathLike, flag: str) -> None:
    """Stop a TRAINING loader that was handed a checkpoint of the pre-rename code (see the module docstring).

    train_online reads extra["dvd_dynamics"] and extra["opt"]["dvd"] and skips what it does not find, so an old
    checkpoint would lose its fine-tuned world model and that optimiser without a message. Raises ValueError naming
    the keys and the converter; a converted or new checkpoint passes.
    """
    found = legacy_checkpoint_keys(payload)
    if found:
        raise ValueError(
            f"{flag} {path}: this checkpoint was written by the pre-rename research code (it holds {', '.join(found)}). "
            f"This trainer reads only the new names, so the fine-tuned world model and its optimiser would be skipped "
            f"silently and training would continue from the pretrained world model. Convert it first and pass the "
            f"converted file:  {CONVERT_CHECKPOINT}")


def refuse_unconverted_resume_state(state: Mapping[str, Any], path: str | os.PathLike) -> None:
    """The same guard for <run>/resume/state.json of baselines/dsrl_mimicgen, whose V / world-model file is named by
    the key "dvd_checkpoint": under the old key it would not be loaded."""
    found = [key for key in RESUME_STATE_RENAMES if key in state]
    if found:
        raise ValueError(
            f"--resume: {path} was written by the pre-rename research code (key {found[0]!r}). This trainer reads "
            f"{RESUME_STATE_RENAMES[found[0]]!r}, so V and the world model of the run would be skipped silently. Rename "
            f"that key in the file (the .pt it names holds no renamed container), or start a fresh run")


def load_checkpoint(path: str | os.PathLike, map_location: Any = "cpu", *, strict: bool = True) -> dict[str, Any]:
    """torch.load of a DICE-RL checkpoint followed by upgrade_checkpoint: the loader evaluation code should use."""
    import torch
    return upgrade_checkpoint(torch.load(path, map_location=map_location, weights_only=False), strict=strict)


def sidecar(ckpt: str | os.PathLike) -> Path:
    """The V / world-model sidecar of a DSRL checkpoint <step>.zip: <step>_dvd.pt, or <step>_svg.pt from the old code."""
    ckpt = Path(ckpt)
    stem = ckpt.with_suffix("") if ckpt.suffix == ".zip" else ckpt
    new, old = Path(f"{stem}{SIDECAR_SUFFIX}"), Path(f"{stem}{LEGACY_SIDECAR_SUFFIX}")
    if new.exists() and old.exists():
        raise ValueError(f"both {new.name} and {old.name} exist next to {ckpt}; refusing to pick one")
    if new.exists():
        return new
    if old.exists():
        return old
    raise FileNotFoundError(f"no sidecar for {ckpt}: neither {new} nor {old} exists")


ASSET_KINDS = ("prior", "value", "wm", "dataset", "pixel_cache", "demo_store")
ASSET_FIELDS = {"base_output": "prior", "base_checkpoint": "prior", "value_model": "value", "dino_wm_ckpt": "wm"}


def asset_roots() -> dict[str, Path]:
    """Where each asset kind lives under the scripts/env.sh roots (the expressions the trainers themselves use)."""
    from .common import CACHE_ROOT, CKPT_ROOT, DATA_ROOT
    return {"prior": CKPT_ROOT / "priors", "value": CKPT_ROOT / "value", "wm": CKPT_ROOT / "wm", "dataset": DATA_ROOT,
            "pixel_cache": Path(os.environ.get("PIXEL_CACHE_DIR", str(CACHE_ROOT / "pixel_cache"))),
            "demo_store": Path(os.environ.get("DSRL_DEMO_CACHE", str(CACHE_ROOT / "dsrl_demo_store")))}


def _last_index(parts: tuple[str, ...], name: str) -> int:
    return len(parts) - 1 - parts[::-1].index(name)


def _candidates(path: Path, kind: str, root: Path, task: str | None) -> list[Path]:
    parts, name = path.parts, path.name
    if kind == "prior":
        out = []
        if "priors" in parts[:-1]:
            out.append(root.joinpath(*parts[_last_index(parts[:-1], "priors") + 1:]))
        if "checkpoints" in parts[1:-1]:
            out.append(root.joinpath(*parts[_last_index(parts[:-1], "checkpoints") - 1:]))
        else:
            out.append(root / name)
        return out
    if kind == "wm":
        direct = [root / path.parent.name / name, root / name]
        if not task or any(candidate.exists() for candidate in direct):
            return direct
        # the same file name is used by several tasks, so a file found by name alone must sit in a directory of this task
        nested = sorted(root.glob(f"*{task}*/{name}"))
        if len(nested) > 1:
            raise FileNotFoundError(f"world model {path} is not on this machine and {len(nested)} files below {root} "
                                    f"match its name and task: {[str(p) for p in nested]}; pass the path explicitly")
        return direct + nested
    if kind in ("pixel_cache", "demo_store"):
        return [root / name if path.suffix else root]
    return [root / name]


def remap_asset_path(path: str | os.PathLike | None, kind: str, override: str | os.PathLike | None = None, *,
                     task: str | None = None, roots: Mapping[str, Path] | None = None) -> Path | None:
    """Resolve an asset path stored in a run's config.json on the machine that reads it.

    Order: an explicit override wins, and must exist (FileNotFoundError naming it otherwise: a mistyped argument must
    not surface as an unrelated error deep inside a loader); else the stored path if it exists; else the same asset
    below the env.sh root of its kind (asset_roots()). kind is one of ASSET_KINDS:

      prior        a prior run dir, or <prior>/checkpoints/<step>/pretrained_model inside one -> $CKPT_ROOT/priors/<prior>/...
      value        a V directory                                                    -> $CKPT_ROOT/value/<name>
      wm           a world-model file -> $CKPT_ROOT/wm/<parent dir>/<file>, else $CKPT_ROOT/wm/<file>, else (only with
                   task=) the one $CKPT_ROOT/wm/*<task>*/<file>: the research machine kept one task's file outside
                   any per-task directory, and file names repeat across tasks
      dataset      a LeRobot dataset root                                           -> $DATA_ROOT/<name>
      pixel_cache  the pixel cache dir or one pixel_cache_*.npy in it               -> $CACHE_ROOT/pixel_cache[/<file>]
      demo_store   the DSRL demo store dir or one file in it                        -> $CACHE_ROOT/dsrl_demo_store[/<file>]

    An empty stored path with no override is returned as None (nothing was configured). A remap is printed; if no
    candidate exists the error names what was tried, so a wrong machine layout fails here, not deep in a loader.
    """
    if kind not in ASSET_KINDS:
        raise ValueError(f"unknown asset kind {kind!r}; one of {ASSET_KINDS}")
    if override is not None and str(override) != "":
        if not Path(override).exists():
            raise FileNotFoundError(f"{kind} override {override} does not exist. It was given explicitly, so nothing "
                                    f"else is tried (stored path of the run: {str(path) if path else 'none'})")
        return Path(override)
    if path is None or str(path) == "":
        return None
    path = Path(path)
    if path.exists():
        return path
    root = Path((roots if roots is not None else asset_roots())[kind])
    tried = _candidates(path, kind, root, task)
    for candidate in tried:
        if candidate.exists():
            print(f"[legacy] {kind} path {path} is not on this machine; using {candidate}", flush=True)
            return candidate
    raise FileNotFoundError(f"{kind} asset {path} (stored in the run's config) does not exist here, and neither does "
                            f"any of {[str(p) for p in dict.fromkeys(tried)]}. Put the asset under the scripts/env.sh "
                            f"roots or pass its path explicitly")


def remap_config_assets(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None,
                        fields: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Copy of a config with its stored asset paths (ASSET_FIELDS, or the given subset) passed through remap_asset_path.

    overrides maps a field name to an explicit path (None / missing = no override). Name only the fields the caller
    will actually open: an asset that is absent here and not needed must not make the caller fail.
    """
    overrides = dict(overrides or {})
    names = tuple(ASSET_FIELDS) if fields is None else tuple(fields)
    unknown = sorted(set(names) - set(ASSET_FIELDS)) + sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(f"not asset fields being resolved: {unknown}; resolving {names} of {sorted(ASSET_FIELDS)}")
    out = dict(config)
    for name in names:
        resolved = remap_asset_path(out.get(name), ASSET_FIELDS[name], overrides.get(name), task=out.get("task"))
        if resolved is not None:
            out[name] = str(resolved)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="convert artifacts of the pre-rename (svg) code to this repository's names")
    sub = parser.add_subparsers(dest="command", required=True)
    cfg = sub.add_parser("config", help="print a config.json with the new names")
    cfg.add_argument("config", type=Path)
    ckpt = sub.add_parser("checkpoint", help="write a DICE-RL checkpoint with the new names (for --resume, "
                                             "--dvd-dynamics-init, --value-init)")
    ckpt.add_argument("source", type=Path)
    ckpt.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    if args.command == "config":
        json.dump(upgrade_config(json.loads(args.config.read_text())), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    import torch
    if args.target.exists():
        raise SystemExit(f"{args.target} exists; refusing to overwrite")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(load_checkpoint(args.source), args.target)
    print(f"[legacy] {args.source} -> {args.target}", flush=True)


if __name__ == "__main__":
    main()
