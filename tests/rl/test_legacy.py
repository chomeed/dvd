"""online_rl_mimicgen/legacy.py: artifacts written by the pre-rename research tree stay usable.

    python tests/rl/test_legacy.py
    LIVE_TREE=<research tree> LEGACY_RUNS=<dir of runs>[:<dir>...] DSRL_ROOT=<dsrl checkout> python tests/rl/test_legacy.py

Always run (no inputs): the rename tables cover every dvd-named TrainConfig field / DSRL argument of this tree, the
unit behaviour of upgrade_config / upgrade_checkpoint / sidecar / remap_asset_path on synthetic inputs, and the guards
that stop the TRAINERS' checkpoint loaders on an unconverted old checkpoint (refuse_unconverted_*), including that the
three loaders still call them before loading anything.
With LIVE_TREE: the tables equal a fresh mechanical derivation from the two source trees.
With LEGACY_RUNS: every real config.json found is upgraded and must be accepted by this tree's TrainConfig (DICE-RL) or
argument parser (DSRL, needs DSRL_ROOT) with every value unchanged; one real checkpoint of each kind is loaded on CPU.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import re
import tempfile
from pathlib import Path

import _util

_util.use_new_tree()
from online_rl_mimicgen import legacy  # noqa: E402

OLD = "svg"
TRAIN_ONLINE, COMMON = "train_online.py", "common.py"
DSRL_FILES = ("baselines/dsrl_mimicgen.py", "baselines/dsrl_dino.py")


def rename(text: str) -> str:
    return text.replace("svg", "dvd")


def trainconfig_fields(path: Path) -> dict[str, str | None]:
    """TrainConfig field -> source of its default, read from the AST (nothing is imported)."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig":
            return {b.target.id: (ast.unparse(b.value) if b.value is not None else None)
                    for b in node.body if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)}
    raise AssertionError(f"no TrainConfig in {path}")


def parser_args(path: Path) -> dict[str, list | None]:
    """argparse dest -> choices, for every add_argument call of a file (read from the AST, nothing is imported).

    None when the argument has no choices; for a computed choices expression, the string literals it contains.
    """
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            keywords = {k.arg: k.value for k in node.keywords}
            dest = (ast.literal_eval(keywords["dest"]) if "dest" in keywords
                    else next(f for f in flags if f.startswith("--"))[2:].replace("-", "_"))
            choices = keywords.get("choices")
            if choices is not None:
                try:
                    choices = list(ast.literal_eval(choices))
                except ValueError:
                    choices = [c.value for c in ast.walk(choices) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            out[dest] = choices
    return out


def string_constants(path: Path) -> set[str]:
    return {n.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def checkpoint_extra_keys(path: Path) -> set[str]:
    """String keys of the dict literal(s) train_online hands to save_checkpoint as `extra` (8th argument), nested dicts included."""
    keys = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "save_checkpoint":
            for extra in node.args[7:8] + [kw.value for kw in node.keywords if kw.arg == "extra"]:
                keys |= {k.value for d in ast.walk(extra) if isinstance(d, ast.Dict)
                         for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return keys


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def expected_upgrade(config: dict) -> dict:
    """What upgrade_config must return, written out independently of its implementation."""
    out = {}
    for key, value in config.items():
        new = legacy.KEY_RENAMES.get(key, key)
        out[new] = legacy.VALUE_RENAMES.get(new, {}).get(value, value) if isinstance(value, str) else value
    return out


# ---------------------------------------------------------------- tables


def test_tables_cover_this_tree():
    """Every dvd-named field / argument of this tree is the target of exactly one legacy key, and vice versa."""
    fields = trainconfig_fields(_util.PACKAGE / COMMON)
    dvd_fields = {f for f in fields if "dvd" in f}
    assert set(legacy.TRAIN_CONFIG_RENAMES.values()) == dvd_fields, (
        sorted(dvd_fields ^ set(legacy.TRAIN_CONFIG_RENAMES.values())))
    dsrl = {}
    for rel in DSRL_FILES:
        dsrl.update(parser_args(_util.PACKAGE / rel))
    dvd_args = {d for d in dsrl if "dvd" in d}
    assert set(legacy.DSRL_ARG_RENAMES.values()) == dvd_args, sorted(dvd_args ^ set(legacy.DSRL_ARG_RENAMES.values()))
    assert len(set(legacy.KEY_RENAMES.values())) == len(legacy.KEY_RENAMES), "two legacy keys map to one name"
    assert all(OLD in old and OLD not in new for old, new in legacy.KEY_RENAMES.items())

    train_args = parser_args(_util.PACKAGE / TRAIN_ONLINE)
    for key, table in legacy.VALUE_RENAMES.items():
        if key == "method":
            continue
        assert key in fields and train_args.get(key), f"{key} is not a TrainConfig field with choices"
        for old, new in table.items():
            assert new in train_args[key] and old not in train_args[key], (key, old, new, train_args[key])
    dsrl_strings = set().union(*(string_constants(_util.PACKAGE / rel) for rel in DSRL_FILES))
    assert set(legacy.VALUE_RENAMES["method"].values()) <= dsrl_strings
    extra = checkpoint_extra_keys(_util.PACKAGE / TRAIN_ONLINE)
    assert set(legacy.EXTRA_RENAMES.values()) | set(legacy.OPTIMIZER_RENAMES.values()) <= extra, sorted(extra)
    assert legacy.SIDECAR_SUFFIX in (_util.PACKAGE / DSRL_FILES[1]).read_text()
    print(f"[tables] {len(legacy.TRAIN_CONFIG_RENAMES)} TrainConfig fields of {len(fields)}, "
          f"{len(legacy.DSRL_ARG_RENAMES)} DSRL arguments of {len(dsrl)} "
          f"({len(legacy.KEY_RENAMES)} distinct keys), {sum(map(len, legacy.VALUE_RENAMES.values()))} option values, "
          f"checkpoint keys {sorted(legacy.EXTRA_RENAMES)} + opt {sorted(legacy.OPTIMIZER_RENAMES)}")


def test_tables_match_live_tree():
    """The tables equal the mechanical derivation: names that differ between LIVE_TREE and this tree, nothing else."""
    root = _util.live_tree()
    live_pkg = root / "online_rl_mimicgen"

    live_fields, new_fields = trainconfig_fields(live_pkg / COMMON), trainconfig_fields(_util.PACKAGE / COMMON)
    assert {rename(f) for f in live_fields} == set(new_fields) and len(live_fields) == len(new_fields)
    assert {f: rename(f) for f in live_fields if rename(f) != f} == legacy.TRAIN_CONFIG_RENAMES
    changed_defaults = {rename(f): (d, new_fields[rename(f)]) for f, d in live_fields.items() if d != new_fields[rename(f)]}

    live_dsrl, new_dsrl = {}, {}
    for rel in DSRL_FILES:
        live_dsrl.update(parser_args(live_pkg / rel))
        new_dsrl.update(parser_args(_util.PACKAGE / rel))
    assert {rename(d) for d in live_dsrl} == set(new_dsrl)
    assert {d: rename(d) for d in live_dsrl if rename(d) != d} == legacy.DSRL_ARG_RENAMES

    live_train, new_train = parser_args(live_pkg / TRAIN_ONLINE), parser_args(_util.PACKAGE / TRAIN_ONLINE)
    assert {rename(d) for d in live_train} == set(new_train)
    values = {}
    for parsers in ((live_train, new_train), (live_dsrl, new_dsrl)):
        for dest, choices in parsers[0].items():
            old_values = [c for c in choices or [] if isinstance(c, str) and OLD in c.lower()]
            if old_values:
                assert [rename(c) if isinstance(c, str) else c for c in choices] == parsers[1][rename(dest)], dest
                values[rename(dest)] = {c: rename(c) for c in old_values}
    tags = {s for rel in DSRL_FILES for s in string_constants(live_pkg / rel) if re.fullmatch(r"dsrl_\w*svg\w*", s)}
    values["method"] = {t: rename(t) for t in tags}
    assert values == legacy.VALUE_RENAMES, (values, legacy.VALUE_RENAMES)
    # a default that changed is exactly a renamed option value (svg_gate_target 'svg' -> 'dvd')
    assert changed_defaults == {"dvd_gate_target": ("'svg'", "'dvd'")}, changed_defaults

    old_extra = {k for k in checkpoint_extra_keys(live_pkg / TRAIN_ONLINE) if OLD in k}
    assert old_extra == set(legacy.EXTRA_RENAMES) | set(legacy.OPTIMIZER_RENAMES), old_extra
    assert legacy.LEGACY_SIDECAR_SUFFIX in (live_pkg / DSRL_FILES[1]).read_text()
    print(f"[tables] derived from {root}: {len(live_fields)} TrainConfig fields ({len(legacy.TRAIN_CONFIG_RENAMES)} renamed), "
          f"{len(live_train)} train_online + {len(live_dsrl)} DSRL arguments ({len(legacy.DSRL_ARG_RENAMES)} renamed), "
          f"option values {values}, checkpoint keys {sorted(old_extra)}: all equal to legacy.py")


# ---------------------------------------------------------------- unit behaviour


def test_upgrade_config_unit():
    old = {key: i for i, key in enumerate(legacy.KEY_RENAMES)}
    old.update(bon_source="svg", bc_filter_source="svg", actor_gradient_source_normal="svg1",
               actor_gradient_source_critical="q", svg_gate_target="svg", method="dsrl_na_svg_dino",
               output_dir="/runs/dsrl-na-svg1.0_x", wandb_name="ag_enc_svg_scaled_s43", task="square_d0", seed=43)
    new = legacy.upgrade_config(old)
    assert new == expected_upgrade(old) and not any(OLD in k for k in new)
    assert (new["bon_source"], new["actor_gradient_source_normal"], new["dvd_gate_target"], new["method"]) == (
        "dvd", "dvd1", "dvd", "dsrl_na_dvd_dino")
    assert new["output_dir"] == old["output_dir"] and new["wandb_name"] == old["wandb_name"], "names/paths must not change"
    assert legacy.upgrade_config(new) == new, "not idempotent"
    assert legacy.is_legacy(old) and not legacy.is_legacy(new)
    assert "svg_weight" in old, "input must not be mutated"

    def refused(config, **kw):
        try:
            legacy.upgrade_config(config, **kw)
        except ValueError as error:
            return str(error)
        return None
    assert refused({"svg_brand_new_flag": 1}), "unknown legacy key accepted"
    assert refused({"some_new_source": "svg"}), "old option value in an unknown field accepted"
    assert refused({"svg_weight": 1.0, "dvd_weight": 2.0}), "old and new spelling together accepted"
    assert legacy.upgrade_config({"svg_brand_new_flag": 1, "x_source": "svg1"}, strict=False) == {
        "svg_brand_new_flag": 1, "x_source": "svg1"}
    print(f"[unit] upgrade_config: {len(legacy.KEY_RENAMES)} keys + 5 values renamed, idempotent, 3 kinds of leftover refused")


def test_upgrade_checkpoint_unit():
    tensor = object()
    old = {"config": {"svg_weight": 1.0, "bon_source": "svg"}, "actor": {"w": tensor}, "env_steps": 5,
           "extra": {"svg_dynamics": {"wm.x": tensor}, "value_fn": {"v": tensor},
                     "opt": {"actor": 1, "svg": 2, "value": 3}}}
    new = legacy.upgrade_checkpoint(old)
    assert new["config"] == {"dvd_weight": 1.0, "bon_source": "dvd"}
    assert set(new["extra"]) == {"dvd_dynamics", "value_fn", "opt"} and set(new["extra"]["opt"]) == {"actor", "dvd", "value"}
    assert new["extra"]["dvd_dynamics"]["wm.x"] is tensor and new["actor"]["w"] is tensor, "tensors must be shared"
    assert "svg_dynamics" in old["extra"] and "svg" in old["extra"]["opt"], "input must not be mutated"
    assert legacy.upgrade_checkpoint(new) == new, "not idempotent"
    assert legacy.upgrade_checkpoint({"config": {}, "actor": {}}) == {"config": {}, "actor": {}}, "no extra must stay no extra"
    for bad in ({"config": {}, "extra": {"svg_new_thing": 1}}, {"config": {}, "extra": {"opt": {"svg2": 1}}}):
        try:
            legacy.upgrade_checkpoint(bad)
        except ValueError:
            continue
        raise AssertionError(f"unknown legacy checkpoint key accepted: {bad}")
    print("[unit] upgrade_checkpoint: config + extra + opt renamed, tensors shared, idempotent, unknown keys refused")


def test_sidecar_unit():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for step in ("0000001", "0000002", "0000003", "0000004"):
            (tmp / f"{step}.zip").touch()
        (tmp / "0000001_dvd.pt").touch()
        (tmp / "0000002_svg.pt").touch()
        (tmp / "0000003_dvd.pt").touch(); (tmp / "0000003_svg.pt").touch()
        assert legacy.sidecar(tmp / "0000001.zip").name == "0000001_dvd.pt"
        assert legacy.sidecar(tmp / "0000002.zip").name == "0000002_svg.pt"
        assert legacy.sidecar(tmp / "0000002").name == "0000002_svg.pt"
        for step, error in (("0000003", ValueError), ("0000004", FileNotFoundError)):
            try:
                legacy.sidecar(tmp / f"{step}.zip")
            except error:
                continue
            raise AssertionError(f"sidecar({step}) should raise {error.__name__}")
    print("[unit] sidecar: new name, legacy name, both -> ValueError, none -> FileNotFoundError")


def test_remap_asset_path_unit():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        roots = {"prior": tmp / "ckpt/priors", "value": tmp / "ckpt/value", "wm": tmp / "ckpt/wm", "dataset": tmp / "data",
                 "pixel_cache": tmp / "cache/pixel_cache", "demo_store": tmp / "cache/dsrl_demo_store"}
        model = roots["prior"] / "square_40k/checkpoints/040000/pretrained_model"
        model.mkdir(parents=True)
        (roots["value"] / "v_square").mkdir(parents=True)
        for task in ("square_d0", "square_d1"):
            (roots["wm"] / f"mimicgen_{task}_wm").mkdir(parents=True)
            (roots["wm"] / f"mimicgen_{task}_wm/wm_150k.pt").touch()
        (roots["wm"] / "mimicgen_threading_wm").mkdir()
        (roots["wm"] / "mimicgen_threading_wm/wm_168k.pt").touch()
        (roots["dataset"] / "mimicgen_square_d0_224x224").mkdir(parents=True)
        roots["pixel_cache"].mkdir(parents=True)
        (roots["pixel_cache"] / "pixel_cache_x_224.npy").touch()
        roots["demo_store"].mkdir(parents=True)
        gone = "/machine/that/is/gone"

        def remap(path, kind, override=None, task=None):
            return legacy.remap_asset_path(path, kind, override, task=task, roots=roots)

        assert remap(f"{gone}/hdr/priors/square_40k", "prior") == roots["prior"] / "square_40k"
        assert remap(f"{gone}/hdr/priors/square_40k/checkpoints/040000/pretrained_model", "prior") == model
        assert remap(f"{gone}/elsewhere/square_40k/checkpoints/040000/pretrained_model", "prior") == model
        assert remap(f"{gone}/0814/v_square", "value") == roots["value"] / "v_square"
        assert remap(f"{gone}/cache/mimicgen_square_d1_wm/wm_150k.pt", "wm") == roots["wm"] / "mimicgen_square_d1_wm/wm_150k.pt"
        assert remap(f"{gone}/cache/wm_168k.pt", "wm", task="threading") == roots["wm"] / "mimicgen_threading_wm/wm_168k.pt"
        assert remap(f"{gone}/cache/wm_150k.pt", "wm", task="square_d1") == roots["wm"] / "mimicgen_square_d1_wm/wm_150k.pt"
        assert remap(f"{gone}/lerobot/mimicgen_square_d0_224x224", "dataset") == roots["dataset"] / "mimicgen_square_d0_224x224"
        assert remap(f"{gone}/pc/pixel_cache_x_224.npy", "pixel_cache") == roots["pixel_cache"] / "pixel_cache_x_224.npy"
        assert remap(f"{gone}/pc", "pixel_cache") == roots["pixel_cache"]
        assert remap(f"{gone}/demo_store_cache", "demo_store") == roots["demo_store"]
        # precedence: override > stored path that exists > roots; nothing stored -> None
        (tmp / "mine").mkdir()
        assert remap(f"{gone}/0814/v_square", "value", override=tmp / "mine") == tmp / "mine"
        assert remap(str(tmp), "value") == tmp, "an existing stored path must be returned untouched"
        assert remap("", "wm") is None and remap(None, "prior") is None
        assert remap("", "wm", override=str(tmp / "mine")) == tmp / "mine", "an override needs no stored path"
        # an explicit override that does not exist is an error naming it, never a fall-through to the stored path or
        # the roots (here both the stored path and its root candidate exist)
        for stored in (str(roots["value"] / "v_square"), f"{gone}/0814/v_square", ""):
            try:
                remap(stored, "value", override=tmp / "typo")
            except FileNotFoundError as error:
                assert str(tmp / "typo") in str(error) and "override" in str(error), error
            else:
                raise AssertionError(f"a missing override was accepted (stored path {stored!r})")
        # a file name alone never picks a world model: it repeats across tasks (wm_150k.pt exists for two here)
        for path, kind, task in ((f"{gone}/0814/v_missing", "value", None), (f"{gone}/cache/wm_168k.pt", "wm", None),
                                 (f"{gone}/cache/wm_150k.pt", "wm", "square"), (f"{gone}/cache/wm_150k.pt", "wm", "hammer"),
                                 (f"{gone}/x_square_d0_other/wm_150k.pt", "wm", "square_d1x"),
                                 (f"{gone}/priors/square_40k/checkpoints/999999/pretrained_model", "prior", None)):
            try:
                remap(path, kind, task=task)
            except FileNotFoundError:
                continue
            raise AssertionError(f"remap({path}, {kind}, task={task}) should fail: missing or ambiguous")
        try:
            remap("x", "weights")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown kind accepted")

        config = {"base_output": f"{gone}/priors/square_40k", "value_model": f"{gone}/0814/v_missing", "dino_wm_ckpt": "", "seed": 1,
                  "task": "threading"}
        real_roots, legacy.asset_roots = legacy.asset_roots, lambda: roots
        try:
            out = legacy.remap_config_assets(config, fields=("base_output", "dino_wm_ckpt"))
            assert out == {**config, "base_output": str(roots["prior"] / "square_40k")}, out
            out = legacy.remap_config_assets(config, {"value_model": tmp / "mine"}, fields=("value_model",))
            assert out["value_model"] == str(tmp / "mine") and out["base_output"] == config["base_output"]
            flat = legacy.remap_config_assets({**config, "dino_wm_ckpt": f"{gone}/cache/wm_168k.pt"}, fields=("dino_wm_ckpt",))
            assert flat["dino_wm_ckpt"] == str(roots["wm"] / "mimicgen_threading_wm/wm_168k.pt"), "task of the config must be used"
            for bad in (dict(fields=("seed",)), dict(overrides={"value_model": "x"}, fields=("base_output",))):
                try:
                    legacy.remap_config_assets(config, **bad)
                except ValueError:
                    continue
                raise AssertionError(f"remap_config_assets accepted {bad}")
        finally:
            legacy.asset_roots = real_roots
    print("[unit] remap_asset_path: 6 kinds, override > existing stored path > env roots, missing/ambiguous raise, "
          "a missing override raises FileNotFoundError naming it")


def test_converter_cli_unit():
    """python -m online_rl_mimicgen.legacy config|checkpoint: the route for feeding an old checkpoint to a training flag."""
    import contextlib
    import io

    import torch
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = {"svg_weight": 1.0, "bon_source": "svg", "task": "square_d0"}
        (tmp / "config.json").write_text(json.dumps(config))
        torch.save({"config": config, "actor": {"w": torch.ones(2)}, "env_steps": 7,
                    "extra": {"svg_dynamics": {"wm.x": torch.zeros(3)}, "opt": {"svg": {"lr": 1.0}}}}, tmp / "old.pt")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            legacy.main(["config", str(tmp / "config.json")])
            legacy.main(["checkpoint", str(tmp / "old.pt"), str(tmp / "sub/new.pt")])
        assert json.loads(printed.getvalue().split("[legacy]")[0]) == {"dvd_weight": 1.0, "bon_source": "dvd", "task": "square_d0"}
        new = torch.load(tmp / "sub/new.pt", map_location="cpu", weights_only=False)
        assert new["config"]["dvd_weight"] == 1.0 and set(new["extra"]) == {"dvd_dynamics", "opt"} and set(new["extra"]["opt"]) == {"dvd"}
        assert torch.equal(new["extra"]["dvd_dynamics"]["wm.x"], torch.zeros(3)) and new["env_steps"] == 7
        try:
            with contextlib.redirect_stdout(printed):
                legacy.main(["checkpoint", str(tmp / "old.pt"), str(tmp / "sub/new.pt")])
        except SystemExit as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("the converter overwrote an existing file")
    print("[unit] converter CLI: config printed with new names, checkpoint rewritten with new containers, no overwrite")


def test_refuse_unconverted_unit():
    """The guards of the TRAINERS' loaders: an old payload is refused with the converter command, a converted one passes."""
    old = {"config": {"svg_weight": 1.0}, "actor": {}, "env_steps": 5,
           "extra": {"svg_dynamics": {"wm.x": 1}, "value_fn": {"v": 1}, "opt": {"actor": 1, "svg": 2}}}
    assert legacy.legacy_checkpoint_keys(old) == ["extra.svg_dynamics", "extra.opt.svg"]
    for payload, expect in ((old, ("extra.svg_dynamics", "extra.opt.svg")),
                            ({"extra": {"svg_dynamics": {}}}, ("extra.svg_dynamics",)),
                            ({"extra": {"opt": {"svg": {}}}}, ("extra.opt.svg",))):
        try:
            legacy.refuse_unconverted_checkpoint(payload, "/runs/x/checkpoints/600000.pt", "--resume")
        except ValueError as error:
            text = str(error)
            assert all(key in text for key in expect), text
            assert legacy.CONVERT_CHECKPOINT in text and "--resume /runs/x/checkpoints/600000.pt" in text, text
        else:
            raise AssertionError(f"an unconverted checkpoint was accepted: {sorted(payload.get('extra', {}))}")
    # what must pass: a converted payload, a new one, a Q-arm payload of the old code (no renamed container: nothing
    # would be skipped), a payload without extra
    for payload in (legacy.upgrade_checkpoint(old), {"config": {"dvd_weight": 1.0}, "extra": {"dvd_dynamics": {}, "opt": {"dvd": {}}}},
                    {"config": {"svg_weight": 0.0}, "extra": {"encoder": {}, "opt": {"actor": 1, "critic": 2}}},
                    {"config": {}, "actor": {}}, {"extra": None}):
        legacy.refuse_unconverted_checkpoint(payload, "x.pt", "--resume")
        assert legacy.legacy_checkpoint_keys(payload) == []
    # the command the message names is the converter exercised in test_converter_cli_unit
    assert legacy.CONVERT_CHECKPOINT.startswith("python -m online_rl_mimicgen.legacy checkpoint ") and callable(legacy.main)

    assert set(legacy.RESUME_STATE_RENAMES.values()) <= string_constants(_util.PACKAGE / DSRL_FILES[0])
    try:
        legacy.refuse_unconverted_resume_state({"checkpoint": "a.zip", "svg_checkpoint": "a_svg.pt"}, "run/resume/state.json")
    except ValueError as error:
        assert "svg_checkpoint" in str(error) and "dvd_checkpoint" in str(error) and "run/resume/state.json" in str(error)
    else:
        raise AssertionError("an old resume bundle was accepted")
    legacy.refuse_unconverted_resume_state({"checkpoint": "a.zip", "dvd_checkpoint": "a_dvd.pt"}, "state.json")
    legacy.refuse_unconverted_resume_state({"checkpoint": "a.zip", "dvd_checkpoint": None}, "state.json")
    print("[unit] refuse_unconverted_checkpoint / _resume_state: old containers refused with the converter command; "
          "converted, new and container-free payloads pass")


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", "")) == name]


def test_trainer_loaders_call_the_guards():
    """Static: each training loader that reads a renamed container calls its guard BEFORE the first load_state_dict
    that follows the torch.load / json read. (The loaders sit inside train() / main(), which need a GPU and a simulator,
    so this is checked on the source: guard call present, right flag, and earlier in the file than the first load.)"""
    train = (_util.PACKAGE / TRAIN_ONLINE).read_text()
    tree = ast.parse(train)
    guards = {ast.literal_eval(c.args[2]): c for c in _calls(tree, "refuse_unconverted_checkpoint")}
    assert set(guards) == {"--resume", "--dvd-dynamics-init"}, sorted(guards)
    lines = train.splitlines()

    def first_line(needle: str, after: int = 0) -> int:
        return next(i for i, line in enumerate(lines[after:], after + 1) if needle in line)

    # --resume: guard on the payload variable, after its torch.load, before anything is restored from it
    resume = guards["--resume"]
    payload_var = resume.args[0].id
    loaded = first_line(f"{payload_var} = torch.load(ck_path")
    restored = first_line(f'actor.load_state_dict({payload_var}["actor"])', loaded)
    assert loaded < resume.lineno < restored, (loaded, resume.lineno, restored)
    assert first_line('if "dvd_dynamics" in ex and dvd_term is not None', loaded) > resume.lineno
    # --dvd-dynamics-init: guard before the lookup of extra.dvd_dynamics
    init = guards["--dvd-dynamics-init"]
    loaded = first_line(f"{init.args[0].id} = torch.load(config.dvd_dynamics_init")
    assert loaded < init.lineno < first_line('.get("dvd_dynamics")', loaded), (loaded, init.lineno)
    # the containers the trainer reads under a renamed name are exactly the ones the guard knows
    read = {c.value for n in ast.walk(tree) if isinstance(n, ast.Compare) and len(n.comparators) == 1
            and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == "ex"
            for c in [n.left] if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert {k for k in read if "dvd" in k} == set(legacy.EXTRA_RENAMES.values()), sorted(read)

    dsrl = (_util.PACKAGE / DSRL_FILES[0]).read_text()
    dsrl_lines = dsrl.splitlines()
    (guard,) = _calls(ast.parse(dsrl), "refuse_unconverted_resume_state")
    read_state = next(i for i, line in enumerate(dsrl_lines, 1) if "state = json.loads(state_path.read_text())" in line)
    use = next(i for i, line in enumerate(dsrl_lines, 1) if 'state.get("dvd_checkpoint")' in line)
    assert read_state < guard.lineno < use, (read_state, guard.lineno, use)
    # dsrl_dino has no resume path to guard: it refuses --resume outright
    assert "--resume is not supported" in (_util.PACKAGE / DSRL_FILES[1]).read_text()
    print(f"[guards] train_online: --resume guard at line {resume.lineno} (payload loaded {first_line(payload_var + ' = torch.load(ck_path')}, "
          f"first restore {restored}), --dvd-dynamics-init guard at line {init.lineno}; dsrl_mimicgen --resume guard at line "
          f"{guard.lineno}; dsrl_dino refuses --resume")


def test_asset_roots_follow_common():
    """asset_roots() is built from the same module constants the trainers resolve their defaults from."""
    from online_rl_mimicgen import common
    roots = legacy.asset_roots()
    assert set(roots) == set(legacy.ASSET_KINDS)
    assert roots["prior"] == common.CKPT_ROOT / "priors" and roots["dataset"] == common.DATA_ROOT
    for task, spec in common.TASKS.items():
        assert spec["il_output"].parent == roots["prior"] and spec["dataset_root"].parent == roots["dataset"], task
    print(f"[roots] prior {roots['prior']}  value {roots['value']}  wm {roots['wm']}  dataset {roots['dataset']}")


# ---------------------------------------------------------------- real artifacts (LEGACY_RUNS)


def real_configs() -> tuple[dict, dict, int]:
    runs = _util.legacy_run_dirs()
    if not runs:
        _util.skip("LEGACY_RUNS is not set (directories holding run directories of the research tree)")
    dice, dsrl, other = {}, {}, 0
    for run in runs:
        config = _util.read_config(run)
        method = str((config or {}).get("method", ""))
        if config and method.startswith("dsrl_"):
            dsrl[run] = config
        elif config and {"task", "output_dir", "base_checkpoint"} <= set(config):
            dice[run] = config
        else:
            other += 1
    return dice, dsrl, other


def test_real_dicerl_configs():
    """Every DICE-RL config.json of the research tree -> a TrainConfig of this tree with every value unchanged."""
    dice, _, other = real_configs()
    if not dice:
        _util.skip("no DICE-RL config.json under LEGACY_RUNS")
    from online_rl_mimicgen.common import TrainConfig
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    n_legacy = renamed_keys = renamed_values = 0
    for run, old in dice.items():
        new = legacy.upgrade_config(old)
        want = expected_upgrade(old)
        assert set(new) == set(want) and all(same(new[k], want[k]) for k in want), run
        assert set(new) <= fields, f"{run}: keys TrainConfig does not have: {sorted(set(new) - fields)}"
        built = dataclasses.asdict(TrainConfig(**new))
        assert all(same(built[k], new[k]) for k in new), run
        assert legacy.upgrade_config(new) == new and not legacy.is_legacy(new), run
        n_legacy += legacy.is_legacy(old)
        renamed_keys += sum(k in legacy.KEY_RENAMES for k in old)
        renamed_values += sum(not same(new[legacy.KEY_RENAMES.get(k, k)], v) for k, v in old.items())
    sizes = sorted({len(c) for c in dice.values()})
    print(f"[real] {len(dice)} DICE-RL configs ({n_legacy} legacy; {sizes[0]}-{sizes[-1]} keys, TrainConfig has {len(fields)}): "
          f"all accepted by TrainConfig, {renamed_keys} keys and {renamed_values} values renamed, every other value "
          f"identical ({other} non-RL config.json ignored)")


def argv_from_config(parser, config: dict) -> list[str]:
    import argparse
    argv = []
    for action in parser._actions:
        if not action.option_strings or action.dest not in config:
            continue
        value = config[action.dest]
        if isinstance(action, argparse._StoreTrueAction):
            argv += [action.option_strings[0]] if value else []
        elif isinstance(action, argparse._StoreAction) and action.nargs is None:
            if value is not None:
                argv.append(f"{action.option_strings[0]}={value}")
        else:
            raise AssertionError(f"argument {action.option_strings} has an action this test does not round-trip")
    return argv


def test_real_dsrl_configs():
    """Every DSRL config.json of the research tree -> argv this tree's parser accepts, parsing back to the same values."""
    _, dsrl, _ = real_configs()
    if not dsrl:
        _util.skip("no DSRL config.json under LEGACY_RUNS")
    try:
        from online_rl_mimicgen.baselines.dsrl_dino import build_dino_parser
    except (ImportError, SystemExit) as error:
        _util.skip(f"online_rl_mimicgen.baselines.dsrl_dino is not importable here ({str(error)[:120]})")
    parser = build_dino_parser()
    dests = {a.dest for a in parser._actions if a.option_strings}
    methods, renamed_keys = {}, 0
    for run, old in dsrl.items():
        new = legacy.upgrade_config(old)
        want = expected_upgrade(old)
        assert set(new) == set(want) and all(same(new[k], want[k]) for k in want), run
        assert set(new) - {"method"} <= dests, f"{run}: keys the parser does not have: {sorted(set(new) - dests - {'method'})}"
        assert new["method"] in ("dsrl_na_dvd_dino", "dsrl_na_q_dino") and bool(new.get("q_critic")) == (new["method"] == "dsrl_na_q_dino"), run
        parsed = vars(parser.parse_args(argv_from_config(parser, new)))
        for key, value in new.items():
            if key != "method":
                got = str(parsed[key]) if isinstance(parsed[key], Path) else parsed[key]
                assert same(got, value), f"{run}: {key} parsed back as {got!r}, config has {value!r}"
        methods[old["method"]] = methods.get(old["method"], 0) + 1
        renamed_keys += sum(k in legacy.KEY_RENAMES for k in old)
    print(f"[real] {len(dsrl)} DSRL configs {methods}: all accepted by build_dino_parser() ({len(dests)} arguments) and "
          f"parsed back to identical values; {renamed_keys} keys renamed")


def test_real_asset_paths():
    """Stored prior / V / world-model paths of real configs, read as if on another machine, land on the same files.

    Each stored path that exists here is made unreachable (a missing directory is put in front of it) and resolved
    under the env roots; the result must be the very file the stored path points at. Assets that are not under the
    roots are counted, not failed: the roots hold one campaign's assets, the run directories span several.
    """
    dice, dsrl, _ = real_configs()
    if not os.environ.get("CKPT_ROOT"):
        _util.skip("CKPT_ROOT is not set (env root holding priors/ value/ wm/)")
    same_file, absent, wrong = {}, {}, []
    for run, config in {**dice, **dsrl}.items():
        for name, kind in legacy.ASSET_FIELDS.items():
            stored = config.get(name)
            if not stored or not Path(stored).exists():
                continue
            try:
                found = legacy.remap_asset_path(f"/no-such-machine{stored}", kind, task=config.get("task"))
            except FileNotFoundError:
                absent[name] = absent.get(name, 0) + 1
                continue
            if found.resolve() == Path(stored).resolve():
                same_file[name] = same_file.get(name, 0) + 1
            else:
                wrong.append(f"{run.name}: {name} {stored} -> {found}")
    print(f"[real] stored asset paths resolved under the env roots to the identical file: {same_file}; "
          f"not under the roots (left to an explicit override): {absent}; resolved to a DIFFERENT file: {len(wrong)}")
    assert not wrong, wrong[:5]
    assert same_file, "nothing resolved: CKPT_ROOT holds none of these runs' assets"


def _container_keys(obj, prefix="", depth=0):
    if isinstance(obj, dict) and depth < 3:
        for key, value in obj.items():
            yield f"{prefix}{key}"
            if prefix + str(key) != "config":
                yield from _container_keys(value, f"{prefix}{key}.", depth + 1)


def test_real_dicerl_checkpoint():
    """One real DICE-RL checkpoint with a fine-tuned world model: loads, no old key is left, tensors untouched."""
    dice, _, _ = real_configs()
    import torch
    for run, config in sorted(dice.items()):
        ckpts = sorted((run / "checkpoints").glob("[0-9]*.pt"))
        if config.get("bon_source") == "svg" and config.get("svg_finetune_dynamics") and ckpts and ckpts[-1].stem != "000000":
            break
    else:
        _util.skip("no legacy DICE-RL checkpoint past step 0 under LEGACY_RUNS")
    raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    new = legacy.load_checkpoint(ckpts[-1])
    old_keys = sorted(k for k in _container_keys(raw) if OLD in k.split(".")[-1])
    left = sorted(k for k in _container_keys(new) if OLD in k.lower())
    assert not left, left
    assert set(old_keys) == {"extra.svg_dynamics", "extra.opt.svg"}, old_keys
    assert set(legacy.legacy_checkpoint_keys(raw)) == set(old_keys) and legacy.legacy_checkpoint_keys(new) == []
    try:
        legacy.refuse_unconverted_checkpoint(raw, ckpts[-1], "--resume")
    except ValueError as error:
        assert legacy.CONVERT_CHECKPOINT in str(error)
    else:
        raise AssertionError("the trainer's --resume guard accepted an unconverted research checkpoint")
    legacy.refuse_unconverted_checkpoint(new, ckpts[-1], "--resume")
    dynamics = new["extra"]["dvd_dynamics"]
    assert set(dynamics) == set(raw["extra"]["svg_dynamics"]) and all(
        torch.equal(dynamics[k], raw["extra"]["svg_dynamics"][k]) for k in dynamics)
    assert all(torch.equal(new["actor"][k], raw["actor"][k]) for k in raw["actor"])
    from online_rl_mimicgen.common import TrainConfig
    config = TrainConfig(**new["config"])
    assert config.bon_source == "dvd" and config.dvd_dynamics == raw["config"]["svg_dynamics"]
    state_keys = [k for part in ("actor", "critics") for k in new[part]] + list(dynamics) + list(new["extra"].get("value_fn", {}))
    assert not any(OLD in k.lower() for k in state_keys), "a parameter is named after the method"
    print(f"[real] {run.name}/{ckpts[-1].name}: legacy containers {old_keys} renamed, {len(dynamics)} world-model tensors and "
          f"{len(raw['actor'])} actor tensors identical, {len(state_keys)} parameter names free of the old name, TrainConfig ok; "
          f"the trainer's --resume guard refuses the raw payload and passes the converted one")


def test_real_dsrl_checkpoint():
    """One real DSRL checkpoint: the legacy sidecar is found, and the .zip weights load without its pickled attributes."""
    _, dsrl, _ = real_configs()
    for run in sorted(dsrl):
        zips = sorted((run / "checkpoints").glob("[0-9]*.zip"))
        if dsrl[run].get("method") == "dsrl_na_svg_dino" and zips:
            break
    else:
        _util.skip("no legacy DSRL checkpoint under LEGACY_RUNS")
    side = legacy.sidecar(zips[-1])
    assert side.name == zips[-1].stem + legacy.LEGACY_SIDECAR_SUFFIX and side.exists()
    try:
        from online_rl_mimicgen.baselines import dsrl_dino  # noqa: F401  (puts the stable-baselines3 fork on sys.path)
        from stable_baselines3.common.save_util import load_from_zip_file
    except (ImportError, SystemExit) as error:
        _util.skip(f"sidecar ok ({side.name}); the stable-baselines3 fork is not importable here ({str(error)[:100]})")
    data, params, _ = load_from_zip_file(str(zips[-1]), load_data=False, device="cpu")
    names = [f"{part}.{key}" for part, state in params.items() if isinstance(state, dict) for key in state]
    assert data is None and "policy" in params and not any(OLD in n.lower() for n in names)
    try:
        load_from_zip_file(str(zips[-1]), device="cpu")
        pickled = "also loads"
    except ModuleNotFoundError as error:
        pickled = f"does NOT load in this tree ({error}), which is why eval_dsrl_ckpt passes load_data=False"
    print(f"[real] {run.name[:60]}.../{zips[-1].name}: sidecar {side.name}; weights {sorted(params)} "
          f"({len(names)} tensors, none named after the method); the pickled class data {pickled}")


def test_real_dsrl_weights_load():
    """Old DSRL weights load, exact_match, into this tree's DSRLDVDDino built the way eval_dsrl_ckpt builds it.

    One DVD-arm and one Q-arm checkpoint. The simulator and the prior are replaced by an env that only carries the
    checkpoint's own observation / action spaces, so this runs on CPU; the policy_kwargs below mirror
    baselines/eval_dsrl_ckpt.py and must be kept in step with it.
    """
    _, dsrl, _ = real_configs()
    picks = {}
    for run in sorted(dsrl):
        method = dsrl[run].get("method")
        if method not in picks and method in ("dsrl_na_svg_dino", "dsrl_na_q_dino") and dsrl[run].get("image_actor"):
            zips = sorted((run / "checkpoints").glob("[0-9]*.zip"))
            if zips:
                picks[method] = zips[-1]
    if not picks:
        _util.skip("no legacy image-actor DSRL checkpoint under LEGACY_RUNS")
    try:
        from online_rl_mimicgen.baselines.dsrl_dino import DSRLDVDDino, PixelProprioSlice
        from stable_baselines3.common.save_util import load_from_zip_file
        from stable_baselines3.common.vec_env import DummyVecEnv
    except (ImportError, SystemExit) as error:
        _util.skip(f"online_rl_mimicgen.baselines.dsrl_dino is not importable here ({str(error)[:120]})")
    import base64
    import zipfile

    import cloudpickle
    import gymnasium
    import torch

    class SpacesOnly(gymnasium.Env):
        def __init__(self, observation_space, action_space):
            self.observation_space, self.action_space = observation_space, action_space

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

    for old_method, ckpt in sorted(picks.items()):
        cfg = legacy.upgrade_config(json.loads((ckpt.parents[1] / "config.json").read_text()))
        with zipfile.ZipFile(ckpt) as archive:
            raw = json.loads(archive.read("data").decode())
        env = SpacesOnly(*(cloudpickle.loads(base64.b64decode(raw[k][":serialized:"])) for k in ("observation_space", "action_space")))
        net = [cfg["layer_size"]] * cfg["num_layers"]
        model = DSRLDVDDino(
            "MlpPolicy", DummyVecEnv([lambda: env]), diffusion_policy=None, diffusion_act_dim=(env.action_space.shape[0] // 7, 7),
            noise_critic_grad_steps=cfg["noise_critic_steps"], critic_backup_combine_type="min", learning_rate=cfg["lr"],
            buffer_size=1, learning_starts=0, batch_size=cfg["batch_size"], tau=cfg["tau"],
            gamma=cfg["gamma"] ** cfg["action_steps"], train_freq=cfg["train_freq"], gradient_steps=cfg["utd"],
            ent_coef="auto", target_update_interval=1, target_entropy=cfg["target_entropy"], seed=0, device="cpu", verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=net, qf=net), log_std_init=0.0, post_linear_modules=[torch.nn.LayerNorm],
                               activation_fn={"mish": torch.nn.Mish, "tanh": torch.nn.Tanh}[cfg.get("activation", "tanh")],
                               n_critics=cfg["n_critics"], share_features_extractor=False,
                               features_extractor_class=PixelProprioSlice,
                               features_extractor_kwargs=dict(out_dim=cfg["pixel_feature_dim"], shift_pad=cfg["pixel_shift_pad"])))
        _, params, _ = load_from_zip_file(str(ckpt), load_data=False, device="cpu")
        model.set_parameters(params, exact_match=True, device="cpu")
        loaded = model.policy.state_dict()
        assert set(loaded) == set(params["policy"]) and all(torch.equal(loaded[k], params["policy"][k]) for k in loaded)
        print(f"[real] {old_method} -> {cfg['method']}: {ckpt.parents[1].name[:48]}.../{ckpt.name} obs {env.observation_space.shape} "
              f"act {env.action_space.shape}: {len(loaded)} policy tensors loaded with exact_match and equal to the file")


if __name__ == "__main__":
    _util.main(globals())
