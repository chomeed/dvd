"""configs/ + scripts/run.sh resolve every campaign cell to the configuration the reference research code resolved.

    python tests/configs/test_configs.py
    DSRL_ROOT=<dsrl checkout> LEGACY_RUNS=<dir of reference runs>[:<dir>...] python tests/configs/test_configs.py

golden/<method>/<cell>.json holds, per cell of _cells.py (12 DICE-RL, 20 DSRL), what the reference launcher and the
reference trainer produced at SEED=99 for 600k and 1M env steps: argv, run name, wandb group, the one environment
variable the launcher sets per arm, and the resolved configuration (301 TrainConfig fields for DICE-RL, 91 parser
arguments for DSRL). make_golden.py wrote them; its docstring lists the normalisations (svg -> dvd names, asset
paths as <CKPT_ROOT> / <SPOOL_ROOT> / <RUNS_ROOT> placeholders).

Always run (bash only, no assets, no GPU): the argv that scripts/run.sh would launch equals the reference argv
TOKEN FOR TOKEN, with the same run name, group and environment, for all 32 cells x 2 budgets; the default DISTILL of
configs/dice_rl/dvd.sh is the paper recipe; seeds and bad inputs behave; configs/campaign/*.txt list every campaign
cell with the campaign's three seeds and scripts/run_campaign.sh --list names them as the reference runs are named.
With this repository importable (torch; for DSRL also the SB3 fork under DSRL_ROOT): the trainers' own argument
code resolves that argv to the reference configuration KEY BY KEY, and TRAIN_ONLINE_DRYRUN=1 prints the same.
With assets under the scripts/env.sh roots (test_installed_assets): each installed prior resolves to the reference
checkpoint, every world model and V directory a campaign cell names exists, every task has its state scaler
(<demo dataset>/meta/stats.json), and a pixel cache that exists has the dataset's frame count. Three outcomes, kept
apart: FAIL lists what is MISSING or wrong; a PASS may still list what is NOT BUILT YET or not fetched for training
(pixel caches, the data/ and videos/ of a demo dataset: enough to evaluate, not to train); and with no asset at all
the test prints "NO ASSETS FOUND - nothing was checked" and SKIPs, which is not a pass.
Also always run: scripts/env.sh keeps to its own checkout whatever CODE_ROOT the shell holds; scripts/fetch_assets.py
would put every published asset exactly where configs/tasks.sh looks for it; scripts/smoke_train.sh launches the
campaign cell with only its budget changed.
With LEGACY_RUNS: every reference run directory that is a campaign cell is compared, config.json against the
configuration this repository resolves for the same task, arm, seed and budget.

The priors are the only assets argument resolution reads (the newest checkpoint directory below the prior), so the
hermetic tests build an empty stand-in tree with the reference checkpoint directory names.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rl"))
sys.path.insert(0, str(REPO / "scripts"))

import _util  # noqa: E402  (tests/rl/_util.py: Skip, main, use_new_tree, legacy_run_dirs)
from _cells import CELLS, GOLDEN_SEED, GOLDEN_STEPS, TASKS, Cell, default_variant  # noqa: E402

os.environ.setdefault("PIXEL_IMG", "224")
os.environ.setdefault("DSRL_ROOT", str(Path(os.environ.get("THIRD_PARTY", REPO / "third_party")) / "dsrl"))   # as scripts/env.sh
_util.use_new_tree()
from online_rl_mimicgen import legacy  # noqa: E402

DSRL = "online_rl_mimicgen.baselines.dsrl_dino"
ROOTS = ("ASSET_ROOT", "DATA_ROOT", "CKPT_ROOT", "CACHE_ROOT", "SPOOL_ROOT", "RUNS_ROOT")
PLACEHOLDERS = ("CKPT_ROOT", "SPOOL_ROOT", "RUNS_ROOT")
# variables that would change what a config resolves to if they leaked in from the caller's shell
SCRUB = ("DISTILL", "WANDB_RUN_GROUP", "PYTORCH_CUDA_ALLOC_CONF", "G", "TRAIN_ONLINE_DRYRUN", "RUN_PRINT_ARGV", "PYTHONPATH")


def golden(cell: Cell) -> dict:
    return json.loads((HERE / cell.golden).read_text())


def expected(cell: Cell, steps: int) -> dict:
    """The reference name / argv / config of a cell at 600k (as stored) or 1M (the stored differences applied)."""
    data = golden(cell)
    out = {"name": data["name"], "group": data["group"], "module": data["module"], "env": data["env"],
           "argv": list(data["argv"]), "config": dict(data["config"])}
    if steps != data["source"]["steps"]:
        other = data[f"steps_{steps}"]
        out["name"] = other["name"]
        out["config"].update(other["config"])
        for index, token in other["argv"]:
            out["argv"][index] = token
    if cell.method == "dsrl":                    # written to config.json by main(), not a parser argument
        out["config"]["method"] = "dsrl_na_q_dino" if out["config"]["q_critic"] else "dsrl_na_dvd_dino"
    return out


class Sandbox:
    """Stand-in asset roots (empty priors with the reference checkpoint directory names) and a clean environment."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="test_configs_")
        self.root = Path(self._tmp.name).resolve()
        self.roots = {"ASSET_ROOT": self.root, "DATA_ROOT": self.root / "datasets", "CKPT_ROOT": self.root / "checkpoints",
                      "CACHE_ROOT": self.root / "cache", "SPOOL_ROOT": self.root / "spool", "RUNS_ROOT": self.root / "runs"}
        for cell in CELLS:
            if cell.method == "dice_rl":
                checkpoint = golden(cell)["config"]["base_checkpoint"].replace("<CKPT_ROOT>", str(self.roots["CKPT_ROOT"]))
                (Path(checkpoint) / "config.json").parent.mkdir(parents=True, exist_ok=True)
                (Path(checkpoint) / "config.json").write_text("{}\n")
        self.project = golden(CELLS[0])["config"]["wandb_project"]

    def env(self, **extra: str) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in SCRUB and key not in ROOTS}
        env.update({key: str(value) for key, value in self.roots.items()})
        env.update(WANDB_PROJECT=self.project, WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1",
                   PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
        env.update(extra)
        return env

    def run(self, cell: Cell, steps: int, *, seed: int = GOLDEN_SEED, variant: str | None = None, task: str | None = None,
            **extra: str) -> subprocess.CompletedProcess:
        variant = cell.variant if variant is None else variant
        if variant:
            extra = {"DISTILL": variant, **extra}
        return subprocess.run(["bash", str(REPO / "scripts" / "run.sh"), str(REPO / cell.config_path), task or cell.task,
                               str(seed), str(steps)], env=self.env(**extra), capture_output=True, text=True, cwd=str(REPO))

    def launch(self, cell: Cell, steps: int, **kwargs) -> dict:
        """What scripts/run.sh would start for a cell: module, name, group, environment, argv (raw tokens)."""
        done = self.run(cell, steps, RUN_PRINT_ARGV="1", **kwargs)
        assert done.returncode == 0, f"{cell.id}: scripts/run.sh exited {done.returncode}: {done.stderr.strip()}"
        lines = done.stdout.split("\n")[:-1]
        meta = dict(line[1:].split("=", 1) for line in lines if line.startswith("@"))
        key, _, value = meta.pop("ENV").partition("=")
        return {"module": meta["MODULE"], "name": meta["NAME"], "group": meta["GROUP"], "env": {key: value or None},
                "argv": [line for line in lines if not line.startswith("@")]}

    def placeholders(self, value):
        """A resolved value with this sandbox's roots written as the golden files write them."""
        if isinstance(value, str):
            for key in PLACEHOLDERS:
                value = value.replace(str(self.roots[key]), f"<{key}>")
        return value

    def close(self) -> None:
        self._tmp.cleanup()


def differences(got: dict, want: dict) -> list[str]:
    out = []
    for key in sorted(set(got) | set(want)):
        if key not in want:
            out.append(f"{key}: resolved here ({got[key]!r}) but not a reference key")
        elif key not in got:
            out.append(f"{key}: reference {want[key]!r}, missing here")
        elif got[key] != want[key]:
            out.append(f"{key}: reference {want[key]!r}, here {got[key]!r}")
    return out


def resolve(module: str, argv: list[str]) -> dict:
    """The trainer's own argument resolution (scripts/resolve_config.py); SKIP when the trainer cannot be imported."""
    import resolve_config
    if module == DSRL and not (Path(os.environ["DSRL_ROOT"]) / "stable-baselines3" / "stable_baselines3" / "dsrl").is_dir():
        _util.skip(f"DSRL_ROOT={os.environ['DSRL_ROOT']} holds no stable-baselines3 fork (environment.md)")
    if importlib.util.find_spec("torch") is None:
        _util.skip("the main environment is not installed (no torch)")
    return resolve_config.resolve(module, argv)


def test_golden_files():
    """One golden file per cell and nothing else; each is tied to the committed launcher snapshot and holds no path."""
    files = {str(path.relative_to(HERE)) for path in (HERE / "golden").rglob("*.json")}
    assert files == {cell.golden for cell in CELLS}, sorted(files ^ {cell.golden for cell in CELLS})
    for cell in CELLS:
        text = (HERE / cell.golden).read_text()
        data = json.loads(text)
        source = data["source"]
        assert data["cell"] == {"method": cell.method, "config": cell.config_path, "task": cell.task,
                                "variant": cell.variant, "campaign": cell.campaign}, cell.id
        assert (source["launcher"], source["arm"], source["seed"], source["steps"]) == \
               (cell.launcher, cell.live_arm, GOLDEN_SEED, GOLDEN_STEPS[0]), cell.id
        snapshot = REPO / "reference" / "launchers" / cell.launcher
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source["launcher_sha256"], \
            f"{cell.id}: reference/launchers/{cell.launcher} is not the launcher this golden file was generated from"
        assert data["name"] == source["name"].replace(legacy.OLD, legacy.NEW), cell.id
        assert f"steps_{GOLDEN_STEPS[1]}" in data, cell.id
        assert sorted(data[f"steps_{GOLDEN_STEPS[1]}"]["config"]) == ["output_dir", "total_env_steps", "wandb_name"], cell.id
        strings = [v for v in list(data["config"].values()) + data["argv"] if isinstance(v, str)]
        assert not [v for v in strings if v.startswith("/") or legacy.OLD in v.lower()], cell.id
    print(f"    {len(CELLS)} golden files: {sum(c.method == 'dice_rl' for c in CELLS)} DICE-RL, "
          f"{sum(c.method == 'dsrl' for c in CELLS)} DSRL ({sum(c.campaign for c in CELLS)} cells ran in the campaign)")


def test_argv_token_for_token():
    """scripts/run.sh would launch exactly the reference argv, module, run name, wandb group and environment."""
    box = Sandbox()
    try:
        tokens = 0
        for cell in CELLS:
            for steps in GOLDEN_STEPS:
                want, got = expected(cell, steps), box.launch(cell, steps)
                got["argv"] = [box.placeholders(token) for token in got["argv"]]
                for field in ("module", "name", "group", "env"):
                    assert got[field] == want[field], f"{cell.id} @{steps}: {field} {got[field]!r}, reference {want[field]!r}"
                bad = [(i, a, b) for i, (a, b) in enumerate(zip(got["argv"], want["argv"])) if a != b]
                assert len(got["argv"]) == len(want["argv"]) and not bad, \
                    f"{cell.id} @{steps}: {len(got['argv'])} tokens vs {len(want['argv'])}; first differences {bad[:3]}"
                tokens += len(got["argv"])
        print(f"    {len(CELLS)} cells x {len(GOLDEN_STEPS)} budgets: {tokens} argv tokens identical")
    finally:
        box.close()


def test_default_variant_is_the_paper_recipe():
    """configs/dice_rl/dvd.sh without DISTILL is the reference ARM=dvd of each task; the other value is its A/B twin."""
    box = Sandbox()
    try:
        for cell in CELLS:
            if cell.config != "dvd" or cell.method != "dice_rl":
                continue
            paper = cell.variant == default_variant(cell.task)
            assert (cell.live_arm == "dvd") == paper, cell.id
            if paper:
                assert box.launch(cell, GOLDEN_STEPS[0], variant="") == box.launch(cell, GOLDEN_STEPS[0]), cell.id
            assert golden(cell)["group"] == f"{cell.task}_dicerl_dvd" + ("" if paper else f"_{cell.variant}"), cell.id
    finally:
        box.close()


def test_seed_task_and_bad_inputs():
    """SEED only changes the seed and the run name; unknown tasks, variants and misplaced DISTILL are refused."""
    box = Sandbox()
    try:
        for cell in (CELLS[0], next(c for c in CELLS if c.method == "dsrl")):
            a, b = (([box.placeholders(token) for token in box.launch(cell, GOLDEN_STEPS[0], seed=seed)["argv"]])
                    for seed in (GOLDEN_SEED, 43))
            changed = [(x, y) for x, y in zip(a, b) if x != y]
            assert len(a) == len(b) and len(changed) == 3, (cell.id, changed)
            assert all(x.replace(f"_s{GOLDEN_SEED}", "_s43") == y or (x, y) == (str(GOLDEN_SEED), "43") for x, y in changed), changed
        dvd = next(c for c in CELLS if c.method == "dice_rl" and c.config == "dvd")
        q = next(c for c in CELLS if c.method == "dice_rl" and c.config == "q")
        ours = next(c for c in CELLS if c.method == "dsrl" and c.config == "ours")
        refused = [box.run(dvd, 600000, task="square_d2", RUN_PRINT_ARGV="1"),
                   box.run(dvd, 600000, variant="distill", RUN_PRINT_ARGV="1"),
                   box.run(q, 600000, variant="tdonly", RUN_PRINT_ARGV="1"),
                   box.run(ours, 600000, variant="tddistill", RUN_PRINT_ARGV="1"),
                   box.run(ours, 600000, seed="43x", RUN_PRINT_ARGV="1")]
        assert [done.returncode for done in refused] == [2] * len(refused), [(d.returncode, d.stderr) for d in refused]
        assert all(done.stdout == "" and done.stderr.strip() for done in refused)
    finally:
        box.close()


def _campaign_lines(method: str) -> list[tuple[Cell, int, int]]:
    """configs/campaign/<method>.txt as (cell, seed, steps), one entry per run."""
    by_key = {(c.config_path, c.task, c.variant): c for c in CELLS}
    out = []
    for line in (REPO / "configs" / "campaign" / f"{method}.txt").read_text().splitlines():
        fields = line.split("#")[0].split()
        if not fields:
            continue
        config, task, seed, steps, *extra = fields
        assert all(item.startswith("DISTILL=") for item in extra), line
        variant = extra[0].split("=", 1)[1] if extra else (default_variant(task) if config.endswith("dice_rl/dvd.sh") else "")
        assert (config, task, variant) in by_key, f"configs/campaign/{method}.txt names no cell of _cells.py: {line}"
        out.append((by_key[(config, task, variant)], int(seed), int(steps)))
    return out


def test_campaign_lists():
    """configs/campaign/*.txt = every campaign cell x (seed 43 @600k, 44 @600k, 45 @1M), with the reference run names."""
    plan = {(43, 600_000), (44, 600_000), (45, 1_000_000)}
    box = Sandbox()
    try:
        for method in ("dice_rl", "dsrl"):
            lines = _campaign_lines(method)
            cells = [c for c in CELLS if c.method == method and c.campaign]
            assert len(lines) == len(set(lines)) == 3 * len(cells), (method, len(lines), len(cells))
            for cell in cells:
                assert {(seed, steps) for c, seed, steps in lines if c == cell} == plan, cell.id
            listed = subprocess.run(["bash", str(REPO / "scripts" / "run_campaign.sh"), str(REPO / "configs" / "campaign" / f"{method}.txt"),
                                     "0", "--list"], env=box.env(DISTILL="tddistill"), capture_output=True, text=True, cwd=str(REPO))
            assert listed.returncode == 0, listed.stdout[-2000:] + listed.stderr[-2000:]
            rows = [line.split()[2:] for line in listed.stdout.splitlines() if " run " in line]
            want = [expected(cell, steps)["name"].replace(f"_s{GOLDEN_SEED}", f"_s{seed}") for cell, seed, steps in lines]
            assert [row[1] for row in rows] == want, f"{method}: scripts/run_campaign.sh --list names differ from the reference names"
            distils = sum("--value-distill-robometer" in expected(c, s)["argv"] and "--no-value-distill" not in expected(c, s)["argv"]
                          for c, _, s in lines)
            assert sum(len(row) > 2 for row in rows) == distils, (method, distils)
            print(f"    configs/campaign/{method}.txt: {len(lines)} runs = {len(cells)} cells x 3 seeds, {distils} need a teacher")
    finally:
        box.close()


def _check_resolved(method: str) -> None:
    box = Sandbox()
    try:
        keys = set()
        for cell in CELLS:
            if cell.method != method:
                continue
            for steps in GOLDEN_STEPS:
                want, launch = expected(cell, steps), box.launch(cell, steps)
                got = {key: box.placeholders(value) for key, value in resolve(launch["module"], launch["argv"]).items()}
                bad = differences(got, want["config"])
                assert not bad, f"{cell.id} @{steps}: {len(bad)} of {len(want['config'])} keys differ:\n  " + "\n  ".join(bad)
                keys.add(len(got))
        print(f"    {sum(c.method == method for c in CELLS)} cells x {len(GOLDEN_STEPS)} budgets: every one of {sorted(keys)} "
              f"resolved keys equals the reference")
    finally:
        box.close()


def test_resolved_config_dice_rl():
    """train_online.parse_args() on the cell's argv == the reference TrainConfig, all 301 fields."""
    _check_resolved("dice_rl")


def test_resolved_config_dsrl():
    """dsrl_dino.build_dino_parser() on the cell's argv == the reference parser result, all arguments."""
    _check_resolved("dsrl")


def test_dry_run_prints_the_same():
    """TRAIN_ONLINE_DRYRUN=1 bash scripts/run.sh ... prints the resolved configuration and creates nothing."""
    box = Sandbox()
    try:
        picks = [next(c for c in CELLS if c.method == m and c.config == a) for m, a in
                 (("dice_rl", "dvd"), ("dice_rl", "q"), ("dsrl", "oursdist"), ("dsrl", "q"))]
        checked, skipped = [], None
        for cell in picks:
            launch = box.launch(cell, GOLDEN_STEPS[0])
            try:
                want = resolve(launch["module"], launch["argv"])
            except _util.Skip as reason:        # the DSRL trainer needs the SB3 fork; the DICE-RL cells are still checked
                skipped = reason
                continue
            done = box.run(cell, GOLDEN_STEPS[0], TRAIN_ONLINE_DRYRUN="1")
            lines = [line for line in done.stdout.splitlines() if line.startswith("DRYRUN_CONFIG ")]
            assert done.returncode == 0 and len(lines) == 1, f"{cell.id}: exit {done.returncode}\n{done.stderr[-2000:]}"
            assert json.loads(lines[0][len("DRYRUN_CONFIG "):]) == want, cell.id
            checked.append(f"{cell.method}/{cell.id}")
        assert not box.roots["RUNS_ROOT"].exists(), "a dry run created the runs directory"
        if not checked:
            raise skipped
        print(f"    dry run == in-process resolution for {', '.join(checked)}" + (f"; not checked: {skipped}" if skipped else ""))
    finally:
        box.close()


def _installed_roots() -> dict[str, Path]:
    assets = Path(os.environ.get("ASSET_ROOT", REPO / "assets"))
    return {"DATA_ROOT": Path(os.environ.get("DATA_ROOT", assets / "datasets")),
            "CKPT_ROOT": Path(os.environ.get("CKPT_ROOT", assets / "checkpoints")),
            "CACHE_ROOT": Path(os.environ.get("CACHE_ROOT", assets / "cache")),
            "SPOOL_ROOT": Path(os.environ.get("SPOOL_ROOT", assets / "spool")),
            "RUNS_ROOT": Path(os.environ.get("RUNS_ROOT", assets / "runs"))}


def _tasks_table(task: str, roots: dict[str, Path]) -> dict[str, str]:
    """The variables configs/tasks.sh sets for a task under the given roots (the file is sourced by bash, as the configs do)."""
    names = ("PRIOR", "PRIOR_DIR", "WM_BASE", "WM_CONT", "V_PV4", "V_DSRL", "V_MC", "SPOOL", "DEMO_DATASET", "PIXEL_CACHE_FILE")
    script = f'source "{REPO}/configs/tasks.sh" && printf "%s\\n" ' + " ".join(f'"${name}"' for name in names)
    env = {"PATH": os.environ.get("PATH", ""), "TASK": task, **{key: str(value) for key, value in roots.items()}}
    done = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == 0, f"configs/tasks.sh for {task}: {done.stderr.strip()}"
    return dict(zip(names, done.stdout.split("\n")))


def test_installed_assets():
    """On a machine with assets: everything a campaign cell loads is in place. See the module docstring for the outcomes."""
    roots = _installed_roots()
    tables = {task: _tasks_table(task, roots) for task in TASKS}
    anything = ([path for sub in ("priors", "wm", "value") for path in (roots["CKPT_ROOT"] / sub).glob("*")]
                + [Path(t["DEMO_DATASET"]) for t in tables.values() if Path(t["DEMO_DATASET"]).exists()]
                + [Path(t["PIXEL_CACHE_FILE"]) for t in tables.values() if Path(t["PIXEL_CACHE_FILE"]).exists()])
    if not anything:
        _util.skip(f"NO ASSETS FOUND - nothing was checked. No prior, world model or V under {roots['CKPT_ROOT']} and no demo "
                   f"dataset under {roots['DATA_ROOT']}: fetch them (python scripts/fetch_assets.py --tier eval --dry-run) or set "
                   f"ASSET_ROOT / CKPT_ROOT / DATA_ROOT / CACHE_ROOT as for scripts/env.sh")
    from online_rl_mimicgen.common import find_latest_checkpoint
    missing, later, checked = [], [], 0
    for cell in CELLS:
        if not cell.campaign:
            continue
        config = golden(cell)["config"]
        prior = Path((config.get("base_output") or config["base_checkpoint"].split("/checkpoints/")[0])
                     .replace("<CKPT_ROOT>", str(roots["CKPT_ROOT"])))
        if cell.method == "dice_rl":
            want = Path(config["base_checkpoint"].replace("<CKPT_ROOT>", str(roots["CKPT_ROOT"])))
            try:
                got = find_latest_checkpoint(prior)
                if got != want.resolve():
                    missing.append(f"{cell.id}: prior resolves to {got}, the reference used {want.relative_to(prior)}")
            except FileNotFoundError as error:
                missing.append(f"{cell.id}: {error}")
        elif not (prior / "checkpoints").is_dir():
            missing.append(f"{cell.id}: no prior at {prior}")
        for key in ("value_model", "dino_wm_ckpt"):
            if config.get(key):
                checked += 1
                path = Path(config[key].replace("<CKPT_ROOT>", str(roots["CKPT_ROOT"])))
                if not path.exists():
                    missing.append(f"{cell.id}: {key} {path} does not exist")
    for task, table in tables.items():           # per task: the state scaler, the rest of the demo dataset, the pixel cache
        dataset, cache = Path(table["DEMO_DATASET"]), Path(table["PIXEL_CACHE_FILE"])
        checked += 2
        if not (dataset / "meta" / "stats.json").is_file():
            missing.append(f"{task}: no state scaler, {dataset}/meta/stats.json does not exist "
                           f"(python scripts/fetch_assets.py --tier eval --task {task} --only dataset)")
        elif not ((dataset / "data").is_dir() and (dataset / "videos").is_dir()):
            later.append(f"{task}: the demo dataset holds meta/ only, enough to evaluate; training needs all of it "
                         f"(python scripts/fetch_assets.py --tier train --task {task} --only dataset)")
        if not cache.exists():
            later.append(f"{task}: pixel cache NOT BUILT YET, every RL cell needs it (bash scripts/build_pixel_cache.sh {task})")
        elif (dataset / "meta" / "info.json").is_file():
            import numpy
            frames = json.loads((dataset / "meta" / "info.json").read_text())["total_frames"]
            shape = numpy.load(cache, mmap_mode="r").shape        # reads the header only
            if shape != (frames, 2, 224, 224, 3):
                missing.append(f"{task}: pixel cache {cache} has shape {shape}, the dataset needs ({frames}, 2, 224, 224, 3): "
                               f"it was built from another dataset; remove it and rebuild")
    for line in later:
        print(f"    NOT READY FOR TRAINING  {line}")
    hints = []                                   # the two assets no Hub repository holds (environment.md, section 5)
    if any("square_d1_45k_statefix" in line for line in missing):
        hints.append("the fixed square_d1 prior is made locally: python scripts/make_square_d1_statefix.py")
    if any("v_dino_mc_return_" in line for line in missing):
        hints.append("the Monte Carlo V directories (DSRL oursmc) are unpublished: TASK=<task> RECIPE=mc bash value/fit_v.sh "
                     "refits an equivalent one")
    assert not missing, (f"{len(missing)} asset(s) MISSING or wrong:\n  " + "\n  ".join(missing)
                         + "".join(f"\n  hint: {hint}" for hint in hints))
    print(f"    priors, world models and V directories of all {sum(c.campaign for c in CELLS)} campaign cells, and the state "
          f"scaler of all {len(TASKS)} tasks, are in place under CKPT_ROOT={roots['CKPT_ROOT']} and DATA_ROOT={roots['DATA_ROOT']} "
          f"({checked} files / directories); "
          + (f"{len(later)} item(s) still to prepare before training, listed above" if later else
             "datasets and pixel caches are complete: ready to train"))


def test_env_sh_keeps_its_own_checkout():
    """scripts/env.sh sets CODE_ROOT to the checkout it lies in whatever the shell holds, says so once, and puts this
    checkout's packages first on PYTHONPATH; scripts/run.sh then launches from here."""
    other = tempfile.mkdtemp(prefix="other_checkout_")
    try:
        env = {"PATH": os.environ.get("PATH", ""), "CODE_ROOT": other, "PYTHONPATH": f"{other}/rl:/somewhere/else"}
        done = subprocess.run(["bash", "-c", f'source "{REPO}/scripts/env.sh" && printf "%s\\n" "$CODE_ROOT" "$PYTHONPATH" '
                               f'"$ASSET_ROOT" && source "{REPO}/scripts/env.sh" && printf "%s\\n" "$PYTHONPATH"'],
                              env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        code_root, first, assets, second = done.stdout.split("\n")[:4]
        assert Path(code_root).resolve() == REPO and Path(assets).resolve() == REPO / "assets", (code_root, assets)
        assert first.split(":")[0] == f"{code_root}/rl" and first.endswith(f"{other}/rl:/somewhere/else"), first
        assert second == first, "sourcing scripts/env.sh twice grew PYTHONPATH"
        warnings = [line for line in done.stderr.splitlines() if "CODE_ROOT" in line]
        assert len(warnings) == 1 and other in warnings[0], done.stderr        # once: the second source finds it right
        box = Sandbox()
        try:
            cell = CELLS[0]
            a = box.launch(cell, GOLDEN_STEPS[0])
            b = box.launch(cell, GOLDEN_STEPS[0], CODE_ROOT=other)
            assert a == b, "a CODE_ROOT preset in the shell changed what scripts/run.sh launches"
        finally:
            box.close()
        print("    CODE_ROOT preset to another directory: replaced with one warning; this checkout's rl/ is first on PYTHONPATH")
    finally:
        os.rmdir(other)


def test_fetch_table_matches_configs():
    """scripts/fetch_assets.py targets exactly the paths configs/tasks.sh, the golden configs, robometer/teachers.sh and
    dinowm/common_ws.py name. No network: only the table is read."""
    import re
    import fetch_assets
    box = Sandbox()
    try:
        roots = {key: box.roots[key] for key in ("DATA_ROOT", "CKPT_ROOT", "CACHE_ROOT", "SPOOL_ROOT")}
        items = fetch_assets.table()
        by_target = {}
        for item in items:
            assert item.kind in ("repo", "folder", "file", "hf-cache", "derive", "build", "unpublished"), item
            assert item.group in fetch_assets.GROUPS and set(item.tiers) <= set(fetch_assets.TIERS), item
            assert ".." not in Path(item.target).parts and not item.target.startswith("/"), item
            if item.kind in ("repo", "folder", "file", "hf-cache"):
                assert re.fullmatch(r"[0-9a-f]{40}", item.revision), f"{item.what}: no recorded revision"
            if item.root != "HF_CACHE" and item.target:
                by_target.setdefault(str(box.roots[item.root] / item.target), []).append(item)
        rollouts = dict(re.findall(r'"(\w+)": "(mimicgen_\w+_mtdit_flow_\w+)"', (REPO / "dinowm" / "common_ws.py").read_text()))
        assert set(rollouts) == set(TASKS), rollouts
        for task in TASKS:
            table = _tasks_table(task, roots)
            step = fetch_assets.TASKS[task]["step"]
            for cell in CELLS:                   # the prior checkpoint directory the reference configs resolved
                if cell.task == task and cell.method == "dice_rl":
                    want = golden(cell)["config"]["base_checkpoint"].replace("<CKPT_ROOT>", str(roots["CKPT_ROOT"]))
                    assert want == f"{table['PRIOR_DIR']}/checkpoints/{step}/pretrained_model", (cell.id, want)
            expect = {f"{table['PRIOR_DIR']}/checkpoints/{step}/pretrained_model": ("derive",) if task == "square_d1" else ("repo",),
                      str(Path(table["WM_BASE"]).parent): ("folder",), str(Path(table["WM_CONT"]).parent): ("folder",),
                      table["V_PV4"]: ("folder",), table["V_DSRL"]: ("folder",), table["V_MC"]: ("unpublished",),
                      table["DEMO_DATASET"]: ("repo",), table["DEMO_DATASET"] + "/meta": ("folder",),
                      table["PIXEL_CACHE_FILE"]: ("build",),
                      f"{roots['DATA_ROOT']}/{rollouts[task]}_success": ("repo",),
                      f"{roots['DATA_ROOT']}/{rollouts[task]}_failure": ("repo",),
                      f"{roots['CKPT_ROOT']}/rm_scores/{task}/predictions.parquet": ("file",)}
            for target, kinds in expect.items():
                got = tuple(item.kind for item in by_target.get(target, []))
                assert got == kinds, f"{task}: {target} is the target of {got or 'no item'}, expected {kinds}"
            for name in ("WM_BASE", "WM_CONT"):  # the file the trainer opens is the file the folder holds
                item = by_target[str(Path(table[name]).parent)][0]
                assert Path(table[name]).name == f"wm_dinov2s14_{item.folder}.pt", (table[name], item.folder)
            lora = subprocess.run(["bash", "-c", f'source "{REPO}/robometer/teachers.sh" && teacher {task} && echo "$TEACHER_LORA"'],
                                  capture_output=True, text=True).stdout.strip()
            assert tuple(i.repo for i in by_target[f"{roots['CKPT_ROOT']}/robometer/{lora}"]) == (lora,), (task, lora)
        seg = subprocess.run(["bash", "-c", f'source "{REPO}/robometer/teachers.sh" && teacher square_d1_seg && echo "$TEACHER_LORA"'],
                             capture_output=True, text=True).stdout.strip()
        assert seg == fetch_assets.SEG_LORA["name"] and f"{roots['CKPT_ROOT']}/robometer/{seg}" in by_target
        for cell in CELLS:                       # every asset a campaign cell names lies below a fetch target
            if not cell.campaign:
                continue
            for key in ("value_model", "dino_wm_ckpt"):
                value = golden(cell)["config"].get(key)
                if value:
                    path = Path(value.replace("<CKPT_ROOT>", str(roots["CKPT_ROOT"])))
                    assert str(path) in by_target or str(path.parent) in by_target, f"{cell.id}: no fetch item for {key} {value}"
        for tier, more in zip(fetch_assets.TIERS, fetch_assets.TIERS[1:]):
            smaller = {(i.root, i.target) for i in fetch_assets.select(tier, list(TASKS), set(), set()) if i.what != "demo dataset, meta/ only"}
            assert smaller <= {(i.root, i.target) for i in fetch_assets.select(more, list(TASKS), set(), set())}, (tier, more)
        saved = os.environ.pop("HF_NAMESPACE", None)
        try:
            assert re.fullmatch(r"[\w.-]+", fetch_assets.default_namespace())
        finally:
            if saved is not None:
                os.environ["HF_NAMESPACE"] = saved
        counts = {kind: sum(i.kind == kind for i in items) for kind in ("repo", "folder", "file", "hf-cache", "derive", "build", "unpublished")}
        print(f"    {len(items)} items ({', '.join(f'{n} {k}' for k, n in counts.items())}): every path of configs/tasks.sh is the "
              f"target of exactly one of them per tier")
    finally:
        box.close()


def test_smoke_argv_is_the_cell_plus_budget():
    """scripts/smoke_train.sh would launch the campaign cell's argv (seed 7, 2000 env steps, run directory under
    $RUNS_ROOT/smoke) followed by its budget flags and nothing else."""
    box = Sandbox()
    try:
        tails = {"dice_rl": ["--save-freq", "1000"],
                 "dsrl": ["--save-freq", "1000", "--init-rollout-chunks", "32", "--async-eval-episodes", "3", "--async-eval-workers", "1"]}
        checked = 0
        for cell in CELLS:
            cell_launch = box.launch(cell, 2000, seed=7)
            extra = {"DISTILL": cell.variant} if cell.variant else {}
            done = subprocess.run(["bash", str(REPO / "scripts" / "smoke_train.sh"), cell.method, cell.config, cell.task],
                                  env=box.env(RUN_PRINT_ARGV="1", **extra), capture_output=True, text=True, cwd=str(REPO))
            assert done.returncode == 0, f"{cell.id}: scripts/smoke_train.sh exited {done.returncode}: {done.stderr.strip()}"
            lines = done.stdout.split("\n")[:-1]
            meta = dict(line[1:].split("=", 1) for line in lines if line.startswith("@"))
            argv = [line for line in lines if not line.startswith("@")]
            tail = tails[cell.method]
            smoke_root = str(box.roots["RUNS_ROOT"] / "smoke")
            want = [token.replace(str(box.roots["RUNS_ROOT"]), smoke_root) for token in cell_launch["argv"]]
            assert argv[:-len(tail)] == want and argv[-len(tail):] == tail, f"{cell.id}: the smoke argv is not the cell's plus {tail}"
            assert meta["MODULE"] == cell_launch["module"] and meta["NAME"] == cell_launch["name"], cell.id
            assert meta["RUN"] == f"{smoke_root}/{meta['NAME']}" and "_2k_s7" in meta["NAME"], meta
            distils = "--value-distill-robometer" in argv and "--no-value-distill" not in argv
            assert meta["DISTILS"] == ("yes" if distils else "no"), cell.id
            checked += 1
        assert not box.roots["RUNS_ROOT"].exists(), "printing the smoke argv created the runs directory"
        print(f"    {checked} cells: smoke argv == the cell's argv at seed 7 / 2000 steps under RUNS_ROOT/smoke, plus the budget flags")
    finally:
        box.close()


def _cell_of_run(config: dict) -> tuple[str, str, str] | None:
    """(method, config file, variant) of a reference run, read from its upgraded config.json (never from its name)."""
    method = config.get("method")
    if method == "dice_rl":
        if not float(config.get("dvd_weight") or 0.0):
            return "dice_rl", "q", ""
        return "dice_rl", "dvd", "tddistill" if config.get("value_distill_robometer") else "tdonly"
    if method == "dsrl_na_q_dino":
        return "dsrl", "q", ""
    if method == "dsrl_na_dvd_dino":
        if config.get("value_scratch"):
            return "dsrl", "randv", ""
        if "mc_return" in str(config.get("value_model")):
            return "dsrl", "oursmc", ""
        return "dsrl", ("ours" if config.get("no_value_distill") else "oursdist"), ""
    return None


def test_reference_runs():
    """Every reference run directory that is a campaign cell: its config.json == what this repository resolves for it."""
    runs = _util.legacy_run_dirs()
    if not runs:
        _util.skip("LEGACY_RUNS is not set (directories holding run directories of the reference campaign)")
    from make_golden import PATH_KEYS, normalise_path
    cells = {(c.method, c.config, c.variant, c.task): c for c in CELLS}
    box = Sandbox()
    try:
        matched, other, failures = [], 0, []
        for run in runs:
            raw = _util.read_config(run)
            if raw is None or raw.get("task") not in TASKS:
                other += 1
                continue
            try:
                config = legacy.upgrade_config(raw)
            except ValueError:                  # an older run with fields this repository does not have
                other += 1
                continue
            kind = _cell_of_run(config)
            cell = cells.get((*kind, config["task"])) if kind else None
            if cell is None:
                other += 1
                continue
            steps, seed = int(config["total_env_steps"]), int(config["seed"])
            launch = box.launch(cell, steps, seed=seed)
            if launch["name"] != run.name.replace(legacy.OLD, legacy.NEW) or config.get("wandb_project") != box.project:
                other += 1                      # same arm, but not a run of this campaign (another recipe or era)
                continue
            got = {key: box.placeholders(value) for key, value in resolve(launch["module"], launch["argv"]).items()}
            want, names, notes = dict(config), (run.name, launch["name"]), {"spool": None, "wm_layout": None}
            for key in PATH_KEYS:
                if want.get(key):
                    want[key] = normalise_path(key, want[key], cell.task, names, notes)
            want["wandb_name"] = str(want["wandb_name"]).replace(legacy.OLD, legacy.NEW)
            bad = differences(got, want)
            source = golden(cell)["source"]     # the run used the spool directory and world-model file the launcher names
            bad += [f"{key}: the run used {notes[key]!r}, the launcher {source[key]!r}"
                    for key in notes if notes[key] != source[key]]
            matched.append(f"{cell.method}/{cell.id} s{seed} {steps // 1000}k")
            if bad:
                failures.append(f"{run.name}: {len(bad)} of {len(want)} keys differ:\n    " + "\n    ".join(bad))
        print(f"    {len(matched)} reference runs are campaign cells ({len(set(m.split(' ')[0] for m in matched))} distinct "
              f"cells); {other} other run directories ignored")
        assert not failures, f"{len(failures)} run(s) differ:\n  " + "\n  ".join(failures)
        assert matched, "LEGACY_RUNS holds no run of the campaign"
    finally:
        box.close()


if __name__ == "__main__":
    _util.main(globals())
