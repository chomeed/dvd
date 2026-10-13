#!/usr/bin/env python3
"""Success-rate table of a set of runs, per cell (method x arm x task) and seed, with the evaluation protocol named.

    python scripts/results_table.py [RUNS_DIR ...] [--bon16-dir DIR] [--runs] [--json FILE]

    RUNS_DIR      directories holding run directories (default: $RUNS_ROOT, as scripts/env.sh sets it)
    --bon16-dir   also read DICE-RL evals stored as <DIR>/<run name>_<step>.json (the reference machine's layout);
                  this repository writes them to <run>/evals/<step>.json (scripts/eval_loop.sh)
    --runs        add one line per run (number of evals, last evaluated step, spread inside the window)
    --json FILE   write the per-run numbers as JSON

Two protocols, never to be mixed in one comparison; both are 100 episodes on env seeds 10000-10099:

    DICE-RL   best of 16 prior draws ranked by the run's own critic, deterministic residual; every 50k checkpoint,
              evaluated after the fact (eval/eval_bon16.py)
    DSRL      ONE stochastic noise draw per chunk, no best-of-N; every 25k checkpoint, evaluated during the run
              (<run>/evals/<step>.json, mirrored as kind=eval rows of metrics.jsonl)

Per run:   win   mean of the evals from 350k to 600k env steps (6 for DICE-RL, 11 for DSRL)
           600k  the eval of the 600k checkpoint
           and for 1M runs: the eval of the 1M checkpoint and the mean of the evals from 650k to 1M
Per cell:  the mean of `win` and of `600k` over the runs that have reached 600k (n is printed).

A cell is identified from each run's config.json (method, critic, V initialisation, distillation), never from its
name; runs of the pre-rename research code are read through online_rl_mimicgen/legacy.py. Only evals of exactly
100 episodes on seeds 10000+ count (DICE-RL: strategy bon16); others are reported in the `bad` column of --runs.
What a single number is worth: one 100-episode eval has a binomial standard error of 3 to 5 points; across the three
seeds of a reference DSRL cell the `win` values span 3 points in the median and up to 9, single 600k evals 8.5 in
the median and up to 21. Compare window means, not single checkpoints or peaks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from online_rl_mimicgen import legacy  # noqa: E402

TASKS = ("square_d0", "square_d1", "threading_d0", "hammer_cleanup_d1")
WINDOW, LATE, AT600 = (345_000, 605_000), (645_000, 1_005_000), (595_000, 605_000)
PROTOCOL = {
    "DICE-RL": "best of 16 prior draws ranked by the run's critic, deterministic residual; 100 episodes, env seeds "
               "10000-10099; every 50k checkpoint",
    "DSRL": "one noise draw per chunk, no best-of-N; 100 episodes, env seeds 10000-10099; every 25k checkpoint",
}
ARMS = {"DICE-RL": ("DVD, TD only", "DVD, TD + distill", "Q"),
        "DSRL": ("DVD, TD only", "DVD, TD + distill", "DVD, MC-return V", "DVD, random V", "Q")}


def cell_of(config: dict) -> tuple[str, str] | None:
    """(method, arm) of a run, from its config."""
    method = config.get("method")
    if method == "dice_rl":
        if not float(config.get("dvd_weight") or 0.0):
            return "DICE-RL", "Q"
        return "DICE-RL", "DVD, TD + distill" if config.get("value_distill_robometer") else "DVD, TD only"
    if method == "dsrl_na_q_dino":
        return "DSRL", "Q"
    if method == "dsrl_na_dvd_dino":
        if config.get("value_scratch"):
            return "DSRL", "DVD, random V"
        if "mc_return" in str(config.get("value_model")):
            return "DSRL", "DVD, MC-return V"
        return "DSRL", "DVD, TD only" if config.get("no_value_distill") else "DVD, TD + distill"
    return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def dsrl_evals(run: Path) -> tuple[dict[int, float], int]:
    evals, bad = {}, 0
    for path in sorted((run / "evals").glob("*.json")):
        row = read_json(path)
        if not re.fullmatch(r"\d{6,7}", path.stem) or not isinstance(row, dict) or "success_rate" not in row:
            continue
        if int(row.get("episodes", 0)) != 100 or int(row.get("seed_start", 10000)) != 10000:
            bad += 1
            continue
        evals[int(path.stem)] = float(row["success_rate"])
    try:
        lines = (run / "metrics.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:                       # the trainer's own record of the same evals (fills in a lost JSON)
        if '"kind": "eval"' in line:
            row = json.loads(line)
            if int(row.get("episodes", 0)) == 100:
                evals.setdefault(int(row["env_steps"]), float(row["success_rate"]))
    return evals, bad


def dice_rl_evals(run: Path, flat: Path | None) -> tuple[dict[int, float], int]:
    evals, bad = {}, 0
    files = [(path, path.stem) for path in (run / "evals").glob("*.json")]
    if flat is not None:
        files += [(path, path.stem[len(run.name) + 1:]) for path in flat.glob(f"{run.name}_*.json")]
    for path, stem in sorted(files):
        if not re.fullmatch(r"\d{6,7}", stem):          # .partial.json, another run sharing the prefix
            continue
        row = read_json(path)
        if not isinstance(row, dict) or "success_rate" not in row:
            continue
        if row.get("strategy") != "bon16" or int(row.get("episodes", 0)) != 100 or int(row.get("seed_start", -1)) != 10000:
            bad += 1
            continue
        evals[int(row.get("env_steps") or stem)] = float(row["success_rate"])
    return evals, bad


def summarise(evals: dict[int, float], total: int) -> dict:
    def inside(span):
        return [rate for step, rate in sorted(evals.items()) if span[0] <= step <= span[1]]
    window, late, at600 = inside(WINDOW), inside(LATE), inside(AT600)
    last = max(evals) if evals else None
    return {
        "evals": len(evals), "last_step": last, "last_rate": evals[last] if last is not None else None,
        "finished": last is not None and last >= total - 1000,
        "final": evals[last] if last is not None and last >= total - 1000 else None,
        "at600": at600[-1] if at600 else None,
        "win": statistics.mean(window) if window else None, "win_n": len(window),
        "win_sd": statistics.pstdev(window) if len(window) > 1 else None,
        "late": statistics.mean(late) if late else None, "late_n": len(late),
    }


def collect(dirs: list[Path], flat: Path | None) -> list[dict]:
    rows = []
    for parent in dirs:
        for config_path in sorted(parent.glob("*/config.json")):
            run = config_path.parent
            raw = read_json(config_path)
            if not isinstance(raw, dict):
                continue
            config = legacy.upgrade_config(raw, strict=False)
            cell = cell_of(config)
            if cell is None or config.get("task") not in TASKS:
                continue
            evals, bad = dsrl_evals(run) if cell[0] == "DSRL" else dice_rl_evals(run, flat)
            total = int(config.get("total_env_steps") or 0)
            rows.append({"run": run.name, "method": cell[0], "arm": cell[1], "task": config["task"],
                         "seed": int(config.get("seed", -1)), "total": total, "bad": bad, **summarise(evals, total)})
    return rows


def num(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def seed_entry(row: dict) -> str:
    """win (600k) for a 600k run; win (600k; final, late mean) for a longer one; * = the last checkpoint has no eval yet."""
    mark = "" if row["finished"] else "*"
    if row["last_step"] is None:
        return f"{mark}no eval"
    if row["at600"] is None:                       # the window is still incomplete: show where the run is instead
        return f"{mark}{num(row['last_rate'], 0)} at {round(row['last_step'] / 1000)}k"
    inner = num(row["at600"], 0)
    if row["total"] > 600_000:
        inner += f"; {num(row['final'], 0)}, {num(row['late'])}"
    return f"{mark}{num(row['win'])} ({inner})"


def table(rows: list[dict], per_run: bool) -> str:
    out = []
    for method in ("DICE-RL", "DSRL"):
        mine = [row for row in rows if row["method"] == method]
        if not mine:
            continue
        seeds = sorted({row["seed"] for row in mine})
        out += [f"## {method}: {PROTOCOL[method]}", "",
                "Per seed: mean of the 350k-600k evals (the 600k eval; for longer runs also the final eval and the "
                "650k-1M mean). `*` = not finished.", "",
                "| arm | task | " + " | ".join(f"s{seed}" for seed in seeds) + " | mean 350k-600k | mean @600k |",
                "|---|---|" + "---|" * (len(seeds) + 2)]
        for arm in ARMS[method]:
            for task in TASKS:
                cell = [row for row in mine if row["arm"] == arm and row["task"] == task]
                if not cell:
                    continue
                entries = []
                for seed in seeds:
                    found = [row for row in cell if row["seed"] == seed]
                    entries.append(" / ".join(seed_entry(row) for row in found) if found else "")
                done = [row for row in cell if row["at600"] is not None]          # runs whose 350k-600k window is complete
                wins, at600 = [row["win"] for row in done], [row["at600"] for row in done]
                out.append(f"| {arm} | {task} | " + " | ".join(entries) + f" | {num(statistics.mean(wins)) if wins else '-'}"
                           f" (n={len(wins)}) | {num(statistics.mean(at600)) if at600 else '-'} (n={len(at600)}) |")
        out.append("")
        if per_run:
            out += ["| run | steps | evals | last eval at | win (n, sd) | 600k | final | bad |", "|---|---|---|---|---|---|---|---|"]
            for row in sorted(mine, key=lambda r: (ARMS[method].index(r["arm"]), TASKS.index(r["task"]), r["seed"], r["total"])):
                out.append(f"| {row['run']} | {row['total'] // 1000}k | {row['evals']} | {row['last_step'] or '-'} | "
                           f"{num(row['win'])} ({row['win_n']}, {num(row['win_sd'])}) | {num(row['at600'], 0)} | "
                           f"{num(row['final'], 0)} | {row['bad']} |")
            out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--bon16-dir", type=Path, default=None)
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    dirs = args.dirs
    if not dirs:
        default = os.environ.get("RUNS_ROOT") or str(Path(os.environ.get("ASSET_ROOT", Path(__file__).resolve().parents[1] / "assets")) / "runs")
        dirs = [Path(default)]
    rows = collect(dirs, args.bon16_dir)
    if not rows:
        raise SystemExit(f"no DICE-RL or DSRL run directory (a directory with a config.json) directly below {[str(d) for d in dirs]}")
    print(table(rows, args.runs))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
