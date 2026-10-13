#!/usr/bin/env python3
"""Mirror the DICE-RL best-of-16 checkpoint evals to wandb, comparable with the DSRL runs' in-run evals.

DICE-RL runs train with their in-loop eval off; scripts/eval_loop.sh evaluates every 50k checkpoint afterwards and
writes <run>/evals/<step>.json, which never reaches wandb. DSRL runs log their evals inside the training run as
  eval/env_steps, eval/success_rate (percent), eval/episodes, eval/avg_length, eval/avg_return   (x axis: eval/env_steps).
This script logs the SAME keys, same scale and same x axis for every DICE-RL run, so one wandb chart of
eval/success_rate shows both. A training run cannot safely take a second writer, so each run gets a COMPANION run:
  name   <training run name>_eval-bo16          id  <training run id>-eval           project  the training run's
  group  the training run's group (read from wandb), so group averages line up with the training groups
  config the training run's config + eval_protocol / train_run_id / train_run_name / arm;  job_type eval-bon16;  tags eval, bon16
PROTOCOL DIFFERENCE, kept visible in the name, tags and config: DICE-RL evals are best-of-16 prior draws (its method),
DSRL evals are a single noise draw; both are 100 episodes on env seeds 10000-10099.
An eval is mirrored once, only if it is this run's (file name = the checkpoint step, ckpt_path inside the run dir), has
exactly 100 episodes on seeds 10000+ and is not stale (written after its checkpoint when that file still exists).
Logged steps are remembered in <work dir>/state.json and re-read from wandb if that file is lost.

Where things are (scripts/env.sh roots; every one can be given explicitly):
  --runs-dir   run directories                                  [$RUNS_ROOT]
  --evals-dir  flat directory of <run name>_<step>.json evals   [unset: <run>/evals/<step>.json]   (reference layout)
  --work-dir   state.json, lock, the companions' wandb files    [<runs dir>/logs/wandb_eval_mirror]
The training runs must exist on the wandb server: train with WANDB_MODE=online (scripts/env.sh defaults to offline),
or `wandb sync` the offline runs first. This script refuses to run unless WANDB_MODE is online.

OPERATING NOTES:
  - one writer only: every non-dry mode takes an exclusive lock (<work dir>/lock) and exits if another instance holds it;
  - wandb is authoritative: before any write the companion's own history is read and already-present steps are skipped, so a lost
    state file, a crash between log and save, or a wandb hiccup cannot duplicate rows (a failed read skips the run for that pass);
  - the training run id is re-read every pass: a run relaunched under the same name gets a new companion (ALERT names the old one);
  - config key `arm` / tags (TD+distill | TD-only | Q, derived from config.json) label the arm: the training GROUP names mean
    different arms on different tasks (bare <task>_dicerl_dvd is TD+distill on square_d0 but TD-only on the other three);
  - keep the chart x axis on eval/env_steps (rows are appended in arrival order; wandb's Step is just the row index);
  - eval/avg_length and eval/avg_return exist for DICE-RL only (the DSRL eval JSONs do not carry them); eval/ckpt_step is bookkeeping;
  - never delete a companion run in the wandb UI (its id cannot be re-created); a rewritten eval JSON is reported, not re-logged.

usage: WANDB_MODE=online python scripts/wandb_eval_mirror.py --dry-run            # what would be logged
       WANDB_MODE=online python scripts/wandb_eval_mirror.py --once [--only NAME] # one pass (optionally one run)
       WANDB_MODE=online python scripts/wandb_eval_mirror.py --loop 300           # a pass every 300 s
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from online_rl_mimicgen import legacy  # noqa: E402  (runs of the pre-rename research code are read through it)

# set by main() from the arguments / the scripts/env.sh roots
RUNS = EVALS = WORK = STATE = None
KEYS = ("success_rate", "episodes", "avg_length", "avg_return")          # the keys DSRL logs under eval/ (plus env_steps)
PROTOCOL = "best-of-16 prior draws ranked by the run's critic, deterministic residual, 100 episodes, env seeds 10000-10099"


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def train_run_id(run_dir: Path) -> str | None:
    """The training run's wandb id: the newest wandb/run-<stamp>-<id> dir (one per launch; relaunches get a new dir)."""
    ds = sorted(glob.glob(str(run_dir / "wandb" / "*run-*-*")), key=os.path.getmtime)
    m = re.fullmatch(r"(?:offline-)?run-\d{8}_\d{6}-([a-z0-9]+)", os.path.basename(ds[-1])) if ds else None
    return m.group(1) if m else None


def valid_evals(name: str, run_dir: Path) -> dict[int, dict]:
    """ckpt step -> row for every valid eval of this exact run."""
    out: dict[int, dict] = {}
    if EVALS is None:                                                   # this repository's layout: <run>/evals/<step>.json
        pattern, files = r"(\d{6,7})\.json", glob.glob(str(run_dir / "evals" / "*.json"))
    else:                                                               # flat: <evals dir>/<run name>_<step>.json
        pattern, files = re.escape(name) + r"_(\d{6,7})\.json", glob.glob(str(EVALS / f"{glob.escape(name)}_*.json"))
    for f in files:
        m = re.fullmatch(pattern, os.path.basename(f))
        if not m:
            continue
        try:
            o = json.loads(Path(f).read_text())
            ck = o.get("ckpt_path") or ""
            if os.path.realpath(os.path.dirname(os.path.dirname(ck))) != os.path.realpath(run_dir):
                continue                                                # another run's checkpoint
            if int(o.get("episodes", 0)) != 100 or o.get("strategy") != "bon16" or int(o.get("seed_start", -1)) != 10000:
                continue
            if os.path.basename(ck) != f"{m.group(1)}.pt" or not math.isfinite(float(o["success_rate"])):
                continue
            if os.path.exists(ck) and os.path.getmtime(f) - float(o.get("seconds") or 0) < os.path.getmtime(ck):
                continue                                                # predates this checkpoint file: an earlier attempt's eval
            steps = int(o.get("env_steps", int(m.group(1))))
            row = {"env_steps": steps}
            for k in KEYS:
                if k in o:
                    row[k] = float(o[k]) if k != "episodes" else int(o[k])
            if int(m.group(1)) in out:
                log(f"skip {os.path.basename(f)}: a second file for step {int(m.group(1))}")
                continue
            out[int(m.group(1))] = row
        except Exception as e:                                          # noqa: BLE001
            log(f"skip {os.path.basename(f)}: {type(e).__name__}: {e}")
    return dict(sorted(out.items()))


def arm_of(cfg: dict) -> str:
    """the arm, from the config (never from the run or group name)"""
    cfg = legacy.upgrade_config(cfg, strict=False)
    if not float(cfg.get("dvd_weight") or 0.0):
        return "Q"
    return "TD+distill" if cfg.get("value_distill_robometer") else "TD-only"


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:                                                   # noqa: BLE001
        return {}


def save_state(st: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_name(STATE.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    tmp.replace(STATE)


class Mirror:
    def __init__(self, dry: bool):
        self.dry = dry
        self.state = load_state()
        self._api = None
        self.meta: dict[str, dict] = {}                                 # run name -> {id, entity, project, group}
        self._warned: dict[str, float] = {}
        self._arm = "?"

    @property
    def api(self):
        if self._api is None:
            import wandb
            self._api = wandb.Api(timeout=60)
        return self._api

    def warn(self, key: str, msg: str) -> None:
        """log msg at most once per hour per key"""
        now = time.time()
        if now - self._warned.get(key, 0) >= 3600:
            self._warned[key] = now
            log(msg)

    def train_meta(self, name: str, run_dir: Path, cfg: dict) -> dict | None:
        rid = train_run_id(run_dir)                                     # re-read every pass: a relaunch has a new id
        if not rid:
            return None
        old = self.meta.get(name)
        if old and old["id"] == rid:
            return old
        if old:
            log(f"ALERT {name}: the training run id changed {old['id']} -> {rid} (relaunch); new evals go to {rid}-eval, the old "
                f"companion {old['id']}-eval stays in the group until it is removed together with the dead run")
        project = cfg.get("wandb_project") or os.environ.get("WANDB_PROJECT") or "dvd"
        try:
            r = self.api.run(f"{self.api.default_entity}/{project}/{rid}")
            if r.name != name:
                self.warn(f"name:{name}", f"{name}: wandb run {rid} is named {r.name!r}; not mirroring")
                return None
            m = dict(id=rid, entity=r.entity, project=r.project, group=r.group)
        except Exception as e:                                          # noqa: BLE001
            self.warn(f"meta:{name}", f"{name}: cannot read the training run {rid} from wandb ({type(e).__name__}: {e}); will retry")
            return None
        self.meta[name] = m
        return m

    def remote_steps(self, m: dict) -> set[int] | None:
        """ckpt steps present in the companion's own wandb history: set() if the companion does not exist yet, None if the read
        failed (then nothing is written this pass)."""
        import wandb
        key = f"{m['id']}-eval"
        try:
            r = self.api.run(f"{m['entity']}/{m['project']}/{key}")
        except wandb.errors.CommError as e:
            if "not find" in str(e).lower() or "not found" in str(e).lower():
                return set()
            self.warn(f"hist:{key}", f"{key}: cannot read the companion from wandb ({e}); skipped this pass")
            return None
        except Exception as e:                                          # noqa: BLE001
            self.warn(f"hist:{key}", f"{key}: cannot read the companion from wandb ({type(e).__name__}: {e}); skipped this pass")
            return None
        try:
            steps = {int(row["eval/ckpt_step"]) for row in r.scan_history(keys=["eval/ckpt_step"]) if row.get("eval/ckpt_step") is not None}
            if r.config.get("arm") is None and not self.dry:           # companions created before the arm label existed
                r.config["arm"] = self._arm
                r.tags = sorted(set(r.tags) | {self._arm})
                r.update()
            return steps
        except Exception as e:                                          # noqa: BLE001
            self.warn(f"hist:{key}", f"{key}: cannot read the companion's history ({type(e).__name__}: {e}); skipped this pass")
            return None

    def one(self, run_dir: Path) -> int:
        name = run_dir.name
        try:
            cfg = json.loads((run_dir / "config.json").read_text())
        except Exception:                                               # noqa: BLE001
            return 0
        if cfg.get("method") != "dice_rl":
            return 0
        evs = valid_evals(name, run_dir)
        if not evs:
            return 0
        m = self.train_meta(name, run_dir, cfg)
        if m is None:
            return 0
        key = f"{m['id']}-eval"
        self._arm = arm_of(cfg)
        ent = self.state.setdefault(key, dict(name=name, steps=[], rates={}))
        ent.setdefault("rates", {})
        for s_, r_ in evs.items():                                      # a JSON rewritten after it was mirrored: report, never re-log
            was = ent["rates"].get(str(s_))
            if was is not None and abs(was - r_["success_rate"]) > 1e-9:
                self.warn(f"rewrite:{key}:{s_}", f"ALERT {name}: eval of step {s_} changed {was} -> {r_['success_rate']} after it was "
                                                 f"mirrored; wandb keeps {was} (history is append-only)")
        cand = {s_: r_ for s_, r_ in evs.items() if s_ not in set(ent["steps"])}
        needs_label = not ent.get("labelled")
        if not cand and not needs_label:
            return 0
        remote = self.remote_steps(m)                                   # authoritative; also adds the arm label once
        if remote is None:
            return 0
        if not self.dry:
            ent["labelled"] = True
        new = {s_: r_ for s_, r_ in cand.items() if s_ not in remote}
        known = sorted(set(ent["steps"]) | (remote & set(evs)))
        if not new:
            if known != ent["steps"] or needs_label:
                ent["steps"] = known
                for s_ in known:
                    ent["rates"].setdefault(str(s_), evs[s_]["success_rate"]) if s_ in evs else None
                if not self.dry:
                    save_state(self.state)
            return 0
        if self.dry:
            log(f"DRY {name} [{self._arm}] -> {m['entity']}/{m['project']}/{key} (group {m['group']}): would log "
                + ", ".join(f"{r_['env_steps']}:{r_['success_rate']:.0f}" for r_ in new.values()))
            return len(new)
        import wandb
        run = wandb.init(
            entity=m["entity"], project=m["project"], id=key, resume="allow", name=f"{name}_eval-bo16", group=m["group"],
            job_type="eval-bon16", tags=["eval", "bon16", self._arm], dir=str(WORK), reinit=True, mode="online",
            config={**cfg, "eval_protocol": PROTOCOL, "train_run_id": m["id"], "train_run_name": name, "arm": self._arm},
            settings=wandb.Settings(console="off", x_disable_stats=True, x_disable_meta=True, silent=True))
        if getattr(run, "disabled", False):
            raise RuntimeError("wandb is disabled in this environment; nothing was logged")
        try:
            run.define_metric("eval/env_steps")
            run.define_metric("eval/*", step_metric="eval/env_steps")
            for s_, r_ in new.items():                                  # increasing checkpoint step
                run.log({**{f"eval/{k}": v for k, v in r_.items()}, "eval/ckpt_step": s_})
        finally:
            run.finish()
        ent["steps"] = sorted(set(known) | set(new))
        for s_ in ent["steps"]:
            if s_ in evs:
                ent["rates"].setdefault(str(s_), evs[s_]["success_rate"])
        save_state(self.state)
        log(f"{name} [{self._arm}]: mirrored {len(new)} eval(s) to {m['entity']}/{m['project']}/{key} (group {m['group']}): "
            + ", ".join(f"{r_['env_steps'] // 1000}k={r_['success_rate']:.0f}" for r_ in new.values()))
        return len(new)

    def sweep(self, only: str | None) -> int:
        n = 0
        if not self.dry:
            self.state = load_state()                                   # another (locked-out) writer cannot exist, but stay honest
        for d in sorted(c.parent for c in RUNS.glob("*/config.json")):  # one() keeps the DICE-RL runs
            if only is None or d.name == only:
                try:
                    n += self.one(d)
                except Exception as e:                                  # noqa: BLE001  (one run's trouble never stops the daemon)
                    self.warn(f"err:{d.name}", f"{d.name}: {type(e).__name__}: {e}")
        return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", type=int, metavar="SECONDS")
    ap.add_argument("--only", help="one training run name")
    ap.add_argument("--runs-dir", type=Path, default=None, help="directory holding the run directories [$RUNS_ROOT]")
    ap.add_argument("--evals-dir", type=Path, default=None, help="flat directory of <run name>_<step>.json evals "
                                                                 "[unset: <run>/evals/<step>.json]")
    ap.add_argument("--work-dir", type=Path, default=None, help="state and lock [<runs dir>/logs/wandb_eval_mirror]")
    a = ap.parse_args()
    global RUNS, EVALS, WORK, STATE
    assets = Path(os.environ.get("ASSET_ROOT", Path(__file__).resolve().parents[1] / "assets"))
    RUNS = a.runs_dir or Path(os.environ.get("RUNS_ROOT", assets / "runs"))
    EVALS = a.evals_dir
    WORK = a.work_dir or RUNS / "logs" / "wandb_eval_mirror"
    STATE = WORK / "state.json"
    os.environ.setdefault("WANDB_SILENT", "true")
    if not a.dry_run:
        if os.environ.get("WANDB_MODE", "online") not in ("online", "run"):
            sys.exit(f"WANDB_MODE={os.environ['WANDB_MODE']}: the mirror must run online")
        WORK.mkdir(parents=True, exist_ok=True)
        lock = open(WORK / "lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit("another wandb_eval_mirror instance holds the lock; exiting")
    mir = Mirror(a.dry_run)
    if a.loop:
        log(f"wandb_eval_mirror started (pid {os.getpid()}, every {a.loop} s)")
        while True:
            mir.sweep(a.only)
            time.sleep(a.loop)
    n = mir.sweep(a.only)
    log(f"{'would mirror' if a.dry_run else 'mirrored'} {n} eval(s)")


if __name__ == "__main__":
    main()
