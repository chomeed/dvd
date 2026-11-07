"""CPU test of the prior-training stage (prior/): the wrapper replays the released runs, LeRobot resolves its argv to
the released train_config.json files, and the recipe builds the released network and normaliser. No training step.

  always (bash, the standard library; pyarrow and torch for the synthetic checks)
    prior/train_prior.sh --print       every stage of every task against prior/released_runs.json, the as-run record:
                                       the train argv is the recorded argv with exactly the documented substitutions
                                       and removals (square_d0: its reconstruction with the paths substituted); the
                                       SMOKE_STEPS variant changes only the budget, run directory and job name
    refusals and resume                missing / unknown TASK or STAGE, bad WANDB_MODE or SMOKE_STEPS, a missing, meta-only
                                       or 26-D training set, a run directory without checkpoints/last; a run directory
                                       with checkpoints/last gives the resume argv, a finished one is left alone
    one release step per task          the wrapper, prior/released_runs.json, scripts/fetch_assets.py, configs/tasks.sh,
                                       the DICE-RL golden configs and rl/ name the same prior directory and step; the
                                       demo dataset is fetched at the revision each prior was trained on; the fetch table
                                       lists the 9-D training set where the wrapper reads it
    prior/state9_dataset.py            on a synthetic LeRobot v3 dataset whose files are relative links into blobs/, as
                                       in a Hugging Face snapshot: state and statistics cut, everything else equal,
                                       videos resolved, --check, --meta-only, refusals
    prior/install_prior.py             on a synthetic run directory: only the release step lands, find_latest_checkpoint
                                       resolves to it, a second install is refused, --verify
  with LeRobot importable (the main environment)
    resume                             the resume argv resolves to the first resolution plus the resume fields only
  with the released priors under $CKPT_ROOT/priors (and their demo datasets, CLIP in the Hugging Face cache)
    resolved configuration             LeRobot's own parse of each task's argv equals the released train_config.json
                                       except EXACTLY the listed keys (ALLOWED below; an allowance that no longer
                                       differs is stale and fails); the installed train_config.json is the released file
    statistics                         the 9-D copy of each demo dataset's meta/ carries the released normaliser exactly;
                                       install_prior.py --verify passes on all four released priors
    architecture                       the recipe builds 465 tensors with the released names and shapes, the released
                                       config.json, postprocessor and normaliser (listed exceptions only)
Run: python tests/pipeline/test_prior_recipe.py
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import _pipeline_util as U

REPO = U.REPO
WRAPPER = REPO / "prior" / "train_prior.sh"
RUNS = json.loads((REPO / "prior" / "released_runs.json").read_text())["tasks"]
FAKE = {"DATA_ROOT": "/assets/data", "CKPT_ROOT": "/assets/ckpt", "CACHE_ROOT": "/assets/cache", "RUNS_ROOT": "/assets/runs",
        "SPOOL_ROOT": "/assets/spool"}
DROPPED = ("--policy.n_state_dims", "--env.type", "--env.task", "--env.discover_packages_path")
# Keys of the resolved configuration allowed to differ from the released train_config.json (flattened, dotted; an
# entry covers the key and everything below it). For each task the set of entries that DO differ must equal this set.
ALLOWED_ALL = {"dataset.root": "path: the roots of scripts/env.sh",
               "output_dir": "path: $RUNS_ROOT/prior_train/<run>",
               "policy.device": "cuda in the argv; this test hides CUDA, so LeRobot switches to cpu"}
ALLOWED_FORK = {"dataset.repo_id": "the 9-D copy replaces --policy.n_state_dims=9",
                "policy.n_state_dims": "field of the unavailable LeRobot fork",
                "env": "in-loop evaluation dropped: the env would feed its 26 / 37-D state to the 9-D policy",
                "wandb.project": "$WANDB_PROJECT", "wandb.mode": "$WANDB_MODE (LeRobot goes online when unset)",
                "wandb.run_id": "assigned by wandb when the run starts"}
NEW_IN_062 = ("parallelism", "accelerator", "ema", "checkpoint_format", "dataloader_multiprocessing_context",
              "dataset.repo_type", "dataset.exclude_episodes")      # fields the fork's 0.6.0 schema does not have
FORK_TOKENIZER = "openai/clip-vit-base-patch16"


def _namespace() -> str:
    """The default HF_NAMESPACE written in scripts/env.sh (the account the released repositories live under)."""
    return re.search(r"^export HF_NAMESPACE=\$\{HF_NAMESPACE:-([\w.-]+)\}", (REPO / "scripts/env.sh").read_text(), re.M).group(1)


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("TASK", "STAGE", "SMOKE_STEPS", "G", "HF_DATASETS_CACHE", "ASSET_ROOT", *FAKE)}
    env.update(FAKE, HF_NAMESPACE="ns", WANDB_PROJECT="proj", WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    env.update(extra)
    return env


def _run(*args, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(WRAPPER), *args], env=_env(**extra), capture_output=True, text=True)


def _printed(task: str, stage: str, **extra) -> list:
    done = _run("--print", TASK=task, STAGE=stage, **extra)
    assert done.returncode == 0, (task, stage, extra, done.stderr)
    return done.stdout.splitlines()


def _as_run(task: str, data: str, runs: str, ns: str, project: str, mode: str) -> list:
    """The released run's argv (or square_d0's reconstruction) with the substitutions the wrapper header documents."""
    rec = RUNS[task]
    job, ds = rec["job_name"], f"mimicgen_{task}_224x224_robotstate9"
    out = []
    for token in rec["argv"] or rec["argv_reconstructed"]:
        key, _, value = token.partition("=")
        if key in DROPPED:
            want = {"--policy.n_state_dims": "9", "--env.type": "mimicgen", "--env.task": task,
                    "--env.discover_packages_path": "lerobot_env_mimicgen"}[key]
            assert rec["argv"] and value == want, (task, token)
            continue
        if key == "--dataset.repo_id":
            assert value.split("/")[-1] == (ds if rec["argv"] is None else f"mimicgen_{task}_224x224"), (task, token)
            token = f"{key}={ns}/{ds}"
        elif key == "--dataset.root":
            token = f"{key}={data}/{ds}"
        elif key == "--output_dir":
            token = f"{key}={runs}/prior_train/{job}"
        elif key == "--policy.repo_id":
            token = f"{key}={ns}/{value.split('/', 1)[1]}"
        elif key == "--wandb.project":
            out += [f"{key}={project}", f"--wandb.mode={mode}"]
            continue
        out.append(token)
    return out


def _flat(node, prefix=""):
    out = {}
    if isinstance(node, dict) and node:
        for key, value in node.items():
            out.update(_flat(value, f"{prefix}.{key}" if prefix else key))
    else:
        out[prefix] = node
    return out


def _covering(key: str, entries) -> str | None:
    for entry in sorted(entries, key=len, reverse=True):
        if key == entry or key.startswith(entry + "."):
            return entry
    return None


def _released(task: str) -> Path:
    rec = RUNS[task]
    return U.root("CKPT_ROOT") / "priors" / rec["prior"] / "checkpoints" / rec["release_step"] / "pretrained_model"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(jobs: list, tmp: Path) -> None:
    """Run tests/pipeline/_prior_worker.py on CPU with the Hub offline and the datasets cache inside tmp."""
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_DATASETS_CACHE=str(tmp / "hf_datasets"))
    done = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).with_name("_prior_worker.py")),
                           str(tmp / "jobs.json")], env=env, capture_output=True, text=True, cwd=str(tmp))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]


def _train_argv(task: str, **extra) -> list:
    """The lerobot_train argv the wrapper prints for task (without the env line, python and the launcher)."""
    lines = _printed(task, "train", **extra)
    assert lines[1:3] == ["python", "prior/lerobot_train.py"], lines[:3]
    return lines[3:]


def _need_lerobot():
    if importlib.util.find_spec("lerobot") is None or importlib.util.find_spec("draccus") is None:
        U.skip("LeRobot is not importable (the main environment of environment.md, section 2)")


def test_wrapper_prints_the_released_argv():
    assert subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0, "bash -n prior/train_prior.sh"
    data, runs, ckpt = FAKE["DATA_ROOT"], FAKE["RUNS_ROOT"], FAKE["CKPT_ROOT"]
    for task, rec in RUNS.items():
        job, ds = rec["job_name"], f"{data}/mimicgen_{task}_224x224_robotstate9"
        assert _printed(task, "data") == ["python", "prior/state9_dataset.py", "--task", task, "--src",
                                          f"{data}/mimicgen_{task}_224x224", "--out", ds]
        assert _printed(task, "install") == ["python", "prior/install_prior.py", "--task", task, "--run",
                                             f"{runs}/prior_train/{job}", "--step", rec["release_step"], "--out",
                                             f"{ckpt}/priors/{rec['prior']}", "--dataset", ds]
        lines = _printed(task, "train")
        want = _as_run(task, data, runs, "ns", "proj", "offline")
        assert lines == [f"HF_DATASETS_CACHE={FAKE['CACHE_ROOT']}/hf_datasets", "python", "prior/lerobot_train.py",
                         *want], (task, lines)
        options = dict(t.split("=", 1) for t in want)
        assert (options["--steps"], options["--save_freq"], options.get("--log_freq", "200")) == \
            (str(rec["steps"]), str(rec["save_freq"]), str(rec["log_freq"])), task
        assert int(rec["release_step"]) % rec["save_freq"] == 0 and int(rec["release_step"]) <= rec["steps"], task
        smoke = _printed(task, "train", SMOKE_STEPS="200")[3:]
        changed = {a.split("=")[0] for a, b in zip(want, smoke) if a != b}
        assert len(smoke) == len(want) and changed == {"--steps", "--save_freq", "--output_dir", "--job_name"}, (task, changed)
        assert f"--output_dir={runs}/smoke/prior_{job}" in smoke and f"--job_name=smoke_{job}" in smoke
        assert "--steps=200" in smoke and "--save_freq=200" in smoke
        print(f"{task}: {len(want)} tokens = the {'recorded' if rec['argv'] else 'reconstructed'} argv"
              + (f" minus {', '.join(DROPPED)}; paths, namespace and wandb substituted" if rec["argv"] else
                 " with its paths substituted") + f"; release {rec['release_step']} of {rec['steps']}")
    assert "--wandb.enable=true" not in _train_argv("square_d0"), "square_d0 ran without wandb"


def test_wrapper_refusals_and_resume():
    for extra, needle in (({"STAGE": "train"}, "TASK"), ({"TASK": "threading", "STAGE": "train"}, "TASK must be"),
                          ({"TASK": "square_d0"}, "STAGE"), ({"TASK": "square_d0", "STAGE": "fit"}, "unknown STAGE"),
                          ({"TASK": "square_d1", "STAGE": "train", "WANDB_MODE": "dryrun"}, "WANDB_MODE"),
                          ({"TASK": "square_d1", "STAGE": "install", "SMOKE_STEPS": "5"}, "SMOKE_STEPS belongs"),
                          ({"TASK": "square_d1", "STAGE": "train", "SMOKE_STEPS": "1e3"}, "positive integer")):
        done = _run("--print", **extra)
        assert done.returncode != 0 and needle in done.stderr, (extra, done.returncode, done.stderr)
    assert _run("--prnt", TASK="square_d0", STAGE="train").returncode == 2
    assert _run("--print", TASK="square_d0", STAGE="train", WANDB_MODE="dryrun").returncode == 0, "square_d0 has no wandb"

    tmp = Path(tempfile.mkdtemp(prefix="test_prior_wrapper_"))
    try:
        roots = {"DATA_ROOT": str(tmp / "data"), "RUNS_ROOT": str(tmp / "runs"), "CKPT_ROOT": str(tmp / "ckpt"),
                 "CACHE_ROOT": str(tmp / "cache")}
        task, rec = "square_d1", RUNS["square_d1"]
        ds, run = tmp / "data" / f"mimicgen_{task}_224x224_robotstate9", tmp / "runs" / "prior_train" / rec["job_name"]

        def train(needle, code=1):
            done = _run(TASK=task, STAGE="train", **roots)
            assert done.returncode == code and needle in (done.stderr + done.stdout), (needle, done.returncode, done.stderr)

        train("run TASK=square_d1 STAGE=data")                      # no training set yet
        (ds / "meta").mkdir(parents=True)
        info = {"features": {"observation.state": {"dtype": "float32", "shape": [9]}}}
        (ds / "meta" / "info.json").write_text(json.dumps(info))
        train("meta/ only")
        (ds / "data").mkdir()
        info["features"]["observation.state"]["shape"] = [26]
        (ds / "meta" / "info.json").write_text(json.dumps(info))
        train("trained on [9]")
        info["features"]["observation.state"]["shape"] = [9]
        (ds / "meta" / "info.json").write_text(json.dumps(info))
        run.mkdir(parents=True)
        train("holds no checkpoints/last")                           # died before its first checkpoint
        step = run / "checkpoints" / "005000" / "pretrained_model"
        step.mkdir(parents=True)
        (step / "train_config.json").write_text("{}")
        (run / "checkpoints" / "last").symlink_to("005000")
        last = run / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        argv = _train_argv(task, **roots)
        assert argv[-2:] == [f"--config_path={last}", "--resume=true"] and argv[:-2] == _as_run(
            task, roots["DATA_ROOT"], roots["RUNS_ROOT"], "ns", "proj", "offline"), argv[-3:]
        (run / "checkpoints" / "060000" / "pretrained_model").mkdir(parents=True)
        shutil.copy(step / "train_config.json", run / "checkpoints" / "060000" / "pretrained_model")
        (run / "checkpoints" / "last").unlink()
        (run / "checkpoints" / "last").symlink_to("060000")
        train("already reached step 60000", code=0)                 # a finished run is left alone

        py = sys.executable
        for argv, needle in ((["prior/state9_dataset.py"], "--task"), (["prior/state9_dataset.py", "--task", "square"], "invalid choice"),
                             (["prior/state9_dataset.py", "--task", task, "--meta-only"], "explicit --out"),
                             (["prior/install_prior.py", "--task", task, "--run", str(run), "--step", "last", "--out", str(tmp / "p")], "numbered"),
                             (["prior/install_prior.py", "--task", task, "--run", str(run), "--step", "045000", "--out", str(tmp / "p")], "no checkpoint"),
                             (["prior/install_prior.py", "--task", task, "--run", str(run), "--step", "005000", "--out", str(tmp)], "exists")):
            done = subprocess.run([py, "-W", "ignore", str(REPO / argv[0]), *argv[1:]], env=_env(), capture_output=True, text=True)
            assert done.returncode != 0 and needle in done.stderr, (argv, done.returncode, done.stderr[-400:])
        print("refused: no / unknown TASK or STAGE, WANDB_MODE dryrun, SMOKE_STEPS with install or not an integer; no, "
              "meta-only or 26-D training set; a run directory without checkpoints/last; state9_dataset.py without a known "
              "--task; install of `last`, of a missing step, into an existing directory. checkpoints/last present -> "
              "the same argv plus --config_path=<last>/train_config.json --resume=true; last at the final step -> nothing to do")
    finally:
        shutil.rmtree(tmp)


def test_release_steps_agree():
    """One prior directory and release step per task, everywhere it is named; the training revision is what is fetched."""
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "prior"))
    import fetch_assets
    import state9_dataset
    U.use_repo()
    from online_rl_mimicgen import common
    golden = REPO / "tests" / "configs" / "golden" / "dice_rl"
    items = fetch_assets.table()
    for task, rec in RUNS.items():
        install = _printed(task, "install")
        out, step = install[install.index("--out") + 1], install[install.index("--step") + 1]
        tasks_sh = subprocess.run(["bash", "-c", f'source "{REPO}/scripts/env.sh" && source "{REPO}/configs/tasks.sh" && echo "$PRIOR"'],
                                  env=_env(TASK=task), capture_output=True, text=True).stdout.strip()
        assert Path(out).name == rec["prior"] == tasks_sh == fetch_assets.TASKS[task]["prior"] == common.TASKS[task]["il_output"].name, task
        assert step == rec["release_step"] == fetch_assets.TASKS[task]["step"], task
        base = json.loads((golden / f"q__{task}.json").read_text())["config"]["base_checkpoint"]
        assert base == f"<CKPT_ROOT>/priors/{rec['prior']}/checkpoints/{step}/pretrained_model", (task, base)
        dim = rec["dataset"]["state_dim"]
        assert dim == state9_dataset.SOURCE_DIM[task] == common.TASKS[task]["state_dim"], task
        assert rec["dataset"]["revision"] == fetch_assets.TASKS[task]["demo_rev"], f"{task}: fetched revision != training revision"
        assert rec["dataset"]["repo"] == f"mimicgen_{task}_224x224"
        ds = _printed(task, "data")[-1]
        built = [i for i in items if i.kind == "build" and i.root == "DATA_ROOT" and i.target == Path(ds).name and i.task == task]
        assert len(built) == 1 and built[0].tiers == ("regenerate",) and built[0].group == "dataset", (task, built)
        assert f"TASK={task} STAGE=data bash prior/train_prior.sh" in built[0].note, built[0].note
        whole = [i for i in items if i.task == task and i.kind == "repo" and i.target == f"mimicgen_{task}_224x224"]
        assert len(whole) == 1 and "train" in whole[0].tiers and whole[0].revision == rec["dataset"]["revision"], task
        print(f"{task}: priors/{rec['prior']}/checkpoints/{step} in the wrapper, released_runs.json, configs/tasks.sh, "
              f"fetch_assets.py, rl/ and the golden configs; demo dataset fetched at the training revision "
              f"{rec['dataset']['revision'][:8]} ({dim}-D state); 9-D copy listed as a local build of the regenerate tier")


def _synthetic_dataset(root: Path, width: int = 23) -> Path:
    """A 2-episode LeRobot v3 layout whose files are relative links into blobs/, like a Hugging Face snapshot."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    snap, blobs = root / "snapshot", root / "blobs"
    blobs.mkdir(parents=True)

    def put(relative: str, write) -> None:
        blob = blobs / relative.replace("/", "_")
        write(blob)
        (snap / relative).parent.mkdir(parents=True, exist_ok=True)
        (snap / relative).symlink_to(os.path.relpath(blob, (snap / relative).parent))

    rows = 7
    state = [[float(10 * r + c) for c in range(width)] for r in range(rows)]
    data = pa.table({"observation.state": pa.array(state, type=pa.list_(pa.float32())),
                     "action": pa.array([[0.5 * r] * 7 for r in range(rows)], type=pa.list_(pa.float32())),
                     "episode_index": pa.array([0] * 3 + [1] * 4, type=pa.int64()), "index": pa.array(range(rows), type=pa.int64())})
    data = data.replace_schema_metadata({b"pandas": b'{"columns": []}'})
    put("data/chunk-000/file-000.parquet", lambda p: pq.write_table(data, p))
    stat = lambda f: [[f(r) for r in range(width)] for _ in range(2)]
    episodes = pa.table({"episode_index": [0, 1], "length": [3, 4], **{f"stats/observation.state/{k}": stat(f)
                         for k, f in (("min", float), ("max", lambda r: r + 1.0), ("mean", lambda r: r + 0.5))},
                         "stats/observation.state/count": [[3], [4]], "stats/action/min": [[0.0] * 7, [0.0] * 7]})
    put("meta/episodes/chunk-000/file-000.parquet", lambda p: pq.write_table(episodes, p))
    put("meta/tasks.parquet", lambda p: pq.write_table(pa.table({"task_index": [0], "task": ["do it"]}), p))
    info = {"codebase_version": "v3.0", "total_episodes": 2, "total_frames": rows,
            "features": {"observation.state": {"dtype": "float32", "shape": [width], "names": None},
                         "action": {"dtype": "float32", "shape": [7], "names": None},
                         "observation.images.agentview": {"dtype": "video", "shape": [224, 224, 3]}}}
    put("meta/info.json", lambda p: p.write_text(json.dumps(info)))
    stats = {"observation.state": {"min": [0.0] * width, "max": [float(r) for r in range(width)], "count": [rows]},
             "action": {"min": [0.0] * 7, "max": [3.0] * 7, "count": [rows]}}
    put("meta/stats.json", lambda p: p.write_text(json.dumps(stats)))
    put("videos/observation.images.agentview/chunk-000/file-000.mp4", lambda p: p.write_bytes(b"\0video" * 100))
    put("README.md", lambda p: p.write_text("readme"))
    return snap


def test_state9_dataset_rules():
    if importlib.util.find_spec("pyarrow") is None:
        U.skip("pyarrow is not installed (main environment)")
    import pyarrow.parquet as pq
    py, tool = sys.executable, str(REPO / "prior" / "state9_dataset.py")
    tmp = Path(tempfile.mkdtemp(prefix="test_state9_"))

    def call(*argv):
        return subprocess.run([py, "-W", "ignore", tool, *argv], env=_env(), capture_output=True, text=True)

    try:
        src, out = _synthetic_dataset(tmp / "src"), tmp / "out" / "ds9"
        (out.parent / "ds9.partial").mkdir(parents=True)                  # left by an interrupted build
        done = call("--task", "square_d0", "--src", str(src), "--out", str(out))
        assert done.returncode == 0, done.stderr
        assert not (out.parent / "ds9.partial").exists() and not (out / "README.md").exists()
        info = json.loads((out / "meta/info.json").read_text())
        assert info["features"]["observation.state"]["shape"] == [9] and info["features"]["action"]["shape"] == [7]
        a, b = pq.read_table(os.path.realpath(src / "data/chunk-000/file-000.parquet")), pq.read_table(out / "data/chunk-000/file-000.parquet")
        assert b.column("observation.state").to_pylist() == [row[:9] for row in a.column("observation.state").to_pylist()]
        assert a.drop_columns(["observation.state"]).equals(b.drop_columns(["observation.state"])) and a.schema == b.schema
        assert b.schema.metadata == a.schema.metadata, "schema metadata kept"
        e_src = pq.read_table(os.path.realpath(src / "meta/episodes/chunk-000/file-000.parquet"))
        e_out = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet")
        for name in ("min", "max", "mean"):
            assert e_out.column(f"stats/observation.state/{name}").to_pylist() == \
                [v[:9] for v in e_src.column(f"stats/observation.state/{name}").to_pylist()]
        assert e_out.column("stats/observation.state/count").to_pylist() == [[3], [4]]
        assert e_out.column("stats/action/min").equals(e_src.column("stats/action/min"))
        stats = json.loads((out / "meta/stats.json").read_text())
        assert stats["observation.state"] == {"min": [0.0] * 9, "max": [float(r) for r in range(9)], "count": [7]}
        assert stats["action"] == json.loads((src / "meta/stats.json").read_text())["action"]
        video = out / "videos/observation.images.agentview/chunk-000/file-000.mp4"
        assert video.is_file() and not video.is_symlink() and video.read_bytes() == b"\0video" * 100, "hardlink or copy of the blob"
        assert call("--task", "square_d0", "--src", str(src), "--out", str(out), "--check").returncode == 0
        again = call("--task", "square_d0", "--src", str(src), "--out", str(out))
        assert again.returncode == 0 and "already built" in again.stdout, again.stdout + again.stderr
        b2 = b.set_column(0, b.schema.field(0), b.column(0).combine_chunks().take([0, 1, 2, 3, 4, 5, 5]))
        pq.write_table(b2, out / "data/chunk-000/file-000.parquet")      # one state row now wrong
        bad = call("--task", "square_d0", "--src", str(src), "--out", str(out), "--check")
        assert bad.returncode != 0 and "observation.state is not the source's first 9" in bad.stdout, bad.stdout
        refused = call("--task", "square_d0", "--src", str(src), "--out", str(out))
        assert refused.returncode != 0 and "remove it yourself" in refused.stderr, refused.stderr

        link = call("--task", "square_d0", "--src", str(src), "--out", str(tmp / "linked"), "--videos", "symlink")
        assert link.returncode == 0 and (tmp / "linked/videos/observation.images.agentview/chunk-000/file-000.mp4").resolve() == \
            Path(os.path.realpath(src / "videos/observation.images.agentview/chunk-000/file-000.mp4")), link.stderr
        meta = call("--task", "square_d0", "--src", str(src), "--out", str(tmp / "meta9"), "--meta-only")
        assert meta.returncode == 0 and not (tmp / "meta9/data").exists() and not (tmp / "meta9/videos").exists(), meta.stderr
        assert call("--task", "square_d0", "--src", str(src), "--out", str(tmp / "meta9"), "--check").returncode == 0
        for task, needle in (("square_d1", "has 23 dims; the square_d1 demo dataset has 26"),):
            done = call("--task", task, "--src", str(src), "--out", str(tmp / "x"))
            assert done.returncode != 0 and needle in done.stderr, done.stderr
        (src / "meta" / "relative_stats.json").write_text("{}")
        done = call("--task", "square_d0", "--src", str(src), "--out", str(tmp / "y"))
        assert done.returncode != 0 and "does not know how to cut" in done.stderr, done.stderr
        print("synthetic 23-D dataset (relative links into blobs/): state and its statistics cut to 9, every other "
              "column / statistic / schema metadata equal, video hardlinked or copied from the blob (symlink mode: "
              "absolute link to it), README left out, leftover .partial removed; --check passes, a corrupted copy fails "
              "it and is not overwritten; --meta-only writes meta/ only; a 23-D source for square_d1 and an unknown "
              "meta file are refused")
    finally:
        shutil.rmtree(tmp)


def _fake_checkpoint(directory: Path, task: str) -> dict:
    """A minimal pretrained_model whose header, config and normaliser satisfy install_prior.py's checks; -> its statistics."""
    import torch
    from safetensors.torch import save_file
    directory.mkdir(parents=True)
    config = {"type": "multi_task_dit", "n_obs_steps": 2, "timestep_embed_dim": 4, "hidden_dim": 3,
              "input_features": {"observation.state": {"type": "STATE", "shape": [9]},
                                 "observation.images.a": {"type": "VISUAL", "shape": [3, 8, 8]},
                                 "observation.images.b": {"type": "VISUAL", "shape": [3, 8, 8]}}}
    (directory / "config.json").write_text(json.dumps(config))
    width = 4 + 2 * (2 * 2 + 9 + 3)
    save_file({"noise_predictor.transformer_blocks.0.adaLN_modulation.1.weight": torch.zeros(6, width),
               "observation_encoder.vision_encoder.model.vision_model.post_layernorm.weight": torch.zeros(2)},
              str(directory / "model.safetensors"))
    train = {"dataset": {"repo_id": f"ns/mimicgen_{task}_224x224_robotstate9"}, "steps": 60000, "policy": {}}
    (directory / "train_config.json").write_text(json.dumps(train))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (directory / name).write_text("{}")
    stats = {"observation.state": {"min": [0.0] * 9, "max": [1.0] * 9, "count": [5]}, "action": {"min": [0.0] * 7, "count": [5]}}
    save_file({f"{feature}.{key}": torch.tensor(value, dtype=torch.float32) for feature, block in stats.items()
               for key, value in block.items()}, str(directory / "policy_preprocessor_step_4_normalizer_processor.safetensors"))
    (directory / "tokenizer").mkdir()
    (directory / "tokenizer" / "tokenizer.json").write_text("{}")
    return stats


def test_install_prior_rules():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None:
        U.skip("torch / safetensors are not installed (main environment)")
    tmp = Path(tempfile.mkdtemp(prefix="test_install_prior_"))
    try:
        task = "threading_d0"
        run = tmp / "runs" / "prior_train" / RUNS[task]["job_name"]
        stats = None
        for step in ("050000", "055000", "060000"):
            stats = _fake_checkpoint(run / "checkpoints" / step / "pretrained_model", task)
            (run / "checkpoints" / step / "training_state").mkdir()
            (run / "checkpoints" / step / "training_state" / "optimizer_state.safetensors").write_bytes(b"x")
        (run / "checkpoints" / "last").symlink_to("060000")
        ds = tmp / "data" / f"mimicgen_{task}_224x224_robotstate9"
        (ds / "meta").mkdir(parents=True)
        (ds / "meta" / "stats.json").write_text(json.dumps(stats))
        out = tmp / "ckpt" / "priors" / RUNS[task]["prior"]
        argv = ["--task", task, "--run", str(run), "--step", "055000", "--out", str(out), "--dataset", str(ds)]
        tool = str(REPO / "prior" / "install_prior.py")
        done = subprocess.run([sys.executable, "-W", "ignore", tool, *argv], env=_env(), capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        assert files == [f"checkpoints/055000/pretrained_model/{n}" for n in
                         ("config.json", "model.safetensors", "policy_postprocessor.json", "policy_preprocessor.json",
                          "policy_preprocessor_step_4_normalizer_processor.safetensors", "tokenizer/tokenizer.json",
                          "train_config.json")], files
        assert not (out.parent / (out.name + ".partial")).exists() and "find_latest_checkpoint" in done.stdout
        U.use_repo()
        from online_rl_mimicgen.common import find_latest_checkpoint
        assert find_latest_checkpoint(out) == (out / "checkpoints/055000/pretrained_model").resolve()
        assert find_latest_checkpoint(run) == (run / "checkpoints/060000/pretrained_model").resolve(), \
            "a run directory resolves to its LAST step: why only the release step is installed"
        again = subprocess.run([sys.executable, "-W", "ignore", tool, *argv], env=_env(), capture_output=True, text=True)
        assert again.returncode != 0 and "never replaces a prior" in again.stderr, again.stderr
        verify = subprocess.run([sys.executable, "-W", "ignore", tool, "--task", task, "--verify", str(out), "--dataset", str(ds)],
                                env=_env(), capture_output=True, text=True)
        assert verify.returncode == 0 and "all checks passed" in verify.stdout, verify.stdout + verify.stderr
        wrong = subprocess.run([sys.executable, "-W", "ignore", tool, "--task", "square_d1", "--verify", str(out)],
                               env=_env(), capture_output=True, text=True)
        assert wrong.returncode != 0 and "not a square_d1 prior" in wrong.stdout, wrong.stdout
        print("synthetic run with steps 050000 / 055000 / 060000 and last -> 060000: only 055000/pretrained_model is "
              "installed (no training_state, no last), find_latest_checkpoint resolves to it (the run directory itself "
              "would give 060000); a second install is refused; --verify passes, and fails for another task")
    finally:
        shutil.rmtree(tmp)


def test_resume_argv_resolves():
    """The resume argv the wrapper prints resolves in LeRobot to the first resolution plus the resume fields only."""
    _need_lerobot()
    tmp = Path(tempfile.mkdtemp(prefix="test_prior_resume_"))
    try:
        task, roots = "square_d1", {"DATA_ROOT": str(tmp / "data"), "RUNS_ROOT": str(tmp / "runs")}
        _worker([{"mode": "resolve", "out": str(tmp / "first"), "argv": _train_argv(task, **roots)}], tmp)
        run = tmp / "runs" / "prior_train" / RUNS[task]["job_name"]
        (run / "checkpoints" / "005000" / "pretrained_model").mkdir(parents=True)
        shutil.copy(tmp / "first" / "train_config.json", run / "checkpoints" / "005000" / "pretrained_model")
        (run / "checkpoints" / "last").symlink_to("005000")
        argv = _train_argv(task, **roots)
        assert argv[-1] == "--resume=true"
        _worker([{"mode": "resolve", "out": str(tmp / "second"), "argv": argv}], tmp)
        a, b = (_flat(json.loads((tmp / d / "train_config.json").read_text())) for d in ("first", "second"))
        differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert differ == {"resume", "checkpoint_path", "policy.pretrained_path"}, differ
        assert b["resume"] is True and b["checkpoint_path"] == str(run / "checkpoints" / "last")
        print(f"{task}: the resume argv resolves to the first configuration with resume true, checkpoint_path "
              f"<run>/checkpoints/last and the policy read from its pretrained_model; nothing else changes")
    finally:
        shutil.rmtree(tmp)


def test_wandb_shim_lets_lerobot_log():
    """LeRobot 0.6.2's wandb logger calls Run.get_url, which the pinned wandb lacks; prior/lerobot_train.py adds it."""
    _need_lerobot()
    if importlib.util.find_spec("wandb") is None:
        U.skip("wandb is not installed (main environment)")
    source = (Path(importlib.util.find_spec("lerobot").submodule_search_locations[0]) / "common" / "wandb_utils.py").read_text()
    tmp = Path(tempfile.mkdtemp(prefix="test_prior_wandb_"))
    try:
        argv = _train_argv("square_d1", DATA_ROOT=str(tmp / "data"), RUNS_ROOT=str(tmp / "runs"))
        _worker([{"mode": "wandb", "out": str(tmp / "out"), "argv": argv}], tmp)
        report = json.loads((tmp / "out" / "wandb.json").read_text())
        assert report["with_shim"] == "ok", report
        if not report["run_has_get_url"]:
            assert "wandb.run.get_url()" in source and report["shim_added"], report
            assert report["without_shim"].startswith("AttributeError") and "get_url" in report["without_shim"], report
        print(f"wandb {report['wandb']}: Run.get_url {'present' if report['run_has_get_url'] else 'MISSING'}; LeRobot's "
              f"WandBLogger without the shim: {report['without_shim']}; with prior/lerobot_train.py's shim it starts "
              f"and logs (disabled mode)")
    finally:
        shutil.rmtree(tmp)


def test_recipe_resolves_to_released_train_config():
    _need_lerobot()
    present = [task for task in RUNS if (_released(task) / "train_config.json").is_file()]
    if not present:
        U.skip(f"no released prior below {U.root('CKPT_ROOT')}/priors (python scripts/fetch_assets.py --tier eval --only prior)")
    tmp = Path(tempfile.mkdtemp(prefix="test_prior_resolve_"))
    try:
        ns = _namespace()
        roots = {"DATA_ROOT": str(tmp / "data"), "RUNS_ROOT": str(tmp / "runs")}
        mode = {"WANDB_PROJECT": "proj", "WANDB_MODE": "offline", "HF_NAMESPACE": ns}
        jobs = [{"mode": "resolve", "out": str(tmp / task), "argv": _train_argv(task, **roots, **mode)} for task in RUNS]
        _worker(jobs, tmp)
        got = {task: _flat(json.loads((tmp / task / "train_config.json").read_text())) for task in RUNS}
        for task in present:
            rec, path = RUNS[task], _released(task) / "train_config.json"
            assert _sha256(path) == rec["train_config_sha256"], f"{path} is not the released train_config.json"
            released = _flat(json.loads(path.read_text()))
            mine = got[task]
            fork = rec["argv"] is not None
            allowed = {**ALLOWED_ALL, **(ALLOWED_FORK if fork else {})}
            used, unexplained = set(), []
            for key in sorted(set(mine) | set(released)):
                if key.startswith(("policy.input_features", "policy.output_features")):
                    continue                                  # filled from the dataset: test_recipe_builds_released_architecture
                if mine.get(key, "<absent>") == released.get(key, "<absent>"):
                    continue
                entry = _covering(key, allowed)
                if entry is not None:
                    used.add(entry)
                elif fork and _covering(key, NEW_IN_062) and key not in released:
                    assert mine[key] == got["square_d0"][key], (task, key)    # the value a 0.6.2-trained prior records
                    used.add(_covering(key, NEW_IN_062))
                else:
                    unexplained.append(f"{key}: released {released.get(key, '<absent>')!r}, recipe {mine.get(key, '<absent>')!r}")
            assert not unexplained, (task, unexplained)
            want = set(allowed) | (set(NEW_IN_062) if fork else set())
            assert used == want, f"{task}: allowances that no longer differ (stale) {sorted(want - used)}"
            ds = f"mimicgen_{task}_224x224_robotstate9"
            assert mine["dataset.root"] == f"{roots['DATA_ROOT']}/{ds}"
            assert mine["output_dir"] == f"{roots['RUNS_ROOT']}/prior_train/{rec['job_name']}"
            assert (mine["policy.device"], released["policy.device"]) == ("cpu", "cuda")
            assert mine["dataset.repo_id"] == f"{ns}/{ds}"
            if fork:
                assert released["dataset.repo_id"] == f"{ns}/mimicgen_{task}_224x224" and released["policy.n_state_dims"] == 9
                assert (released["env.type"], released["env.task"], mine["env"]) == ("mimicgen", task, None)
                assert (mine["wandb.project"], mine["wandb.mode"], mine["wandb.run_id"]) == ("proj", "offline", None)
                assert (released["wandb.project"], released["wandb.run_id"]) == (rec["wandb"]["project"], rec["wandb"]["run_id"])
            print(f"{task}: LeRobot resolves the wrapper's argv to the released train_config.json (SHA-256 "
                  f"{rec['train_config_sha256'][:12]}) except exactly {', '.join(sorted(want))}")
        if len(present) < len(RUNS):
            print(f"    not installed, so not compared: {sorted(set(RUNS) - set(present))}")
    finally:
        shutil.rmtree(tmp)


def _meta_copies(tmp: Path) -> dict:
    """{task: 9-D meta-only copy of its installed demo dataset} for every task whose demo meta/ and prior are present."""
    out = {}
    for task in RUNS:
        demo = U.root("DATA_ROOT") / f"mimicgen_{task}_224x224"
        if not ((demo / "meta" / "stats.json").is_file() and (_released(task) / "config.json").is_file()):
            continue
        target = tmp / "data" / f"mimicgen_{task}_224x224_robotstate9"
        done = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "prior/state9_dataset.py"), "--task", task,
                               "--src", str(demo), "--out", str(target), "--meta-only"], env=_env(), capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        out[task] = target
    return out


def test_released_priors_carry_the_9d_statistics():
    tmp = Path(tempfile.mkdtemp(prefix="test_prior_stats_"))
    try:
        copies = _meta_copies(tmp)
        if not copies:
            U.skip(f"no released prior with its demo dataset meta/ below {U.root('CKPT_ROOT')} and {U.root('DATA_ROOT')}")
        for task, copy in copies.items():
            demo = U.root("DATA_ROOT") / f"mimicgen_{task}_224x224"
            check = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "prior/state9_dataset.py"), "--task", task,
                                    "--src", str(demo), "--out", str(copy), "--check", "--prior", str(_released(task))],
                                   env=_env(), capture_output=True, text=True)
            assert check.returncode == 0, check.stdout + check.stderr
            prior_dir = U.root("CKPT_ROOT") / "priors" / RUNS[task]["prior"]
            verify = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "prior/install_prior.py"), "--task", task,
                                     "--verify", str(prior_dir), "--dataset", str(copy)], env=_env(), capture_output=True, text=True)
            assert verify.returncode == 0 and "all checks passed" in verify.stdout, verify.stdout + verify.stderr
            print(f"{task}: the 9-D copy of the demo dataset's meta/ carries exactly the released normaliser "
                  f"(observation.state and action); install_prior.py --verify passes on priors/{RUNS[task]['prior']}")
    finally:
        shutil.rmtree(tmp)


def test_recipe_builds_released_architecture():
    _need_lerobot()
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(FORK_TOKENIZER, "config.json"), str):
        U.skip(f"{FORK_TOKENIZER} is not in the Hugging Face cache (HF_HOME; scripts/fetch_assets.py --tier eval --only base)")
    from safetensors.numpy import load_file
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    import make_square_d1_statefix
    tmp = Path(tempfile.mkdtemp(prefix="test_prior_build_"))
    try:
        copies = _meta_copies(tmp)
        if not copies:
            U.skip(f"no released prior with its demo dataset meta/ below {U.root('CKPT_ROOT')} and {U.root('DATA_ROOT')}")
        roots = {"DATA_ROOT": str(tmp / "data"), "RUNS_ROOT": str(tmp / "runs"), "HF_NAMESPACE": _namespace()}
        _worker([{"mode": "build", "out": str(tmp / "out" / task), "argv": _train_argv(task, **roots)} for task in copies], tmp)
        for task in copies:
            mine, rel, fork = tmp / "out" / task, _released(task), RUNS[task]["argv"] is not None
            width = RUNS[task]["dataset"]["state_dim"]
            header = {k: v["shape"] for k, v in make_square_d1_statefix.safetensors_header(rel / "model.safetensors").items()}
            assert json.loads((mine / "weights.json").read_text()) == header, f"{task}: tensor names or shapes differ"
            a, b = (json.loads((d / "config.json").read_text()) for d in (mine, rel))
            assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"device"} and b["device"] == "cuda", task
            assert json.loads((mine / "policy_postprocessor.json").read_text()) == json.loads((rel / "policy_postprocessor.json").read_text())
            sa, sb = (json.loads((d / "policy_preprocessor.json").read_text())["steps"] for d in (mine, rel))
            assert [s["registry_name"] for s in sa] == [s["registry_name"] for s in sb], task
            for x, y in zip(sa, sb):
                if x == y:
                    continue
                if x["registry_name"] == "device_processor":
                    assert {**x["config"], "device": "cuda"} == y["config"], (task, x, y)
                elif x["registry_name"] == "tokenizer_processor" and fork:   # 0.6.2 saves the tokenizer beside the weights
                    assert {**x["config"], "tokenizer_name": FORK_TOKENIZER} == y["config"] and "artifacts" not in y, (task, y)
                else:
                    raise AssertionError(f"{task}: preprocessor step {x['registry_name']} differs: {x} != {y}")
            norm = "policy_preprocessor_step_4_normalizer_processor.safetensors"
            na, nb = load_file(str(mine / norm)), load_file(str(rel / norm))
            assert set(na) == set(nb) and all(np.array_equal(na[k], nb[k]) for k in na), f"{task}: normaliser differs"
            unnorm = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            ua, ub = load_file(str(mine / unnorm)), load_file(str(rel / unnorm))
            differ = sorted(k for k in ua if k not in ub or ua[k].shape != ub[k].shape or not np.array_equal(ua[k], ub[k]))
            assert set(ua) == set(ub), task
            if fork:      # the state fix never touched the postprocessor: it still holds the full state (unused there)
                assert differ == sorted(f"observation.state.{s}" for s in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")), (task, differ)
                assert all(ub[k].shape == (width,) and np.array_equal(ua[k], ub[k][:9]) for k in differ), task
            else:
                assert not differ, (task, differ)
            fa, fb = (json.loads((d / "train_config.json").read_text())["policy"] for d in (mine, rel))
            assert fa["output_features"] == fb["output_features"], task
            changed = {k for k in set(fa["input_features"]) | set(fb["input_features"]) if fa["input_features"].get(k) != fb["input_features"].get(k)}
            assert changed == ({"observation.state"} if fork else set()), (task, changed)
            if fork:
                assert (fa["input_features"]["observation.state"]["shape"], fb["input_features"]["observation.state"]["shape"]) == ([9], [width])
            print(f"{task}: {len(header)} tensors with the released names and shapes; config.json equal but device; "
                  f"postprocessor json equal; normaliser equal ({len(na)} tensors); preprocessor json "
                  + ("differs in device and the tokenizer saved beside the weights; the unnormaliser's 9 unused "
                     f"{width}-D state statistics are their first 9 here; train_config features: state [9] for [{width}]"
                     if fork else "differs in device only; unnormaliser equal; train_config features equal"))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    U.main(globals())
