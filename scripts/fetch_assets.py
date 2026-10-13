#!/usr/bin/env python3
"""Download the published assets to the paths configs/tasks.sh and environment.md (section 5) expect.

    python scripts/fetch_assets.py --tier eval|train|regenerate [--task TASK ...] [--dry-run | --check [--sha256]]
    bash scripts/fetch_assets.sh ...          # the same after `source scripts/env.sh`, with the HF offline switches off

What a tier holds (each tier includes the ones before it; sizes are for all four tasks):

  eval        evaluate the base prior or a trained checkpoint                                      about 5.3 GB
              the two base models (Hugging Face cache), meta/ of the demo dataset (state scaler), the frozen prior,
              the base and the continued world model, the V directories
  train       run RL cells                                                                         about 41 GB more
              the whole demo dataset and the task's online RoboMeter teacher LoRA (9.4 GB each; needed by the
              TD + distill cells only: --skip teacher leaves them out)
  regenerate  refit world models and V from the prior's rollouts                                   about 11 GB more
              the success / failure rollout datasets, the RoboMeter score tables, the offline square_d1 seg LoRA

--task        square_d0 square_d1 threading_d0 hammer_cleanup_d1, or all [default]
--only / --skip   comma-separated groups: base dataset prior wm value teacher rollouts scores cache runs
--dry-run     print every item (repository, folder, revision, target path, size, what is at the target now) and
              download nothing
--check       like --dry-run, but exit 1 unless every published item of the selection is in place with the Hub's
              file names and sizes (and the square_d1 state fix verifies); --sha256 also hashes every file (LFS
              files: SHA-256, others: git blob id). Locally built and unpublished items never fail it.
--latest      use each repository's current head instead of the revision recorded on 2026-10-12
--namespace   Hugging Face account that hosts the project repositories [HF_NAMESPACE, else the default of scripts/env.sh]
--no-hub      (--dry-run / --check) do not contact the Hub: recorded sizes, existence of the target only

Every item is one of
  repo / folder / file   downloaded: a whole repository, one folder of it (renamed to the target directory) or one
                         file. Files already at the target with the Hub's size are not downloaded again; a
                         repository's README.md and .gitattributes are left out.
  hf-cache               a public base model, into the Hugging Face cache ($HF_HOME) at the repository's head; the
                         revision the reference runs used is printed next to it
  derive                 made locally from a downloaded item: the square_d1 prior's 9-D state fix
                         (scripts/make_square_d1_statefix.py; run automatically after the download)
  build                  made locally by a command of this repository (pixel caches, frame caches); only reported
  unpublished            exists on the reference machine only; reported with what replaces it, never an error

Several repositories are private: log in first with an account that can read them (`hf auth login`, or HF_TOKEN).
Nothing is ever uploaded, and nothing outside the target paths, the Hugging Face cache and a .fetch_tmp_* staging
directory next to each target is written. An interrupted download resumes on the next call.
Afterwards:  python tests/configs/test_configs.py     (test_installed_assets lists what a campaign cell still misses)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIERS = ("eval", "train", "regenerate")
GROUPS = ("base", "dataset", "prior", "wm", "value", "teacher", "rollouts", "scores", "cache", "runs")
RECORDED = "2026-10-12"

# Per task: what configs/tasks.sh, dinowm/common_ws.py (ROLLOUTS) and robometer/teachers.sh name, and the Hub revision of
# every repository as listed on RECORDED (the files installed on the reference machine are these revisions).
TASKS = {
    "square_d0": dict(
        prior="square_d0_40k_rs9", step="040000", prior_repo="mimicgen_square_d0_mtdit_flow_40k_robotstate9",
        prior_rev="43fd6615124d129b1865a0b988e9a22f631b208a", prior_private=False, prior_mb=999.7,
        wm="150k", wm_rev="9ee80c7d0916b805de3a7704056e8a6dd5e07464", v_rev="39b005be45b025ac70bbd6319c7929683392d53b",
        seg=False, demo_rev="7214ccb9aabdc0bb8c1d3fc7e7907c6c2c4e6f5d", demo_mb=714.1, meta_mb=9.8,
        rollouts="mimicgen_square_d0_224x224_mtdit_flow_40k", success_rev="68707f4c72b6125b5886c8934b5e7b10937ce962",
        failure_rev="4bb156e59b94af7abf2192483429dc4bc4fba409", success_mb=67.2, failure_mb=165.0,
        lora="robometer-4b-lora-square-d0-seg", lora_rev="2d350c78a5ad2f348c22269359b2e1163a0f1158", lora_private=False,
        mc_ran=True, pixel_gb=46.2, frames_gb=16.9),
    "square_d1": dict(
        prior="square_d1_45k_statefix", step="045000", prior_repo="mimicgen_square_d1_flow_matching_prior_45k",
        prior_rev="c36377958364d0a8d15204316dd7ef32b2984044", prior_private=True, prior_mb=996.0,
        wm="150k", wm_rev="cb52a38c598fff144a8e03ec966985514f360f56", v_rev="fc87dd3ec7ea2491fda7384866fb7478f14f3956",
        seg=True, demo_rev="85ac2ae2a78b1158aef50a9ed5588621244556f6", demo_mb=650.9, meta_mb=10.1,
        rollouts="mimicgen_square_d1_224x224_mtdit_flow_45k", success_rev="ed9241f1c81cb0bf4e7ab173ffd9d9527352aa3e",
        failure_rev="9e12ebc87f8de6b766dadb504a31988b734143c2", success_mb=79.9, failure_mb=171.6,
        lora="robometer-4b-lora-square-d1", lora_rev="8dafd9137629d01504117d0146186dcbed758d28", lora_private=True,
        mc_ran=True, pixel_gb=45.9, frames_gb=17.6),
    "threading_d0": dict(
        prior="threading_d0_55k_statefix", step="055000", prior_repo="mimicgen_threading_d0_flow_matching_prior_55k_statefix",
        prior_rev="24d7bdd29163da680ee338a4d0a80a6ff1dd9c4c", prior_private=True, prior_mb=996.0,
        wm="168k", wm_rev="6e97c1c82d6bbd57bb19209e0b8cb3f9eaf9db90", v_rev="2c5bd96745aff545eb6ecaaa1706f8be790dc6ea",
        seg=True, demo_rev="f8d4e7697a8685ba6c98806bba6a99b32dc4f044", demo_mb=807.0, meta_mb=12.6,
        rollouts="mimicgen_threading_d0_224x224_mtdit_flow_55k", success_rev="0cd0bd2dddabcf3ef9ab775f4012cffbce808e52",
        failure_rev="a8960a5c0f9d3ba372c03c6030f2821c135405e6", success_mb=81.9, failure_mb=149.5,
        lora="robometer-4b-lora-threading-d0-seg", lora_rev="602e7cac5f850605e3b3cf71a8641caf6fe8b667", lora_private=False,
        mc_ran=False, pixel_gb=67.6, frames_gb=18.8),
    "hammer_cleanup_d1": dict(
        prior="hammer_cleanup_d1_35k_statefix", step="035000",
        prior_repo="mimicgen_hammer_cleanup_d1_flow_matching_prior_35k_statefix",
        prior_rev="36633fa7a31ce4f576141416cafaae8d0587d0fa", prior_private=True, prior_mb=996.0,
        wm="217k", wm_rev="fd9a64ae0412c638526e3562042882d86d4aa688", v_rev="25ce1f0fdd4cefd6ff7eb3416d3c1b1c592c8d7a",
        seg=False, demo_rev="7477bc233b7031f4ebb0022e0fdbc6849510f44d", demo_mb=1436.5, meta_mb=11.5,
        rollouts="mimicgen_hammer_cleanup_d1_224x224_mtdit_flow_35k", success_rev="9a0233786867b6430129ae0caa5015cbf9605b42",
        failure_rev="dbd1af4a88630c2fed14401a9919514ed08b70ce", success_mb=146.4, failure_mb=243.3,
        lora="robometer-4b-lora-hammer-cleanup-seg", lora_rev="037f662e74b36658fcbb266487bcd1c9ad1a5e52", lora_private=False,
        mc_ran=True, pixel_gb=86.4, frames_gb=24.0),
}
UNFIXED_SQUARE_D1 = "square_d1_45k"                  # where the published (un-fixed) square_d1 prior is kept
SEG_LORA = dict(name="robometer-4b-lora-square-d1-seg-c095", rev="9701d9eedccce15f167fed7ac05aa0d423d26cab")
# The base models every run loads by name, with the revision and the files the reference machine's cache holds.
BASE_MODELS = (
    dict(repo="facebook/dinov2-small", rev="ed25f3a31f01632728cabb09d1542f84ab7b0056", mb=88.3,
         files=("config.json", "model.safetensors"), what="DINOv2-S/14 (world model and V trunk)"),
    dict(repo="openai/clip-vit-base-patch16", rev="57c216476eefef5ab752ec549e440a49ae4ae5f3", mb=602.4,
         files=("config.json", "preprocessor_config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "special_tokens_map.json"),
         what="CLIP ViT-B/16 (vision and text encoder of every prior)"),
)
IGNORED = (".gitattributes", "README.md")           # repository housekeeping: not downloaded, not required at a target


@dataclass(frozen=True)
class Item:
    tiers: tuple          # the tiers that include it
    group: str            # one of GROUPS (--only / --skip)
    task: str | None      # None: needed whatever the task
    what: str
    kind: str             # repo | folder | file | hf-cache | derive | build | unpublished
    root: str             # DATA_ROOT | CKPT_ROOT | CACHE_ROOT | RUNS_ROOT | HF_CACHE
    target: str           # path below the root (a directory; kind file: the file)
    repo: str = ""        # repository name below the namespace; a full id for the public base models
    repo_type: str = "model"
    revision: str = ""
    folder: str = ""      # kind folder: the folder in the repository; kind file: the file in the repository
    files: tuple = ()     # kind hf-cache: the files to fetch
    mb: float = 0.0       # size recorded on RECORDED
    private: bool = False
    note: str = ""        # who needs it; for derive / build / unpublished: how to get it

    def repo_id(self, namespace: str) -> str:
        return self.repo if "/" in self.repo else f"{namespace}/{self.repo}"


def default_namespace() -> str:
    """HF_NAMESPACE, else the default written in scripts/env.sh (the one place that names the account)."""
    if os.environ.get("HF_NAMESPACE"):
        return os.environ["HF_NAMESPACE"]
    match = re.search(r"^export HF_NAMESPACE=\$\{HF_NAMESPACE:-([\w.-]+)\}", (REPO / "scripts" / "env.sh").read_text(), re.M)
    if not match:
        raise SystemExit("HF_NAMESPACE is not set and scripts/env.sh names no default; pass --namespace")
    return match.group(1)


def roots() -> dict[str, Path]:
    """The asset roots exactly as scripts/env.sh derives them."""
    assets = Path(os.environ.get("ASSET_ROOT", REPO / "assets"))
    return {"DATA_ROOT": Path(os.environ.get("DATA_ROOT", assets / "datasets")),
            "CKPT_ROOT": Path(os.environ.get("CKPT_ROOT", assets / "checkpoints")),
            "CACHE_ROOT": Path(os.environ.get("CACHE_ROOT", assets / "cache")),
            "RUNS_ROOT": Path(os.environ.get("RUNS_ROOT", assets / "runs"))}


def table() -> list[Item]:
    """Every asset of the campaign, in the order it is fetched."""
    every, train_up, regen = TIERS, TIERS[1:], TIERS[2:]
    out = [Item(every, "base", None, m["what"], "hf-cache", "HF_CACHE", m["repo"], repo=m["repo"], revision=m["rev"],
                files=m["files"], mb=m["mb"], note="read by name at start-up; must be cached before HF_OFFLINE=1")
           for m in BASE_MODELS]
    for task, t in TASKS.items():
        demo, tag = f"mimicgen_{task}_224x224", f"ph_conv_{t['wm']}"
        cont = f"{tag}_cont{t['wm']}"
        wm_repo, v_repo = f"{demo}_dinowm_ph", f"v_dino_robometer_pred_progress_{task}"
        v_dir = f"value/{v_repo}_{tag}_mqattn_pv4"
        prior_dir = f"priors/{t['prior']}/checkpoints/{t['step']}/pretrained_model"
        out.append(Item(("eval",), "dataset", task, "demo dataset, meta/ only", "folder", "DATA_ROOT", f"{demo}/meta",
                        repo=demo, repo_type="dataset", revision=t["demo_rev"], folder="meta", mb=t["meta_mb"],
                        note="meta/stats.json is the state scaler of every evaluator"))
        out.append(Item(train_up, "dataset", task, "demo dataset (1000 episodes)", "repo", "DATA_ROOT", demo, repo=demo,
                        repo_type="dataset", revision=t["demo_rev"], mb=t["demo_mb"],
                        note="state scaler, expert buffer, source of the pixel cache"))
        if task == "square_d1":
            unfixed = f"priors/{UNFIXED_SQUARE_D1}/checkpoints/{t['step']}/pretrained_model"
            out.append(Item(every, "prior", task, "frozen prior as published (un-fixed, 26-D state)", "repo", "CKPT_ROOT",
                            unfixed, repo=t["prior_repo"], revision=t["prior_rev"], mb=t["prior_mb"], private=True,
                            note="input of the state fix; no cell loads it"))
            out.append(Item(every, "prior", task, "frozen prior with the 9-D state fix", "derive", "CKPT_ROOT", prior_dir,
                            note="python scripts/make_square_d1_statefix.py   (the fixed prior is on no Hub repository)"))
        else:
            out.append(Item(every, "prior", task, "frozen prior", "repo", "CKPT_ROOT", prior_dir, repo=t["prior_repo"],
                            revision=t["prior_rev"], mb=t["prior_mb"], private=t["prior_private"]))
        base_user = ("every V was fitted on it; DICE-RL DVD loads it" if task == "hammer_cleanup_d1"
                     else "every V was fitted on it; no RL cell loads it")
        cont_user = "DSRL DVD arms" if task == "hammer_cleanup_d1" else "DICE-RL DVD and the DSRL DVD arms"
        for name, user in ((tag, base_user), (cont, cont_user)):
            out.append(Item(every, "wm", task, f"world model {name}", "folder", "CKPT_ROOT", f"wm/{demo}_dinowm_{name}",
                            repo=wm_repo, revision=t["wm_rev"], folder=name, mb=57.6, private=True, note=user))
        pv4_user = "DICE-RL DVD" + ("" if t["seg"] else "; DSRL ours / oursdist / randv")
        out.append(Item(every, "value", task, "V, RoboMeter progress (pv4)", "folder", "CKPT_ROOT", v_dir, repo=v_repo,
                        revision=t["v_rev"], folder=f"actual_pred_{tag}_mqattn_pv4", mb=13.5, private=True, note=pv4_user))
        if t["seg"]:
            out.append(Item(every, "value", task, "V, progress from the segmented LoRA (pv4_seg)", "folder", "CKPT_ROOT",
                            f"{v_dir}_seg", repo=v_repo, revision=t["v_rev"], folder=f"actual_pred_{tag}_mqattn_pv4_seg",
                            mb=13.5, private=True, note="DSRL ours / oursdist / randv"))
        out.append(Item(every, "value", task, "V, Monte Carlo return (mc)", "unpublished", "CKPT_ROOT",
                        f"value/v_dino_mc_return_{task}_{tag}_mqattn_pv4",
                        note=("DSRL oursmc" if t["mc_ran"] else "DSRL oursmc (defined, never run in the campaign)") +
                             f". Refit an equivalent, not identical, one: TASK={task} RECIPE=mc bash value/fit_v.sh "
                             f"(needs the regenerate tier and the frame cache)"))
        out.append(Item(train_up, "teacher", task, "online RoboMeter teacher LoRA", "repo", "CKPT_ROOT",
                        f"robometer/{t['lora']}", repo=t["lora"], revision=t["lora_rev"], mb=9442.8,
                        private=t["lora_private"], note="TD + distill cells only (DICE-RL dvd with DISTILL=tddistill, DSRL oursdist)"))
        out.append(Item(train_up, "cache", task, "demo pixel cache", "build", "CACHE_ROOT",
                        f"pixel_cache/pixel_cache_{demo}_224.npy", mb=t["pixel_gb"] * 1000,
                        note=f"bash scripts/build_pixel_cache.sh {task}   (every RL cell; needs the whole demo dataset)"))
        for split in ("success", "failure"):
            out.append(Item(regen, "rollouts", task, f"prior rollouts, {split} (100 episodes)", "repo", "DATA_ROOT",
                            f"{t['rollouts']}_{split}", repo=f"{t['rollouts']}_{split}", repo_type="dataset",
                            revision=t[f"{split}_rev"], mb=t[f"{split}_mb"], note="frame cache, world model, V fit"))
        out.append(Item(regen, "scores", task, "RoboMeter score table (pv4 and mc recipes)", "file", "CKPT_ROOT",
                        f"rm_scores/{task}/predictions.parquet", repo=v_repo, revision=t["v_rev"],
                        folder="data/predictions.parquet", mb=0.13, private=True))
        if t["seg"]:
            how = ("the same LoRA scored the same rows again, so the scores equal rm_scores/threading_d0 in another row "
                   "order" if task == "threading_d0" else "scored by the segmented LoRA, teacher square_d1_seg")
            out.append(Item(regen, "scores", task, "RoboMeter score table of the pv4_seg recipe", "unpublished", "CKPT_ROOT",
                            f"rm_scores/{task}_seg/predictions.parquet",
                            note=f"{how}. Rescore: robometer/score_rollouts.py with --rows (its docstring), teacher from "
                                 f"robometer/serve.sh"))
        if task == "square_d1":
            out.append(Item(regen, "teacher", task, "segmented LoRA (offline rescoring for pv4_seg)", "repo", "CKPT_ROOT",
                            f"robometer/{SEG_LORA['name']}", repo=SEG_LORA["name"], revision=SEG_LORA["rev"], mb=9442.8,
                            private=True, note="teacher square_d1_seg; no cell distils from it online"))
        out.append(Item(regen, "cache", task, "prior-rollout frame cache", "build", "CACHE_ROOT",
                        f"wm_frames/{task}_frames_u8.npy", mb=t["frames_gb"] * 1000,
                        note=f"WS_TASK={task} python dinowm/build_cache.py   (about 30 s; needs both rollout datasets)"))
    out.append(Item(train_up, "cache", None, "DSRL demo stores (0.6-1.2 GB per task)", "build", "CACHE_ROOT", "dsrl_demo_store",
                    note="(nothing to run) the first DSRL run of a task builds its store in about 20 min; ahead of time: "
                         "python -m online_rl_mimicgen.baselines.build_demo_store. Stores are not portable between machines"))
    out.append(Item(every, "runs", None, "checkpoints, config.json and eval results of the reference runs", "unpublished",
                    "RUNS_ROOT", "", note="reference/results_*.md holds their numbers; no reference checkpoint can be "
                                          "re-evaluated on another machine until they are published"))
    return out


def select(tier: str, tasks: list[str], only: set[str], skip: set[str]) -> list[Item]:
    return [item for item in table()
            if tier in item.tiers and (item.task is None or item.task in tasks)
            and (not only or item.group in only) and item.group not in skip]


# ---------------------------------------------------------------------------------------------------------------- Hub
class Hub:
    """Read-only access to the Hub: file listings (cached per repository) and downloads."""

    def __init__(self, namespace: str, latest: bool, workers: int):
        from huggingface_hub import HfApi
        self.api, self.namespace, self.latest, self.workers = HfApi(), namespace, latest, workers
        self._listings: dict = {}
        self.warnings: list[str] = []
        try:
            self.user = self.api.whoami()["name"]
        except Exception:                                   # no token, or no network: public repositories still work
            self.user = None

    def listing(self, item: Item) -> tuple[str, dict[str, tuple[int, str | None, str | None]]]:
        """(commit, {path in the repository: (size, lfs sha256 or None, git blob id or None)}) at the item's revision."""
        from huggingface_hub.errors import RevisionNotFoundError
        repo_id = item.repo_id(self.namespace)
        revision = None if self.latest or item.kind == "hf-cache" else (item.revision or None)
        key = (repo_id, item.repo_type, revision)
        if key not in self._listings:
            try:
                info = self.api.repo_info(repo_id, repo_type=item.repo_type, revision=revision, files_metadata=True)
            except RevisionNotFoundError:
                self.warnings.append(f"{repo_id}: the recorded revision {revision[:12]} is not in this repository; its "
                                     f"current head is used")
                info = self.api.repo_info(repo_id, repo_type=item.repo_type, files_metadata=True)
            files = {s.rfilename: (int(s.size or 0), getattr(s.lfs, "sha256", None) if s.lfs else None, s.blob_id)
                     for s in info.siblings}
            self._listings[key] = (info.sha, files)
        return self._listings[key]

    def files(self, item: Item) -> tuple[str, dict[str, tuple]]:
        """(revision, {path RELATIVE TO THE TARGET: (repository path, size, sha256, blob id)}) of what an item holds."""
        sha, listing = self.listing(item)
        out = {}
        for path, (size, lfs, blob) in listing.items():
            if Path(path).name in IGNORED:
                continue
            if item.kind == "repo":
                out[path] = (path, size, lfs, blob)
            elif item.kind == "folder" and path.startswith(item.folder + "/"):
                out[path[len(item.folder) + 1:]] = (path, size, lfs, blob)
            elif item.kind == "file" and path == item.folder:
                out[""] = (path, size, lfs, blob)
            elif item.kind == "hf-cache" and path in item.files:
                out[path] = (path, size, lfs, blob)
        return sha, out

    def download(self, item: Item, sha: str, wanted: dict[str, tuple], target: Path) -> None:
        """Fetch `wanted` ({relative path: (repository path, ...)}) into a staging directory, then move it to the target."""
        from huggingface_hub import snapshot_download
        stage = target.parent / f".fetch_tmp_{target.name or 'file'}"
        stage.mkdir(parents=True, exist_ok=True)
        snapshot_download(item.repo_id(self.namespace), repo_type=item.repo_type, revision=sha,
                          allow_patterns=[source for source, *_ in wanted.values()], local_dir=stage,
                          max_workers=self.workers)
        for relative, (source, *_) in wanted.items():
            final = target / relative if relative else target
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage / source), str(final))
        shutil.rmtree(stage, ignore_errors=True)

    def cache_model(self, item: Item) -> str:
        """A base model into the Hugging Face cache at its head (so that it is found by name offline); returns the revision."""
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(item.repo, allow_patterns=list(item.files), max_workers=self.workers)).name


def hub_cache() -> Path:
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.environ.get("HF_HUB_CACHE", home / "hub"))


def cached_base_model(item: Item) -> tuple[int, str]:
    """(how many of the item's files the Hugging Face cache resolves BY NAME, i.e. as an offline run would; the snapshot)."""
    from huggingface_hub import try_to_load_from_cache
    found = {}
    for name in item.files:
        path = try_to_load_from_cache(item.repo, name)
        if isinstance(path, str):
            found[name] = path
    snapshots = sorted({Path(path).parents[len(Path(name).parts) - 1].name for name, path in found.items()})
    return len(found), "/".join(snapshots)


def git_blob_id(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(target: Path, files: dict[str, tuple], hashes: bool) -> tuple[dict[str, tuple], list[str]]:
    """(the files still to download, one line per file that is absent or differs) of a target against the Hub listing."""
    todo, problems = {}, []
    for relative, entry in files.items():
        _, size, lfs, blob = entry
        path = target / relative if relative else target
        if not path.is_file():
            todo[relative] = entry
            problems.append(f"missing: {relative or path.name}")
        elif path.stat().st_size != size:
            todo[relative] = entry
            problems.append(f"size {path.stat().st_size} != {size}: {relative or path.name}")
        elif hashes and ((lfs and sha256(path) != lfs) or (not lfs and blob and git_blob_id(path) != blob)):
            todo[relative] = entry
            problems.append(f"content differs from the Hub's: {relative or path.name}")
    return todo, problems


def exists(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def statefix_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_square_d1_statefix
    return make_square_d1_statefix


def statefix_state(out: Path) -> tuple[bool, str]:
    if not (out / "config.json").is_file():
        return False, "missing"
    with contextlib.redirect_stdout(io.StringIO()) as log:
        good = statefix_module().verify(out, None, False)
    if good:
        return True, "present, verified" + (" (SHA-256 warnings)" if "warn" in log.getvalue() else "")
    return False, "present but FAILS scripts/make_square_d1_statefix.py --verify"


def handle(item: Item, target: Path, hub: Hub | None, namespace: str, root: dict, report_only: bool, hashes: bool):
    """Report on one item and, unless report_only, fetch or make it. -> (source, size in MB, state, bucket, MB to fetch)."""
    if item.kind in ("repo", "folder", "file"):
        source = item.repo_id(namespace) + (f" / {item.folder}" if item.folder else "")
        if hub is None:
            present = exists(target)
            return (source, item.mb, "present (not compared with the Hub)" if present else "missing",
                    "present" if present else "to fetch", 0.0 if present else item.mb)
        sha, files = hub.files(item)
        if not files:
            raise RuntimeError(f"{source} holds no such file at revision {sha[:12]}")
        source += f" @{sha[:12]}"
        size = sum(entry[1] for entry in files.values()) / 1e6
        todo, problems = compare(target, files, hashes)
        todo_mb = sum(entry[1] for entry in todo.values()) / 1e6
        if not todo:
            return (source, size, f"present: {len(files)} file(s), names, sizes" + (" and hashes" if hashes else "") +
                    " equal the Hub's", "present", 0.0)
        if report_only:
            if len(todo) == len(files) and not exists(target):
                return source, size, f"missing ({len(files)} file(s))", "to fetch", todo_mb
            return (source, size, f"INCOMPLETE: {len(files) - len(todo)} of {len(files)} file(s) in place; "
                    + "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else ""), "incomplete", todo_mb)
        print(f"  downloading {len(todo)} file(s), {fmt_size(todo_mb)}: {source}", flush=True)
        hub.download(item, sha, todo, target)
        left, problems = compare(target, files, False)
        if left:
            raise RuntimeError("after the download: " + "; ".join(problems[:4]))
        return source, size, f"fetched {len(todo)} file(s), {fmt_size(todo_mb)}", "fetched", 0.0

    if item.kind == "hf-cache":
        source = f"{item.repo} (the reference runs used revision {item.revision[:12]})"
        have, snapshot = cached_base_model(item)
        bucket = "present" if have == len(item.files) else "to fetch"
        if bucket == "to fetch" and hub is not None and not report_only:
            print(f"  caching {item.repo}", flush=True)
            hub.cache_model(item)
            have, snapshot = cached_base_model(item)
            bucket = "fetched"
        state = f"{have} of {len(item.files)} files in the cache"
        if have:
            state += f", snapshot {snapshot[:12]}" + ("" if snapshot == item.revision else
                                                      f" -- NOT the reference revision {item.revision[:12]}")
        if hub is not None:
            head, _ = hub.listing(item)
            if head != item.revision:
                hub.warnings.append(f"{item.repo}: its head is now {head[:12]}; the reference runs used "
                                    f"{item.revision[:12]} (environment.md, section 2)")
        return source, item.mb, state, bucket, item.mb if bucket == "to fetch" else 0.0

    if item.kind == "derive":
        good, state = statefix_state(target)
        unfixed = root["CKPT_ROOT"] / "priors" / UNFIXED_SQUARE_D1 / "checkpoints" / TASKS["square_d1"]["step"] / "pretrained_model"
        if state == "missing" and not report_only and (unfixed / "config.json").is_file():
            fix = statefix_module()
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    fix.build(unfixed, target, "hardlink", False)
                except SystemExit:                           # another file system: copy the weights instead
                    try:
                        fix.build(unfixed, target, "copy", True)
                    except SystemExit as error:
                        raise RuntimeError(str(error)) from None
            good, state = statefix_state(target)
            state = state.replace("present", "made")
        return "made locally from the item above", 0.0, state, "present" if good else "local step", 0.0

    if item.kind == "build":
        present = exists(target)
        return "built locally", item.mb, "present" if present else "not built yet", "present" if present else "local step", 0.0

    present = bool(item.target) and exists(target)
    return ("NOT ON THE HUB", 0.0, "present here" if present else "unpublished: only on the reference machine",
            "present" if present else "unpublished", 0.0)


def fmt_size(mb: float) -> str:
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1000:.0f} kB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--task", nargs="+", default=["all"], choices=(*TASKS, "all"), metavar="TASK")
    parser.add_argument("--only", default="", metavar="GROUPS")
    parser.add_argument("--skip", default="", metavar="GROUPS")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads per item [8]")
    args = parser.parse_args()

    only, skip = ({g for g in value.split(",") if g} for value in (args.only, args.skip))
    if (only | skip) - set(GROUPS):
        parser.error(f"unknown group(s) {sorted((only | skip) - set(GROUPS))}; groups: {' '.join(GROUPS)}")
    if args.sha256 and not args.check:
        parser.error("--sha256 belongs to --check")
    report_only = args.dry_run or args.check
    if args.no_hub and not report_only:
        parser.error("--no-hub belongs to --dry-run or --check")
    tasks = list(TASKS) if "all" in args.task else args.task
    namespace = args.namespace or default_namespace()
    root = roots()
    items = select(args.tier, tasks, only, skip)

    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
    if offline and not report_only:
        raise SystemExit("HF_HUB_OFFLINE is set (HF_OFFLINE=1 before scripts/env.sh), so nothing can be downloaded. Run\n"
                         "  bash scripts/fetch_assets.sh ...     (clears the offline switches for this command), or\n"
                         "  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE python scripts/fetch_assets.py ...")
    hub = None
    if not args.no_hub and not offline:
        hub = Hub(namespace, args.latest, args.workers)
    mode = "dry run" if args.dry_run else "check" if args.check else "download"
    print(f"{mode}: tier {args.tier}, task(s) {' '.join(tasks)}; Hugging Face namespace {namespace}"
          + ("" if hub else "; the Hub is NOT contacted (recorded sizes, existence only)")
          + (f"; logged in as {hub.user}" if hub and hub.user else ""))
    for name in ("DATA_ROOT", "CKPT_ROOT", "CACHE_ROOT"):
        print(f"  {name}={root[name]}")
    if hub and not hub.user and any(item.private for item in items):
        print("  NOT LOGGED IN: the private repositories below will fail. Run `hf auth login` (or set HF_TOKEN) with an "
              f"account that can read {namespace}/*")

    totals = dict.fromkeys(("present", "fetched", "to fetch", "incomplete", "failed", "local step", "unpublished"), 0)
    fetch_mb, steps = 0.0, []
    for item in items:
        if item.root == "HF_CACHE":
            target = hub_cache() / ("models--" + item.repo.replace("/", "--"))
        else:
            target = root[item.root] / item.target if item.target else root[item.root]
        try:
            source, size, state, bucket, todo_mb = handle(item, target, hub, namespace, root, report_only, args.sha256)
        except KeyboardInterrupt:
            raise SystemExit("\ninterrupted; the next call resumes (staging directories .fetch_tmp_* are kept)")
        except Exception as error:                           # one failing repository must not hide the others
            kind = type(error).__name__
            hint = (" (not found or no access: private repositories need `hf auth login` with an account that can read "
                    f"{namespace})" if kind in ("RepositoryNotFoundError", "GatedRepoError") else "")
            source = item.repo_id(namespace) if item.repo else ""
            size, bucket, todo_mb = item.mb, "failed", 0.0
            state = f"FAILED {kind}{hint}: {(str(error).splitlines() or [''])[0][:160]}"
        totals[bucket] += 1
        fetch_mb += todo_mb
        if bucket == "local step":
            steps.append(item.note)
        print(f"[{item.tiers[0]:<10}] {item.task or 'all tasks':<17} {item.what}")
        print(f"    {item.kind:<11} {source}{'   ' + fmt_size(size) if size else ''}{'   private' if item.private else ''}")
        print(f"    -> {target}")
        print(f"    {state}")
        if item.note and item.kind in ("repo", "folder", "file", "hf-cache"):
            print(f"    for: {item.note}")
        elif item.note and bucket in ("local step", "unpublished"):
            print(f"    how: {item.note}")

    print()
    for warning in (hub.warnings if hub else []):
        print(f"WARNING {warning}")
    print("summary: " + ", ".join(f"{count} {name}" for name, count in totals.items() if count)
          + (f"; {fmt_size(fetch_mb)} to download" if report_only and fetch_mb else ""))
    if steps:
        print("local steps still to run:")
        for line in dict.fromkeys(steps):
            print(f"  {line}")
    print("then check what a campaign cell still misses:  python tests/configs/test_configs.py")
    if totals["failed"]:
        return 1
    if args.check and (totals["to fetch"] or totals["incomplete"] or
                       any(item.kind == "derive" and not statefix_state(root[item.root] / item.target)[0] for item in items)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
