#!/usr/bin/env python3
"""Derive the training set of a task's prior from its demo dataset: the same dataset with a 9-D observation.state.

    python prior/state9_dataset.py --task TASK [--src DIR] [--out DIR] [--videos hardlink|copy|symlink]
    python prior/state9_dataset.py --task TASK --check [--src DIR] [--out DIR] [--prior DIR]
    python prior/state9_dataset.py --task TASK --meta-only --out DIR        # meta/ only: for CPU checks, not training
    TASK=<task> STAGE=data bash prior/train_prior.sh                        # the first line with the default paths

WHY. Every released prior conditions on the first 9 dims of the demo dataset's observation.state: eef_pos 3 |
eef_quat 4 | gripper_qpos 2. The rest is privileged object state (square_d0 23 = 9 + 14, square_d1 26 = 9 + 17,
threading_d0 and hammer_cleanup_d1 37 = 9 + 28). square_d0's prior was trained by stock LeRobot on such a copy of its
dataset ("robotstate9"); the other three by a LeRobot fork that cut the state inside the policy after normalising it
(--policy.n_state_dims=9), which is not available. Stock LeRobot on this copy builds the same 9-wide network with the
same normaliser: min-max normalisation is elementwise, so normalising 26 / 37 dims and keeping 9 equals normalising 9.

WHAT IS WRITTEN to --out [$DATA_ROOT/mimicgen_<task>_224x224_robotstate9] from --src [$DATA_ROOT/mimicgen_<task>_224x224]:
  data/chunk-*/file-*.parquet       observation.state (fixed width D, the width info.json declares) cut to its first 9
                                    values; every other column, the row order and the file layout unchanged
  meta/info.json                    features["observation.state"].shape [D] -> [9] (its names cut too, if listed)
  meta/stats.json                   observation.state: every statistic of length D cut to its first 9; count unchanged
  meta/episodes/chunk-*/*.parquet   the per-episode columns stats/observation.state/<stat> cut the same way
  meta/tasks.parquet                copied
  videos/**/*.mp4                   hardlinked [default; a copy where the two directories are on different file
                                    systems], copied, or symlinked (absolute link). Always the RESOLVED source file: in a
                                    Hugging Face snapshot every file is a relative link into blobs/, so linking or copying
                                    the link itself would leave a dangling one
  left out                          README.md, .gitattributes, .cache/ (download bookkeeping)
Anything else in the source (another file under meta/, another top-level directory) is refused: it might hold state
statistics this script does not know how to cut. The source width must be the task's (23 / 26 / 37 / 37).
The copy is written as <out>.partial and renamed when it passed --check. An existing --out is never overwritten: one
that passes --check is reported as already built (exit 0), any other is refused. A leftover <out>.partial is removed.

--check      compares --out with --src and writes nothing: shapes; every cut statistic and every cut state value equals
             the first 9 entries of the source's; every other column, statistic and info field is equal; frame and
             episode counts; every video resolves to a file of the source's size. With --prior <pretrained_model dir>:
             the dataset's observation.state and action statistics equal the prior's normaliser tensors exactly (they
             do, to the bit, for all four released priors). Exit 1 on any difference.
--meta-only  writes meta/ only: enough for LeRobotDatasetMetadata, i.e. to build the policy and its processors on CPU
             (tests/pipeline/test_prior_recipe.py). Needs an explicit --out; prior/train_prior.sh refuses it as a
             training set. --check of a meta-only copy checks meta/ only.

Needs pyarrow and numpy (and safetensors for --prior), all in the main environment. CPU; seconds per task.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

STATE = "observation.state"
KEEP = 9
SOURCE_DIM = {"square_d0": 23, "square_d1": 26, "threading_d0": 37, "hammer_cleanup_d1": 37}   # TaskSpec state_dim
SKIPPED = {"README.md", ".gitattributes", ".cache"}           # top-level entries never copied
META_FILES = {"info.json", "stats.json", "tasks.parquet", "episodes"}
REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    assets = Path(os.environ.get("ASSET_ROOT", REPO / "assets"))
    return Path(os.environ.get("DATA_ROOT", assets / "datasets"))


def default_src(task: str) -> Path:
    return data_root() / f"mimicgen_{task}_224x224"


def default_out(task: str) -> Path:
    return data_root() / f"mimicgen_{task}_224x224_robotstate9"


def real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def stat_column(name: str) -> bool:
    """A per-episode statistic of the state that has one value per state dim (count has one value per episode)."""
    return name.startswith(f"stats/{STATE}/") and not name.endswith("/count")


def cut_list(array, width: int):
    """A fixed-width list column (pyarrow Array or ChunkedArray) cut to its first KEEP values; same Arrow type."""
    import pyarrow.compute as pc
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if array.null_count:
        raise SystemExit(f"{array.null_count} null rows in a list column of width {width}")
    lengths = pc.min_max(pc.list_value_length(array)).as_py()
    if lengths["min"] != width or lengths["max"] != width:
        raise SystemExit(f"list column has lengths {lengths['min']}..{lengths['max']}, info.json declares {width}")
    return pc.list_slice(array, 0, KEEP).cast(array.type)


def cut_table(table, names, width: int):
    for name in names:
        index = table.schema.get_field_index(name)
        table = table.set_column(index, table.schema.field(index), cut_list(table.column(index), width))
    return table


def source_layout(src: Path, task: str) -> tuple[dict, int]:
    """(info.json, state width) of a source dataset, after the refusals of the module docstring."""
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"no dataset at {src} (meta/info.json missing). Fetch it first:\n"
                         f"  python scripts/fetch_assets.py --tier train --task {task} --only dataset")
    unknown = sorted(p.name for p in src.iterdir() if p.name not in SKIPPED | {"data", "meta", "videos"})
    unknown += sorted(f"meta/{p.name}" for p in (src / "meta").iterdir() if p.name not in META_FILES)
    if unknown:
        raise SystemExit(f"{src} holds entries this script does not know how to cut: {unknown}")
    info = json.loads((src / "meta" / "info.json").read_text())
    feature = info["features"].get(STATE)
    if not feature or len(feature.get("shape", [])) != 1:
        raise SystemExit(f"{src}: no 1-D {STATE} feature in meta/info.json")
    width = feature["shape"][0]
    if width != SOURCE_DIM[task]:
        raise SystemExit(f"{src}: {STATE} has {width} dims; the {task} demo dataset has {SOURCE_DIM[task]}")
    names = feature.get("names")
    if names is not None and not (isinstance(names, list) and len(names) == width):
        raise SystemExit(f"{src}: {STATE} names {names!r} are not a list of {width}; refusing to guess how to cut them")
    return info, width


def cut_info(info: dict) -> dict:
    out = json.loads(json.dumps(info))
    out["features"][STATE]["shape"] = [KEEP]
    if out["features"][STATE].get("names") is not None:
        out["features"][STATE]["names"] = out["features"][STATE]["names"][:KEEP]
    return out


def cut_stats(stats: dict, width: int) -> dict:
    out = json.loads(json.dumps(stats))
    for key, value in out[STATE].items():
        if key == "count":
            continue
        if not (isinstance(value, list) and len(value) == width):
            raise SystemExit(f"stats.json: {STATE}.{key} is not a list of {width}")
        out[STATE][key] = value[:KEEP]
    return out


def files(root: Path, sub: str, pattern: str) -> list[Path]:
    """Relative paths of the files below root/sub matching pattern (links followed), sorted."""
    return sorted(p.relative_to(root) for p in (root / sub).rglob(pattern)) if (root / sub).is_dir() else []


def place_video(src: Path, dst: Path, how: str) -> str:
    source = real(src)
    if how == "symlink":
        dst.symlink_to(source)
        return "symlinked"
    if how == "hardlink":
        try:
            os.link(source, dst)
            return "hardlinked"
        except OSError:
            pass                                               # another file system, or links not allowed: copy
    shutil.copyfile(source, dst)
    return "copied"


def build(task: str, src: Path, out: Path, videos: str, meta_only: bool) -> None:
    import pyarrow.parquet as pq

    info, width = source_layout(src, task)
    if out.resolve() == src.resolve():
        raise SystemExit("--src and --out are the same directory")
    if out.exists() or out.is_symlink():
        problems = verify(src, out, None, meta_only)
        if not problems:
            print(f"already built: {out} ({STATE} {width} -> {KEEP}, passes --check)")
            return
        raise SystemExit(f"{out} exists and does not pass --check ({problems[0]}{' ...' if len(problems) > 1 else ''}); "
                         f"it is never overwritten: remove it yourself to rebuild")
    partial = out.with_name(out.name + ".partial")
    if partial.exists() or partial.is_symlink():
        shutil.rmtree(partial) if partial.is_dir() and not partial.is_symlink() else partial.unlink()
    (partial / "meta").mkdir(parents=True)

    (partial / "meta" / "info.json").write_text(json.dumps(cut_info(info), indent=4))
    stats = json.loads((src / "meta" / "stats.json").read_text())
    (partial / "meta" / "stats.json").write_text(json.dumps(cut_stats(stats, width), indent=4))
    shutil.copyfile(real(src / "meta" / "tasks.parquet"), partial / "meta" / "tasks.parquet")
    episode_files = files(src, "meta/episodes", "*.parquet")
    if not episode_files:
        raise SystemExit(f"{src}: no meta/episodes/*.parquet")
    for relative in episode_files:
        table = pq.read_table(real(src / relative))
        table = cut_table(table, [name for name in table.column_names if stat_column(name)], width)
        (partial / relative).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, partial / relative)
    done = {"data files": 0}
    if not meta_only:
        data_files = files(src, "data", "*.parquet")
        if not data_files:
            raise SystemExit(f"{src} holds no data/*.parquet: fetch the whole dataset (--tier train), not meta/ only")
        for relative in data_files:
            table = cut_table(pq.read_table(real(src / relative)), [STATE], width)
            (partial / relative).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, partial / relative)
            done["data files"] += 1
        video_files = files(src, "videos", "*.mp4")
        if not video_files and any(f.get("dtype") == "video" for f in info["features"].values()):
            raise SystemExit(f"{src} holds no videos/**/*.mp4 although info.json declares video features")
        for relative in video_files:
            (partial / relative).parent.mkdir(parents=True, exist_ok=True)
            how = place_video(src / relative, partial / relative, videos)
            done[f"videos {how}"] = done.get(f"videos {how}", 0) + 1
    problems = verify(src, partial, None, meta_only)
    if problems:
        raise SystemExit(f"the copy does not pass --check, kept as {partial}:\n  " + "\n  ".join(problems[:10]))
    partial.rename(out)
    print(f"wrote {out}\n  from {src}: {STATE} {width} -> {KEEP} dims; {len(episode_files)} episode-metadata file(s), "
          + ", ".join(f"{n} {k}" for k, n in done.items()) + ("; meta/ only (not a training set)" if meta_only else ""))


def verify(src: Path, out: Path, prior: Path | None, meta_only: bool) -> list[str]:
    """Every difference between --out and what this script writes from --src (and the prior's normaliser); [] = pass."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    problems: list[str] = []
    if not (out / "meta" / "info.json").is_file():
        return [f"{out}/meta/info.json missing"]
    info = json.loads((src / "meta" / "info.json").read_text())
    width = info["features"][STATE]["shape"][0]
    if json.loads((out / "meta" / "info.json").read_text()) != cut_info(info):
        problems.append("meta/info.json differs from the source's with the state cut to 9")
    if json.loads((out / "meta" / "stats.json").read_text()) != cut_stats(json.loads((src / "meta" / "stats.json").read_text()), width):
        problems.append("meta/stats.json differs from the source's with the state statistics cut to 9")
    if real(out / "meta" / "tasks.parquet").read_bytes() != real(src / "meta" / "tasks.parquet").read_bytes():
        problems.append("meta/tasks.parquet differs")

    def compare(relative: Path, names) -> None:
        a, b = pq.read_table(real(src / relative)), pq.read_table(real(out / relative))
        if a.schema != b.schema or a.num_rows != b.num_rows:
            problems.append(f"{relative}: schema or row count differs")
            return
        for name in names:
            if not pc.list_slice(a.column(name).combine_chunks(), 0, KEEP).cast(a.schema.field(name).type).equals(
                    b.column(name).combine_chunks()):
                problems.append(f"{relative}: {name} is not the source's first {KEEP} values")
        if not a.drop_columns(list(names)).equals(b.drop_columns(list(names))):
            problems.append(f"{relative}: a column other than the state differs")

    groups = [("meta/episodes", "*.parquet")] + ([] if meta_only else [("data", "*.parquet")])
    for sub, pattern in groups:
        have, want = files(out, sub, pattern), files(src, sub, pattern)
        if have != want:
            problems.append(f"{sub}: {len(have)} file(s), the source has {len(want)}")
            continue
        for relative in want:
            names = [STATE] if sub == "data" else [n for n in pq.read_schema(real(src / relative)).names if stat_column(n)]
            compare(relative, names)
    if meta_only:
        if (out / "data").exists() or (out / "videos").exists():
            problems.append("a meta-only copy holds data/ or videos/")
    else:
        have, want = files(out, "videos", "*.mp4"), files(src, "videos", "*.mp4")
        if have != want:
            problems.append(f"videos: {len(have)} file(s), the source has {len(want)}")
        for relative in sorted(set(have) & set(want)):
            target = real(out / relative)
            if not target.is_file() or target.stat().st_size != real(src / relative).stat().st_size:
                problems.append(f"{relative}: does not resolve to a file of the source's size")
    if prior is not None:
        problems += normaliser_problems(json.loads((out / "meta" / "stats.json").read_text()), prior)
    return problems


def normaliser_problems(stats: dict, prior: Path) -> list[str]:
    """The prior's normaliser observation.state.* and action.* tensors against a dataset's stats.json (exact)."""
    import numpy as np
    from safetensors.numpy import load_file

    found = sorted(prior.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(found) != 1:
        return [f"{prior}: {len(found)} normaliser files, expected 1"]
    tensors, problems = load_file(str(found[0])), []
    for feature in (STATE, "action"):
        keys = {key[len(feature) + 1:] for key in tensors if key.startswith(feature + ".") and key.count(".") == feature.count(".") + 1}
        if keys != set(stats[feature]):
            problems.append(f"normaliser {feature} statistics {sorted(keys)} != dataset's {sorted(stats[feature])}")
            continue
        for key in sorted(keys):
            want = np.asarray(stats[feature][key], dtype=np.float32).reshape(-1)
            got = tensors[f"{feature}.{key}"].reshape(-1)
            if got.shape != want.shape:
                problems.append(f"normaliser {feature}.{key} has shape {got.shape}, the dataset statistic {want.shape}")
            elif not np.array_equal(got, want):
                problems.append(f"normaliser {feature}.{key} differs from the dataset statistic by up to "
                                f"{float(np.max(np.abs(got.astype(np.float64) - want))):.3g}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(SOURCE_DIM))
    parser.add_argument("--src", type=Path, default=None, help="the task's demo dataset [$DATA_ROOT/mimicgen_<task>_224x224]")
    parser.add_argument("--out", type=Path, default=None, help="[$DATA_ROOT/mimicgen_<task>_224x224_robotstate9]")
    parser.add_argument("--videos", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--meta-only", action="store_true", help="write meta/ only (needs --out)")
    parser.add_argument("--check", action="store_true", help="compare --out with --src; write nothing")
    parser.add_argument("--prior", type=Path, default=None, help="with --check: a prior's pretrained_model dir")
    args = parser.parse_args()
    src, out = args.src or default_src(args.task), args.out or default_out(args.task)
    if args.meta_only and args.out is None:
        parser.error("--meta-only needs an explicit --out (a meta-only copy at the default path would block the real one)")
    if args.prior is not None and not args.check:
        parser.error("--prior belongs to --check")
    if args.check:
        source_layout(src, args.task)
        if not out.exists():
            raise SystemExit(f"{out} does not exist")
        meta_only = args.meta_only or not (out / "data").exists()
        problems = verify(src, out, args.prior, meta_only)
        for line in problems:
            print(f"  FAIL {line}")
        if problems:
            raise SystemExit(f"FAILED: {out} is not the 9-D copy of {src}" + (" with the prior's normaliser" if args.prior else ""))
        print(f"ok: {out} is the 9-D copy of {src}" + (" (meta/ only)" if meta_only else "")
              + (f"; its statistics equal the normaliser of {args.prior}" if args.prior else ""))
        return
    build(args.task, src, out, args.videos, args.meta_only)


if __name__ == "__main__":
    sys.exit(main())
