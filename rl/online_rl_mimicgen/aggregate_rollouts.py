#!/usr/bin/env python3
"""Merge per-shard rollout datasets into one dataset per outcome, then optionally push."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path


def merge_rollout_info(shard_roots: list[Path], outcome: str) -> dict:
    """Concatenate the shards' sidecar metadata, renumbering episodes as aggregation does."""
    merged: dict = {"outcome": outcome, "shards": [], "episodes": []}
    for root in shard_roots:
        info = json.loads((root / "meta/rollout_info.json").read_text())
        merged["shards"].append(
            {
                "root": str(root),
                "seed_start": info["seed_start"],
                "episodes_attempted": info["episodes_attempted"],
                "episodes_kept": info["counts"][outcome],
            }
        )
        for episode in info["episodes"]:
            merged["episodes"].append(
                {
                    "episode_index": len(merged["episodes"]),
                    "seed": episode["seed"],
                    "length": episode["length"],
                }
            )
        for key in ("checkpoint", "base_checkpoint", "task", "policy", "action_steps",
                    "inference_steps", "residual_scale"):
            merged.setdefault(key, info[key])
    merged["episodes_attempted_total"] = sum(s["episodes_attempted"] for s in merged["shards"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="parent dir for the merged datasets")
    parser.add_argument("--outcomes", nargs="+", default=["success", "failure"])
    parser.add_argument("--expect-episodes", type=int, default=100)
    parser.add_argument("--reference-root", type=Path, required=True, help="the task's demo dataset (file packing)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="skip aggregation and push the already-merged datasets, so what was verified is what ships",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.aggregate import aggregate_datasets

    reference = json.loads((args.reference_root / "meta/info.json").read_text())
    prefix_name = args.repo_prefix.split("/")[-1]

    results = {}
    for outcome in args.outcomes:
        shard_repo_ids, shard_roots = [], []
        for shard in range(args.shards):
            root = args.shard_root / f"shard{shard}" / f"{prefix_name}_shard{shard}_{outcome}"
            if not (root / "meta/info.json").exists():
                raise FileNotFoundError(f"Missing shard dataset: {root}")
            shard_repo_ids.append(f"{args.repo_prefix}_shard{shard}_{outcome}")
            shard_roots.append(root)

        repo_id = f"{args.repo_prefix}_{outcome}"
        root = args.output_root / repo_id.split("/")[-1]

        if args.push_only:
            info = json.loads((root / "meta/info.json").read_text())
            results[outcome] = {
                "repo_id": repo_id,
                "root": str(root),
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
            }
            continue

        if root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{root} already exists; pass --overwrite")
            shutil.rmtree(root)

        logging.info("Aggregating %d shards into %s", len(shard_roots), root)
        aggregate_datasets(
            repo_ids=shard_repo_ids,
            aggr_repo_id=repo_id,
            roots=shard_roots,
            aggr_root=root,

            chunk_size=reference["chunks_size"],
            data_files_size_in_mb=reference["data_files_size_in_mb"],
            video_files_size_in_mb=reference["video_files_size_in_mb"],
        )

        info = json.loads((root / "meta/info.json").read_text())
        if info["total_episodes"] != args.expect_episodes:
            raise RuntimeError(
                f"{repo_id}: aggregated {info['total_episodes']} episodes, expected {args.expect_episodes}"
            )
        rollout_info = merge_rollout_info(shard_roots, outcome)
        if len(rollout_info["episodes"]) != info["total_episodes"]:
            raise RuntimeError(f"{repo_id}: sidecar episode count disagrees with the dataset")
        (root / "meta/rollout_info.json").write_text(json.dumps(rollout_info, indent=2) + "\n")

        results[outcome] = {
            "repo_id": repo_id,
            "root": str(root),
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "episodes_attempted_total": rollout_info["episodes_attempted_total"],
        }
        logging.info("%s: %d episodes / %d frames", repo_id, info["total_episodes"], info["total_frames"])

    if args.push:
        for outcome, result in results.items():
            logging.info("Pushing %s", result["repo_id"])
            dataset = LeRobotDataset(result["repo_id"], root=Path(result["root"]))
            dataset.push_to_hub(
                tags=["lerobot", "mimicgen", "robosuite", "v3.0", "rollout", outcome, "dice-rl"],
                license="cc-by-4.0",
                tag_version=True,
                push_videos=True,
                private=args.private,
                upload_large_folder=True,
            )
            results[outcome]["pushed"] = True

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
