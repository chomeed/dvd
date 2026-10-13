"""Decode a demo dataset's videos once into a uint8 (N, cameras, IMG, IMG, 3) array the expert buffer mmaps.

The interface is two environment variables (no arguments besides -h / --help):

  DATASET   the task's LeRobot demo dataset directory, e.g. $DATA_ROOT/mimicgen_<task>_224x224      (required)
  OUT       the .npy to write [<DATASET>/pixel_cache_<IMG>.npy]. The trainers look for
            $CACHE_ROOT/pixel_cache/pixel_cache_mimicgen_<task>_224x224_224.npy (PIXEL_CACHE / PIXEL_CACHE_DIR override)

IMG comes from PIXEL_IMG, which scripts/env.sh sets to 224. One file per task, 46-86 GB, written through a memmap.
Frames are stored in decode order, which is the dataset's global frame index; the count per camera is checked against
meta/info.json, because a mismatch would misalign every image with its state. The file is created at its full size
before decoding, so an interrupted run leaves a right-sized file that is partly zeros: delete it and start again.

scripts/build_pixel_cache.sh TASK is the wrapper to use: it sets both variables from configs/tasks.sh, checks the
dataset and the free space, never overwrites, and writes to <file>.partial until the decode has finished.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from online_rl_mimicgen.pixel import CAMERAS, IMG  # noqa: E402

USAGE = "usage: DATASET=<LeRobot dataset dir> [OUT=<file.npy>] python -m online_rl_mimicgen.pixel_cache"
if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] in ("-h", "--help"):
        print(f"{USAGE}\n\n{__doc__}")
        raise SystemExit(0)
    raise SystemExit(f"{USAGE}   (no arguments: got {' '.join(sys.argv[1:])!r})")
if "DATASET" not in os.environ:
    raise SystemExit(f"{USAGE}   (DATASET is not set)")
DATASET = Path(os.environ["DATASET"])


_SUF = "" if len(CAMERAS) == 2 else f"_{len(CAMERAS)}cam"
OUT = Path(os.environ.get("OUT", DATASET / f"pixel_cache_{IMG}{_SUF}.npy"))
IDX = np.linspace(0, 223, IMG).astype(np.int64)


def main() -> None:
    import imageio.v2 as imageio

    info = json.loads((DATASET / "meta/info.json").read_text())
    total = int(info["total_frames"])
    print(f"{DATASET.name}: {total} frames, {info['total_episodes']} episodes -> {OUT}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        OUT, mode="w+", dtype=np.uint8, shape=(total, len(CAMERAS), IMG, IMG, 3))

    for cam_i, cam in enumerate(CAMERAS):
        key = f"observation.images.{cam}"
        files = sorted((DATASET / "videos" / key).rglob("*.mp4"))
        written = 0
        for f in files:
            rdr = imageio.get_reader(f, "ffmpeg")
            for frame in rdr:
                if written >= total:
                    break
                out[written, cam_i] = frame[np.ix_(IDX, IDX)]
                written += 1
                if written % 20000 == 0:
                    print(f"  {cam}: {written}/{total}", flush=True)
            rdr.close()
        print(f"  {cam}: decoded {written} frames from {len(files)} files", flush=True)
        if written != total:
            raise RuntimeError(
                f"{cam} decoded {written} frames but the dataset declares {total}. The cache is "
                "indexed by global frame number, so a mismatch would silently misalign every "
                "image with its state.")

    out.flush()

    tab = pads.dataset(DATASET / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index"]).to_pydict()
    print(f"first parquet row: episode {tab['episode_index'][0]} frame {tab['frame_index'][0]}")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
