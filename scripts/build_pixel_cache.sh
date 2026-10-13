#!/usr/bin/env bash
# Build the demo pixel cache of a task: every frame of the 1000 demonstrations decoded once into one uint8 array
# (frames, 2 cameras, 224, 224, 3) that the expert buffer of every RL cell memory-maps.
#
#   bash scripts/build_pixel_cache.sh TASK [--print]
#
#   TASK     square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1
#   --print  show the command and the paths, build nothing
#
#   reads    $DATA_ROOT/mimicgen_<task>_224x224/            (the whole dataset: scripts/fetch_assets.py --tier train)
#   writes   $CACHE_ROOT/pixel_cache/pixel_cache_mimicgen_<task>_224x224_224.npy
#            46.2 GB square_d0, 45.9 GB square_d1, 67.6 GB threading_d0, 86.4 GB hammer_cleanup_d1 (246 GB for all four)
#
# = DATASET=<dataset> OUT=<file> python -m online_rl_mimicgen.pixel_cache with the two paths of configs/tasks.sh, and
# three guards the module does not have:
#   - it refuses when the dataset is incomplete or the file system lacks the space (size = frames x 301056 + 128 bytes)
#   - it never overwrites: an existing cache of the right shape is left alone, any other file must be removed by hand
#   - the array is written as <file>.partial and renamed when the decode finished. The module creates the file at its
#     full size before decoding, so without this an interrupted build leaves a right-sized cache full of zeros.
# Both trainers find the file by itself (DICE-RL: PIXEL_CACHE overrides the file; DSRL: PIXEL_CACHE_DIR the directory).
# CPU only. The videos are decoded by the ffmpeg binary bundled with imageio-ffmpeg (requirements.txt), not the system
# one. The reference build of square_d1 took 3 min 16 s (152,400 frames); no other build time was logged.
# This wrapper has been verified with --print and its refusals only; the module is the reference machine's unchanged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$CODE_ROOT/scripts/_tree_guard.sh"
if [ $# -lt 1 ] || [ $# -gt 2 ] || { [ $# -eq 2 ] && [ "$2" != --print ]; }; then
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2
fi
TASK=$1; PRINT=${2:-}
source "$CODE_ROOT/configs/tasks.sh"          # validates TASK; sets DEMO_DATASET and PIXEL_CACHE_FILE
OUT=$PIXEL_CACHE_FILE
export PIXEL_CAMERAS=agentview,eye_in_hand OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1

if [ -n "$PRINT" ]; then
  printf '%s\n' "DATASET=$DEMO_DATASET" "OUT=$OUT.partial   (renamed to $OUT when complete)" \
    "PIXEL_IMG=$PIXEL_IMG PIXEL_CAMERAS=$PIXEL_CAMERAS" "cd $CODE_ROOT && python -m online_rl_mimicgen.pixel_cache"
  exit 0
fi
for need in meta/info.json data videos/observation.images.agentview videos/observation.images.eye_in_hand; do
  [ -e "$DEMO_DATASET/$need" ] || {
    echo "$DEMO_DATASET/$need is missing: download the whole demo dataset first (python scripts/fetch_assets.py --tier train --task $TASK --only dataset)" >&2
    exit 1; }
done
require_this_tree
FRAMES=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_frames"])' "$DEMO_DATASET/meta/info.json")
BYTES=$(( FRAMES * 2 * 224 * 224 * 3 + 128 ))
shape_of() { python -c 'import numpy, sys; a = numpy.load(sys.argv[1], mmap_mode="r"); print(*a.shape, a.dtype)' "$1" 2>/dev/null || echo unreadable; }
WANT="$FRAMES 2 224 224 3 uint8"
if [ -e "$OUT" ]; then
  if [ "$(shape_of "$OUT")" = "$WANT" ]; then echo "already built: $OUT ($FRAMES frames)"; exit 0; fi
  echo "$OUT exists but holds ($(shape_of "$OUT")), the dataset needs ($WANT): remove it yourself to rebuild" >&2; exit 1
fi
mkdir -p "$(dirname "$OUT")"
FREE=$(( $(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}') * 1024 ))
if [ "$FREE" -lt $(( BYTES + BYTES / 50 )) ]; then
  echo "not enough space under $(dirname "$OUT"): $(( FREE / 1000000000 )) GB free, the cache needs $(( BYTES / 1000000000 )) GB" >&2; exit 1
fi
echo "[pixel cache] $TASK: $FRAMES frames, $(( BYTES / 1000000000 )) GB -> $OUT"
rm -f "$OUT.partial"                           # left by an interrupted build of this script
cd "$CODE_ROOT"
DATASET=$DEMO_DATASET OUT=$OUT.partial python -m online_rl_mimicgen.pixel_cache
[ "$(shape_of "$OUT.partial")" = "$WANT" ] || { echo "the build wrote ($(shape_of "$OUT.partial")), expected ($WANT); kept as $OUT.partial" >&2; exit 1; }
mv "$OUT.partial" "$OUT"
echo "[pixel cache] done: $OUT"
