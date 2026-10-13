#!/usr/bin/env bash
# Evaluate one DSRL checkpoint with the protocol of every DSRL number: ONE noise draw per chunk (no best-of-N),
# 100 episodes on env seeds 10000-10099.
#
#   bash scripts/eval_dsrl.sh --ckpt <run>/checkpoints/<step>.zip --out <json> [--episodes 100] [--seed-start 10000]
#                             [the evaluator's other options: --base-output, --dataset-root, ...]
#   bash scripts/eval_dsrl.sh --help          # the evaluator's own help: every option
#
# = python -m online_rl_mimicgen.baselines.eval_dsrl_ckpt "$@", started in $CODE_ROOT with this checkout's code. Use it
# instead of the bare module call: `python -m` puts the working directory first on the import path, so the same command
# typed inside another tree that holds an online_rl_mimicgen/ directory would run that tree's code. Relative paths
# (--ckpt, --out, --base-output, --dataset-root, --config) are taken relative to the directory you call this from.
#
# G=<gpu> pins the GPU. A run of the research code (old names, the reference machine's absolute paths in config.json) is
# accepted as it is. The run's config.json is looked up two levels above the REAL checkpoint file, so a symlinked
# checkpoint is evaluated with the config.json beside the link's target. DSRL runs score their own checkpoints while
# they train (configs/dsrl/_common.sh); this is for re-evaluating one. DICE-RL checkpoints: scripts/eval_loop.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$CODE_ROOT/scripts/_tree_guard.sh"
[ $# -ge 1 ] || { sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
ARGS=()
while [ $# -gt 0 ]; do                  # paths become absolute before the cd
  case $1 in
    --ckpt|--out|--base-output|--dataset-root|--config)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
      ARGS+=("$1" "$(realpath -m -s -- "$2")"); shift 2 ;;
    --ckpt=*|--out=*|--base-output=*|--dataset-root=*|--config=*)
      ARGS+=("${1%%=*}" "$(realpath -m -s -- "${1#*=}")"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
export WANDB_MODE=disabled
cd "$CODE_ROOT"
require_this_tree
exec python -m online_rl_mimicgen.baselines.eval_dsrl_ckpt "${ARGS[@]}"
