#!/usr/bin/env bash
# Run ONE cell of the campaign in the foreground:
#
#   bash scripts/run.sh configs/<method>/<arm>.sh TASK SEED [STEPS]
#
#   TASK   square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1
#   SEED   the campaign used 43 and 44 at STEPS=600000 and 45 at STEPS=1000000
#   STEPS  env steps (default 600000)
#
# Environment (besides everything scripts/env.sh honours):
#   G=<gpu>                 pin the run to one GPU (CUDA device and EGL render device)
#   DISTILL=tdonly|tddistill   configs/dice_rl/dvd.sh only: the V-training variant (default: the task's paper recipe)
#   TRAIN_ONLINE_DRYRUN=1   print "DRYRUN_CONFIG <json>", the configuration the trainer resolves from the cell's
#                           argv, and exit before any environment, model or run directory is created
#   RUN_PRINT_ARGV=1        print @MODULE= @NAME= @GROUP= @ENV= lines and then the argv, one token per line, and exit
#
# The run directory is $RUNS_ROOT/<run name>; both trainers refuse to start in a populated one.
# The code that runs is always this checkout's: scripts/env.sh ignores a CODE_ROOT left in the shell by another
# checkout, and before python starts (dry run included) require_this_tree aborts with exit 3 unless `python` resolves
# online_rl_mimicgen under $CODE_ROOT/rl. A first, short run of a cell on a new machine: scripts/smoke_train.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$CODE_ROOT/scripts/_tree_guard.sh"
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
CFG=$(realpath "$1"); TASK=$2; SEED=$3; STEPS=${4:-600000}
case $SEED$STEPS in *[!0-9]*) echo "SEED and STEPS must be integers (got '$SEED' '$STEPS')" >&2; exit 2;; esac
[ -f "$CFG" ] || { echo "no such config: $1" >&2; exit 2; }

source "$CFG"          # sets MODULE NAME GROUP ARGS (and exports what the cell needs in its environment)
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$GROUP}

if [ -n "${RUN_PRINT_ARGV:-}" ]; then
  printf '%s\n' "@MODULE=$MODULE" "@NAME=$NAME" "@GROUP=$WANDB_RUN_GROUP" \
    "@ENV=PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}" "${ARGS[@]}"
  exit 0
fi
cd "$CODE_ROOT"
require_this_tree
if [ -n "${TRAIN_ONLINE_DRYRUN:-}" ]; then
  # train_online prints its own DRYRUN_CONFIG and returns; dsrl_dino has no dry run, so its parser is called for it
  if [ "$MODULE" = online_rl_mimicgen.train_online ]; then exec python -m "$MODULE" "${ARGS[@]}"; fi
  exec python scripts/resolve_config.py "$MODULE" "${ARGS[@]}"
fi
mkdir -p "$RUNS_ROOT"
echo "[run] $MODULE -> $RUNS_ROOT/$NAME (wandb $WANDB_MODE: $WANDB_PROJECT / $WANDB_RUN_GROUP)"
exec python -m "$MODULE" "${ARGS[@]}"
