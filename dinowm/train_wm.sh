#!/usr/bin/env bash
# Train the production world model of one task with the recipe that produced the released checkpoints.
#
#   TASK=<task> STAGE=base|cont [G=<gpu>] bash dinowm/train_wm.sh [--print]
#
#   task               steps    base model      continuation              data seed of the continuation
#   square_d0          150000   ph_conv_150k    ph_conv_150k_cont150k     1
#   square_d1          150000   ph_conv_150k    ph_conv_150k_cont150k     0
#   threading_d0       168000   ph_conv_168k    ph_conv_168k_cont168k     1
#   hammer_cleanup_d1  217000   ph_conv_217k    ph_conv_217k_cont217k     1
#
#   base  from scratch: peak LR 5e-4, data seed 0 (the trainer's defaults). square_d1 also wrote snapshots at 44k / 80k.
#   cont  --init the base model, fresh optimiser and schedule, peak LR 1e-4, the same number of steps again,
#         snapshots at +50k / +100k. Needs the base model in $CKPT_ROOT/wm (trained here or downloaded).
#
# Needs the task's frame cache (WS_TASK=<task> python dinowm/build_cache.py). Both stages pass --resume, so rerunning
# the command continues an interrupted run. Models land in $CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_<tag>/.
# Who uses which: every V is fitted through the BASE model; DICE-RL loads the continuation on square_d0, square_d1 and
# threading_d0 and the BASE model on hammer_cleanup_d1; DSRL loads the continuation on all four tasks.
# --print shows the command (one token per line) instead of running it.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/scripts/env.sh" ]; then source "$ROOT/scripts/env.sh"; fi
: "${CKPT_ROOT:?source scripts/env.sh first (CKPT_ROOT is not set)}" "${CACHE_ROOT:?CACHE_ROOT is not set}"
TASK=${TASK:?set TASK=square_d0|square_d1|threading_d0|hammer_cleanup_d1}
STAGE=${STAGE:?set STAGE=base|cont}
BASE_EXTRA=()
case "$TASK" in
  square_d0)         STEPS=150000 CONT_SEED=1 ;;
  square_d1)         STEPS=150000 CONT_SEED=0 BASE_EXTRA=(--snapshot-steps 44000,80000) ;;
  threading_d0)      STEPS=168000 CONT_SEED=1 ;;
  hammer_cleanup_d1) STEPS=217000 CONT_SEED=1 ;;
  *) echo "unknown TASK '$TASK'" >&2; exit 2 ;;
esac
K=$((STEPS / 1000))
TAG=ph_conv_${K}k
ARGS=(--task "$TASK" --act conv --steps "$STEPS")
case "$STAGE" in
  base) ARGS+=(${BASE_EXTRA[@]+"${BASE_EXTRA[@]}"}) ;;
  cont) ARGS+=(--init "$CKPT_ROOT/wm/mimicgen_${TASK}_224x224_dinowm_${TAG}/wm_dinov2s14_${TAG}.pt"
               --lr 1e-4 --seed "$CONT_SEED" --tag-suffix "_cont${K}k" --snapshot-steps 50000,100000) ;;
  *) echo "unknown STAGE '$STAGE' (base|cont)" >&2; exit 2 ;;
esac
ARGS+=(--resume)
if [ "${1:-}" = "--print" ]; then printf '%s\n' "WS_TASK=$TASK" python dinowm/train_wm_ph.py "${ARGS[@]}"; exit 0; fi
if [ -n "${G:-}" ]; then export CUDA_VISIBLE_DEVICES=$G; fi
# torch.compile needs an executable cache directory; /tmp is mounted noexec on some machines
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torch_inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
echo "[train_wm] $TASK $STAGE: python dinowm/train_wm_ph.py ${ARGS[*]}"
WS_TASK=$TASK exec python "$ROOT/dinowm/train_wm_ph.py" "${ARGS[@]}"
