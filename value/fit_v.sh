#!/usr/bin/env bash
# Fit one of the campaign's V directories with the recipe that produced the released ones.
#
#   TASK=<task> RECIPE=pv4|pv4_seg|mc [G=<gpu>] bash value/fit_v.sh [--print]
#
#   recipe   target                                   score table (rows)                 output in $CKPT_ROOT/value
#   pv4      RoboMeter progress, first scoring        rm_scores/<task>/                  v_dino_robometer_pred_progress_<task>_<tag>_mqattn_pv4
#   pv4_seg  RoboMeter progress, segmented LoRA       rm_scores/<task>_seg/              v_dino_robometer_pred_progress_<task>_<tag>_mqattn_pv4_seg
#            (square_d1 and threading_d0 only)
#   mc       MC success return gamma^(t_r - t)        rm_scores/<task>/ (row set only)   v_dino_mc_return_<task>_<tag>_mqattn_pv4
#
#   <tag> = the task's BASE world model: ph_conv_150k (square_d0, square_d1), ph_conv_168k (threading_d0),
#           ph_conv_217k (hammer_cleanup_d1).
#   Which cell reads which: DICE-RL DVD cells -> pv4 on every task; DSRL ours / oursdist / randv -> pv4 on square_d0 and
#   hammer_cleanup_d1, pv4_seg on square_d1 and threading_d0; DSRL oursmc -> mc.
#   Teacher of each table (robometer/teachers.sh): rm_scores/<task> = teacher <task>; rm_scores/square_d1_seg = teacher
#   square_d1_seg; rm_scores/threading_d0_seg = teacher threading_d0 again (the same LoRA, so the same scores in the
#   rescoring's row order).
#
# Needs the frame cache, the base world model and the score table (robometer/score_rollouts.py), or the released ones.
# --print shows the command (one token per line) instead of running it.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/scripts/env.sh" ]; then source "$ROOT/scripts/env.sh"; fi
: "${CKPT_ROOT:?source scripts/env.sh first (CKPT_ROOT is not set)}"
source "$ROOT/robometer/teachers.sh"
TASK=${TASK:?set TASK=square_d0|square_d1|threading_d0|hammer_cleanup_d1}
RECIPE=${RECIPE:?set RECIPE=pv4|pv4_seg|mc}
case "$TASK" in
  square_d0|square_d1) TAG=ph_conv_150k ;;
  threading_d0)        TAG=ph_conv_168k ;;
  hammer_cleanup_d1)   TAG=ph_conv_217k ;;
  *) echo "unknown TASK '$TASK'" >&2; exit 2 ;;
esac
SCORES=$TASK TEACHER=$TASK SUFFIX=mqattn_pv4 EXTRA=()
case "$RECIPE" in
  pv4) ;;
  mc)  EXTRA=(--target mc) ;;
  pv4_seg)
    SUFFIX=mqattn_pv4_seg SCORES=${TASK}_seg
    case "$TASK" in
      square_d1)    TEACHER=square_d1_seg ;;
      threading_d0) TEACHER=threading_d0 ;;
      *) echo "pv4_seg exists for square_d1 and threading_d0 only: the first score table of $TASK already comes from its segmented LoRA (use RECIPE=pv4)" >&2; exit 2 ;;
    esac ;;
  *) echo "unknown RECIPE '$RECIPE' (pv4|pv4_seg|mc)" >&2; exit 2 ;;
esac
teacher "$TEACHER"
ARGS=(--task "$TASK" --wm "$CKPT_ROOT/wm/mimicgen_${TASK}_224x224_dinowm_${TAG}/wm_dinov2s14_${TAG}.pt" --tag "$TAG"
      --per-view --queries 4 --out-dim 768 --suffix "$SUFFIX"
      --scores "$CKPT_ROOT/rm_scores/$SCORES/predictions.parquet" --teacher "$TEACHER_LORA"
      ${EXTRA[@]+"${EXTRA[@]}"})
if [ "${1:-}" = "--print" ]; then printf '%s\n' "WS_TASK=$TASK" python value/fit_v_mqattn.py "${ARGS[@]}"; exit 0; fi
if [ -n "${G:-}" ]; then export CUDA_VISIBLE_DEVICES=$G; fi
export PYTHONUNBUFFERED=1
echo "[fit_v] $TASK $RECIPE: python value/fit_v_mqattn.py ${ARGS[*]}"
WS_TASK=$TASK exec python "$ROOT/value/fit_v_mqattn.py" "${ARGS[@]}"
