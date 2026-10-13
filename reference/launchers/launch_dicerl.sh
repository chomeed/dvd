#!/usr/bin/env bash
# One DICE-RL (train_online.py --method dice_rl) run of the post_iclr2027 campaign (2026-10-08). User: "could you also run
# dice-rl?" -> "All 4 tasks, campaign-style" (DVD + Q, s43/s44 at 600k, s45 at 1M), "After the DSRL queue", "Paper DICE-RL" DVD
# recipe, "Paper pixq" Q recipe, "Prune like DSRL".
# ARM=dvd: the paper/supplementary dpinitvtd DICE-RL recipe (wandb EXPERIMENT1 cells; argv verified token for token against the
#   wandb-metadata of q3afuix6 square_d0 s43, xbatc6cj threading s44, pqp5gbbm hammer s43 and the 3dspr2q9 square_d1 argv):
#   best-of-16 prior draws ranked by gamma^8 V(f), residual actor trained through the DINO-WM, offline progress-pretrained pv4 V
#   fine-tuned online with TD. square_d0 adds online RoboMeter progress distillation (rm_spool_square; the 09-26 rule "square d0
#   td + distill, the other 3 td only"). hammer keeps its own cell: base 217k WM, --svg-rows 256, 1 update per chunk (no
#   --update-every 4 / --compile-parts). Plain pv4 V dirs on every task (not the _pv4_seg dirs the DSRL arm uses).
# ARM=dvdtd (2026-10-08, user: "add it and prioritize it" = a distill vs no-distill A/B on square_d0): square_d0 ONLY; ARM=dvd
#   with every --value-distill-* flag and --value-distill-pred removed, i.e. TD-only like the other tasks' DVD cells. Name token
#   tdonly, wandb group square_d0_dicerl_dvd_tdonly. Everything else is identical (dry-run verified).
# ARM=q: the paper pixq recipe (the 3-seed yonsei cells, e.g. kq0y2xyr): pixel Q critic ensemble (10), full-batch Q rows,
#   1 update per chunk, no compile; best-of-16 by min-Q.
# Only deviations from the references: --save-freq 50000 for Q (the paper pixq saved every 25k; the DVD cells already saved every
#   50k): every saved checkpoint gets a best-of-16 eval and prune_guard.py never deletes an unevaluated one, so 50k halves both the
#   eval load and the disk; saving does not touch training. Plus --total-env-steps 600k/1M, wandb project post_iclr2027, the output
#   location and the seeds.
# Run dirs live in post_iclr2027/dicerl/<name> (NOT directly in post_iclr2027: prune_guard.py's critic-study name check scans
# that level). train_online refuses an existing output dir, so a relaunch can never overwrite a run. RELAUNCH of a dead cell under
# the same name: move its dir OUT of post_iclr2027/dicerl/ (e.g. to dicerl_dead/), wait until no eval names it, then move aside its
# exact-name eval files -- wm_runs/bon16_evals/<name>_<step>.{json,partial.json,log} and post_iclr2027/logs/eval_dicerl_failed/
# <name>_<step> and .rmalert_<name> -- or they would be taken for evals of the new attempt's same-step checkpoints. This launcher
# refuses (exit 2, first log line "refusing:") while any of them remains.
# usage: TASK=square_d0|square_d1|threading_d0|hammer_cleanup_d1 ARM=dvd|dvdtd|q SEED=43 STEPS=600000 G=<gpu> bash launch_dicerl.sh
#        NAME_ONLY=1 ... prints the run name and exits; TRAIN_ONLINE_DRYRUN=1 ... prints the resolved config and exits.
set -eu
H=/NHNHOME/WORKSPACE/26moe002_B/chomeed; W=$H/wm_runs; R=$H/workspace/ICLR/online_rl_mimicgen; P=$H/post_iclr2027
E=$W/encspace_cache; PRI=$H/hdr_robot/priors
: "${TASK:?}" "${ARM:?}" "${SEED:?}" "${STEPS:?}"
K=$((STEPS / 1000))k
case $TASK in
  square_d0) PRIOR=$PRI/square_d0_40k_rs9; WM=$E/mimicgen_square_d0_224x224_dinowm_ph_conv_150k_cont150k/wm_dinov2s14_ph_conv_150k_cont150k.pt
    VDIR=$R/0814/v_dino_robometer_pred_progress_square_d0_ph_conv_150k_mqattn_pv4;;
  square_d1) PRIOR=$PRI/square_d1_45k_statefix; WM=$E/mimicgen_square_d1_224x224_dinowm_ph_conv_150k_cont150k/wm_dinov2s14_ph_conv_150k_cont150k.pt
    VDIR=$R/0814/v_dino_robometer_pred_progress_square_d1_ph_conv_150k_mqattn_pv4;;
  threading_d0) PRIOR=$PRI/threading_d0_55k_statefix; WM=$E/wm_dinov2s14_ph_conv_168k_cont168k.pt
    VDIR=$R/0814/v_dino_robometer_pred_progress_threading_d0_ph_conv_168k_mqattn_pv4;;
  hammer_cleanup_d1) PRIOR=$PRI/hammer_cleanup_d1_35k_statefix; WM=$E/mimicgen_hammer_cleanup_d1_224x224_dinowm_ph_conv_217k/wm_dinov2s14_ph_conv_217k.pt
    VDIR=$R/0814/v_dino_robometer_pred_progress_hammer_cleanup_d1_ph_conv_217k_mqattn_pv4;;
  *) echo "TASK must be square_d0, square_d1, threading_d0 or hammer_cleanup_d1"; exit 2;;
esac
[ $ARM = dvdtd ] && [ $TASK != square_d0 ] && { echo "ARM=dvdtd is the square_d0 TD-only control; use ARM=dvd for $TASK (already TD-only)"; exit 2; }
case $ARM in
  dvd|dvdtd)
    if [ $TASK = square_d0 ] && [ $ARM = dvd ]; then VTAG=tddistill; else VTAG=tdonly; fi
    NAME=dicerl-dvd_${TASK}_dpinitvtd_${VTAG}_${K}_s$SEED
    GROUP=${TASK}_dicerl_dvd; [ $ARM = dvdtd ] && GROUP=${TASK}_dicerl_dvd_tdonly
    DISTILL=(); [ $TASK = square_d0 ] && [ $ARM = dvd ] && DISTILL=(--value-distill-robometer $W/rm_spool_square --value-distill-every 20
      --value-distill-batch 256 --value-distill-frames 64 --value-distill-capacity 20000 --value-distill-mix succ=0.6,fail=0.4
      --value-distill-target progress)
    DPRED=(); [ $TASK = square_d0 ] && [ $ARM = dvd ] && DPRED=(--value-distill-pred)
    UPD=(--update-every 4 --compile-parts all); ROWS=0
    if [ $TASK = hammer_cleanup_d1 ]; then UPD=(); ROWS=256; fi
    ARGS=(--task $TASK --method dice_rl --total-env-steps $STEPS --residual-scale 1.0 --expert-ratio 0.3 --adaptive-expert-ratio
      --adaptive-expert-ratio-steps 40000 --progressive-exploration-steps 75000 --bon-axis prior --eval-freq 999999999
      --save-freq 50000 --eval-samples 16 --multi-sample-k 4 --value-finetune "${DISTILL[@]}"
      --dino-wm-state-head --svg-proprio-source predicted "${DPRED[@]}" --value-target-tau 0.005 --batch-size 64
      --svg-fit-rows 64 "${UPD[@]}" --bon-source svg --pixel-obs --pixel-feature-dim 128
      --value-model $VDIR --value-variant v_td --value-intervention none --svg-weight 1.0 --actor-gradient-source-normal svg1
      --svg-finetune-dynamics --svg-dynamics dino_wm --svg-rows $ROWS --dino-pool mqattn --no-train-critic --actor-train-encoder
      --dino-wm-ckpt $WM);;
  q)
    NAME=dicerl-q_${TASK}_pixq_${K}_s$SEED
    GROUP=${TASK}_dicerl_q
    ARGS=(--task $TASK --method dice_rl --total-env-steps $STEPS --residual-scale 1.0 --pixel-obs --pixel-feature-dim 128
      --expert-ratio 0.3 --adaptive-expert-ratio --adaptive-expert-ratio-steps 40000 --progressive-exploration-steps 75000
      --bon-axis prior --eval-freq 999999999 --save-freq 50000 --eval-samples 16 --multi-sample-k 4);;
  *) echo "ARM must be dvd, dvdtd or q"; exit 2;;
esac
NSTEP=(); [ $ARM = q ] && NSTEP=(--nstep 1)            # the yonsei pixq argv ends with --nstep 1 (= the default)
if [ -n "${NAME_ONLY:-}" ]; then echo $NAME; exit 0; fi
: "${G:?}"
OUT=$P/dicerl/$NAME
if [ -z "${TRAIN_ONLINE_DRYRUN:-}" ] && { ls -1A $W/bon16_evals 2>/dev/null | grep -qE "^${NAME}_[0-9]{6,7}\.(json|partial\.json|log)(\.tmp[0-9]*)?$" ||
     ls -1A $P/logs/eval_dicerl_failed 2>/dev/null | grep -qE "^(\.rmalert_)?${NAME}(_[0-9]{6,7})?$"; }; then
  echo "refusing: evals / eval-failure markers of an earlier $NAME exist ($W/bon16_evals/${NAME}_<step>.*, $P/logs/eval_dicerl_failed/${NAME}_<step> or .rmalert_${NAME}); move them aside first"
  exit 2
fi
mkdir -p $P/dicerl
source $W/eval_env.sh
unset WS_TASK RESUME VALUEFT VALUETAU UTD BATCH NSTEP
export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G PIXEL_IMG=224 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online WANDB_RUN_GROUP=$GROUP
[ $ARM != q ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # task_dino.sbatch:78 (DVD cells only)
cd $R
exec $CONDA_PREFIX/bin/python -m online_rl_mimicgen.train_online "${ARGS[@]}" \
  --output-dir $OUT --wandb-project post_iclr2027 --wandb-name $NAME --step-penalty 0.0 --eval-strategy greedy \
  --eval-episodes 0 --no-save-replay --base-output $PRIOR --seed $SEED "${NSTEP[@]}"
