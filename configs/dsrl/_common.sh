# Shared part of the five DSRL cells (sourced by ours.sh, oursmc.sh, oursdist.sh, randv.sh and q.sh; not a cell).
#
# DSRL-NA: a SAC actor in the NOISE space of the frozen flow prior (the 56-D noise of the 8 executed actions; the
# other positions are resampled), an image actor (IMPALA encoder), demonstrations mixed into every batch (share
# annealed 1.0 -> 0.1 over the first 40k env steps), exploration ramped over 75k steps. Every 25k env steps a
# checkpoint is saved and scored in the background by eval_dsrl_ckpt: 100 episodes, env seeds 10000-10099, ONE noise
# draw per chunk (no best-of-N), logged as eval/success_rate and written to <run>/evals/<step>.json and metrics.jsonl.
#
# Reference: reference/launchers/launch_hammer_dsrl.sh. COMMON below is its lines 47-52 with the launcher defaults
# SAVE=25000, EVAL_EPS=100, WARMUP=1000 written out; dsrl_dvd_cell is its lines 70-76 (run name and the ARGS of the
# four DVD arms) and 86-87 (argv tail), token for token, with svg -> dvd in seven flags and in the run name
# (dsrl-na-svg1.0_..._svgrowsall_... there) and the asset paths taken from configs/tasks.sh.

MODULE=online_rl_mimicgen.baselines.dsrl_dino
source "$(dirname "${BASH_SOURCE[0]}")/../tasks.sh"
if [ -n "${DISTILL:-}" ]; then
  echo "DISTILL belongs to configs/dice_rl/dvd.sh; in DSRL the arm decides (ours = TD only, oursdist = TD + distill)" >&2
  return 2
fi

K=$((STEPS / 1000))k
COMMON=(--task "$TASK" --variant na --seed "$SEED" --base-output "$PRIOR_DIR"
  --total-env-steps "$STEPS" --save-freq 25000 --image-actor --pixel-feature-dim 128 --pixel-shift-pad 4
  --target-entropy 0.0 --init-rollout-chunks 1000 --action-magnitude 1.0 --lr 1e-4 --layer-size 256 --num-layers 3
  --actor-grad-clip 10 --ent-coef-lr 1.5e-3 --activation mish --log-std-min -5 --executed-noise-only resample
  --progressive-exploration-steps 75000 --expert-ratio 0.3 --adaptive-expert-ratio
  --async-eval-episodes 100 --async-eval-workers 2 --async-eval-seed-start 10000)

# The four DVD arms. Set before calling:
#   VDIR      V directory          RTOK    its run-name token          VINIT   V-init name token ("" = none)
#   NODISTILL 1 = TD only (--no-value-distill), 0 = TD + RoboMeter distillation every 20 V steps
#   SCRATCH   1 = re-initialise V's heads and its readout (--value-scratch --value-pool-scratch)
#   SUFFIX    run-name suffix
# The seven --value-distill-* flags are passed in every arm, as in the reference; --no-value-distill switches them off.
dsrl_dvd_cell() {
  local dtag=
  if [ "$NODISTILL" = 1 ]; then dtag=_nodistill; fi
  NAME=dsrl-na-dvd1.0_${TASK}_${WM_CONT_TAG}_${RTOK}${VINIT:+_$VINIT}_img_bs64_dvdrowsall_ue4_utd2as2_alr1.5e-3_lsm5_mish_ramp75k_exec56resample_demo1.0to0.1x40k${dtag}_${K}_s$SEED$SUFFIX
  ARGS=("${COMMON[@]}" --dvd-weight 1.0 --value-model "$VDIR" --dino-pool mqattn
    --dino-wm-ckpt "$WM_CONT"
    --dino-wm-arch 6,6,2048 --dvd-proprio-source predicted --dvd-rows 0 --dvd-fit-rows 64 --value-target-tau 0.005
    --value-distill-robometer "$SPOOL" --value-distill-every 20 --value-distill-batch 256 --value-distill-frames 64
    --value-distill-capacity 20000 --value-distill-mix succ=0.6,fail=0.4 --value-distill-target progress --value-distill-pred)
  if [ "$NODISTILL" = 1 ]; then ARGS+=(--no-value-distill); fi
  if [ "$SCRATCH" = 1 ]; then ARGS+=(--value-scratch --value-pool-scratch); fi
  ARGS+=(--batch-size 64 --train-freq 4 --utd 2 --dvd-actor-steps 2 --dvd-fit-utd 0.25 --value-lr 1e-4 --dvd-dynamics-lr 1e-4)
  ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-name "$NAME" --output-dir "$RUNS_ROOT/$NAME")
}
