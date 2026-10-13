#!/usr/bin/env bash
# One DSRL hammer_cleanup_d1 run (2026-09-30). ARM=ours: dpinitvtd V, TD only (no RoboMeter distillation), the
# 300k reference config (wm_runs/dsrl-na-svg1.0_hammer..._dpinitvtd_..._nodistill_300k_s43) at 600k. ARM=q: DSRL-NA with
# 10 Q^A + 10 Q^W critics, the other machine's working Q recipe (wandb iclr2027/o9yxajsc) on dsrl_dino --q-critic.
# Every checkpoint (25k) is evaluated in the background: 100 single-draw episodes, seeds 10000+, logged as eval/success_rate.
# ARM=oursmc (2026-10-01, user: "queue same but mc return initialized V"): ours with V (readout + 3 heads) pretrained offline on
# Monte Carlo returns gamma^(t_success - t) (0 on failures) instead of RoboMeter progress; same architecture, WM, recipe.
# ARM=oursdist (2026-10-01, user: "variant ... where we also have robometer distillation"): ours (dpinitvtd) with online
# RoboMeter progress distillation every 20 V steps, served by robometer_server.py on wm_runs/rm_spool_hammer (the
# hammer_cleanup_seg LoRA that also labelled V's offline targets); the 09-25 TD+distill 300k s43 recipe at 600k/1M.
# ARM=randv (2026-10-01, user: "could you also try randomly initialized V? with _randv suffix"): ours with a FULLY random V
# (--value-scratch: V MLP re-initialised; --value-pool-scratch: mqattn readout re-initialised, shape from the pv4 blob) -- the
# 09-24 "random initialized V (both encoder and head)" flags. The V dir only supplies architecture, state scaler and pool shape.
# Names drop the V-init token and end in _randv; wandb group hammer_cleanup_d1_ours_dsrl_randv. As in every DSRL random-V run,
# the init is the same for all seeds (torch.manual_seed(0) in check_equivalence runs before V is built).
# TASK=square_d0 (2026-10-01, user: "queue the same process for square d0" -- the critic study needs a matched Q/DVD pair there):
# same recipes with square_d0's prior, WM (ph_conv_150k_cont150k) and pv4 V dir, exactly the ICLR square_d0 nodistill config;
# names/groups swap the task (and WM tag) only. Default TASK=hammer_cleanup_d1 keeps every hammer name and argument unchanged.
# TASK=square_d1|threading_d0 (2026-10-01, user: "queue mcinit and randv for square d0 and square d1" / "the ablation study on the
# other two tasks"): matched to the ICLR TD-only ours runs there (verified config diff: only task, prior, WM, V dir, spool differ).
# Their dpinit V is the *_mqattn_pv4_seg dir (offline labels from the segmented RoboMeter LoRA, as hammer's/square_d0's pv4 already
# were), name token mqattn_pv4seg; the MC-return V keeps the plain _mqattn_pv4 dir/token. threading's WM sits at a flat path.
# usage: [TASK=hammer_cleanup_d1|square_d0|square_d1|threading_d0] ARM=ours|oursmc|oursdist|randv|q SEED=43 G=<gpu> [STEPS=600000 SAVE=25000 EVAL_EPS=100 WARMUP=1000 PROJECT=post_iclr2027 OUT=<dir>] bash launch_hammer_dsrl.sh
set -eu
H=/NHNHOME/WORKSPACE/26moe002_B/chomeed; W=$H/wm_runs; R=$H/workspace/ICLR/online_rl_mimicgen; P=$H/post_iclr2027
: "${ARM:?}" "${SEED:?}" "${G:?}"
TASK=${TASK:-hammer_cleanup_d1}
VSFX=; WMCKPT=
case $TASK in
  hammer_cleanup_d1) PRIOR=$H/hdr_robot/priors/hammer_cleanup_d1_35k_statefix; WMTAG=ph_conv_217k_cont217k; VBASE=ph_conv_217k
    SPOOL=$W/rm_spool_hammer;;
  square_d0) PRIOR=$H/hdr_robot/priors/square_d0_40k_rs9; WMTAG=ph_conv_150k_cont150k; VBASE=ph_conv_150k; SPOOL=$W/rm_spool_square;;
  square_d1) PRIOR=$H/hdr_robot/priors/square_d1_45k_statefix; WMTAG=ph_conv_150k_cont150k; VBASE=ph_conv_150k
    SPOOL=$W/rm_spool_square_d1; VSFX=_seg;;
  threading_d0) PRIOR=$H/hdr_robot/priors/threading_d0_55k_statefix; WMTAG=ph_conv_168k_cont168k; VBASE=ph_conv_168k
    SPOOL=$W/rm_spool; VSFX=_seg; WMCKPT=$W/encspace_cache/wm_dinov2s14_ph_conv_168k_cont168k.pt;;
  *) echo "TASK must be hammer_cleanup_d1, square_d0, square_d1 or threading_d0"; exit 2;;
esac
WMCKPT=${WMCKPT:-$W/encspace_cache/mimicgen_${TASK}_224x224_dinowm_${WMTAG}/wm_dinov2s14_${WMTAG}.pt}
RTOK=mqattn_pv4${VSFX:+seg}                      # run-name readout token of the dpinit-family V dirs
STEPS=${STEPS:-600000}; SAVE=${SAVE:-25000}; EVAL_EPS=${EVAL_EPS:-100}; WARMUP=${WARMUP:-1000}
PROJECT=${PROJECT:-post_iclr2027}; OUT=${OUT:-$P}
source $W/eval_env.sh
export DSRL_DEMO_CACHE=${DSRL_DEMO_CACHE:-$W/demo_store_cache}   # reuse the already-encoded demo stores (read-only, mmap)
export PYTHONPATH="$R:$R/0811:${PYTHONPATH:-}" PIXEL_IMG=224 CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
K=$((STEPS / 1000))k
COMMON=(--task $TASK --variant na --seed $SEED --base-output $PRIOR
  --total-env-steps $STEPS --save-freq $SAVE --image-actor --pixel-feature-dim 128 --pixel-shift-pad 4
  --target-entropy 0.0 --init-rollout-chunks $WARMUP --action-magnitude 1.0 --lr 1e-4 --layer-size 256 --num-layers 3
  --actor-grad-clip 10 --ent-coef-lr 1.5e-3 --activation mish --log-std-min -5 --executed-noise-only resample
  --progressive-exploration-steps 75000 --expert-ratio 0.3 --adaptive-expert-ratio
  --async-eval-episodes $EVAL_EPS --async-eval-workers 2 --async-eval-seed-start 10000)
if [ "$ARM" = ours ] || [ "$ARM" = oursmc ] || [ "$ARM" = oursdist ] || [ "$ARM" = randv ]; then
  DTAG=_nodistill; NODISTILL=--no-value-distill; SCRATCH=(); SUFFIX=
  if [ "$ARM" = ours ]; then
    VINIT=dpinitvtd; VDIR=$R/0814/v_dino_robometer_pred_progress_${TASK}_${VBASE}_mqattn_pv4${VSFX}
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${TASK}_ours_dsrl}
  elif [ "$ARM" = oursdist ]; then
    VINIT=dpinitvtd; VDIR=$R/0814/v_dino_robometer_pred_progress_${TASK}_${VBASE}_mqattn_pv4${VSFX}
    DTAG=; NODISTILL=
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${TASK}_ours_online_distill_dsrl}
  elif [ "$ARM" = randv ]; then
    VINIT=; VDIR=$R/0814/v_dino_robometer_pred_progress_${TASK}_${VBASE}_mqattn_pv4${VSFX}
    SCRATCH=(--value-scratch --value-pool-scratch); SUFFIX=_randv
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${TASK}_ours_dsrl_randv}
  else
    VINIT=mcinitvtd; VDIR=$R/0814/v_dino_mc_return_${TASK}_${VBASE}_mqattn_pv4; RTOK=mqattn_pv4
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${TASK}_ours_mcinit_dsrl}
  fi
  NAME=${NAME:-dsrl-na-svg1.0_${TASK}_${WMTAG}_${RTOK}${VINIT:+_$VINIT}_img_bs64_svgrowsall_ue4_utd2as2_alr1.5e-3_lsm5_mish_ramp75k_exec56resample_demo1.0to0.1x40k${DTAG}_${K}_s$SEED$SUFFIX}
  ARGS=(--svg-weight 1.0 --value-model $VDIR --dino-pool mqattn
    --dino-wm-ckpt $WMCKPT
    --dino-wm-arch 6,6,2048 --svg-proprio-source predicted --svg-rows 0 --svg-fit-rows 64 --value-target-tau 0.005
    --value-distill-robometer $SPOOL --value-distill-every 20 --value-distill-batch 256 --value-distill-frames 64
    --value-distill-capacity 20000 --value-distill-mix succ=0.6,fail=0.4 --value-distill-target progress --value-distill-pred
    $NODISTILL "${SCRATCH[@]}" --batch-size 64 --train-freq 4 --utd 2 --svg-actor-steps 2 --svg-fit-utd 0.25 --value-lr 1e-4 --svg-dynamics-lr 1e-4)
elif [ "$ARM" = q ]; then
  NAME=${NAME:-dsrl-na-q_${TASK}_img_bs256_ue1_ncrit10_ncs1_ramp75k_mish_lsm5_alr1.5e-3_pd2_buf100k_exec56resample_demo1.0to0.1x40k_${K}_s$SEED}
  export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${TASK}_q_dsrl}
  ARGS=(--q-critic --n-critics 10 --tau 0.01 --critic-grad-clip 10 --noise-critic-steps 1 --policy-delay 2 --buffer-size 100000
    --batch-size 256 --train-freq 1 --utd 1 --svg-actor-steps 1)
else
  echo "ARM must be ours or q"; exit 2
fi
cd $R
exec $CONDA_PREFIX/bin/python -m online_rl_mimicgen.baselines.dsrl_dino "${COMMON[@]}" "${ARGS[@]}" \
  --wandb-project "$PROJECT" --wandb-name $NAME --output-dir $OUT/$NAME
