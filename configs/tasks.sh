# Per-task asset table. Sourced by every config and by scripts/robometer_teacher.sh; needs TASK and the
# scripts/env.sh roots. It is the only place that maps a task to the files a run loads.
#
#   task               frozen prior                     WM tag (steps)  V for DSRL   online RoboMeter teacher (LoRA)
#   square_d0          square_d0_40k_rs9                ph_conv_150k    pv4          robometer-4b-lora-square-d0-seg
#   square_d1          square_d1_45k_statefix           ph_conv_150k    pv4_seg      robometer-4b-lora-square-d1  (NON-seg)
#   threading_d0       threading_d0_55k_statefix        ph_conv_168k    pv4_seg      robometer-4b-lora-threading-d0-seg
#   hammer_cleanup_d1  hammer_cleanup_d1_35k_statefix   ph_conv_217k    pv4          robometer-4b-lora-hammer-cleanup-seg
#
# Encodes the `case $TASK` tables of the reference launchers (reference/launchers/launch_dicerl_ab.sh lines 46-56 and
# 59, launch_hammer_dsrl.sh lines 29-40) with the reference machine's directories replaced by the roots:
#   hdr_robot/priors/<prior>                      -> $CKPT_ROOT/priors/<prior>
#   wm_runs/encspace_cache/[<dir>/]<file>         -> $CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_<tag>/<file>
#                                                    (the reference keeps threading_d0's files outside any directory)
#   <research tree>/0814/<V dir>                  -> $CKPT_ROOT/value/<V dir>
#   wm_runs/rm_spool_square | rm_spool | rm_spool_square_d1 | rm_spool_hammer
#                                                 -> $SPOOL_ROOT/square_d0 | threading_d0 | square_d1 | hammer_cleanup_d1
#
# World models: every V was fitted on the BASE model <tag>; the runs load the CONTINUATION <tag>_cont<steps>, except
# DICE-RL on hammer_cleanup_d1, which loads the base model.
# V directories: *_mqattn_pv4 (RoboMeter progress labels) is what DICE-RL loads on every task. DSRL loads the same
# directory on square_d0 and hammer_cleanup_d1 and the *_pv4_seg refit on square_d1 and threading_d0 (rescored
# labels; from a different LoRA only on square_d1). v_dino_mc_return_* (Monte Carlo return labels) is the DSRL oursmc arm.
# Teachers: the LoRA that must answer on $SPOOL_ROOT/<task> during TD + distill runs (last column). On square_d1 it
# is the non-segmented LoRA, the one that labelled the plain pv4 V (not the seg LoRA of pv4_seg). robometer/teachers.sh
# holds the LoRAs with their instruction sentences; scripts/robometer_teacher.sh start TASK serves one.
# Demo data: DEMO_DATASET and PIXEL_CACHE_FILE are NOT passed to the trainers; the code derives the same two paths
# from the roots (TASKS in rl/online_rl_mimicgen/common.py; the pixel cache from CACHE_ROOT). They are here for the
# scripts that prepare and check them (scripts/build_pixel_cache.sh, tests/configs/test_configs.py).
# scripts/fetch_assets.py downloads everything in this table that is published, to these paths.

case ${TASK:-} in
  square_d0)         PRIOR=square_d0_40k_rs9;              WM_STEPS=150k; V_SEG= ;;
  square_d1)         PRIOR=square_d1_45k_statefix;         WM_STEPS=150k; V_SEG=_seg ;;
  threading_d0)      PRIOR=threading_d0_55k_statefix;      WM_STEPS=168k; V_SEG=_seg ;;
  hammer_cleanup_d1) PRIOR=hammer_cleanup_d1_35k_statefix; WM_STEPS=217k; V_SEG= ;;
  *)
    echo "TASK must be square_d0, square_d1, threading_d0 or hammer_cleanup_d1 (got '${TASK:-}')" >&2
    return 2 ;;
esac

PRIOR_DIR=$CKPT_ROOT/priors/$PRIOR
WM_BASE_TAG=ph_conv_$WM_STEPS
WM_CONT_TAG=ph_conv_${WM_STEPS}_cont$WM_STEPS
WM_BASE=$CKPT_ROOT/wm/mimicgen_${TASK}_224x224_dinowm_$WM_BASE_TAG/wm_dinov2s14_$WM_BASE_TAG.pt
WM_CONT=$CKPT_ROOT/wm/mimicgen_${TASK}_224x224_dinowm_$WM_CONT_TAG/wm_dinov2s14_$WM_CONT_TAG.pt
V_PV4=$CKPT_ROOT/value/v_dino_robometer_pred_progress_${TASK}_${WM_BASE_TAG}_mqattn_pv4
V_DSRL=$V_PV4$V_SEG
V_DSRL_TOKEN=mqattn_pv4${V_SEG:+seg}               # run-name token of V_DSRL
V_MC=$CKPT_ROOT/value/v_dino_mc_return_${TASK}_${WM_BASE_TAG}_mqattn_pv4
SPOOL=$SPOOL_ROOT/$TASK
DEMO_DATASET=$DATA_ROOT/mimicgen_${TASK}_224x224
PIXEL_CACHE_FILE=$CACHE_ROOT/pixel_cache/pixel_cache_mimicgen_${TASK}_224x224_224.npy
