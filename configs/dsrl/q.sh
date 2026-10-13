# DSRL baseline with Q critics: DSRL-NA as published, 10 action-space critics Q^A + 10 noise-space critics Q^W,
# batch 256, one update per env chunk, policy delay 2, replay of 100k chunks. No world model, no V.
#
#   bash scripts/run.sh configs/dsrl/q.sh TASK SEED [STEPS]
#
# The campaign ran all four tasks, seeds 43 and 44 at 600k env steps and 45 at 1M.
# Reference: reference/launchers/launch_hammer_dsrl.sh ARM=q, lines 78-81 (name, wandb group, ARGS) and 86-87 (argv
# tail), token for token, with svg -> dvd in one flag (--dvd-actor-steps, the actor updates per step: the flag is
# shared with the DVD arms); the rest is configs/dsrl/_common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME=dsrl-na-q_${TASK}_img_bs256_ue1_ncrit10_ncs1_ramp75k_mish_lsm5_alr1.5e-3_pd2_buf100k_exec56resample_demo1.0to0.1x40k_${K}_s$SEED
GROUP=${TASK}_q_dsrl
ARGS=("${COMMON[@]}" --q-critic --n-critics 10 --tau 0.01 --critic-grad-clip 10 --noise-critic-steps 1 --policy-delay 2 --buffer-size 100000
  --batch-size 256 --train-freq 1 --utd 1 --dvd-actor-steps 1)
ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-name "$NAME" --output-dir "$RUNS_ROOT/$NAME")
