# DICE-RL baseline with a Q critic ("pixq"): the same residual actor and best-of-16 exploration, but the critic is
# an ensemble of 10 pixel Q networks (IMPALA encoder), exploration and evaluation rank the 16 prior draws by min-Q,
# and no world model or V is loaded. One update per chunk on full-batch Q rows, no torch.compile.
#
#   bash scripts/run.sh configs/dice_rl/q.sh TASK SEED [STEPS]
#
# The campaign ran it on all four tasks, seeds 43 and 44 at 600k env steps and 45 at 1M. Checkpoints every 50k env
# steps (the paper's pixq cells saved every 25k; saving does not touch training); the in-loop eval is off
# (scripts/eval_loop.sh evaluates every checkpoint).
#
# Reference: reference/launchers/launch_dicerl_ab.sh, ARM=q. Lines 80-84 (name, wandb group, ARGS) and 103-105 (argv
# tail) are encoded below token for token; no flag of this cell was renamed. Line 87 of the launcher appends
# "--nstep 1", but its line 98 unsets that variable before use, so no reference Q run was launched with the flag
# (the process argv ends in --seed <n>); 1 is the default, and the argv below is the one that ran.
# tests/configs/test_configs.py checks argv, name, group and the 301 resolved config keys against that launcher.

MODULE=online_rl_mimicgen.train_online
source "$(dirname "${BASH_SOURCE[0]}")/../tasks.sh"
if [ -n "${DISTILL:-}" ]; then
  echo "DISTILL selects the V-training variant of configs/dice_rl/dvd.sh; the Q cell has none" >&2; return 2
fi

NAME=dicerl-q_${TASK}_pixq_$((STEPS / 1000))k_s$SEED
GROUP=${TASK}_dicerl_q

ARGS=(--task "$TASK" --method dice_rl --total-env-steps "$STEPS" --residual-scale 1.0 --pixel-obs --pixel-feature-dim 128
  --expert-ratio 0.3 --adaptive-expert-ratio --adaptive-expert-ratio-steps 40000 --progressive-exploration-steps 75000
  --bon-axis prior --eval-freq 999999999 --save-freq 50000 --eval-samples 16 --multi-sample-k 4)
ARGS+=(--output-dir "$RUNS_ROOT/$NAME" --wandb-project "$WANDB_PROJECT" --wandb-name "$NAME" --step-penalty 0.0 --eval-strategy greedy
  --eval-episodes 0 --no-save-replay --base-output "$PRIOR_DIR" --seed "$SEED")
