# DICE-RL + DVD ("ours"): residual actor on the frozen prior, trained through the DINO world model by ascending
# gamma^8 V(f(s, a)); exploration and evaluation pick the best of 16 prior draws by the same V(f). V is the offline
# progress-pretrained pv4 readout + 3 heads ("dpinit"), fine-tuned online.
#
#   bash scripts/run.sh configs/dice_rl/dvd.sh TASK SEED [STEPS]          # the task's paper recipe
#   DISTILL=tdonly|tddistill bash scripts/run.sh configs/dice_rl/dvd.sh TASK SEED [STEPS]
#
# DISTILL  how V is trained online
#   tdonly     TD only
#   tddistill  TD + RoboMeter progress distillation every 20 updates. Needs the task's teacher answering on
#              $SPOOL_ROOT/<task> for the whole run (scripts/robometer_teacher.sh start TASK); a run whose log shows
#              "[rm-distill] DISABLED" lost its teacher and is not a tddistill run any more.
#   default    square_d0: tddistill; square_d1, threading_d0, hammer_cleanup_d1: tdonly  (the paper recipe)
# The campaign ran both values on every task (the non-default one is the distill A/B twin), seeds 43 and 44 at 600k
# env steps and 45 at 1M. --value-distill-every counts UPDATES: hammer_cleanup_d1 updates once per chunk and so
# distils about 4x more often per env step than the three --update-every 4 tasks.
#
# hammer_cleanup_d1 is its own cell: the BASE world model (no continuation), DVD on 256 rows per update, one update
# per chunk and no torch.compile (no --update-every 4 / --compile-parts all).
# Checkpoints every 50k env steps; the in-loop eval is off (scripts/eval_loop.sh evaluates every checkpoint).
#
# Reference: reference/launchers/launch_dicerl_ab.sh. ARM=dvd is DISTILL unset, ARM=dvdtd is square_d0 with
# DISTILL=tdonly, ARM=dvdd is the other three tasks with DISTILL=tddistill. Lines 61-78 (name, wandb group, the
# DISTILL / DPRED / UPD / ROWS switches, ARGS) and 101-105 (allocator setting, argv tail) are encoded below token for
# token, with svg -> dvd in six flags and two option values and the asset paths taken from configs/tasks.sh.
# tests/configs/test_configs.py checks argv, name, group and the 301 resolved config keys against that launcher.

MODULE=online_rl_mimicgen.train_online
source "$(dirname "${BASH_SOURCE[0]}")/../tasks.sh"

DISTILL=${DISTILL:-}
PAPER=tdonly
if [ "$TASK" = square_d0 ]; then PAPER=tddistill; fi
DISTILL=${DISTILL:-$PAPER}
case $DISTILL in
  tdonly|tddistill) ;;
  *) echo "DISTILL must be tdonly or tddistill (got '$DISTILL')" >&2; return 2 ;;
esac

NAME=dicerl-dvd_${TASK}_dpinitvtd_${DISTILL}_$((STEPS / 1000))k_s$SEED
# wandb group, as in the reference: the bare name is the task's paper recipe, the A/B twin carries its variant. So
# <task>_dicerl_dvd is TD + distill on square_d0 and TD only on the other tasks: group by config, not by this name.
GROUP=${TASK}_dicerl_dvd
if [ "$DISTILL" != "$PAPER" ]; then GROUP=${TASK}_dicerl_dvd_$DISTILL; fi

WM=$WM_CONT; ROWS=0
if [ "$TASK" = hammer_cleanup_d1 ]; then WM=$WM_BASE; ROWS=256; fi

ARGS=(--task "$TASK" --method dice_rl --total-env-steps "$STEPS" --residual-scale 1.0 --expert-ratio 0.3 --adaptive-expert-ratio
  --adaptive-expert-ratio-steps 40000 --progressive-exploration-steps 75000 --bon-axis prior --eval-freq 999999999
  --save-freq 50000 --eval-samples 16 --multi-sample-k 4 --value-finetune)
if [ "$DISTILL" = tddistill ]; then
  ARGS+=(--value-distill-robometer "$SPOOL" --value-distill-every 20
    --value-distill-batch 256 --value-distill-frames 64 --value-distill-capacity 20000 --value-distill-mix succ=0.6,fail=0.4
    --value-distill-target progress)
fi
ARGS+=(--dino-wm-state-head --dvd-proprio-source predicted)
if [ "$DISTILL" = tddistill ]; then ARGS+=(--value-distill-pred); fi
ARGS+=(--value-target-tau 0.005 --batch-size 64
  --dvd-fit-rows 64)
if [ "$TASK" != hammer_cleanup_d1 ]; then ARGS+=(--update-every 4 --compile-parts all); fi
ARGS+=(--bon-source dvd --pixel-obs --pixel-feature-dim 128
  --value-model "$V_PV4" --value-variant v_td --value-intervention none --dvd-weight 1.0 --actor-gradient-source-normal dvd1
  --dvd-finetune-dynamics --dvd-dynamics dino_wm --dvd-rows "$ROWS" --dino-pool mqattn --no-train-critic --actor-train-encoder
  --dino-wm-ckpt "$WM")
ARGS+=(--output-dir "$RUNS_ROOT/$NAME" --wandb-project "$WANDB_PROJECT" --wandb-name "$NAME" --step-penalty 0.0 --eval-strategy greedy
  --eval-episodes 0 --no-save-replay --base-output "$PRIOR_DIR" --seed "$SEED")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # DVD cells only, as in the reference launcher
