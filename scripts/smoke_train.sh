#!/usr/bin/env bash
# Smoke test of ONE cell on a new machine: a 2000-env-step run that goes through everything a real run goes through
# (assets load, collection, updates, a checkpoint, an evaluation of that checkpoint), then says which stage was reached.
#
#   bash scripts/smoke_train.sh METHOD ARM TASK
#
#   METHOD  dice_rl | dsrl
#   ARM     dice_rl: dvd | q          dsrl: ours | oursmc | oursdist | randv | q        (= configs/<METHOD>/<ARM>.sh)
#   TASK    square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1
#
# NOT EXECUTED ON THE REFERENCE MACHINE: its memory was committed to the campaign when this was written. Verified
# there by dry run only (TRAIN_ONLINE_DRYRUN=1: the resolved configuration equals the cell's except for the keys listed
# below; RUN_PRINT_ARGV=1). The first real run of this script is itself a test of the script.
#
# The cell is the campaign cell, argv token for token (configs/<METHOD>/<ARM>.sh through the same variables as
# scripts/run.sh), with only the budget changed:
#   both      STEPS = SMOKE_STEPS [2000] (the config's own parameter: --total-env-steps and the "2k" in the run name),
#             SEED = 7 unless SEED is set (no campaign run uses it), run directory $RUNS_ROOT/smoke/<run name>,
#             WANDB_MODE=disabled (nothing is logged anywhere), --save-freq SMOKE_SAVE [1000]
#   dice_rl   nothing else. Updates start with the first transition, as in a real run; kind=train rows every 500 steps.
#             The in-loop eval stays off, as in the campaign; afterwards the last checkpoint is scored the campaign's
#             way:  python eval/eval_bon16.py --ckpt <last> --episodes SMOKE_EVAL_EPISODES [3]
#   dsrl      --init-rollout-chunks SMOKE_WARMUP_CHUNKS [32]  (prior-only warm-up: 256 env steps instead of 8000, so
#             that updates start inside the budget), --async-eval-episodes SMOKE_EVAL_EPISODES [3] and
#             --async-eval-workers 1 (the run scores its own checkpoints, as in the campaign, on 3 episodes not 100)
# Everything else is the cell: the same prior, world model, V, demo share, exploration ramp, torch.compile where the
# cell compiles. The numbers a smoke run prints mean nothing; only "did every stage happen" does.
#
# What it needs: the cell's assets (python tests/configs/test_configs.py), including the task's pixel cache; for DSRL
# the demo store, which the first DSRL run of a task builds in about 20 min and later runs reuse; for a TD + distill
# cell (dice_rl dvd on square_d0 by default, or DISTILL=tddistill; dsrl oursdist) the task's teacher
# (scripts/robometer_teacher.sh start TASK), otherwise use DISTILL=tdonly / arm ours. Memory at start-up is the
# cell's, not smaller: an estimated 31-46 GiB of RAM for dice_rl dvd and 23-37 GiB for dice_rl q (the reference
# machine's projected 600k peaks minus the replay growth it measured; never measured at start-up, and not known for
# dsrl); GPU memory as in environment.md, section 1. About 1 GB of disk.
#
# Environment: G=<gpu>; DISTILL (dice_rl dvd); SEED; SMOKE_STEPS, SMOKE_SAVE, SMOKE_EVAL_EPISODES, SMOKE_WARMUP_CHUNKS;
# TRAIN_ONLINE_DRYRUN=1 prints "DRYRUN_CONFIG <json>" for the smoke argv and starts nothing; RUN_PRINT_ARGV=1 prints the
# argv. Exit status: 0 only if collection, updates, a checkpoint and an evaluation all happened.
# Nothing is overwritten: an existing $RUNS_ROOT/smoke/<run name> must be removed by hand before a second try.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$CODE_ROOT/scripts/_tree_guard.sh"
if [ $# -ne 3 ]; then sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; fi
METHOD=$1; ARM=$2; TASK=$3
CFG=$CODE_ROOT/configs/$METHOD/$ARM.sh
case $METHOD in dice_rl|dsrl) ;; *) echo "METHOD must be dice_rl or dsrl (got '$METHOD')" >&2; exit 2;; esac
case $ARM in _*) echo "'$ARM' is not an arm" >&2; exit 2;; esac
[ -f "$CFG" ] || { echo "no arm '$ARM' for $METHOD: configs/$METHOD/ holds $(cd "$CODE_ROOT/configs/$METHOD" && ls [!_]*.sh | sed 's/\.sh$//' | tr '\n' ' ')" >&2; exit 2; }
SEED=${SEED:-7}; STEPS=${SMOKE_STEPS:-2000}; SAVE=${SMOKE_SAVE:-1000}
EPISODES=${SMOKE_EVAL_EPISODES:-3}; WARMUP=${SMOKE_WARMUP_CHUNKS:-32}
case $SEED$STEPS$SAVE$EPISODES$WARMUP in *[!0-9]*) echo "SEED and the SMOKE_* values must be integers" >&2; exit 2;; esac
if [ "$STEPS" -lt 1000 ] || [ "$SAVE" -gt "$STEPS" ] || [ "$EPISODES" -lt 1 ]; then
  echo "need SMOKE_STEPS >= 1000 (the run name counts in thousands), SMOKE_SAVE <= SMOKE_STEPS, SMOKE_EVAL_EPISODES >= 1" >&2; exit 2
fi
if [ "$METHOD" = dsrl ] && [ $(( WARMUP * 8 )) -ge "$STEPS" ]; then
  echo "SMOKE_WARMUP_CHUNKS x 8 env steps must be below SMOKE_STEPS, or no update happens" >&2; exit 2
fi

export RUNS_ROOT=$RUNS_ROOT/smoke WANDB_MODE=disabled
source "$CFG"          # sets MODULE NAME GROUP ARGS, exactly as for scripts/run.sh
export WANDB_RUN_GROUP=smoke
ARGS+=(--save-freq "$SAVE")
if [ "$METHOD" = dsrl ]; then
  ARGS+=(--init-rollout-chunks "$WARMUP" --async-eval-episodes "$EPISODES" --async-eval-workers 1)
fi
RUN=$RUNS_ROOT/$NAME
DISTILS=no
if printf '%s\n' "${ARGS[@]}" | grep -qx -- '--value-distill-robometer' && ! printf '%s\n' "${ARGS[@]}" | grep -qx -- '--no-value-distill'; then
  DISTILS=yes
fi

if [ -n "${RUN_PRINT_ARGV:-}" ]; then
  printf '%s\n' "@MODULE=$MODULE" "@NAME=$NAME" "@RUN=$RUN" "@DISTILS=$DISTILS" \
    "@ENV=PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}" "${ARGS[@]}"
  exit 0
fi
cd "$CODE_ROOT"
require_this_tree
if [ -n "${TRAIN_ONLINE_DRYRUN:-}" ]; then
  if [ "$MODULE" = online_rl_mimicgen.train_online ]; then exec python -m "$MODULE" "${ARGS[@]}"; fi
  exec python scripts/resolve_config.py "$MODULE" "${ARGS[@]}"
fi
if [ -e "$RUN" ]; then echo "$RUN exists (an earlier smoke run): look at it, then remove it yourself to run again" >&2; exit 1; fi
if [ $DISTILS = yes ] && ! bash "$CODE_ROOT/scripts/robometer_teacher.sh" check "$TASK"; then
  echo "this cell distils from the $TASK RoboMeter teacher and none answers: bash scripts/robometer_teacher.sh start $TASK" >&2
  echo "(or smoke-test the TD-only twin: DISTILL=tdonly for dice_rl dvd, arm ours for dsrl)" >&2
  exit 1
fi

mkdir -p "$RUNS_ROOT"
echo "[smoke] $METHOD $ARM $TASK: $STEPS env steps, checkpoint every $SAVE, eval on $EPISODES episodes -> $RUN"
rc=0
python -m "$MODULE" "${ARGS[@]}" || rc=$?
echo "[smoke] trainer exit status $rc"

LAST=
if [ "$METHOD" = dice_rl ]; then
  for f in "$RUN"/checkpoints/*.pt; do
    s=$(basename "$f" .pt)
    if [[ $s =~ ^[0-9]+$ ]] && [ $((10#$s)) -gt 0 ]; then LAST=$f; fi          # the glob is sorted: the highest step wins
  done
  if [ $rc -eq 0 ] && [ -n "$LAST" ]; then
    mkdir -p "$RUN/evals"
    OUT=$RUN/evals/$(basename "$LAST" .pt)
    echo "[smoke] best-of-16 eval of $(basename "$LAST") on $EPISODES episodes (log $OUT.log)"
    python eval/eval_bon16.py --ckpt "$LAST" --episodes "$EPISODES" --out "$OUT.json" > "$OUT.log" 2>&1 \
      || echo "[smoke] the evaluator failed: $(tail -n 1 "$OUT.log")"
  fi
fi

# What happened, read from what the run wrote. The four stages must all be there.
RC=$rc python - "$RUN" "$METHOD" <<'PY'
import glob, json, os, sys
run, method = sys.argv[1], sys.argv[2]
rows = []
try:
    with open(os.path.join(run, "metrics.jsonl")) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
except OSError:
    pass
train = [r for r in rows if r.get("kind") == "train"]
episodes = [r for r in rows if r.get("kind") == "episode"]
steps = max([int(r.get("env_steps", 0)) for r in rows] or [0])
updates = max([int(r.get("updates", 0)) for r in train] or [0])
pattern = "*.pt" if method == "dice_rl" else "*.zip"
checkpoints = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "checkpoints", pattern))
                     if os.path.basename(p).split(".")[0].isdigit() and int(os.path.basename(p).split(".")[0]) > 0)
evals = {}
for path in sorted(glob.glob(os.path.join(run, "evals", "*.json"))):
    if path.endswith(".partial.json"):
        continue
    try:
        result = json.load(open(path))
        evals[os.path.basename(path)] = f"{result['success_rate']:.0f}% of {result.get('episodes', '?')} episodes"
    except (OSError, ValueError, KeyError):
        pass
stages = [("collection", bool(episodes) or steps > 0, f"{steps} env steps, {len(episodes)} finished episodes"),
          ("updates", updates > 0, f"{updates} gradient updates in {len(train)} logged windows"),
          ("checkpoint", bool(checkpoints), ", ".join(checkpoints) or "none"),
          ("evaluation", bool(evals), "; ".join(f"{k}: {v}" for k, v in evals.items()) or "no result file")]
for name, reached, detail in stages:
    print(f"[smoke] {'reached' if reached else 'MISSING'}  {name:<11} {detail}")
ok = all(reached for _, reached, _ in stages) and os.environ["RC"] == "0"
print(f"[smoke] {'PASS' if ok else 'FAIL'}: {run}" + ("" if ok else "   (the trainer's own messages are above; evals/*.log for the evaluation)"))
sys.exit(0 if ok else 1)
PY
