#!/usr/bin/env bash
# Best-of-16 evaluation of every saved checkpoint of DICE-RL runs. (DSRL runs evaluate their own checkpoints while
# they train: --async-eval-episodes 100 in configs/dsrl/_common.sh.)
#
#   bash scripts/eval_loop.sh GPU [--watch SECONDS] [RUN ...]
#
#   GPU       index of the GPU that evaluates (an eval needs about 12 GiB of GPU memory)
#   RUN       run directories, or run names below $RUNS_ROOT; default: every DICE-RL run in $RUNS_ROOT
#   --watch   after each pass wait SECONDS and look for new checkpoints, until every run's last checkpoint (the one at
#             total_env_steps) has an eval or was given up on. Without it: one pass over what exists now. A run that
#             died before its last checkpoint keeps the watch waiting; end it with Ctrl-C.
#
# For every <run>/checkpoints/<step>.pt with step > 0, lowest step first across the runs, one eval at a time:
#
#   python eval/eval_bon16.py --ckpt <run>/checkpoints/<step>.pt --episodes 100 --out <run>/evals/<step>.json
#
# i.e. the protocol of every DICE-RL number: 100 episodes, env seeds 10000-10099, best of 16 prior draws ranked by the
# run's own critic (gamma^8 V(f) for DVD, min-Q for Q), deterministic residual. Output of an eval: <run>/evals/<step>.log.
# Rules, as in the reference eval daemon:
#   - an eval counts only if its JSON is NEWER than the checkpoint file
#   - a checkpoint younger than MIN_AGE seconds [300] waits for the next pass (it may still be being written)
#   - a failed eval is retried on later passes, MAX_TRIES [3] attempts per checkpoint (marker <run>/evals/<step>.failed);
#     an interrupted eval resumes from <step>.partial.json unless that file predates the checkpoint
# Nothing here deletes or rewrites a checkpoint, and nothing stops a process.
# EVAL_BON16 overrides the evaluator script [$CODE_ROOT/eval/eval_bon16.py]; EVAL_ARGS adds arguments to it.
set -uo pipefail
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ $# -ge 1 ] || { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
export G=$1; shift
WATCH=0
if [ "${1:-}" = --watch ]; then WATCH=${2:?--watch needs SECONDS}; shift 2; fi
source "$HERE/env.sh"
EVAL=${EVAL_BON16:-$CODE_ROOT/eval/eval_bon16.py}
MIN_AGE=${MIN_AGE:-300}; MAX_TRIES=${MAX_TRIES:-3}
[ -f "$EVAL" ] || { echo "no evaluator at $EVAL" >&2; exit 2; }
export WANDB_MODE=disabled OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
say(){ echo "$(date '+%F %T') $*"; }
field(){ python -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1/config.json" "$2" 2>/dev/null; }

RUNS=()
if [ $# -gt 0 ]; then
  for r in "$@"; do [ -d "$r" ] || r=$RUNS_ROOT/$r; RUNS+=("${r%/}"); done
else
  for r in "$RUNS_ROOT"/*/; do RUNS+=("${r%/}"); done
fi
is_dice_rl(){ [ -f "$1/config.json" ] && [ "$(field "$1" method)" = dice_rl ]; }

tries(){ local m=$1/evals/$2.failed; if [ -f "$m" ] && [ "$m" -nt "$1/checkpoints/$2.pt" ]; then wc -l < "$m"; else echo 0; fi; }
evaluated(){ [ -f "$1/evals/$2.json" ] && [ "$1/evals/$2.json" -nt "$1/checkpoints/$2.pt" ]; }
todo(){ # "<step> <stem> <run dir>" of every checkpoint that still needs an eval and may be tried now, lowest step first
  local r f stem
  for r in "${RUNS[@]}"; do
    is_dice_rl "$r" || continue
    for f in "$r"/checkpoints/*.pt; do
      stem=$(basename "$f" .pt)
      [[ $stem =~ ^[0-9]{6,7}$ ]] && [ $((10#$stem)) -gt 0 ] || continue
      evaluated "$r" "$stem" && continue
      [ "$(tries "$r" "$stem")" -lt "$MAX_TRIES" ] || continue
      [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge "$MIN_AGE" ] || continue
      echo "$((10#$stem)) $stem $r"
    done
  done | sort -n
}
unfinished(){ # runs whose last checkpoint is not saved yet, or is saved and still has tries left without an eval
  local r f stem total last
  for r in "${RUNS[@]}"; do
    is_dice_rl "$r" || continue
    total=$(field "$r" total_env_steps); last=
    for f in "$r"/checkpoints/*.pt; do
      stem=$(basename "$f" .pt)
      [[ $stem =~ ^[0-9]{6,7}$ ]] && [ $((10#$stem)) -ge $(( ${total:-0} - 1000 )) ] && last=$stem
    done
    if [ -z "$last" ]; then echo "$(basename "$r")"; continue; fi
    evaluated "$r" "$last" || [ "$(tries "$r" "$last")" -ge "$MAX_TRIES" ] || echo "$(basename "$r")"
  done
}

[ ${#RUNS[@]} -gt 0 ] || { echo "no run directory given and none in $RUNS_ROOT" >&2; exit 2; }
failed=0
while :; do
  while read -r step stem run; do
    [ -n "${run:-}" ] || continue
    evaluated "$run" "$stem" && continue
    mkdir -p "$run/evals"
    out=$run/evals/$stem; ckpt=$run/checkpoints/$stem.pt
    resume=()
    if [ -f "$out.partial.json" ] && ! [ "$out.partial.json" -nt "$ckpt" ]; then resume=(--no-resume); fi
    say "eval  $(basename "$run") $stem on GPU $G"
    # shellcheck disable=SC2086
    python "$EVAL" --ckpt "$ckpt" --episodes 100 --out "$out.json" ${resume[@]+"${resume[@]}"} ${EVAL_ARGS:-} >> "$out.log" 2>&1 < /dev/null
    rc=$?
    if [ $rc -eq 0 ] && evaluated "$run" "$stem"; then
      rm -f "$out.failed"
      say "done  $(basename "$run") $stem: $(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["success_rate"])' "$out.json")%"
    else
      if [ -f "$out.failed" ] && ! [ "$out.failed" -nt "$ckpt" ]; then rm -f "$out.failed"; fi   # marker of an older checkpoint file
      echo "$(date '+%F %T') exit $rc" >> "$out.failed"
      failed=$((failed + 1))
      say "FAILED $(basename "$run") $stem (exit $rc, attempt $(tries "$run" "$stem") of $MAX_TRIES; see $out.log)"
    fi
  done < <(todo)
  [ "$WATCH" -gt 0 ] || break
  left=$(unfinished)
  [ -n "$left" ] || break
  sleep "$WATCH"
done
say "eval loop ends: $failed failed attempt(s)$( [ "$WATCH" -gt 0 ] || echo "; checkpoints younger than ${MIN_AGE}s or saved later are left for the next call")"
[ $failed -eq 0 ]
