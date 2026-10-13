#!/usr/bin/env bash
# Run a list of cells on ONE GPU, one after another:
#
#   bash scripts/run_campaign.sh CELLS_FILE GPU [--list]
#
#   CELLS_FILE   one run per line: <config> <task> <seed> <env steps> [VAR=value ...]; blank lines and # comments are
#                skipped. configs/campaign/dice_rl.txt and configs/campaign/dsrl.txt are the reference campaign.
#                Any subset works:  bash scripts/run_campaign.sh <(grep square_d0 configs/campaign/dsrl.txt) 0
#   GPU          index of the GPU that trains (CUDA device and EGL render device)
#   --list       print every run with its name and what would happen to it, start nothing
#
# Per line, in order:
#   skip      its run directory $RUNS_ROOT/<name> exists: the run is finished, running or dead. Nothing is ever
#             overwritten, resumed or stopped here; to redo a run, move its directory away yourself.
#   skip      it distils (TD + distill) and no RoboMeter teacher answers on the task's spool
#             (bash scripts/robometer_teacher.sh start TASK, in another terminal)
#   run       bash scripts/run.sh ... in the foreground, output to $RUNS_ROOT/logs/<name>.log
# A run that exits non-zero is reported and the list continues. Exit status 0 = every line was run or already had a
# run directory; 1 = at least one line failed, was malformed or was skipped for a missing teacher.
#
# This is a plain sequential runner: it does not watch memory, disk or other users of the GPU. Start one per GPU, and
# read environment.md for what one run needs. DICE-RL checkpoints are evaluated by scripts/eval_loop.sh (it can run
# alongside); DSRL runs evaluate their own checkpoints.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ $# -lt 2 ] || [ $# -gt 3 ] || { [ $# -eq 3 ] && [ "$3" != --list ]; }; then
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2
fi
CELLS=$1; export G=$2; LIST=${3:-}
source "$HERE/env.sh"
[ -r "$CELLS" ] || { echo "cannot read $CELLS" >&2; exit 2; }
mapfile -t LINES < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$CELLS" | sed '/^[[:space:]]*$/d')
[ ${#LINES[@]} -gt 0 ] || { echo "$CELLS lists no run" >&2; exit 2; }
say(){ echo "$(date '+%F %T') $*"; }

pending=0
for line in "${LINES[@]}"; do
  read -r cfg task seed steps rest <<< "$line"
  vars=()
  for kv in $rest; do
    case $kv in [A-Z_]*=*) vars+=("$kv");; *) say "BAD LINE (not VAR=value: $kv): $line"; pending=$((pending + 1)); continue 2;; esac
  done
  if [ -z "${steps:-}" ]; then say "BAD LINE (need <config> <task> <seed> <env steps>): $line"; pending=$((pending + 1)); continue; fi
  case $cfg in /*) ;; *) cfg=$CODE_ROOT/$cfg;; esac
  # DISTILL comes from the line only, never from the caller's shell
  cell=(env -u DISTILL ${vars[@]+"${vars[@]}"} bash "$HERE/run.sh" "$cfg" "$task" "$seed" "$steps")
  if ! argv=$(RUN_PRINT_ARGV=1 "${cell[@]}" 2>&1); then
    say "BAD LINE ($(echo "$argv" | tail -1)): $line"; pending=$((pending + 1)); continue
  fi
  name=$(sed -n 's/^@NAME=//p' <<< "$argv")
  distils=no
  if grep -qx -- '--value-distill-robometer' <<< "$argv" && ! grep -qx -- '--no-value-distill' <<< "$argv"; then distils=yes; fi

  if [ -e "$RUNS_ROOT/$name" ]; then
    say "skip  $name: $RUNS_ROOT/$name exists"; continue
  fi
  if [ -n "$LIST" ]; then
    say "run   $name$([ $distils = yes ] && echo "  (needs the $task teacher)")"; continue
  fi
  if [ $distils = yes ] && ! bash "$HERE/robometer_teacher.sh" check "$task"; then
    say "skip  $name: TD + distill, and no RoboMeter teacher answers for $task"; pending=$((pending + 1)); continue
  fi
  mkdir -p "$RUNS_ROOT/logs"
  say "start $name on GPU $G (log $RUNS_ROOT/logs/$name.log)"
  "${cell[@]}" > "$RUNS_ROOT/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ $rc -eq 0 ]; then say "done  $name"; else say "FAILED $name (exit $rc): $(tail -1 "$RUNS_ROOT/logs/$name.log")"; pending=$((pending + 1)); fi
done
say "end of list: $pending line(s) failed, malformed or skipped for a missing teacher"
[ $pending -eq 0 ]
