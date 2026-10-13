#!/usr/bin/env bash
# Start the RoboMeter teacher of one spool (stays in the foreground; run it under nohup / tmux, one per teacher):
#
#   SPOOL=<name> [G=<gpu>] [MODEL=<LoRA checkpoint dir or hf id>] bash robometer/serve.sh [--print]
#
#   name = threading_d0 | square_d0 | hammer_cleanup_d1 | square_d1 | square_d1_seg     (robometer/teachers.sh)
#
# Serves $SPOOL_ROOT/<name> with the teacher's LoRA and its training sentence; <spool>/READY appears when the model is
# loaded (about 11 GiB of GPU memory, 2-3 GiB of RAM). Needed for robometer/score_rollouts.py and for every online
# "TD+distill" cell, which passes --value-distill-robometer $SPOOL_ROOT/<task>; TD-only cells need no server.
# Runs in the RoboMeter environment ($ROBOMETER_ROOT, its own python 3.10 uv venv), never in the main one.
# The four per-task teachers can equally be started with scripts/robometer_teacher.sh start <task> (same LoRA, spool
# and sentence); this script also knows square_d1_seg (offline rescoring) and the directory a local fine-tune writes.
# --print shows the command instead of running it.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/scripts/env.sh" ]; then source "$ROOT/scripts/env.sh"; fi
: "${CKPT_ROOT:?source scripts/env.sh first (CKPT_ROOT is not set)}" "${SPOOL_ROOT:?SPOOL_ROOT is not set}"
: "${ROBOMETER_ROOT:?ROBOMETER_ROOT is not set (the robometer checkout)}"
source "$ROOT/robometer/teachers.sh"
teacher "${SPOOL:?set SPOOL=<teacher name>}"
MODEL=${MODEL:-$(teacher_model)}
CMD=(uv run --no-sync python "$ROOT/robometer/robometer_server.py" --model "$MODEL" --spool "$SPOOL_ROOT/$SPOOL"
     --task "$TEACHER_TEXT")
if [ "${1:-}" = "--print" ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
case "$MODEL" in
  /*|./*) [ -d "$MODEL" ] || { echo "LoRA checkpoint $MODEL not found (set MODEL=...)" >&2; exit 1; } ;;
esac
mkdir -p "$SPOOL_ROOT/$SPOOL"
READY="$SPOOL_ROOT/$SPOOL/READY"          # written by the server (its pid) once the model is loaded
if [ -f "$READY" ]; then
  pid=$(head -n 1 "$READY" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "a server (pid $pid) already serves $SPOOL_ROOT/$SPOOL; not starting a second one" >&2; exit 1
  fi
  rm -f "$READY"                          # stale: a client must not wait on a dead server
fi
cd "$ROBOMETER_ROOT"
export UNSLOTH_DISABLE_STATISTICS=1 PYTHONUNBUFFERED=1
if [ -n "${G:-}" ]; then export CUDA_VISIBLE_DEVICES=$G; fi
echo "[serve] teacher $SPOOL: $MODEL -> $SPOOL_ROOT/$SPOOL  task: $TEACHER_TEXT"
# the main environment's PYTHONPATH (python 3.12 source trees) must not leak into RoboMeter's python 3.10 environment
exec env -u PYTHONPATH "${CMD[@]}"
