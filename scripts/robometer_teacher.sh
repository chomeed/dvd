#!/usr/bin/env bash
# The RoboMeter teacher of a task: the process that scores episodes for online V distillation (TD + distill cells:
# configs/dice_rl/dvd.sh with DISTILL=tddistill, configs/dsrl/oursdist.sh). Trainer and teacher exchange frames and
# scores through the spool directory $SPOOL_ROOT/<task>; the teacher runs in RoboMeter's own Python 3.10 environment.
#
#   bash scripts/robometer_teacher.sh start TASK     # serve in the foreground (about 11 GiB of GPU memory); G=<gpu> pins it
#   bash scripts/robometer_teacher.sh check TASK     # exit 0 if a teacher answers a 16-frame test clip within 120 s
#   bash scripts/robometer_teacher.sh show  TASK     # print the server command (LoRA, spool, instruction sentence)
#
# `start` and `show` are robometer/serve.sh for the task's online teacher (SPOOL=<task>); robometer/teachers.sh says
# which LoRA that is and where it is read from. `check` is what scripts/run_campaign.sh asks before a distilling run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ $# -eq 2 ] || { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
ACTION=$1; TASK=$2
source "$CODE_ROOT/configs/tasks.sh"          # validates TASK, sets SPOOL

case $ACTION in
  start) SPOOL=$TASK exec bash "$CODE_ROOT/robometer/serve.sh" ;;
  show)  SPOOL=$TASK exec bash "$CODE_ROOT/robometer/serve.sh" --print ;;
  check)
    [ -d "$SPOOL/req" ] && [ -d "$SPOOL/res" ] || { echo "no teacher has served $SPOOL yet" >&2; exit 1; }
    SPOOL_DIR=$SPOOL exec python - <<'PY'
import os, sys, time, uuid
from pathlib import Path
import numpy as np
spool, rid = Path(os.environ["SPOOL_DIR"]), f"check_{uuid.uuid4().hex[:8]}"
tmp = spool / "req" / f"{rid}.tmp"
with open(tmp, "wb") as handle:
    np.save(handle, np.zeros((16, 224, 224, 3), np.uint8))
tmp.rename(spool / "req" / f"{rid}.npy")
start = time.time()
while time.time() - start < 120:
    for ext in ("npy", "err"):
        answer = spool / "res" / f"{rid}.{ext}"
        if answer.exists():
            time.sleep(0.5)
            answer.unlink()
            sys.exit(0 if ext == "npy" else 1)
    time.sleep(1)
(spool / "req" / f"{rid}.npy").unlink(missing_ok=True)       # nobody took it: do not leave it for a later teacher
print(f"no answer on {spool} within 120 s", file=sys.stderr)
sys.exit(1)
PY
    ;;
  *) echo "unknown action '$ACTION' (start | check | show)" >&2; exit 2 ;;
esac
