# DSRL + DVD, TD + online RoboMeter distillation: configs/dsrl/ours.sh without --no-value-distill, so V is also
# regressed on the teacher's progress scores every 20 V steps. Needs the task's teacher answering on
# $SPOOL_ROOT/<task> for the whole run (scripts/robometer_teacher.sh start TASK).
#
#   bash scripts/run.sh configs/dsrl/oursdist.sh TASK SEED [STEPS]
#
# The campaign ran square_d0 and hammer_cleanup_d1 (seeds 43 and 44 at 600k env steps, 45 at 1M). The launcher also
# defines square_d1 and threading_d0, which have no reference result. On square_d1 the spool's teacher is not the
# LoRA that labelled the pv4_seg V this arm loads there (configs/tasks.sh); on threading_d0 it is the same one.
# Reference: reference/launchers/launch_hammer_dsrl.sh ARM=oursdist (lines 54, 58-61); the rest is _common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
VDIR=$V_DSRL; RTOK=$V_DSRL_TOKEN; VINIT=dpinitvtd; NODISTILL=0; SCRATCH=0; SUFFIX=
GROUP=${TASK}_ours_online_distill_dsrl
dsrl_dvd_cell
