# DSRL + DVD ("ours"), TD only: the noise-space actor ascends gamma^8 V(f(s, a)) through the DINO world model and
# the frozen prior; no Q critic. V is the offline progress-pretrained pv4 readout + 3 heads ("dpinit"), fine-tuned
# online with TD only (the RoboMeter flags are passed and switched off by --no-value-distill, as in the reference).
#
#   bash scripts/run.sh configs/dsrl/ours.sh TASK SEED [STEPS]
#
# V directory: *_mqattn_pv4 on square_d0 and hammer_cleanup_d1, the *_mqattn_pv4_seg refit on square_d1 and
# threading_d0 (configs/tasks.sh). The campaign ran all four tasks, seeds 43 and 44 at 600k env steps and 45 at 1M.
# Reference: reference/launchers/launch_hammer_dsrl.sh ARM=ours (lines 54-57); the rest is configs/dsrl/_common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
VDIR=$V_DSRL; RTOK=$V_DSRL_TOKEN; VINIT=dpinitvtd; NODISTILL=1; SCRATCH=0; SUFFIX=
GROUP=${TASK}_ours_dsrl
dsrl_dvd_cell
