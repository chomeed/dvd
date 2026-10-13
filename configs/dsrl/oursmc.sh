# DSRL + DVD with a Monte Carlo initialised V: configs/dsrl/ours.sh, but V (readout + 3 heads) was pretrained
# offline on Monte Carlo returns gamma^(t_success - t) (0 on failed episodes) instead of RoboMeter progress. Same
# architecture, world model and recipe; TD only online.
#
#   bash scripts/run.sh configs/dsrl/oursmc.sh TASK SEED [STEPS]
#
# V directory: v_dino_mc_return_<task>_<wm tag>_mqattn_pv4 on every task (no _seg variant; configs/tasks.sh). The
# campaign ran square_d0, square_d1 and hammer_cleanup_d1 (seeds 43 and 44 at 600k env steps, 45 at 1M);
# threading_d0 is defined by the launcher and has no reference result.
# Reference: reference/launchers/launch_hammer_dsrl.sh ARM=oursmc (lines 54, 66-68); the rest is _common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
VDIR=$V_MC; RTOK=mqattn_pv4; VINIT=mcinitvtd; NODISTILL=1; SCRATCH=0; SUFFIX=
GROUP=${TASK}_ours_mcinit_dsrl
dsrl_dvd_cell
