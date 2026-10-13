# DSRL + DVD with a randomly initialised V: configs/dsrl/ours.sh plus --value-scratch (V's heads re-initialised)
# and --value-pool-scratch (the mqattn readout re-initialised). The V directory then supplies only the architecture,
# the state scaler and the readout shape; TD only online.
#
#   bash scripts/run.sh configs/dsrl/randv.sh TASK SEED [STEPS]
#
# The random V is THE SAME FOR EVERY SEED: the trainer's start-up check (check_equivalence) calls
# torch.manual_seed(0) after the run seed is set and before V is built. Every reference random-V run has this
# property; it is kept, so the seeds of this arm differ in everything except V's initial weights.
# The campaign ran square_d0, square_d1 and hammer_cleanup_d1 (seeds 43 and 44 at 600k env steps, 45 at 1M);
# threading_d0 is defined by the launcher and has no reference result. Run names end in _randv and carry no V-init token.
# Reference: reference/launchers/launch_hammer_dsrl.sh ARM=randv (lines 54, 62-65); the rest is _common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
VDIR=$V_DSRL; RTOK=$V_DSRL_TOKEN; VINIT=; NODISTILL=1; SCRATCH=1; SUFFIX=_randv
GROUP=${TASK}_ours_dsrl_randv
dsrl_dvd_cell
