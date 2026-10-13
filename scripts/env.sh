# Environment of every command in this repository:   source scripts/env.sh
# Sourced, never executed. Nothing else in the repository holds a path: every script and config resolves its files
# from the roots below or from its arguments. Set a variable BEFORE sourcing to override its default [in brackets].
#
# Roots
#   CODE_ROOT        this repository: ALWAYS the checkout this file lies in. It cannot be preset; a different
#                    value found in the environment (another checkout's env.sh was sourced in this shell) is replaced
#                    with one warning line, so that a script never runs another checkout's code
#   ASSET_ROOT       parent of the asset roots                                   [$CODE_ROOT/assets]
#   DATA_ROOT        LeRobot datasets: demos and prior rollouts                  [$ASSET_ROOT/datasets]
#   CKPT_ROOT        priors/ wm/ value/ robometer/ rm_scores/                    [$ASSET_ROOT/checkpoints]
#   CACHE_ROOT       pixel_cache/ wm_frames/ dsrl_demo_store/                    [$ASSET_ROOT/cache]
#   SPOOL_ROOT       RoboMeter request/response directories, one per task        [$ASSET_ROOT/spool]
#   RUNS_ROOT        run directories <run name>/ and logs/                       [$ASSET_ROOT/runs]
#   THIRD_PARTY      third-party checkouts (environment.md)                      [$CODE_ROOT/third_party]
#   MIMICGEN_ROOT    NVlabs/mimicgen checkout                                    [$THIRD_PARTY/mimicgen]
#   TASK_ZOO_ROOT    ARISE-Initiative/robosuite-task-zoo checkout                [$THIRD_PARTY/robosuite-task-zoo]
#   DSRL_ROOT        DSRL checkout holding its stable-baselines3 fork            [$THIRD_PARTY/dsrl]
#   ROBOMETER_ROOT   RoboMeter checkout with its own Python 3.10 uv environment  [$THIRD_PARTY/robometer]
#   ROBOSUITE_ROOT   optional: a directory CONTAINING robosuite/ that replaces the pip-installed robosuite 1.4.1 [unset]
# Switches
#   G                GPU index: sets CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID together (the reference convention) [unset]
#   CUDA_VISIBLE_DEVICES, MUJOCO_EGL_DEVICE_ID   used as given when G is unset; the EGL device defaults to the
#                    first CUDA device (robosuite refuses an EGL device that is not a visible CUDA device)
#   HF_OFFLINE       1 = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1, as in every reference run; needs the base models
#                    (environment.md) in the Hugging Face cache already                                           [unset]
#   HF_NAMESPACE     Hugging Face account that hosts the datasets and checkpoints; read by scripts/fetch_assets.py
#                    and robometer/finetune/train_lora.sh (environment.md, section 5)                        [chomeed]
#   WANDB_PROJECT    [dvd]        WANDB_MODE   [offline]   (the reference campaign logged online to post_iclr2027)
# Fixed here (the code reads them at import; a different value silently changes shapes or rendering)
#   PIXEL_IMG=224    side of every camera image; the pixel encoder defaults to 84 without it
#   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
# Read by the code when set, never set here
#   PIXEL_CACHE      DICE-RL: the demo pixel cache FILE  [$CACHE_ROOT/pixel_cache/pixel_cache_mimicgen_<task>_224x224_224.npy]
#   PIXEL_CACHE_DIR  DSRL: the DIRECTORY holding those files                     [$CACHE_ROOT/pixel_cache]
#   DSRL_DEMO_CACHE  DSRL: pre-encoded demo stores                               [$CACHE_ROOT/dsrl_demo_store]
#   PYTORCH_CUDA_ALLOC_CONF   exported by configs/dice_rl/dvd.sh only (expandable_segments:True), as in the reference
#   WANDB_RUN_GROUP  overrides the wandb group a config chooses
#   TRAIN_ONLINE_DRYRUN, RUN_PRINT_ARGV, DISTILL   scripts/run.sh
#   EVAL_BON16, EVAL_ARGS, MIN_AGE, MAX_TRIES      scripts/eval_loop.sh
#   DATASET, OUT     python -m online_rl_mimicgen.pixel_cache (scripts/build_pixel_cache.sh TASK sets both)
#   SMOKE_STEPS, SMOKE_SAVE, SMOKE_EVAL_EPISODES, SMOKE_WARMUP_CHUNKS   scripts/smoke_train.sh
#   WS_TASK          dinowm/ and value/ scripts (mandatory there)
#   LIVE_TREE, LEGACY_RUNS   tests/ (optional inputs from the reference machine)
# Switches of a single script in dinowm/, value/, robometer/ or eval/ are listed in that script's header.

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -n "${CODE_ROOT:-}" ] && [ "$CODE_ROOT" != "$_here" ] && ! [ "$CODE_ROOT" -ef "$_here" ]; then
  echo "scripts/env.sh: CODE_ROOT was $CODE_ROOT; using this checkout, $_here (asset roots already set are kept)" >&2
fi
export CODE_ROOT=$_here
unset _here
export ASSET_ROOT=${ASSET_ROOT:-$CODE_ROOT/assets}
export DATA_ROOT=${DATA_ROOT:-$ASSET_ROOT/datasets}
export CKPT_ROOT=${CKPT_ROOT:-$ASSET_ROOT/checkpoints}
export CACHE_ROOT=${CACHE_ROOT:-$ASSET_ROOT/cache}
export SPOOL_ROOT=${SPOOL_ROOT:-$ASSET_ROOT/spool}
export RUNS_ROOT=${RUNS_ROOT:-$ASSET_ROOT/runs}
export THIRD_PARTY=${THIRD_PARTY:-$CODE_ROOT/third_party}
export MIMICGEN_ROOT=${MIMICGEN_ROOT:-$THIRD_PARTY/mimicgen}
export TASK_ZOO_ROOT=${TASK_ZOO_ROOT:-$THIRD_PARTY/robosuite-task-zoo}
export DSRL_ROOT=${DSRL_ROOT:-$THIRD_PARTY/dsrl}
export ROBOMETER_ROOT=${ROBOMETER_ROOT:-$THIRD_PARTY/robometer}
export HF_NAMESPACE=${HF_NAMESPACE:-chomeed}
export WANDB_PROJECT=${WANDB_PROJECT:-dvd}
export WANDB_MODE=${WANDB_MODE:-offline}

# This repository's packages first, then the source checkouts; an installed copy of any of them is shadowed.
_path=$CODE_ROOT/rl:$CODE_ROOT/envs/lerobot_env_mimicgen/src:$CODE_ROOT/dinowm
if [ -n "${ROBOSUITE_ROOT:-}" ]; then _path=$_path:$ROBOSUITE_ROOT; fi
_path=$_path:$MIMICGEN_ROOT:$TASK_ZOO_ROOT
case "${PYTHONPATH:-}" in
  "$_path"|"$_path":*) ;;                                 # already first (sourced twice): do not grow PYTHONPATH
  *) export PYTHONPATH=$_path${PYTHONPATH:+:$PYTHONPATH} ;;
esac
unset _path

export PIXEL_IMG=224 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
if [ -n "${G:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G
else
  _gpu=${CUDA_VISIBLE_DEVICES:-0}
  export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${_gpu%%,*}}
  unset _gpu
fi
if [ "${HF_OFFLINE:-}" = 1 ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
