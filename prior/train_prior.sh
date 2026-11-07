#!/usr/bin/env bash
# Train the frozen DiT prior of one task with the recipe of the released priors. Optional: the released priors are
# assets (scripts/fetch_assets.py), and every campaign run, rollout set, world model, score table, V and teacher LoRA
# was made from those, not from a retrained one.
#
#   TASK=<task> STAGE=data|train|install [G=<gpu>] [SMOKE_STEPS=<n>] bash prior/train_prior.sh [--print]
#
#   task               run name (--job_name)                                   steps   save  log  release  installed as
#   square_d0          mtdit_square_d0_robotstate9                             40000  10000  200   040000  square_d0_40k_rs9
#   square_d1          mimicgen_square_d1_mtdit_flow_60k_robotstate9           60000   5000  100   045000  square_d1_45k_statefix
#   threading_d0       mimicgen_threading_d0_mtdit_flow_60k_robotstate9        60000   5000  100   055000  threading_d0_55k_statefix
#   hammer_cleanup_d1  mimicgen_hammer_cleanup_d1_mtdit_flow_60k_robotstate9   60000   5000  100   035000  hammer_cleanup_d1_35k_statefix
#
# Common to all four (their train_config.json files): LeRobot's in-tree multi_task_dit policy, --policy.objective=
# flow_matching and every other policy field at its default (CLIP ViT-B/16 image and text encoders, 2 observation
# frames, horizon 32, DiT 512 wide x 6 layers x 8 heads); batch 64, seed 1000 (the default), 8 data workers; Adam lr
# 2e-5 (vision encoder x 0.1), cosine schedule over all steps, gradient clip 10; fp32, one GPU, no image augmentation.
# The run length is the original one even where the release step is earlier: the schedule spans the whole run.
#
#   data     python prior/state9_dataset.py: $DATA_ROOT/mimicgen_<task>_224x224 -> $DATA_ROOT/mimicgen_<task>_224x224_robotstate9,
#            a copy whose observation.state keeps its first 9 dims (eef_pos 3 | eef_quat 4 | gripper_qpos 2), with
#            the statistics cut the same way and the videos hardlinked. CPU, a few seconds, 89-164 MB. Needs the whole
#            demo dataset (python scripts/fetch_assets.py --tier train --task <task> --only dataset).
#   train    LeRobot's trainer with the released run's argv, into $RUNS_ROOT/prior_train/<run>/: python
#            prior/lerobot_train.py, which is python -m lerobot.scripts.lerobot_train plus one shim (the pinned wandb
#            0.29 lacks the Run.get_url that LeRobot's wandb logger calls; without it every run stops at start).
#            One GPU with 40 GB free (39.3 GiB peak), about 3.5-4 h for 60k steps on a B200 (the released runs took
#            8.8-10.4 h, most of it their in-loop evaluation), 2.5 GB per checkpoint (30 GB for 12; square_d0 10 GB).
#            Running the command again resumes from checkpoints/last. The Arrow cache of the dataset goes to
#            $HF_DATASETS_CACHE [$CACHE_ROOT/hf_datasets]; wandb logs to $WANDB_PROJECT in $WANDB_MODE (online,
#            offline or disabled; square_d0 ran without wandb and keeps it off).
#   install  python prior/install_prior.py: copies ONLY checkpoints/<release>/pretrained_model of the run to
#            $CKPT_ROOT/priors/<installed as>/checkpoints/<release>/pretrained_model, the directory configs/tasks.sh
#            names for every RL cell, then checks it (9-D state, conditioning width, normaliser, the checkpoint the
#            RL code resolves). A LeRobot run directory must never be used as the prior directory itself: the RL code
#            prefers its checkpoints/last, i.e. the final step, over every numbered step.
#
# NOTHING IS OVERWRITTEN: data and install refuse an existing target, and LeRobot refuses an existing run directory
# unless it resumes. The install target is where the RELEASED prior lives: move that directory away first, or set
# CKPT_ROOT to another tree (with the world models and V directories linked into it). A retrained prior is a different
# prior: no step is bit-reproducible, and the release steps were picked by hand from a 10-episode in-loop evaluation
# (hammer_cleanup_d1's on purpose not its best). Score candidate checkpoints first, without installing anything:
#   python eval/eval_prior.py --task <task> --prior $RUNS_ROOT/prior_train/<run>/checkpoints/<step>/pretrained_model --out <json>
# (the released priors: 74 / 38 / 64 / 40 % in the row order above).
#
# How the released priors were made (prior/released_runs.json is the record; AGENTS.md section 5.5):
#   square_d0        stock LeRobot on a 9-D copy of its demo dataset: what this script does. Its run logged nothing;
#                    the argv is its train_config.json written as flags (plus the inert --eval.batch_size=50, which
#                    LeRobot otherwise derives from the host's CPU count).
#   the other three  a LeRobot 0.6.0 fork that is not available, on the full 26 / 37-D dataset with
#                    --policy.n_state_dims=9 (the policy kept the first 9 state dims after normalising), followed by
#                    the "statefix" (config and normaliser cut to 9 dims, weights untouched). The stock policy on the
#                    9-D copy builds the same network with the same normaliser and needs no fix. Their recorded argv is
#                    replayed in its order with: --dataset.repo_id and --dataset.root -> the 9-D copy; --output_dir
#                    from the roots; --policy.repo_id under $HF_NAMESPACE (inert: nothing is pushed);
#                    --wandb.project=$WANDB_PROJECT followed by --wandb.mode=$WANDB_MODE (LeRobot goes online when
#                    the mode is unset); --policy.n_state_dims=9 and the three --env.* flags dropped. Without --env.type
#                    there is no in-loop evaluation: LeRobot's would feed the env's 26 / 37-D state to the 9-D policy.
#                    Its --env_eval_freq / --eval.* flags stay, inert.
#
# SMOKE_STEPS=<n>  the same argv with --steps=<n> --save_freq=<n> and job smoke_<run>, into $RUNS_ROOT/smoke/prior_<run>
#                  (STAGE=train only). Running it again with a larger <n> resumes that run.
# --print          show the command (one token per line) instead of running it; nothing is checked or written.
# Nothing here imports online_rl_mimicgen except the final check of install, which takes it from this checkout.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/scripts/env.sh" ]; then source "$ROOT/scripts/env.sh"; fi
: "${CKPT_ROOT:?source scripts/env.sh first (CKPT_ROOT is not set)}" "${DATA_ROOT:?DATA_ROOT is not set}"
: "${RUNS_ROOT:?RUNS_ROOT is not set}" "${CACHE_ROOT:?CACHE_ROOT is not set}" "${HF_NAMESPACE:?HF_NAMESPACE is not set}"
USAGE="usage: TASK=<task> STAGE=data|train|install [G=<gpu>] [SMOKE_STEPS=<n>] bash prior/train_prior.sh [--print]"
PRINT=
case $# in
  0) ;;
  1) if [ "$1" = --print ]; then PRINT=1; else echo "$USAGE" >&2; exit 2; fi ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
TASK=${TASK:?set TASK=square_d0|square_d1|threading_d0|hammer_cleanup_d1}
STAGE=${STAGE:?set STAGE=data|train|install}
source "$ROOT/configs/tasks.sh"                 # validates TASK; sets PRIOR, PRIOR_DIR and DEMO_DATASET
case "$STAGE" in data|train|install) ;; *) echo "unknown STAGE '$STAGE' (data|train|install)" >&2; exit 2 ;; esac

case "$TASK" in
  square_d0)         RUN=mtdit_square_d0_robotstate9                           STEPS=40000 SAVE=10000 RELEASE=040000 ;;
  square_d1)         RUN=mimicgen_square_d1_mtdit_flow_60k_robotstate9         STEPS=60000 SAVE=5000  RELEASE=045000 ;;
  threading_d0)      RUN=mimicgen_threading_d0_mtdit_flow_60k_robotstate9      STEPS=60000 SAVE=5000  RELEASE=055000 ;;
  hammer_cleanup_d1) RUN=mimicgen_hammer_cleanup_d1_mtdit_flow_60k_robotstate9 STEPS=60000 SAVE=5000  RELEASE=035000 ;;
esac
DS=$DATA_ROOT/mimicgen_${TASK}_224x224_robotstate9
OUT=$RUNS_ROOT/prior_train/$RUN
JOB=$RUN
if [ -n "${SMOKE_STEPS:-}" ]; then
  case "$SMOKE_STEPS" in ''|*[!0-9]*|0*) echo "SMOKE_STEPS must be a positive integer (got '$SMOKE_STEPS')" >&2; exit 2 ;; esac
  [ "$STAGE" = train ] || { echo "SMOKE_STEPS belongs to STAGE=train (a smoke run holds no release step to install)" >&2; exit 2; }
  OUT=$RUNS_ROOT/smoke/prior_$RUN JOB=smoke_$RUN STEPS=$SMOKE_STEPS SAVE=$SMOKE_STEPS
fi

case "$STAGE" in
  data)
    CMD=(python prior/state9_dataset.py --task "$TASK" --src "$DEMO_DATASET" --out "$DS")
    if [ -n "$PRINT" ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
    cd "$CODE_ROOT"
    exec "${CMD[@]}" ;;
  install)
    CMD=(python prior/install_prior.py --task "$TASK" --run "$OUT" --step "$RELEASE" --out "$PRIOR_DIR" --dataset "$DS")
    if [ -n "$PRINT" ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
    cd "$CODE_ROOT"
    exec "${CMD[@]}" ;;
esac

# STAGE=train
if [ "$TASK" != square_d0 ]; then
  case "${WANDB_MODE:-}" in
    online|offline|disabled) ;;
    *) echo "WANDB_MODE must be online, offline or disabled (got '${WANDB_MODE:-}'; LeRobot would treat it as online)" >&2; exit 2 ;;
  esac
fi
ARGS=(--dataset.repo_id="$HF_NAMESPACE/mimicgen_${TASK}_224x224_robotstate9" --dataset.root="$DS" --output_dir="$OUT"
      --job_name="$JOB" --batch_size=64 --num_workers=8 --steps="$STEPS" --save_freq="$SAVE")
if [ "$TASK" = square_d0 ]; then
  ARGS+=(--policy.type=multi_task_dit --policy.objective=flow_matching --policy.device=cuda --policy.push_to_hub=false
         --eval.batch_size=50)
else
  ARGS+=(--log_freq=100 --policy.type=multi_task_dit --policy.objective=flow_matching --policy.device=cuda
         --policy.repo_id="$HF_NAMESPACE/$RUN" --policy.push_to_hub=false
         --wandb.enable=true --wandb.disable_artifact=true --wandb.project="$WANDB_PROJECT" --wandb.mode="$WANDB_MODE"
         --env_eval_freq=1000 --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false)
fi
LAST=$OUT/checkpoints/last/pretrained_model/train_config.json
if [ -f "$LAST" ]; then ARGS+=(--config_path="$LAST" --resume=true); fi
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$CACHE_ROOT/hf_datasets}
if [ -n "$PRINT" ]; then printf '%s\n' "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" python prior/lerobot_train.py "${ARGS[@]}"; exit 0; fi

[ -f "$DS/meta/info.json" ] || { echo "no training set at $DS: run TASK=$TASK STAGE=data bash prior/train_prior.sh first" >&2; exit 1; }
[ -d "$DS/data" ] || { echo "$DS holds meta/ only (state9_dataset.py --meta-only): not a training set; remove it and run STAGE=data" >&2; exit 1; }
WIDTH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["features"]["observation.state"]["shape"])' "$DS/meta/info.json")
[ "$WIDTH" = "[9]" ] || { echo "$DS: observation.state has shape $WIDTH, the prior is trained on [9]" >&2; exit 1; }
if [ -e "$OUT" ] && [ ! -f "$LAST" ]; then
  echo "$OUT exists but holds no checkpoints/last (the run stopped before its first checkpoint): remove it to start again" >&2; exit 1
fi
if [ -f "$LAST" ]; then
  DONE=$(basename "$(cd "$OUT/checkpoints/last" && pwd -P)")
  if [ "$DONE" = "$(printf '%06d' "$STEPS")" ]; then echo "[train_prior] $TASK: $OUT already reached step $STEPS (checkpoints/last -> $DONE); nothing to do"; exit 0; fi
  echo "[train_prior] $TASK: resuming $OUT from step $DONE"
fi
if [ -n "${G:-}" ]; then export CUDA_VISIBLE_DEVICES=$G; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$OUT")"
echo "[train_prior] $TASK train: python prior/lerobot_train.py ${ARGS[*]}"
cd "$CODE_ROOT"
exec python prior/lerobot_train.py "${ARGS[@]}"
