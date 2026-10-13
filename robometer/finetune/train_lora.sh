#!/usr/bin/env bash
# Optional: fine-tune the Robometer-4B LoRA teacher of one task (step 3; the released LoRAs are assets).
#
#   bash robometer/finetune/train_lora.sh <name> <rbm_root> <out_dir> [<exp_name>]
#
#   name      RBM dataset prefix (make_rbm_from_seg.py --name-prefix), e.g. mimicgen_square_d0: reads
#             $HF_NAMESPACE/<name>_{train,test}_seg_rfm from the Hub
#   rbm_root  ROBOMETER_DATASET_PATH. For each of the two datasets it holds a directory <name>_<split>_seg_rfm that
#             CONTAINS the dataset directory of the same name again (the parquet's `frames` paths start with the name):
#               <rbm_root>/<name>_train_seg_rfm/<name>_train_seg_rfm -> <make_rbm out>/train/<name>_train_seg_rfm
#   out_dir   LoRA output; the scoring server loads <out_dir>/<exp_name>/ckpt-latest-*
#   exp_name  default <name without mimicgen_>_seg_lora. Names of the released LoRAs: threading_d0_seg_lora,
#             square_d0_seg_lora, hammer_cleanup_seg_lora (name mimicgen_hammer_cleanup), square_d1_seg_c095_lora
#
# Runs in the RoboMeter environment ($ROBOMETER_ROOT: robometer @ 352d160, python 3.10 uv venv, robometer/env/uv.lock).
# Before training: both datasets are on the Hub, and their success cutoff (0.95) is registered in
# $ROBOMETER_ROOT/robometer/data/dataset_success_cutoff.txt (checked below; robometer/finetune/NOTES.md).
# 1000 steps, r 32 / alpha 64, LR 2e-5, batch 8: about 50 minutes, under 20 GB of GPU memory, about 26 GB of RAM.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "$ROOT/scripts/env.sh" ]; then source "$ROOT/scripts/env.sh"; fi
: "${ROBOMETER_ROOT:?ROBOMETER_ROOT is not set (the robometer checkout)}" "${HF_NAMESPACE:?HF_NAMESPACE is not set}"
: "${CACHE_ROOT:?CACHE_ROOT is not set}"
NAME=$1; export ROBOMETER_DATASET_PATH=$2; OUT=$3; EXP=${4:-${NAME#mimicgen_}_seg_lora}
CUTOFF=$ROBOMETER_ROOT/robometer/data/dataset_success_cutoff.txt
for split in train test; do
  grep -q "^${NAME}_${split}_seg_rfm," "$CUTOFF" || {
    echo "success cutoff of ${NAME}_${split}_seg_rfm is not registered: append '${NAME}_${split}_seg_rfm,0.95' to $CUTOFF" >&2
    echo "(the lines used for the released LoRAs: robometer/finetune/dataset_success_cutoff.additions.txt)" >&2
    exit 1; }
done
export ROBOMETER_PROCESSED_DATASETS_PATH=${ROBOMETER_PROCESSED_DATASETS_PATH:-$CACHE_ROOT/robometer_cache}
export WANDB_MODE=${WANDB_MODE:-offline} PYTHONUNBUFFERED=1
TR=${HF_NAMESPACE}_${NAME}_train_seg_rfm_${NAME}_train_seg_rfm
EV=${HF_NAMESPACE}_${NAME}_test_seg_rfm_${NAME}_test_seg_rfm
CFG=$(mktemp --suffix .yaml)
cat > "$CFG" <<YAML
train_datasets: ["${HF_NAMESPACE}/${NAME}_train_seg_rfm"]
train_subsets: [["${NAME}_train_seg_rfm"]]
eval_datasets: ["${HF_NAMESPACE}/${NAME}_test_seg_rfm"]
eval_subsets: [["${NAME}_test_seg_rfm"]]
max_frames_for_preprocessing: 64
video_frame_sampling: "uniform"
num_proc: 1
num_threads: 8
force_reprocess: true
cache_dir: ${ROBOMETER_PROCESSED_DATASETS_PATH}
precompute_embeddings: false
embeddings_cache_dir: "embeddings"
dinov2_model: "facebook/dinov2-base"
sentence_model: "sentence-transformers/all-MiniLM-L12-v2"
embedding_batch_size: 64
YAML
cd "$ROBOMETER_ROOT"
unset PYTHONPATH          # the main environment's python 3.12 source trees must not leak into this python 3.10 one
uv run --no-sync python -m robometer.data.scripts.preprocess_datasets --config "$CFG"
uv run --no-sync python train.py \
  mode=train trainer_cls=rbm_heads \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true model.use_unsloth=false model.torch_dtype=bfloat16 \
  model.train_language_model=true model.train_progress_head=true \
  model.train_success_head=true model.train_preference_head=true \
  model.train_vision_encoder=false model.peft_vision_encoder=false \
  model.use_multi_image=true model.use_per_frame_progress_token=true \
  model.average_temporal_patches=true model.frame_pooling=mean \
  model.progress_loss_type=discrete model.progress_discrete_bins=10 \
  peft.r=32 peft.lora_alpha=64 peft.lora_dropout=0.05 peft.bias=none \
  'peft.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.exp_name="$EXP" \
  training.max_steps=1000 training.learning_rate=2e-5 \
  training.per_device_train_batch_size=8 training.per_device_eval_batch_size=8 \
  training.gradient_accumulation_steps=1 training.gradient_checkpointing=true \
  training.lr_scheduler_type=cosine training.warmup_ratio=0.1 \
  training.weight_decay=0.01 training.max_grad_norm=10.0 \
  training.max_seq_length=1024 training.bf16=true training.beta=0.1 \
  training.eval_steps=100 training.custom_eval_steps=100 training.save_steps=200 \
  training.predict_pref_progress=true training.num_gpus=1 \
  training.overwrite_output_dir=true \
  training.output_dir="$OUT" \
  "data.train_datasets=[$TR]" "data.eval_datasets=[$EV]" \
  data.dataset_type=rbm data.max_frames=16 data.max_frames_after_preprocessing=64 \
  data.min_frames_per_trajectory=5 data.dataset_preference_ratio=0.7 \
  'data.sample_type_ratio=[1.0,0.0,0.0]' \
  'data.progress_strategy_ratio=[1.0,1.0,1.0,1.0]' \
  'data.preference_strategy_ratio=[1.0,1.0,1.0,1.0]' \
  data.traj_same_source_prob=0.5 data.min_success=0.5 data.max_success=1.0 \
  data.partial_success_threshold=0.2 data.progress_pred_type=absolute_first_frame \
  data.predict_last_frame_partial_progress=false \
  'custom_eval.eval_types=[reward_alignment,policy_ranking]' \
  "custom_eval.reward_alignment=[$EV]" "custom_eval.policy_ranking=[$EV]" \
  custom_eval.policy_ranking_max_tasks=1 \
  'custom_eval.quality_preference=[]' 'custom_eval.confusion_matrix=[]' \
  loss.predict_last_frame_progress=false loss.progress_loss_type=discrete \
  loss.progress_discrete_bins=10 loss.success_positive_weight=1.0 \
  logging.save_best.keep_top_k=1
