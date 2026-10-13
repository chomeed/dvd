# RoboMeter teachers: environment and fine-tune notes

The five LoRA teachers of the campaign are assets (`robometer/teachers.sh` lists them). Serving them needs only the
environment below and `robometer/serve.sh`. Fine-tuning one again is optional; this file records how the released
ones were made and what on the reference machine was never committed anywhere else.

## Environment (separate from the main one)

RoboMeter runs in its own Python 3.10 environment; the RL code never imports it and it never imports the RL code.

```bash
git clone https://github.com/robometer/robometer $ROBOMETER_ROOT
git -C $ROBOMETER_ROOT checkout 352d160389daa964788de1ec933d1925f3a6de4f
cp robometer/env/uv.lock $ROBOMETER_ROOT/uv.lock      # upstream ships no lock file; this is the reference machine's
cd $ROBOMETER_ROOT && uv sync --frozen
```

`uv sync` is upstream's install command; without the lock it picks whatever is current. The reference environment
holds python 3.10.19, torch 2.8.0+cu128, transformers 4.57.6, peft 0.20.0, unsloth 2026.9.2, accelerate 1.14.0,
datasets 4.1.1 and numpy 1.26.4: compare after installing (`uv run --no-sync python -c "import torch, transformers,
peft; print(torch.__version__, transformers.__version__, peft.__version__)"`). The lock also carries the alternative
resolution of upstream's mutually exclusive `vlac` extra (older transformers and peft), which is not what was used; the
exact `uv sync` invocation of the reference machine was not recorded. Models that must be in the Hugging Face cache:
`robometer/Robometer-4B` (the LoRA's initialisation), `Qwen/Qwen3-VL-4B-Instruct`, `unsloth/Qwen3-VL-4B-Instruct`
and `sentence-transformers/all-MiniLM-L6-v2`. Every command is run as `uv run --no-sync ...` from `$ROBOMETER_ROOT`.

## What the scoring server returns (robometer_server.py)

- The progress head was trained on successful trajectories only: the collator masks the progress loss on failures.
  Progress is therefore a time ramp learned from successes and does not by itself separate a timeout from a success.
- The success head was trained on failures too (label 0 on every failure frame) and is the head that carries outcome
  information. The server returns both, shape `(2, T)`.
- `is_discrete_mode` must match training (progress is discretised into 10 bins); the server reads it from the
  checkpoint's config. With the wrong mode the model returns raw per-bin logits without an error.
- `--task` must be the sentence the LoRA was fine-tuned with (`TEACHER_TEXT` in `teachers.sh`).

## Fine-tuning a teacher

1. **Cut the failures at the outcome.** `segment_square.py` (square_d0) and `segment_hammer.py` (hammer_cleanup_d1)
   read a failure-rollout dataset and write a LeRobot dataset of clips that end where the attempt failed; their
   docstrings give the rule. Successes are not segmented. No segmenter for threading_d0 or square_d1 is in this
   repository: their segmented training sets exist only as the RBM datasets of step 2.
2. **Convert to RBM format.** `make_rbm_from_seg.py --out <dir> --specs <segmented failures>,<success rollouts>,<demos>
   --task "<sentence>" --name-prefix mimicgen_<task> --lang-json <embedding json>` writes
   `<dir>/{train,test}/mimicgen_<task>_{train,test}_seg_rfm/`. The released sets hold 160 training trajectories
   (80 failures, 80 successes of which half are prior rollouts and half expert demos) and 40 test trajectories,
   64 agentview frames each. A dataset is labelled a failure when its path contains `failure`.
   Name prefixes used: `mimicgen_threading_d0`, `mimicgen_square_d0`, `mimicgen_hammer_cleanup` (no `_d1`),
   `mimicgen_square_d1`.
3. **Push both sets to the Hub** as `$HF_NAMESPACE/mimicgen_<task>_{train,test}_seg_rfm`. This is required:
   RoboMeter's preprocessing loads them with `load_dataset(repo, name=subset)`. The test set's parquet file must be
   named `train-*.parquet` as well, because the Hub split of both repositories is `train`.
4. **Register the success cutoff.** Append one line per dataset, `<dataset name>,0.95`, to
   `$ROBOMETER_ROOT/robometer/data/dataset_success_cutoff.txt`. The lines used for the released LoRAs are in
   `dataset_success_cutoff.additions.txt`; `train_lora.sh` refuses to start without them. This edit is the only
   change to tracked RoboMeter files on the reference machine.
5. **Lay out the local copy** that `ROBOMETER_DATASET_PATH` points at. Each dataset name appears twice in the path,
   because the parquet's `frames` entries already start with the name:
   `<rbm_root>/<name>/<name>` is a link to `<make_rbm out>/<split>/<name>`.
6. **Train.** `bash robometer/finetune/train_lora.sh mimicgen_<task> <rbm_root> <out_dir> [<exp_name>]` runs the
   preprocessing and then 1000 steps (rank 32, alpha 64, LR 2e-5, batch 8, cosine schedule, bf16), the same
   overrides for every task. About 50 minutes, under 20 GB of GPU memory, about 26 GB of RAM. Each checkpoint is
   9.4 GB (adapter, heads and a merged copy of the 4B model); `logging.save_best.keep_top_k=1` keeps one "best"
   checkpoint instead of five. The server loads `<out_dir>/<exp_name>/ckpt-latest-*`.

## What is and is not recorded

- The `train.py` overrides in `train_lora.sh` are the ones of the four per-task launchers of the reference machine,
  which differed only in dataset names, experiment name and output directory. The first three launchers did not set
  `keep_top_k`.
- Experiment names of the released LoRAs: `threading_d0_seg_lora`, `square_d0_seg_lora`, `hammer_cleanup_seg_lora`,
  `square_d1_seg_c095_lora`.
- The non-segmented square_d1 teacher (`robometer-4b-lora-square-d1`) was not produced by this pipeline on the
  reference machine; it is used as downloaded. The segmented square_d1 LoRA was trained from a different rollout set
  than the one the world model and V use, and its segmentation code is not available.
- The exact argument values passed to `segment_*.py` and `make_rbm_from_seg.py` for the released sets were not logged.
  The defaults of the segmenters (`--ref`, `--near-mm`, ...) are the values in the files as they were last run.
- Fine-tuning is not bit-reproducible; a retrained LoRA gives different scores, hence different V targets.
