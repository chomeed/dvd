# Environment

What has to be installed, at which versions, to run this repository the way the reference runs were run. Everything
here was read from the reference machine on 2026-10-12 (package metadata, checkout HEADs, process arguments); what
could not be verified is listed at the end. `scripts/env.sh` is the only file that sets up paths; its header lists
every variable.

## 1. Reference machine

| | |
|---|---|
| OS | Ubuntu 24.04.3 (kernel 6.8.0, glibc 2.39), inside a Docker container with a 400 GiB memory limit and a 16-CPU quota |
| GPU | 2 x NVIDIA B200 (183 GB each), driver 580.95.05 |
| Python | 3.12.14 (conda-forge), packages from pip |
| torch | 2.11.0 built for CUDA 13.0, cuDNN 9.19.0 (the PyPI wheel) |
| ffmpeg | system 6.1.1 (`/usr/bin/ffmpeg`); torchcodec decodes the dataset videos with its libraries |

Every campaign run so far (48 DSRL, 21 DICE-RL) was launched from one code state (`reference/LIVE_COMMIT`) in this
one environment; the package set did not change between the first and the latest run.

What one run needs (measured on the reference machine while several runs shared it):

| run | GPU memory | host RAM at the end of 600k / 1M steps | wall clock, 600k steps |
|---|---|---|---|
| DICE-RL DVD | 37 GiB | 95-110 / 115-131 GiB (replay grows for the whole run; hammer is the largest) | 8-15 h (hammer 17-30 h) |
| DICE-RL Q | 13 GiB | 66-80 / 79-93 GiB | 11-18 h |
| DSRL DVD | not measured (the reference queue required 30 GB free) | 74-76 GiB incl. its two eval processes | 8.5-17 h |
| DSRL Q | not measured (same gate) | 65 / 79 GiB | 22-34 h |
| RoboMeter teacher | 11 GiB | 2.3 GiB | - |
| one best-of-16 eval (100 episodes) | 12 GiB free was the reference daemon's requirement | 10 GiB budgeted | 10-45 min, by task |

Disk, from file sizes on the reference machine (decimal GB):

| what | size |
|---|---|
| demo pixel caches, needed by every RL cell | 46.2 (square_d0), 45.9 (square_d1), 67.6 (threading_d0), 86.4 (hammer_cleanup_d1): 246 GB for the four |
| one checkpoint | DICE-RL: 0.34 (DVD) or 0.14 (Q) every 50k steps, plus 0.02 at step 0. DSRL: 0.38 (DVD: `.zip` 0.17 + `_dvd.pt` 0.22) or 0.33 (Q: 0.26 + 0.07) every 25k steps |
| one run with every checkpoint kept, 600k / 1M steps | DICE-RL DVD 4.1 / 6.9, DICE-RL Q 1.7 / 2.9, DSRL DVD 9.2 / 15.4, DSRL Q 7.9 / 13.1 |
| the campaign (84 runs) with every checkpoint kept | 669 GB: DSRL 522 (48 runs, 1408 checkpoints), DICE-RL 147 (36 runs, 528 checkpoints) |
| the campaign with only the last checkpoint of each run | 28 GB |
| published assets (section 5) | 5 GB to evaluate, 47 GB to train (38 of them the four teacher LoRAs), 57 GB to regenerate |
| DSRL demo stores; frame caches (regeneration only) | 3.4 GB; 77 GB |

So the whole campaign needs about 960 GB if nothing is deleted and about 320 GB if intermediate checkpoints are
removed as their evaluations arrive. Nothing in this repository deletes a checkpoint. A checkpoint is read again
only to evaluate it, so once `<run>/evals/<step>.json` exists and is newer than the checkpoint file, every
checkpoint but the last of its run can go (the reference machine, with an 800 GiB disk, pruned this way throughout).
This lists the candidates and deletes nothing; a DSRL `<step>.zip` goes together with its `<step>_dvd.pt`:

```bash
for dir in "$RUNS_ROOT"/*/checkpoints; do
  ls "$dir" | grep -E '^[0-9]+\.(pt|zip)$' | sort | sed '$d' | while read -r f; do
    if [ "${dir%/checkpoints}/evals/${f%.*}.json" -nt "$dir/$f" ]; then echo "$dir/$f"; fi
  done
done
```

## 2. Main environment (Python 3.12)

```bash
conda create -n dvd python=3.12.14 -y && conda activate dvd
pip install --no-deps -r requirements.txt
sudo apt-get install -y ffmpeg                # the reference has 6.1.1 (Ubuntu 24.04); any ffmpeg 6.x torchcodec can load
```

`requirements.txt` is the complete package set of the reference environment (a freeze), so it is installed with
`--no-deps`. That flag is required: lerobot 0.6.2 declares `numpy<2.3` while the reference environment runs it with
numpy 2.3.5, and pip refuses to resolve the two together. After the install, `pip check` should report exactly that one
conflict. `pip install --dry-run --no-deps` finds all 138 PyPI pins for Linux x86_64 / CPython 3.12 (checked
2026-10-12); `evdev` is the only one that builds from source and needs a C compiler.

The versions that decide numerical behaviour:

| package | version | note |
|---|---|---|
| torch / torchvision | 2.11.0 / 0.26.0 | the PyPI wheels are the CUDA 13.0 builds (`torch.__version__` is `2.11.0+cu130`); triton 3.6.0 for `torch.compile` |
| torchcodec / av | 0.11.1 / 15.1.0 | video decoding of the LeRobot datasets, with the system ffmpeg 6.1.1 |
| lerobot | 0.6.2 at commit `4aaff99be4a1d81568c08c8f0296b41b40c99ec4` | the frozen prior is its in-tree `multi_task_dit` policy; installed unmodified |
| transformers / tokenizers | 5.5.4 / 0.22.2 | DINOv2-S trunk, CLIP encoders of the prior |
| diffusers / accelerate / safetensors | 0.40.0 / 1.14.0 / 0.8.0 | |
| huggingface_hub / datasets | 1.29.0 / 4.8.5 | |
| numpy / scipy / numba | 2.3.5 / 1.18.1 / 0.67.0 | |
| mujoco / gymnasium | 3.2.7 / 1.3.0 | |
| robosuite | 1.4.1 | see section 3 |
| pandas / pyarrow / h5py | 2.3.3 / 25.0.1 / 3.16.0 | h5py is imported by robosuite |
| wandb | 0.29.0 | |

Base models read by name from the Hugging Face cache at start-up (`HF_HOME`, default `~/.cache/huggingface`), with
the snapshots the reference runs used: `facebook/dinov2-small` at `ed25f3a31f01632728cabb09d1542f84ab7b0056` (world
model and V trunk; `config.json`, `model.safetensors`) and `openai/clip-vit-base-patch16` at
`57c216476eefef5ab752ec549e440a49ae4ae5f3` (vision and text encoder named in each prior's `config.json`; the
config, tokenizer files and `pytorch_model.bin`). They get into the cache in one of two ways, both online:

```bash
bash scripts/fetch_assets.sh --tier eval --only base     # the two models, at their head; says so if the head has moved
# or: any first run without HF_OFFLINE downloads what it loads
export HF_OFFLINE=1                                       # from then on, BEFORE `source scripts/env.sh`
```

Every reference run had `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, which is what `HF_OFFLINE=1` sets: nothing is
looked up on the Hub any more and the snapshots cannot move. With that switch on and an empty cache, the first
model load fails with an offline error, so fill the cache first. Checked on the reference machine: an offline load
of the CLIP encoders reads `pytorch_model.bin` of that snapshot (the repository's head has no `model.safetensors`);
an online load may instead take the converted `model.safetensors` of the repository's pull request 17, whose 400
tensors are identical (maximum difference 0.0).

Rendering is MuJoCo through EGL on the GPU (`MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID` equal to the CUDA device;
`scripts/env.sh` sets both, and `G=<gpu>` pins both). A container that exposes the GPU without NVIDIA's EGL libraries
fails at the first render. The reference container was such a case and worked around it by extracting
`libEGL_nvidia.so.0` and its companions from the driver installer (same version as the host driver) into a
directory, adding that directory to `LD_LIBRARY_PATH`, and pointing `__EGL_VENDOR_LIBRARY_FILENAMES` at a JSON file
`{"file_format_version": "1.0.0", "ICD": {"library_path": "<dir>/libEGL_nvidia.so.0"}}` (`reference/eval_env.sh`
shows the two variables). On a host with a normal driver installation, or a container started with the graphics
capability, none of this is needed.

`PIXEL_IMG=224` is mandatory and set by `scripts/env.sh`: the pixel encoder reads it at import and defaults to 84.

## 3. Third-party checkouts

Source checkouts under `$THIRD_PARTY` (default `third_party/`, git-ignored). None of them is pip-installed;
`scripts/env.sh` puts them on `PYTHONPATH` or passes their location to the code.

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/NVlabs/mimicgen                    && git -C mimicgen checkout 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo && git -C robosuite-task-zoo checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
git clone https://github.com/ajwagen/dsrl                        && git -C dsrl checkout 8d21b9cf55459f022c0e769bca1343edd9d30e5a
git -C dsrl -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init stable-baselines3
git clone https://github.com/robometer/robometer                 && git -C robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f
cd ..
```

| checkout | commit | role | how it is found |
|---|---|---|---|
| NVlabs/mimicgen | `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb` (version 1.0.1) | the task environments (Square_D0/D1, Threading_D0, HammerCleanup_D1) | `MIMICGEN_ROOT` on `PYTHONPATH` |
| ARISE-Initiative/robosuite-task-zoo | `74eab7f88214c21ca1ae8617c2b2f8d19718a9ed` | objects of HammerCleanup | `TASK_ZOO_ROOT` on `PYTHONPATH` |
| ajwagen/dsrl | `8d21b9cf55459f022c0e769bca1343edd9d30e5a` | holds the submodule below | `DSRL_ROOT` |
| ajwagen/stable-baselines3-dsrl (submodule `stable-baselines3`) | `10e5d311b5c42fe571ff3eeb72f622ca616a149d` (version 2.6.0a1) | SAC / DSRL implementation of the DSRL arms | `$DSRL_ROOT/stable-baselines3` is put on `sys.path` by `baselines/dsrl_mimicgen.py` |
| robometer/robometer | `352d160389daa964788de1ec933d1925f3a6de4f` | the RoboMeter teacher (section 4) | `ROBOMETER_ROOT` |

- The stable-baselines3 fork is used from source and must not be pip-installed: it pins `gymnasium<1.3`. Its
  submodule URL in `dsrl/.gitmodules` is an SSH URL; the `insteadOf` rewrite above fetches it over HTTPS. The other
  submodule of `dsrl` (`dppo`) is not used.
- The reference machine's MimicGen copy is byte-identical to the pinned commit (585 files).
- **robosuite.** `requirements.txt` installs the PyPI wheel 1.4.1. The reference machine uses that same wheel
  unpacked on `PYTHONPATH`, with one line changed in `robosuite/utils/binding_utils.py` (line 175, the segmentation
  image in `MjRenderContext.read_pixels`):

  ```python
  # wheel
  seg_img = rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2**16)
  # reference machine
  seg_img = (rgb_img[:, :, 0].astype(np.int32) + rgb_img[:, :, 1].astype(np.int32) * (2**8) + rgb_img[:, :, 2].astype(np.int32) * (2**16))
  ```

  Under numpy 2 the unpatched line overflows uint8. It is reached only when a segmentation image is rendered (the
  scripts that prepare segmented RoboMeter fine-tuning data); RGB rendering, and therefore RL training and
  evaluation, does not execute it. Apply the edit to the installed package, or point `ROBOSUITE_ROOT` at a
  directory containing a patched `robosuite/`, if you render segmentations.
- `lerobot_env_mimicgen` (the gym wrapper around the MimicGen tasks) is this repository's `envs/` directory and is
  found through `PYTHONPATH`; installing it with pip is not needed.

Import check (CPU, no assets):

```bash
source scripts/env.sh
python -c "import torch, numpy, lerobot, robosuite, mimicgen, robosuite_task_zoo, lerobot_env_mimicgen
from online_rl_mimicgen import train_online
from online_rl_mimicgen.baselines import dsrl_dino
import stable_baselines3 as sb3
print(torch.__version__, numpy.__version__, lerobot.__version__, robosuite.__version__, mimicgen.__version__, sb3.__version__)"
# reference: 2.11.0+cu130 2.3.5 0.6.2 1.4.1 1.0.1 2.6.0a1
python tests/configs/test_configs.py     # every campaign cell resolves to the reference configuration
```

Every script takes its code from the checkout it lies in: `scripts/env.sh` sets `CODE_ROOT` from its own location
(a `CODE_ROOT` left in the shell by another checkout is replaced, with one warning line), and the scripts that
start a trainer stop with exit status 3 if `python` would import `online_rl_mimicgen` from anywhere else. Run
`python -m online_rl_mimicgen...` entry points from `$CODE_ROOT` or through `scripts/` (`scripts/eval_dsrl.sh`,
`scripts/build_pixel_cache.sh`), never from inside another tree that holds an `online_rl_mimicgen/` directory.

## 4. RoboMeter teacher (Python 3.10, separate environment)

Needed only for the cells that distil online (`configs/dice_rl/dvd.sh` with `DISTILL=tddistill`, which is the default
on square_d0, and `configs/dsrl/oursdist.sh`), and for relabelling rollouts when V is refitted. The teacher is a
separate process in RoboMeter's own environment; trainer and teacher exchange frames and scores through
`$SPOOL_ROOT/<task>`. `robometer/finetune/NOTES.md` is the full account of this environment and of the teachers.

```bash
cp robometer/env/uv.lock third_party/robometer/uv.lock      # the lock of the reference environment (303 packages)
(cd third_party/robometer && uv sync --frozen)               # uv 0.9.18 on the reference machine
bash scripts/robometer_teacher.sh start square_d0            # = SPOOL=square_d0 bash robometer/serve.sh; G=<gpu> pins it
bash scripts/robometer_teacher.sh check square_d0            # exit 0 once it answers
```

The upstream repository carries no lock file, so `uv sync` without the copied `uv.lock` resolves today's versions.
Key versions of the reference environment: Python 3.10.19, torch 2.8.0+cu128, torchvision 0.23.0+cu128,
transformers 4.57.6, peft 0.20.0, trl 0.20.0, unsloth 2026.9.2, unsloth_zoo 2026.9.1, bitsandbytes 0.50.2,
accelerate 1.14.0, datasets 4.1.1, huggingface_hub 0.36.2, numpy 1.26.4, xformers 0.0.32.post2, qwen-vl-utils 0.0.14,
sentence-transformers 5.7.0, decord 0.6.0, av 17.1.0. These are the lock's default resolution (no extra); the lock
also holds the resolution of upstream's `vlac` extra (transformers 4.51.3, trl 0.16.1), which was not used.

Online teachers, one per task (`robometer/teachers.sh` is the table the scripts read; LoRAs are read from
`$CKPT_ROOT/robometer/<name>/`, where `scripts/fetch_assets.py --tier train` puts them):

| task | LoRA (Hugging Face repo under `$HF_NAMESPACE`) | instruction given to the teacher |
|---|---|---|
| square_d0 | `robometer-4b-lora-square-d0-seg` | Pick up the square nut and place it on the peg. |
| square_d1 | `robometer-4b-lora-square-d1` (snapshot `8dafd9137629`), the NON-segmented LoRA | Pick up the square nut and place it on the peg. |
| threading_d0 | `robometer-4b-lora-threading-d0-seg` | Pick up the needle and thread it through the hole in the tripod. |
| hammer_cleanup_d1 | `robometer-4b-lora-hammer-cleanup-seg` | Open the drawer, place the hammer inside, and close the drawer. |

Each LoRA checkpoint is 9.4 GB (adapter, heads and a merged model). On the RoboMeter side the reference machine's
Hugging Face cache holds `robometer/Robometer-4B` (`beef63bc914c`, the model the LoRAs were trained from),
`Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec70b0`, named as base model in the LoRA configs), `unsloth/Qwen3-VL-4B-Instruct`
(`252d592b59b0`) and `sentence-transformers/all-MiniLM-L6-v2`; which of them serving alone needs was not isolated.
The square_d1 `_pv4_seg` V that DSRL loads was labelled by a different LoRA
(`robometer-4b-lora-square-d1-seg-c095`, teacher `square_d1_seg` of `robometer/serve.sh`); no campaign cell uses that
one online.

## 5. Assets: where they come from and where they go

`configs/tasks.sh` builds every path from the roots of `scripts/env.sh`; the table below is the layout it expects.
`<S>` is 150k (square_d0, square_d1), 168k (threading_d0) or 217k (hammer_cleanup_d1). The Hugging Face
repositories are under `$HF_NAMESPACE` (default `chomeed`, the account that hosts the project's assets).

### Getting them

```bash
hf auth login                                                  # once; several repositories are private (below)
bash scripts/fetch_assets.sh --tier eval --dry-run             # what would be fetched: repository, folder, revision, target, size
bash scripts/fetch_assets.sh --tier train [--task square_d0]   # download; already complete items are skipped
bash scripts/build_pixel_cache.sh square_d0                    # per task, 46-86 GB, CPU only
python tests/configs/test_configs.py                           # test_installed_assets: what a campaign cell still misses
bash scripts/smoke_train.sh dsrl q square_d0                   # a 2000-step run of one cell, before a real one
```

`scripts/fetch_assets.py` (its `--help` is the full description) knows three need levels, each including the one
before, and puts every repository or folder at exactly the path of the table, renaming where the table says so:

| tier | for | downloads (all four tasks) |
|---|---|---|
| `eval` | evaluating the base prior or a checkpoint | 5.3 GB: the two base models (into the Hugging Face cache), `meta/` of each demo dataset, the priors, both world models and the V directories per task |
| `train` | running RL cells | 41 GB more: the whole demo datasets (3.6 GB) and the four online teacher LoRAs (9.4 GB each, TD + distill cells only; `--skip teacher` leaves them out) |
| `regenerate` | refitting world models and V | 11 GB more: the prior-rollout datasets, the score tables, the offline square_d1 seg LoRA |

Each repository is fetched at the revision recorded on 2026-10-12 (`--latest` takes its head); the files installed
on the reference machine are those revisions, checked by SHA-256 for every dataset, prior, world model, V
directory and score table, and for the adapters and heads of the LoRAs. `--check [--sha256]` repeats that
comparison on any machine.

**Login.** Private as of 2026-10-12: the square_d1, threading_d0 and hammer_cleanup_d1 priors, all four
`*_dinowm_ph` world-model repositories, all four `v_dino_robometer_pred_progress_*` repositories and the two
square_d1 LoRAs. Public: the twelve datasets, the square_d0 prior, the other three LoRAs and the two base models.
Without a login a private repository answers "not found". `hf auth login` (huggingface_hub 1.x; older versions:
`huggingface-cli login`) stores a token under `HF_HOME`; `HF_TOKEN=<token>` in the environment works as well. The
account must have read access to `$HF_NAMESPACE`. The RoboMeter environment (section 4) reads the same `HF_HOME`.

### Layout

| path | content | source |
|---|---|---|
| `$DATA_ROOT/mimicgen_<task>_224x224/` | 1000 demonstrations (LeRobot v3.0, 0.65-1.4 GB); `meta/stats.json` is the state scaler, the rest feeds the expert buffer and the pixel cache | dataset repo of the same name |
| `$CKPT_ROOT/priors/square_d0_40k_rs9/checkpoints/040000/pretrained_model/` | frozen prior (1.0 GB) | `mimicgen_square_d0_mtdit_flow_40k_robotstate9` |
| `$CKPT_ROOT/priors/square_d1_45k/checkpoints/045000/pretrained_model/` | the square_d1 prior as published: un-fixed, loaded by no cell | `mimicgen_square_d1_flow_matching_prior_45k` |
| `$CKPT_ROOT/priors/square_d1_45k_statefix/checkpoints/045000/pretrained_model/` | frozen prior | **made locally** from the row above by `scripts/make_square_d1_statefix.py` (below) |
| `$CKPT_ROOT/priors/threading_d0_55k_statefix/checkpoints/055000/pretrained_model/` | frozen prior | `mimicgen_threading_d0_flow_matching_prior_55k_statefix` |
| `$CKPT_ROOT/priors/hammer_cleanup_d1_35k_statefix/checkpoints/035000/pretrained_model/` | frozen prior | `mimicgen_hammer_cleanup_d1_flow_matching_prior_35k_statefix` |
| `$CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_ph_conv_<S>/wm_dinov2s14_ph_conv_<S>.pt` (+ `.json`) | base world model (57.6 MB) | `mimicgen_<task>_224x224_dinowm_ph`, folder `ph_conv_<S>` |
| `$CKPT_ROOT/wm/mimicgen_<task>_224x224_dinowm_ph_conv_<S>_cont<S>/wm_dinov2s14_ph_conv_<S>_cont<S>.pt` (+ `.json`) | continued world model | same repo, folder `ph_conv_<S>_cont<S>` |
| `$CKPT_ROOT/value/v_dino_robometer_pred_progress_<task>_ph_conv_<S>_mqattn_pv4[_seg]/` | V: `config.json`, `state_scaler.json`, `mqattn_pool.pt`, `weights/v_td_seed{0,1,2}.pt` (13 MB) | `v_dino_robometer_pred_progress_<task>`, folder `actual_pred_ph_conv_<S>_mqattn_pv4[_seg]` (`_seg`: square_d1 and threading_d0) |
| `$CKPT_ROOT/value/v_dino_mc_return_<task>_ph_conv_<S>_mqattn_pv4/` | Monte Carlo initialised V (DSRL `oursmc`) | **unpublished**: on the reference machine only |
| `$CKPT_ROOT/robometer/<LoRA>/` | teacher LoRAs (section 4), 9.4 GB each | LoRA repos, whole repository |
| `$CACHE_ROOT/pixel_cache/pixel_cache_mimicgen_<task>_224x224_224.npy` | decoded demo frames (46-86 GB) | built locally: `bash scripts/build_pixel_cache.sh <task>` |
| `$CACHE_ROOT/dsrl_demo_store/` | DSRL's pre-encoded demonstrations (0.6-1.2 GB per task) | built by the first DSRL run of a task (about 20 min), or ahead of time with `python -m online_rl_mimicgen.baselines.build_demo_store`; the key contains the dataset path, so stores do not move between machines |
| `$SPOOL_ROOT/<task>/` | teacher request / response files | created by the teacher |
| `$RUNS_ROOT/<run name>/` | `config.json`, `metrics.jsonl`, `checkpoints/`, `evals/`; `$RUNS_ROOT/logs/`; `$RUNS_ROOT/smoke/` | written by the runs |

Inputs of the offline pipeline only (frame cache -> world model -> scores -> V; `dinowm/`, `robometer/`, `value/`).
No RL cell and no evaluator reads them:

| path | content | source |
|---|---|---|
| `$DATA_ROOT/<rollouts>_success/`, `$DATA_ROOT/<rollouts>_failure/` | the frozen prior's own rollouts, 100 successful and 100 failed episodes per task (67-243 MB each), with `<rollouts>` = `mimicgen_square_d0_224x224_mtdit_flow_40k`, `mimicgen_square_d1_224x224_mtdit_flow_45k`, `mimicgen_threading_d0_224x224_mtdit_flow_55k`, `mimicgen_hammer_cleanup_d1_224x224_mtdit_flow_35k` (`ROLLOUTS` in `dinowm/common_ws.py`) | dataset repos of the same names |
| `$CACHE_ROOT/wm_frames/<task>_frames_u8.npy`, `<task>_meta.npz`, `<task>_cache_checks.json` | those rollouts decoded once (16.9 / 17.6 / 18.8 / 24.0 GB for square_d0 / square_d1 / threading_d0 / hammer_cleanup_d1); read by the world-model trainer, the scorer and the V fit | built locally in about 30 s per task: `WS_TASK=<task> python dinowm/build_cache.py` |
| `$CKPT_ROOT/rm_scores/<task>/predictions.parquet` | RoboMeter scores of the rollouts, 12,800 rows; target of the `pv4` recipe and row set of the `mc` recipe (`value/fit_v.sh`) | `v_dino_robometer_pred_progress_<task>`, file `data/predictions.parquet` |
| `$CKPT_ROOT/rm_scores/square_d1_seg/`, `rm_scores/threading_d0_seg/` | the rescoring behind the two `pv4_seg` V directories | **unpublished**. threading_d0_seg holds the scores of `rm_scores/threading_d0` in another row order; square_d1_seg comes from the segmented LoRA. Rescore with `robometer/score_rollouts.py --rows` |
| `$CKPT_ROOT/robometer/robometer-4b-lora-square-d1-seg-c095/` | the segmented square_d1 LoRA (teacher `square_d1_seg`), offline rescoring only | LoRA repo of the same name |

Which cell loads what: DICE-RL DVD loads the continued world model on square_d0, square_d1 and threading_d0 and the
BASE one on hammer_cleanup_d1, with the plain `_pv4` V everywhere. DSRL loads the continued world model on all four
tasks, with the `_pv4` V on square_d0 and hammer_cleanup_d1 and `_pv4_seg` on square_d1 and threading_d0. The Q cells
load neither. `python tests/configs/test_configs.py` (test_installed_assets) checks all of it against the installed
files: a FAIL lists what is missing, a PASS may still list what is not built yet (pixel caches) or not fetched for
training, and `SKIP ... NO ASSETS FOUND - nothing was checked` means exactly that.

Two differences from the reference machine's layout: there the threading_d0 world models lie directly in the cache
directory instead of a per-task folder (`rl/online_rl_mimicgen/legacy.py` finds the file in either place when a
reference run is evaluated), and the spools are named `rm_spool_square`, `rm_spool` (threading_d0),
`rm_spool_square_d1` and `rm_spool_hammer`.

### The square_d1 prior

The published square_d1 checkpoint says `observation.state` shape `[26]` with `n_state_dims: 9` (a field of the
patched policy it was trained with); the in-tree LeRobot policy used here would build a 26-D network that does not
fit the weights. The prior the runs load has three files edited and the weights untouched:

```bash
python scripts/make_square_d1_statefix.py           # after fetch_assets, which also runs it by itself
python scripts/make_square_d1_statefix.py --verify $CKPT_ROOT/priors/square_d1_45k_statefix/checkpoints/045000/pretrained_model
```

It sets the state shape to `[9]` in `config.json` and `policy_preprocessor.json`, removes `n_state_dims`, cuts the
normalizer's `observation.state.*` statistics to their first 9 entries, and then checks the result: shapes, the
conditioning width stored in the weights (4370 = 256 + 2 x (2 x 768 + 9 + 512), which fits a 9-D state only) and
the SHA-256 of the three edited files against the reference machine's. Run on the published checkpoint it
reproduces the reference machine's fixed prior byte for byte (all 9 files).

### Not available off the reference machine (as of 2026-10-12)

- **Monte Carlo V directories** (4 x 13 MB, DSRL `oursmc`). `TASK=<task> RECIPE=mc bash value/fit_v.sh` refits an
  equivalent, not identical, directory from the regenerate tier. Until they are published or refitted, the three
  `oursmc` campaign cells cannot start.
- **The two `_seg` score tables** (regeneration of the `pv4_seg` V directories only; the V directories themselves
  are published).
- **Checkpoints, `config.json` and eval results of the reference runs.** `reference/results_2026-10-12.md` holds
  their numbers; none of them can be re-evaluated elsewhere until they are published.

## 6. What was not verified

- `requirements.txt` has not been installed into a clean environment. Verified: the versions equal the reference
  environment's freeze, every PyPI pin resolves (`pip install --dry-run --no-deps`), and the code imports and
  resolves all campaign configurations in the reference environment through this repository's `scripts/env.sh`.
- robosuite from the PyPI wheel instead of the reference machine's unpacked copy of the same wheel: equal except
  for the one line above (compared file by file on the reference machine), but not run from a pip installation.
- The RoboMeter environment was not rebuilt from the lock file, and the `uv sync` invocation that built the
  reference one was not recorded; compare the versions above after installing.
- No training step of this repository has been executed: the reference machine's memory was committed to the
  campaign. `scripts/smoke_train.sh` (a 2000-step run of one cell with a checkpoint and an evaluation) was checked
  by dry run only, for all 36 task x arm combinations; its first real run on a new machine is the test.
- `scripts/fetch_assets.py`: every repository, folder and file name was checked against the Hub and against the
  files installed on the reference machine, and the download path was exercised on the small JSON files of 16
  items. No weight file or dataset was downloaded through it, and the LoRA weight shards (2 x 4-5 GB each) were
  compared by name and size only. `scripts/build_pixel_cache.sh` was checked with `--print` and its refusals; the
  module it calls is the reference machine's.
- Results across machines: evaluations replay exactly on the reference machine (each episode reseeds everything
  from its env seed), but nothing has been run on a different GPU model, driver or CUDA version. The world model
  runs under bf16 autocast, DSRL enables TF32 and three DICE-RL DVD cells use `torch.compile`; no deterministic-
  algorithm switch is set anywhere. Expect statistical, not bitwise, agreement (`reference/results_2026-10-12.md`
  and `scripts/results_table.py` say what spread is normal).
