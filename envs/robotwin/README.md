# RoboTwin 2.0 environment (open_laptop, handover_mic; aloha-agilex)

Setup behind the public datasets `chomeed/robotwin2_<task>_224x224` (50 clean episodes) and
`chomeed/robotwin2_<task>_randomized500_224x224` (500 domain-randomized episodes), collection
https://huggingface.co/collections/chomeed/robotwin-6aefe84f24ee9d502298fc7f.

## 1. Checkout and conda environment

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin
git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f      # main, 2026-09-24
conda create -n robotwin python=3.10 -y && conda activate robotwin
pip install -r <dvd>/envs/robotwin/requirements.txt
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

Deviations from RoboTwin's install doc, and why:
- **torch**: the pinned 2.4.1 has no kernels for Blackwell GPUs (sm_100). 2.9.1+cu130 matches a CUDA 13 toolchain,
  which cuRobo's extensions need to build.
- **Skipped**: pytorch3d (only point-cloud FPS), `pip install -e XPolicyLab` (policy serving), azure/openai (LLM
  instruction generation), wandb.
- **Patches**: apply the doc's two source patches (sapien `urdf_loader.py` utf-8 open; mplib `planner.py` screw-plan
  collision check).
- **cuRobo** (v0.7.8, `envs/curobo`): needed to *generate new demonstrations* and to `import envs` at all
  (`envs/robot/robot.py` imports `CuroboPlanner` unconditionally). Converting the official data does not need it.
  Build it before running RoboTwin's own scripts or evaluating a policy in RoboTwin.

Reference versions: Python 3.10, torch 2.9.1+cu130, sapien 3.0.0b1, mplib 0.2.1, numpy 1.26.4, gymnasium 0.29.1.

## 2. Assets

Only these are needed for the two tasks: `assets/embodiments/aloha-agilex` (780 MB) and
`assets/objects/{015_laptop,018_microphone,cube,vis_box}` (118 MB). They can be pulled out of the HF asset zips with
HTTP range reads instead of downloading all 3.96 GB.

## 3. GPU rendering

Same rules as ManiSkill3 (`envs/maniskill3/README.md`, section 2): point `VK_ICD_FILENAMES` at the NVIDIA ICD,
put the driver user-space libs (incl. `libnvidia-rtcore`, `libnvoptix` for the ray-tracing camera shader) on
`LD_LIBRARY_PATH`, **unset `DISPLAY` / `WAYLAND_DISPLAY`**, and check the render device is the GPU, not lavapipe.

Known issue: SAPIEN's bundled OIDN 2.0.1 denoiser fails on B200 (`unsupported device type: CUDA`). RoboTwin's
ray-traced cameras (32 spp + OIDN) then render without denoising, with visible noise. The official datasets were
rendered with the denoiser, so locally rendered frames do not match them: for evaluation, use a GPU where OIDN works,
or the rasterizer shader, and report which.

## 4. Data and conversion

Official data: HF `TianxingChen/RoboTwin2.0` (MIT) @ `3dc3b798`: `aloha-agilex_clean_50.zip` and
`aloha-agilex_randomized_500.zip` per task (only 50 clean episodes exist per task). Generating more needs cuRobo.

| | definition |
|---|---|
| `observation.state` (14) | `[left joint1..6 (rad), left gripper (0 = closed .. 1 = open), right joint1..6, right gripper]`, the PD drive targets RoboTwin records |
| `action` (14) | absolute joint targets, `action[t] = state[t+1]` (last raw frame dropped) |
| cameras | `cam_high` (head), `cam_left_wrist`, `cam_right_wrist`, `cam_front` (static); D435 320x240 squashed to 224x224 (`INTER_AREA`) |
| `aux.endpose` (16) | per arm: end-effector position, quaternion (wxyz), gripper; **not a policy input** |
| fps | 15 (RoboTwin `collect_freq`) |
| task text | open_laptop: "Use one arm to open the laptop." · handover_mic: "Use one arm to grasp the microphone on the table and handover it to the other arm." (per-episode instruction variants in `meta/robotwin_episodes.jsonl`) |
| success | only on the last frame (RoboTwin keeps episodes whose final state passes `check_success()`) |

Notes for training and evaluation:
- Joint targets reach about 2.9 rad, but the RL code assumes actions in [-1, 1] and clamps them there
  (`rl/online_rl_mimicgen/common.py`, `dvd_term.py`). Rescale by fixed joint limits before RL on these tasks.
- Choose cameras explicitly when training (all four are `observation.images.*`, so LeRobot uses all four by default).
  For the world model (2 cameras), `cam_high` + the active wrist is the informative pair; handover uses both wrists.
- Build DataLoaders with `multiprocessing_context="spawn"` if the parent process decoded video first; forked workers
  inherit LeRobot's video decoder handles and fail with "Invalid data found when processing input".
- Evaluation step limits (RoboTwin `env_cfg/task_config/_eval_step_limit.yml`): 700 (open_laptop), 600
  (handover_mic). The optional `--expert_check` in RoboTwin's eval scripts adds a scripted (cuRobo) rollout per seed.
