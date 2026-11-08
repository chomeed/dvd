# MimicGen environment (square_d0, square_d1, threading_d0, hammer_cleanup_d1)

`lerobot_env_mimicgen` is the gym / LeRobot wrapper every campaign run, evaluation and rollout uses
(`src/lerobot_env_mimicgen/environment.py`). Unlike ManiSkill3 and RoboTwin (`envs/maniskill3/`, `envs/robotwin/`),
MimicGen runs in the **main** environment: there is no separate conda env. This page is the task-side summary;
`environment.md` sections 1 to 3 are authoritative for versions and installation.

## 1. Install (main environment)

```bash
pip install --no-deps -r requirements.txt          # includes robosuite 1.4.1 (PyPI wheel), mujoco 3.2.7
mkdir -p third_party && cd third_party
git clone https://github.com/NVlabs/mimicgen                    && git -C mimicgen checkout 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo && git -C robosuite-task-zoo checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
cd .. && source scripts/env.sh                     # PYTHONPATH (mimicgen, task zoo, this wrapper), MUJOCO_GL=egl, asset roots
```

- Nothing here is pip-installed: `scripts/env.sh` puts `MIMICGEN_ROOT`, `TASK_ZOO_ROOT` and `envs/lerobot_env_mimicgen/src`
  on `PYTHONPATH`. **Source it in every shell**; without it imports fail.
- **robosuite-task-zoo is required for hammer_cleanup_d1.** MimicGen's `hammer_cleanup.py` imports from it, and if the
  import fails MimicGen swallows the `ImportError`: `HammerCleanup_D1` is then simply not registered.
- robosuite: one line of `robosuite/utils/binding_utils.py` overflows under numpy 2 when a **segmentation** image is
  rendered (the patch is in `environment.md` section 3). RGB rendering, so RL and evaluation, never reaches it.

## 2. Rendering (MuJoCo EGL)

`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `MUJOCO_EGL_DEVICE_ID` equal to the CUDA device. `G=<gpu>` before
`source scripts/env.sh` sets `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` together. A container without NVIDIA's
EGL libraries fails at the first render; the workaround (driver-matched `libEGL_nvidia.so.0`, `LD_LIBRARY_PATH`,
`__EGL_VENDOR_LIBRARY_FILENAMES`) is in `environment.md` section 2.

Smoke test:

```bash
python -c "
from lerobot_env_mimicgen.environment import MimicGenGymEnv
e = MimicGenGymEnv(task='square_d0', episode_length=400); o, _ = e.reset(seed=10000)
print({k: v.shape for k, v in o['pixels'].items()}, o['agent_pos'].shape)"   # two (224, 224, 3) images, (23,)
```

## 3. Simulator settings (fixed in the wrapper)

| setting | value |
|---|---|
| robot / controller | Panda, `OSC_POSE` (robosuite default config), `control_freq=20` |
| action (7) | `[dpos (3), drot axis-angle (3), gripper (1)]` in [-1, 1]; the controller maps ±1 to ±0.05 m and ±0.5 rad per step (delta control); gripper -1 open, +1 close |
| cameras | `agentview` and `robot0_eye_in_hand` at 224x224, flipped vertically (robosuite renders bottom-up; the datasets were flipped the same way) |
| reward | robosuite's sparse success reward (`reward_shaping=False`) |
| episode end | `terminated` on `_check_success()`; `truncated` at the horizon |
| reset seed | `reset(seed=s)` seeds NumPy's **global** RNG with `s + worker_index`: MimicGen's placement samplers use it. Evaluation seeds are 10000 to 10099 everywhere |

| task | robosuite env | state dim | horizon | object block after the 9 proprio dims |
|---|---|---|---|---|
| square_d0 | `Square_D0` | 23 | 400 | nut (14); the peg is a fixture, not observed |
| square_d1 | `Square_D1` | 26 | 400 | nut (14) + peg position (3) |
| threading_d0 | `Threading_D0` | 37 | 400 | needle (14) + tripod (14) |
| hammer_cleanup_d1 | `HammerCleanup_D1` | 37 | 500 | hammer (14) + cabinet (14) |

Each object contributes pos (3), quat (4), position and quaternion relative to the gripper (3 + 4).

## 4. Observation: only the first 9 state dims are policy input

`agent_pos` is `[eef pos (3), eef quat (4, xyzw as robosuite), gripper finger qpos (2: about +0.02 / -0.02 m open, 0 closed; the second finger is negative), object-state]`. The
priors, world models and value functions read **only the first 9 dims**; the object block is privileged simulator
state. New policies take robot state and cameras only. Assert a prior's `input_features` declare a 9-D
`observation.state`; the three published "statefix" priors needed a config fix for exactly this (`AGENTS.md` section 5.3).

## 5. Datasets

`chomeed/mimicgen_<task>_224x224` on the Hub (LeRobot v3.0, CC-BY-4.0, 1000 demonstrations each, 20 fps). Built from
NVIDIA's released `core/<task>.hdf5` (`amandlek/mimicgen_datasets`): only the cameras were recomputed, by writing
each frame's recorded simulator state back into the live env and re-rendering at 224x224 (the source images are
84x84). All other columns are copied from the hdf5; nothing was re-simulated.

- `observation.state` is the full 23 to 37-D vector of section 4, and `observation.object`, `observation.sim_state` and
  `observation.robot0_*` are extra columns in these older datasets (newer tasks name them `aux.*`). Train priors on
  the 9-D copy: `TASK=<task> STAGE=data bash prior/train_prior.sh` writes `mimicgen_<task>_224x224_robotstate9`.
- `success == done == (reward >= 1)`, on the last frame.

## 6. Task text

| task | sentence in the dataset (prior and RoboMeter training) | `TaskSpec.instruction` (wrapper) |
|---|---|---|
| square_d0, square_d1 | Pick up the square nut and place it on the peg. | ... place it **onto** the peg. |
| threading_d0 | Pick up the needle and thread it through the hole in the tripod. | same |
| hammer_cleanup_d1 | Open the drawer, place the hammer inside, and close the drawer. | Pick up the hammer and place it in the drawer, then close the drawer. |

**Kept on purpose:** every reference RL run and evaluation feeds the prior the **task id** (`"square_d0"`), not a
sentence, and repeats the current frame as the 2-frame history (`AGENTS.md` section 8; measured +4.25 points for the
sentence, -1 for real history, neither significant). Keep both for numbers comparable with the reference; for any new
setup, feed the dataset's sentence and check it at the tokenizer. RoboMeter teachers always get the dataset sentence.

## 7. Pitfalls

- `WS_TASK` must be set for the world-model / value scripts under `dinowm/`; they refuse a missing or wrong value
  (the research code silently defaulted to threading).
- The opt-in `insertion_camera=True` (a third, bore-facing view for threading) is used by no campaign cell; the
  campaign renders exactly two views.
- Timeouts are stored as `done=1` in the RL replay; with γ=0.99 per primitive step and sparse {0, 1} reward every value
  lies in [0, 1], so a negative mean V or Q indicates a bug.
