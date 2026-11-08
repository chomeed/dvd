# ManiSkill3 environment (LiftPegUpright-v1, PegInsertionSide-v1)

Setup that produced the public datasets `chomeed/maniskill3_lift_peg_upright_224x224_robotstate9` (1,200 episodes) and
`chomeed/maniskill3_peg_insertion_side_224x224_robotstate9` (1,049 episodes), collection
https://huggingface.co/collections/chomeed/maniskill3-6aefe84ee68b1425f0bab0e1. Use the same settings for prior
evaluation and rollouts, or the prior sees different images than it was trained on.

## 1. Conda environment (separate from the main `dvd` env)

ManiSkill3 pins its own sapien / torch; never install it into the main environment.

```bash
conda create -n maniskill3 python=3.11 -y && conda activate maniskill3
pip install mani_skill==3.0.1 torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import mani_skill, sapien; print(mani_skill.__version__, sapien.__version__)"   # 3.0.1 3.0.3
```

Reference versions: Python 3.11, mani_skill 3.0.1, sapien 3.0.3, torch 2.8.0+cu128, numpy 1.26.4, gymnasium 1.4.0.
The dataset converter itself runs in the main environment (LeRobot 0.6.2); the two talk through files (`.npz` per episode).

## 2. GPU rendering (Vulkan)

SAPIEN renders cameras through Vulkan. In containers the NVIDIA user-space driver libraries are often not in the
system library path. Source `env.sh.example` (edit the paths) in every shell that runs ManiSkill:

- `VK_ICD_FILENAMES` must point at an ICD json whose `library_path` is the driver's `libGLX_nvidia.so.0`.
- `LD_LIBRARY_PATH` must contain the driver user-space libs (incl. `libnvidia-rtcore`, `libnvoptix` for ray tracing).
- **Unset `DISPLAY`** if it points at an unreachable X server: the NVIDIA ICD tries to connect and hangs silently.
- **Check the device is the GPU**: a software Vulkan driver (lavapipe) may also be registered and renders on the CPU
  without any error. `sapien.render.get_device_summary()` (or the first render's speed) tells which one is used.
- Keep `MS_ASSET_DIR` and demo downloads on a real disk, not a RAM-backed `/tmp`.

Smoke test: create each env with `obs_mode="rgb"` and save one frame per camera; expect two 224x224 images.

## 3. Demonstrations

```bash
python -m mani_skill.utils.download_demo PegInsertionSide-v1 -o $DEMO_DIR    # motion-planning + RL demos
python -m mani_skill.utils.download_demo LiftPegUpright-v1 -o $DEMO_DIR      # RL demos only (no motion-planning set)
```

Source: `haosulab/ManiSkill_Demonstrations` (Apache-2.0). The demo files store no images (`obs_mode none`): every
episode is re-rendered on replay with the cameras of section 4, and only episodes that end in success are kept.
- PegInsertionSide-v1 (motion planning, recorded on `physx_cpu`): converted to the `pd_ee_delta_pose` controller with
  `mani_skill.trajectory.replay_trajectory -c pd_ee_delta_pose`, then replayed open loop (actions only).
- LiftPegUpright-v1 (RL, recorded on `physx_cuda`): does not reproduce open loop on the CPU backend, so each step's
  recorded env state is written back after the action (as `replay_trajectory --use-env-states`) and success is
  re-evaluated on that state. A policy evaluated open loop in this env therefore faces slightly different physics
  than the demos.
- Recorded states name the robot after the uid used at recording (`panda` / `panda_wristcam`); rename that key to
  the replay env's robot before `set_state_dict`.

## 4. Cameras (must match in eval)

Two 224x224 cameras, named like the MimicGen datasets: `observation.images.agentview` and
`observation.images.eye_in_hand`, `shader_pack="default"`.

**agentview** replaces the task's `base_camera` pose and fov (it is neither the task's `base_camera` nor its
`render_camera`):

```python
from mani_skill.utils.sapien_utils import look_at
import numpy as np
AGENTVIEW = {  # eye, target in world frame (m); fov in radians
    "LiftPegUpright-v1":   dict(eye=[0.40, -0.35, 0.45], target=[-0.05, 0.00, 0.15], fov=np.deg2rad(80.0)),
    "PegInsertionSide-v1": dict(eye=[0.45,  0.00, 0.45], target=[-0.05, 0.05, 0.10], fov=np.deg2rad(80.0)),
}
# resolved poses [x, y, z, qw, qx, qy, qz]:
#   LiftPegUpright-v1   [0.40, -0.35, 0.45, 0.315063, -0.226864, 0.077839, 0.918264]
#   PegInsertionSide-v1 [0.45,  0.00, 0.45, 0.047527, -0.299154, 0.014920, 0.952904]
cam = AGENTVIEW[env_id]
sensor_configs = dict(
    shader_pack="default",
    base_camera=dict(width=224, height=224, pose=look_at(cam["eye"], cam["target"]), fov=cam["fov"]),
    hand_camera=dict(width=224, height=224),
)
env = gym.make(env_id, obs_mode="rgb", control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
               render_backend="gpu", robot_uids=ROBOT, sensor_configs=sensor_configs, num_envs=1)
```

**eye_in_hand** (`hand_camera`) is the Panda wrist mount, fov 90 deg:
- PegInsertionSide-v1: `ROBOT = "panda_wristcam"` (the task's only robot; the demos were recorded with it).
- LiftPegUpright-v1: the demos were recorded with plain `"panda"`. `panda_wristcam` adds a 0.564 kg camera body with
  collision, which changes the dynamics on replay, so the datasets keep `"panda"` and add a weightless, collision-free
  sensor camera on `panda_hand` at exactly the pose of `panda_wristcam`'s `camera_link` (same image geometry).

## 5. State, action, text

| | definition |
|---|---|
| `observation.state` (9) | gripper-tip (`panda_hand_tcp`) position (3, world frame) + quaternion (4, **xyzw**, sign fixed to the task's rest orientation hemisphere) + two finger joint positions (2, each 0..0.04 m) |
| `action` (7) | `pd_ee_delta_pose`: dpos (3), drot (3), gripper (1), normalised to [-1, 1] (the controller clips) |
| `aux.object`, `aux.sim_state`, `aux.robot0_*` | object pose / simulator state / extra robot fields: analysis and resets only, **never policy inputs** |
| fps | 20 (control frequency; simulation 100 Hz) |
| task text | LiftPegUpright: "Pick up the peg lying on the table and stand it upright." · PegInsertionSide: "Pick up the peg and insert its orange end into the hole in the box." |

Feed the prior exactly the dataset's task sentence at evaluation and in RL (check it at the tokenizer).

## 6. Episode limits

The default limits are short for an 8-step-chunk policy (LiftPegUpright 50 steps, PegInsertionSide 100). Set
`max_episode_steps` explicitly for evaluation (about 2x the mean demo length) and report it with the results.
