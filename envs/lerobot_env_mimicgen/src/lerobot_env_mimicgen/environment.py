"""MimicGen tasks as a LeRobot / gymnasium environment (robosuite 1.4.1, OSC_POSE, 20 Hz, two 224x224 cameras).

Observation: {"pixels": {"agentview", "eye_in_hand"} uint8 HWC, "agent_pos": float32 state}. The state is
[eef pos 3 | eef quat 4 | gripper qpos 2 | robosuite object-state]; the first 9 dims are the proprio every prior, world
model and V in this repository reads, the rest differs per task (TaskSpec.state_dim, layouts below).
Reward is robosuite's sparse success reward; an episode terminates on success and truncates at TaskSpec.horizon.

TaskSpec.instruction is what LeRobot's own evaluation would feed a language-conditioned policy. It is NOT what the
frozen prior sees in the RL code or in any reference number: FrozenPolicyPrior passes the task id (e.g. "square_d0")
as the text, and the instruction differs from the sentence the priors and RoboMeter LoRAs were trained with on three
of the four campaign tasks (training sentences: eval/eval_prior.py TRAIN_TEXT, robometer/serve.sh).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class TaskSpec:
    env_name: str
    state_dim: int
    horizon: int
    instruction: str


TASKS = {
    "coffee_d0": TaskSpec(
        env_name="Coffee_D0",
        state_dim=66,
        horizon=400,
        instruction="Insert the coffee pod into the coffee machine and close the lid.",
    ),
    "mugcleanup_d0": TaskSpec(
        env_name="MugCleanup_D0",
        state_dim=38,
        horizon=500,
        instruction="Open the drawer, place the mug upright inside, and close the drawer.",
    ),
    # 37 = 9 proprio + 2 x 14 objects (the hammer, and the cabinet whose drawer is the goal) --
    # the SAME layout and width as threading, so every state-slicing helper transfers unchanged.
    # D1, NOT D0: D1 is the wide-initialisation variant, whose drawer is sampled over
    # x=(0.0,0.2), y=(0.2,0.3) and z_rot=(-pi/6,pi/6) instead of being a fixture. Measured spread
    # over 12 resets is 181 x 90 mm, which places this task BETWEEN threading (0 mm, bolted
    # tripod) and square_d1 (397 x 398 mm) on the goal-variance axis. D1 also replaces D0's
    # hardcoded z/y success thresholds with a contact-based check.
    # Requires robosuite_task_zoo to be importable: mimicgen's hammer_cleanup.py imports
    # HammerPlaceEnv from it, and without it mimicgen swallows the ImportError and the env never
    # registers.
    "hammer_cleanup_d1": TaskSpec(
        env_name="HammerCleanup_D1",
        state_dim=37,
        horizon=500,
        instruction="Pick up the hammer and place it in the drawer, then close the drawer.",
    ),
    # state_dim = 9 proprio + object-state. Threading exposes two objects (needle, tripod), each
    # contributing pos(3) quat(4) to_eef_pos(3) to_eef_quat(4) = 14, and unlike Coffee it has no
    # trailing hinge angle: 9 + 28 = 37. Confirmed against the shipped datasets, whose
    # meta/info.json declares observation.state shape [37]. horizon 400 matches
    # mimicgen/__init__.py's ("threading_d0", 400, ...) entry.
    "threading_d0": TaskSpec(
        env_name="Threading_D0",
        state_dim=37,
        horizon=400,
        instruction="Pick up the needle and thread it through the hole in the tripod.",
    ),
    # 26 = 9 proprio + 14 (the square nut: pos 3, quat 4, to_eef_pos 3, to_eef_quat 4)
    # + 3 (the peg's position). The peg block is the giveaway: dims 23:26 have exactly ZERO
    # within-episode travel but vary across episodes, i.e. placed at reset and never moved.
    # So the manipulated object still starts at dim 9 (its position is dims 9:12).
    "square_d1": TaskSpec(
        env_name="Square_D1",
        state_dim=26,
        horizon=400,
        instruction="Pick up the square nut and place it onto the peg.",
    ),
    # Square_D0, the NARROW-initialisation sibling of square_d1. 23 = 9 proprio + 14 (the nut).
    # THREE DIMS SHORTER THAN D1 AND THAT IS THE POINT: D0's peg is a fixture at a fixed pose, so
    # it is not an observable at all, whereas D1 samples it over 397 x 398 mm and carries it in
    # dims 23:26. So square_d0 is a natively fixed-goal task. Confirmed against the shipped
    # dataset: meta/info.json declares observation.state [23].
    # Consequence for --state-repr goal_relative: there is NO goal block to subtract, so that
    # flag is meaningless here and GOAL_RELATIVE_MAPS deliberately has no square_d0 entry.
    "square_d0": TaskSpec(
        env_name="Square_D0",
        state_dim=23,
        horizon=400,
        instruction="Pick up the square nut and place it onto the peg.",
    ),
    # ThreePieceAssembly_D0. 51 = 9 proprio + THREE objects x 14 (pos 3, quat 4, to_eef_pos 3,
    # to_eef_quat 4). Object order follows the env's `objects = [self.base, self.piece_1,
    # self.piece_2]`, so dims 9:23 are the BASE (the fixture), 23:37 piece_1, 37:51 piece_2.
    # NOTE FOR ANY CODE THAT READS dims 9:12 AS "THE OBJECT": on every other task here dim 9 starts
    # the MANIPULATED object; on this one it starts the immobile base. Not a campaign task.
    # Two-stage task (seat piece_1 on the base, then piece_2 on piece_1), horizon 500 from
    # mimicgen/__init__.py's ("three_piece_assembly_d0", 500, ...) entry.
    "three_piece_assembly_d0": TaskSpec(
        env_name="ThreePieceAssembly_D0",
        state_dim=51,
        horizon=500,
        instruction="Assemble the three pieces by stacking them on the base.",
    ),
}


def normalize_task(task: str) -> TaskSpec:
    key = task.lower().replace("-", "_").replace("mug_cleanup", "mugcleanup")
    if key not in TASKS:
        raise ValueError(f"Unsupported MimicGen task {task!r}; choose one of {sorted(TASKS)}")
    return TASKS[key]


class MimicGenGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        episode_length: int,
        observation_height: int = 224,
        observation_width: int = 224,
        render_gpu_device_id: int = 0,
        worker_index: int = 0,
        reward_shaping: bool = False,
        insertion_camera: bool = False,
    ) -> None:
        super().__init__()
        self.specification = normalize_task(task)
        self.task = task
        self.task_description = self.specification.instruction
        self._max_episode_steps = int(episode_length)
        self.observation_height = int(observation_height)
        self.observation_width = int(observation_width)
        self.render_gpu_device_id = int(render_gpu_device_id)
        self.worker_index = int(worker_index)
        # robosuite's DENSE reward: a staged shaping term (reach / grasp / place) on every step,
        # instead of 1.0-on-success-only. Off by default so existing arms are unchanged.
        self.reward_shaping = bool(reward_shaping)
        # INSERTION CAMERA (opt-in). A third view aimed at the tripod bore, which neither
        # agentview (fixed third-person) nor eye_in_hand (down the gripper axis, occluded by the
        # needle tip) can see. Off by default so every existing arm renders exactly two views.
        self.insertion_camera = bool(insertion_camera)
        self._env: Any = None
        self._latest_images: dict[str, np.ndarray] | None = None
        self._elapsed_steps = 0

        self.camera_names = (("agentview", "eye_in_hand", "insertion")
                             if self.insertion_camera else ("agentview", "eye_in_hand"))
        image_space = {
            name: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for name in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_space),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.specification.state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._env is not None:
            return

        # Importing mimicgen registers its environments in robosuite.
        import mimicgen  # noqa: F401
        import robosuite
        from robosuite.controllers import load_controller_config

        self._env = robosuite.make(
            env_name=self.specification.env_name,
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            control_freq=20,
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            camera_depths=False,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_names=["agentview", "robot0_eye_in_hand"],
            render_gpu_device_id=self.render_gpu_device_id,
            reward_shaping=self.reward_shaping,
        )

    # az 260, elev 15, radius 0.15 m, solved on threading_d0 episode 0 frame 188 (closest
    # needle-tripod approach). Local to robot0_right_hand. Opt-in; no campaign cell uses it.
    INSERTION_CAM_POS = (-0.19814982671723147, -0.03568677302792253, 0.07803779959061452)
    INSERTION_CAM_QUAT = (-0.3227273186531255, 0.5177944979352156,
                          0.6117135409417261, -0.5035300185008988)

    def _render_insertion(self) -> np.ndarray:
        """The bore view, rendered by borrowing the wrist camera's pose for one frame.

        robosuite compiles its MJCF before we get the env, so a fourth camera cannot be added at
        runtime; the mount was solved the same way, by moving robot0_eye_in_hand.

        sim.forward() after EVERY pose write is load-bearing: MuJoCo renders from data.cam_xmat,
        which mj_forward derives from model.cam_quat, so writing the quaternion without a
        re-forward renders the STALE matrix -- measured as a 0.0 image diff, i.e. silently the
        wrong view. Restored afterwards so the next eye_in_hand render is unaffected.
        """
        sim = self._env.sim
        cid = sim.model.camera_name2id("robot0_eye_in_hand")
        pos0 = sim.model.cam_pos[cid].copy()
        quat0 = sim.model.cam_quat[cid].copy()
        try:
            sim.model.cam_pos[cid] = self.INSERTION_CAM_POS
            sim.model.cam_quat[cid] = self.INSERTION_CAM_QUAT
            sim.forward()
            img = sim.render(camera_name="robot0_eye_in_hand",
                             width=self.observation_width, height=self.observation_height)
        finally:
            sim.model.cam_pos[cid] = pos0
            sim.model.cam_quat[cid] = quat0
            sim.forward()
        return np.ascontiguousarray(img[::-1])

    def _format_observation(self, raw: dict[str, np.ndarray]) -> dict[str, Any]:
        images = {
            # robosuite returns OpenGL images bottom-up. The converter applied
            # this same flip before encoding the LeRobot demonstrations.
            "agentview": np.ascontiguousarray(raw["agentview_image"][::-1]),
            "eye_in_hand": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1]),
            **({"insertion": self._render_insertion()} if self.insertion_camera else {}),
        }
        state = np.concatenate(
            (
                raw["robot0_eef_pos"],
                raw["robot0_eef_quat"],
                raw["robot0_gripper_qpos"],
                raw["object-state"],
            )
        ).astype(np.float32, copy=False)
        if state.shape != (self.specification.state_dim,):
            raise RuntimeError(
                f"Unexpected {self.specification.env_name} state shape {state.shape}; "
                f"expected {(self.specification.state_dim,)}"
            )
        self._latest_images = images
        return {"pixels": images, "agent_pos": state}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_env()
        assert self._env is not None
        if seed is not None:
            # MimicGen / robosuite 1.4 placement samplers use NumPy's global RNG: this line is what
            # makes env.reset(seed=s) reproduce an episode's initial scene (eval seeds 10000+i).
            np.random.seed(int(seed) + self.worker_index)
        self._elapsed_steps = 0
        raw = self._env.reset()
        return self._format_observation(raw), {"is_success": False, "task": self.task}

    def step(self, action: np.ndarray):
        self._ensure_env()
        assert self._env is not None
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        raw, reward, _done, _info = self._env.step(action)
        self._elapsed_steps += 1
        is_success = bool(self._env._check_success())
        terminated = is_success
        truncated = self._elapsed_steps >= self._max_episode_steps and not terminated
        info = {
            "is_success": is_success,
            "task": self.task,
            "elapsed_steps": self._elapsed_steps,
        }
        return self._format_observation(raw), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._latest_images is None:
            raise RuntimeError("Call reset() before render()")
        return np.concatenate(
            (self._latest_images["agentview"], self._latest_images["eye_in_hand"]), axis=1
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
