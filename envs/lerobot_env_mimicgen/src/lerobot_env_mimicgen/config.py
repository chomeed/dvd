from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .environment import MimicGenGymEnv, normalize_task


@EnvConfig.register_subclass("mimicgen")
@dataclass
class MimicGenEnvConfig(EnvConfig):
    task: str = "coffee_d0"
    fps: int = 20
    episode_length: int | None = None
    observation_height: int = 224
    observation_width: int = 224
    render_gpu_device_id: int = 0
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = normalize_task(self.task)
        if self.episode_length is None:
            self.episode_length = task.horizon

        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(task.state_dim,)),
            "pixels/agentview": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            ),
            "pixels/eye_in_hand": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            ),
        }
        self.features_map = {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/agentview": f"{OBS_IMAGES}.agentview",
            "pixels/eye_in_hand": f"{OBS_IMAGES}.eye_in_hand",
        }

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode_length": self.episode_length,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "render_gpu_device_id": self.render_gpu_device_id,
        }

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")

        def make_one(worker_index: int):
            return lambda: MimicGenGymEnv(worker_index=worker_index, **self.gym_kwargs)

        factories = [make_one(i) for i in range(n_envs)]
        if use_async_envs and n_envs > 1:
            return {self.type: {0: gym.vector.AsyncVectorEnv(factories, context="forkserver")}}

        try:
            from gymnasium.vector import AutoresetMode

            vec = gym.vector.SyncVectorEnv(factories, autoreset_mode=AutoresetMode.SAME_STEP)
        except ImportError:
            vec = gym.vector.SyncVectorEnv(factories)
        return {self.type: {0: vec}}
