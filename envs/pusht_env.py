"""PushT gym environment wrapper for RL training.

Wraps gym_pusht/PushT-v0 as a BaseEnv. Uses state observations
(agent pos + block keypoints) for RL, not pixels.

Auto-resets on episode end (same pattern as CarEnv).
"""

import numpy as np
import gymnasium as gym
import gym_pusht  # noqa: F401 — registers the PushT-v0 environment with gymnasium
from envs.base import BaseEnv
from core.registry import ENV_REGISTRY


@ENV_REGISTRY.register("pusht")
class PushTEnv(BaseEnv):
    """PushT environment: push a T-block to match a target pose.

    Observation: agent position (2) + block keypoints (variable).
    Action: (2,) — dx, dy of end effector.
    Reward: coverage ratio between current and target T-block pose.
    """

    def __init__(self, obs_type="state", max_steps=300, seed=None):
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type=obs_type,
            max_episode_steps=max_steps,
        )
        self._obs_shape = self.env.observation_space.shape
        self._action_shape = self.env.action_space.shape
        self._seed = seed
        self.episodic_return = 0.0
        self.episode_length = 0

    @property
    def observation_shape(self):
        return self._obs_shape

    @property
    def action_shape(self):
        return self._action_shape

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed)
        self.episodic_return = 0.0
        self.episode_length = 0
        return obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episodic_return += reward
        self.episode_length += 1

        if done:
            info["episode"] = {"r": self.episodic_return, "l": self.episode_length}
            info["terminal_obs"] = obs.astype(np.float32).copy()
            obs, _ = self.env.reset(seed=self._seed)

        return obs.astype(np.float32), float(reward), done, info
