from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class BaseAlgo(ABC):
    """Base class for all RL algorithms.

    Every algorithm must implement three things:
    - predict(): inference-time forward pass (used by InferenceServer)
    - update(): training step on a batch (used by Learner)
    - model: the nn.Module whose weights get synced between processes
    """

    @abstractmethod
    def predict(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference on a batch of observations.

        Called by InferenceServer with tensors already on device.
        Must return a dict containing at least 'action'. Other keys
        are algo-specific (e.g. PPO returns 'log_prob', 'value').

        Args:
            obs: Observation tensor of shape (num_envs, obs_dim).

        Returns:
            Dict with 'action' key and any algo-specific extras.
        """
        pass

    @abstractmethod
    def update(self, batch) -> Dict[str, float]:
        """Run a training update on a batch of collected experience.

        Called by Learner after receiving a rollout from the Worker.

        Args:
            batch: A Batch dataclass containing rollout data.

        Returns:
            Dict of metric name -> value for logging.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """The nn.Module used for inference.

        InferenceServer uses this to load/save weights via
        state_dict() and load_state_dict(). Must return the same
        module that predict() uses internally.
        """
        pass
