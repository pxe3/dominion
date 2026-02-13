import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Batch:
    """A batch of rollout data. All tensors have shape (num_steps, num_envs, ...).

    Fields match what RolloutBuffer collects. For algos that don't produce
    values or log_probs (e.g. diffusion BC), those fields will be zeros.
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor

    def to(self, device):
        """Move all tensors to the given device. Returns a new Batch."""
        return Batch(
            **{k: v.to(device) if v is not None else None
               for k, v in self.__dict__.items()}
        )


class RolloutBuffer:
    """Pre-allocated numpy buffer for collecting rollout data.

    Stores transitions step-by-step via add(), then converts to a Batch
    of tensors via get(). Call clear() to reset the pointer between rollouts.
    """

    def __init__(self, num_steps, num_envs, obs_dim, act_dim):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.ptr = 0

        self.states = np.zeros((num_steps, num_envs, obs_dim))
        self.actions = np.zeros((num_steps, num_envs, act_dim))
        self.rewards = np.zeros((num_steps, num_envs))
        self.values = np.zeros((num_steps, num_envs))
        self.dones = np.zeros((num_steps, num_envs))
        self.log_probs = np.zeros((num_steps, num_envs))

    def add(self, states, actions, rewards, values, dones, log_probs):
        """Store one timestep of data across all envs."""
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values if values is not None else 0.0
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs if log_probs is not None else 0.0
        self.ptr += 1

    def get(self):
        """Convert numpy arrays to a Batch of tensors."""
        return Batch(
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs),
        )

    def clear(self):
        """Reset pointer to reuse the buffer for the next rollout."""
        self.ptr = 0
