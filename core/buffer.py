import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Batch:
    """A batch of rollout data. All tensors have shape (num_steps, num_envs, ...).

    Fields match what RolloutBuffer collects. For algos that don't produce
    values or log_probs (e.g. diffusion BC), those fields will be zeros.
    extras is an optional dict of additional tensors (e.g. denoising chains for DPPO).
    """
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    extras: dict = None

    def to(self, device):
        """Move all tensors to the given device. Returns a new Batch."""
        extras = None
        if self.extras is not None:
            extras = {k: v.to(device) for k, v in self.extras.items()}
        return Batch(
            obs=self.obs.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            values=self.values.to(device),
            dones=self.dones.to(device),
            log_probs=self.log_probs.to(device),
            extras=extras,
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

        self.obs = np.zeros((num_steps, num_envs, obs_dim))
        self.actions = np.zeros((num_steps, num_envs, act_dim))
        self.rewards = np.zeros((num_steps, num_envs))
        self.values = np.zeros((num_steps, num_envs))
        self.dones = np.zeros((num_steps, num_envs))
        self.log_probs = np.zeros((num_steps, num_envs))
        self.extras = {}  # lazily allocated on first add() with extras

    def add(self, obs, actions, rewards, values, dones, log_probs, extras=None):
        """Store one timestep of data across all envs."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values if values is not None else 0.0
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs if log_probs is not None else 0.0

        if extras is not None:
            for k, v in extras.items():
                if k not in self.extras:
                    # Lazy allocation: create array matching shape
                    self.extras[k] = np.zeros((self.num_steps, *v.shape))
                self.extras[k][self.ptr] = v

        self.ptr += 1

    def get(self):
        """Convert numpy arrays to a Batch of tensors."""
        extras = None
        if self.extras:
            extras = {k: torch.FloatTensor(v) for k, v in self.extras.items()}
        return Batch(
            torch.FloatTensor(self.obs),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs),
            extras=extras,
        )

    def clear(self):
        """Reset pointer to reuse the buffer for the next rollout."""
        self.ptr = 0
        for k in self.extras:
            self.extras[k][:] = 0.0
