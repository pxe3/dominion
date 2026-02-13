import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from core.buffer import Batch
from algos.base import BaseAlgo
from core.registry import ALGO_REGISTRY
from typing import Dict


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for continuous control.

    Architecture: obs -> backbone (2-layer MLP) -> actor_head (mean)
                                                 -> critic_head (value)
    Actions are sampled from Normal(mean, exp(log_std)).
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        """Full forward pass returning (action_mean, std, value)."""
        features = self.backbone(obs)
        action_mean = self.actor_head(features)
        std = torch.exp(self.log_std)
        value = self.critic_head(features)
        return action_mean, std, value

    def get_log_prob(self, obs, action):
        """Compute log-prob of given actions under current policy. Used in PPO update."""
        mean, std, _ = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


@ALGO_REGISTRY.register("ppo")
class PPO(BaseAlgo):
    """Proximal Policy Optimization with clipped surrogate objective.

    Uses a shared ActorCritic backbone with separate actor/critic heads.
    GAE for advantage estimation, multiple gradient epochs per update.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64, lr=1e-4,
                 gamma=0.99, gae_disc=0.95, eps_clip=0.2, grad_epochs=10):
        self.ac = ActorCritic(obs_dim, action_dim, hidden_dim)
        self.ac_optim = torch.optim.Adam(self.ac.parameters(), lr)

        self.gamma = gamma
        self.gae_disc = gae_disc
        self.eps_clip = eps_clip
        self.grad_epochs = grad_epochs

    @property
    def model(self) -> nn.Module:
        """The ActorCritic module — used by InferenceServer for weight sync."""
        return self.ac

    def predict(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sample actions and compute values/log-probs for a batch of obs.

        Args:
            obs: Tensor of shape (num_envs, obs_dim), already on device.

        Returns:
            Dict with 'action', 'log_prob', 'value' tensors.
        """
        with torch.no_grad():
            features = self.ac.backbone(obs)
            mean = self.ac.actor_head(features)
            std = torch.exp(self.ac.log_std)
            value = self.ac.critic_head(features).squeeze(-1)

            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return {"action": action, "log_prob": log_prob, "value": value}

    def update(self, batch: Batch) -> Dict[str, float]:
        """Run PPO update: compute GAE, then multiple epochs of clipped surrogate + value loss.

        Args:
            batch: Rollout data with obs, actions, rewards, values, dones, log_probs.

        Returns:
            Dict with 'actor_loss', 'critic_loss', 'total_loss' (averaged over epochs).
        """
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        values = batch.values
        dones = batch.dones
        log_probs = batch.log_probs

        # Bootstrap value for GAE: V(s_last) * (1 - done_last)
        v_bootstrap = self.ac.critic_head(
            self.ac.backbone(obs[-1])
        ).squeeze().detach() * (1 - dones[-1])

        advantages = get_gae_vectorized(rewards, values, dones, self.gamma, self.gae_disc, v_bootstrap)
        returns = (advantages + values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten (num_steps, num_envs) -> (num_steps * num_envs)
        obs = obs.flatten(0, 1)
        actions = actions.flatten(0, 1)
        log_probs = log_probs.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()

        # Track metrics across gradient epochs
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.grad_epochs):
            new_values = self.ac.critic_head(self.ac.backbone(obs)).squeeze()
            critic_loss = ((returns - new_values) ** 2).mean()

            new_log_probs = self.ac.get_log_prob(obs, actions)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            full_loss = actor_loss + critic_loss

            self.ac_optim.zero_grad()
            full_loss.backward()
            self.ac_optim.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / self.grad_epochs,
            "critic_loss": total_critic_loss / self.grad_epochs,
            "total_loss": (total_actor_loss + total_critic_loss) / self.grad_epochs,
        }


def get_gae_vectorized(rewards, values, dones, gamma, gae_disc, v_bootstrap):
    """Compute Generalized Advantage Estimation (vectorized across envs).

    Args:
        rewards: (num_steps, num_envs) reward tensor.
        values: (num_steps, num_envs) value predictions.
        dones: (num_steps, num_envs) done flags.
        gamma: Discount factor.
        gae_disc: GAE lambda (discount for advantage accumulation).
        v_bootstrap: (num_envs,) bootstrap value for the last state.

    Returns:
        (num_steps, num_envs) advantage tensor.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1])
    next_val = v_bootstrap

    for t in reversed(range(num_steps)):
        td_error = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = td_error + gamma * gae_disc * gae * (1 - dones[t])
        advantages[t] = gae
        next_val = values[t]
    return advantages
