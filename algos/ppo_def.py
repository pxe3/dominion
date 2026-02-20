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


def get_vtrace(rewards, values, dones, behavior_log_probs, target_log_probs,
               gamma, bootstrap_value, rho_bar=1.0, c_bar=1.0):
    """V-trace off-policy correction (Espeholt et al., 2018 — IMPALA paper).

    The problem: in async RL, the policy that generated a trajectory (behavior
    policy mu) is stale — the learner has updated the weights since. Naive
    policy gradient on stale data gives biased gradients.

    The fix: importance sampling with clipping.
      rho_t = min(rho_bar, pi(a_t|s_t) / mu(a_t|s_t))   — clips value correction
      c_t   = min(c_bar,  pi(a_t|s_t) / mu(a_t|s_t))    — clips trace propagation

    V-trace target:
      v_s = V(s) + sum_{t=s}^{T-1} gamma^{t-s} (prod_{i=s}^{t-1} c_i) * delta_t
      where delta_t = rho_t * (r_t + gamma * V(s_{t+1}) - V(s_t))

    When on-policy (pi == mu), rho=c=1 and this reduces to TD(lambda=1),
    similar to GAE with lambda=1.

    Args:
        rewards: (T, N) rewards.
        values: (T, N) value estimates V(s_t) from behavior policy.
        dones: (T, N) done flags.
        behavior_log_probs: (T, N) log pi_mu(a|s) — policy that generated the data.
        target_log_probs: (T, N) log pi(a|s) — current policy.
        gamma: Discount factor.
        bootstrap_value: (N,) V(s_T) for bootstrapping.
        rho_bar: Clipping threshold for value correction (default 1.0).
        c_bar: Clipping threshold for trace propagation (default 1.0).

    Returns:
        vs: (T, N) V-trace value targets (used for critic loss).
        advantages: (T, N) policy gradient advantages.
    """
    T, N = rewards.shape
    device = rewards.device

    # Importance sampling ratios: pi(a|s) / mu(a|s)
    ratios = torch.exp(target_log_probs - behavior_log_probs)
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)

    # Values at t+1 (shift by 1, bootstrap at the end)
    values_tp1 = torch.zeros(T, N, device=device)
    values_tp1[:-1] = values[1:]
    values_tp1[-1] = bootstrap_value

    # TD errors with importance-weighted correction
    # delta_t = rho_t * (r_t + gamma * V(s_{t+1}) * (1-d_t) - V(s_t))
    deltas = rho * (rewards + gamma * values_tp1 * (1 - dones) - values)

    # Backward recursion for V-trace targets
    # vs - V(s) = delta_t + gamma * c_t * (1-d_t) * (vs_{t+1} - V(s_{t+1}))
    vs_minus_v = torch.zeros(T, N, device=device)
    for t in reversed(range(T)):
        if t == T - 1:
            vs_minus_v[t] = deltas[t]
        else:
            vs_minus_v[t] = deltas[t] + gamma * c[t] * (1 - dones[t]) * vs_minus_v[t + 1]

    vs = (values + vs_minus_v).detach()

    # Policy gradient advantages
    # adv_t = rho_t * (r_t + gamma * v_{t+1} - V(s_t))
    vs_tp1 = torch.zeros(T, N, device=device)
    vs_tp1[:-1] = vs[1:]
    vs_tp1[-1] = bootstrap_value
    advantages = rho * (rewards + gamma * vs_tp1 * (1 - dones) - values)

    return vs, advantages
