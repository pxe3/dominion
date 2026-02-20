"""Diffusion Policy Policy Optimization (DPPO).

Combines diffusion-based action generation with PPO for RL training.
The K-step denoising chain is treated as a K-step MDP where each step
has a tractable Gaussian log-probability, enabling standard policy gradient.

Supports two backbones:
  - horizon=1:  DiffusionMLP for per-step actions (e.g. car env, action_dim=2)
  - horizon>1:  TemporalUNet for action chunks (e.g. PushT, horizon=16)

Key idea:
  - predict(): Run stochastic DDIM (eta > 0) to sample actions, recording
    per-step log-probs and the full denoising chain.
  - update(): Recompute log-probs under current weights using the stored chain,
    then apply PPO clipped surrogate + value loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from algos.base import BaseAlgo
from algos.diffusion_policy import cosine_beta_schedule
from algos.diffusion_nets import TemporalUNet
from algos.ppo_def import get_gae_vectorized
from core.registry import ALGO_REGISTRY
from core.buffer import Batch


class DiffusionMLP(nn.Module):
    """MLP noise predictor for per-step actions (horizon=1).

    Input: noisy_action (B, action_dim) + timestep embedding + obs embedding
    Output: predicted noise (B, action_dim)
    """

    def __init__(self, action_dim, obs_dim, hidden_dim=256, cond_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(action_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, timestep, obs):
        t_norm = timestep.float().unsqueeze(-1) / 1000.0
        cond = self.time_emb(t_norm) + self.obs_emb(obs)
        x = torch.cat([noisy_action, cond], dim=-1)
        return self.mlp(x)


class ValueMLP(nn.Module):
    """State-only value function V(obs).

    Per DPPO paper, the critic does NOT condition on denoising state.
    """

    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class DPPOModel(nn.Module):
    """Wrapper holding both noise predictor and value network.

    Single state_dict for weight sync through InferenceServer.
    Switches between DiffusionMLP (horizon=1) and TemporalUNet (horizon>1).
    """

    def __init__(self, action_dim, obs_dim, hidden_dim=256, cond_dim=128,
                 horizon=1, base_channels=128, channel_mults=(1, 2, 4)):
        super().__init__()
        self.horizon = horizon
        if horizon > 1:
            self.noise_pred = TemporalUNet(
                action_dim=action_dim,
                obs_dim=obs_dim,
                T_p=horizon,
                base_channels=base_channels,
                channel_mults=tuple(channel_mults),
                cond_dim=cond_dim,
            )
        else:
            self.noise_pred = DiffusionMLP(action_dim, obs_dim, hidden_dim, cond_dim)
        self.value_fn = ValueMLP(obs_dim, hidden_dim)


@ALGO_REGISTRY.register("dppo")
class DPPO(BaseAlgo):
    """Diffusion Policy Policy Optimization.

    Trains a diffusion-based policy via PPO. The denoising chain is treated
    as a multi-step MDP with Gaussian transitions, giving tractable log-probs
    for importance sampling.

    When horizon > 1, uses TemporalUNet to predict action chunks and returns
    only the first action for env stepping (receding horizon control).
    """

    def __init__(self, obs_dim, action_dim,
                 hidden_dim=256, cond_dim=128,
                 horizon=1, base_channels=128, channel_mults=(1, 2, 4),
                 num_diffusion_steps=100, num_denoise_steps=5,
                 eta=1.0, min_sampling_std=0.01, min_logprob_std=0.01,
                 gamma_denoising=1.0,
                 lr=3e-4, gamma=0.99, gae_disc=0.95,
                 eps_clip=0.2, grad_epochs=5):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_denoise_steps = num_denoise_steps
        self.eta = eta
        self.min_sampling_std = min_sampling_std
        self.min_logprob_std = min_logprob_std
        self.gamma_denoising = gamma_denoising
        self.gamma = gamma
        self.gae_disc = gae_disc
        self.eps_clip = eps_clip
        self.grad_epochs = grad_epochs

        # Networks
        if isinstance(channel_mults, (list,)):
            channel_mults = tuple(channel_mults)
        self._model = DPPOModel(
            action_dim, obs_dim, hidden_dim, cond_dim,
            horizon, base_channels, channel_mults,
        )
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Noise schedule
        betas = cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self._betas = betas
        self._alphas = alphas
        self._alpha_bar = alpha_bar

        # DDIM sub-step indices (evenly spaced in [0, T-1])
        self._ddim_steps = torch.linspace(0, num_diffusion_steps - 1, num_denoise_steps).long()

    @property
    def model(self) -> nn.Module:
        return self._model

    def _to_device(self, device):
        """Move schedule tensors to match model device."""
        self._betas = self._betas.to(device)
        self._alphas = self._alphas.to(device)
        self._alpha_bar = self._alpha_bar.to(device)
        self._ddim_steps = self._ddim_steps.to(device)

    def _get_ddim_sigma(self, ab_curr, ab_prev):
        """Compute DDIM sigma for stochastic sampling."""
        sigma = self.eta * torch.sqrt(
            (1 - ab_prev) / (1 - ab_curr + 1e-8) *
            (1 - ab_curr / (ab_prev + 1e-8))
        )
        return sigma

    def _noise_shape(self, B, device):
        """Return the right noise shape based on horizon."""
        if self.horizon > 1:
            return (B, self.horizon, self.action_dim)
        return (B, self.action_dim)

    def _chain_shape(self, B, K, device):
        """Return the right chain storage shape based on horizon."""
        if self.horizon > 1:
            return (B, K + 1, self.horizon, self.action_dim)
        return (B, K + 1, self.action_dim)

    def _flat_log_prob(self, x_next, mu, sigma):
        """Compute Gaussian log-prob, handling both per-step and chunk shapes.

        Flattens all non-batch dims so the same formula works for
        (B, action_dim) and (B, horizon, action_dim).
        """
        diff = (x_next - mu).reshape(x_next.shape[0], -1)  # (B, D)
        D = diff.shape[1]
        sigma_clamped = sigma.clamp(min=self.min_logprob_std)
        return -0.5 * (
            D * np.log(2 * np.pi)
            + 2 * torch.log(sigma_clamped) * D
            + (diff ** 2 / (sigma_clamped ** 2)).sum(dim=-1)
        )

    def predict(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sample actions via stochastic DDIM denoising.

        Returns the first action (for env stepping), total log_prob, value,
        and the full denoising chain for log_prob recomputation in update().

        When horizon > 1, denoises a full action chunk (B, H, action_dim)
        but only returns action[:, 0, :] for execution.
        """
        device = obs.device
        self._to_device(device)
        B = obs.shape[0]
        K = len(self._ddim_steps)

        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(*self._noise_shape(B, device), device=device)

            # Store chain
            chain = torch.zeros(*self._chain_shape(B, K, device), device=device)
            chain[:, 0] = x

            total_log_prob = torch.zeros(B, device=device)

            for i in reversed(range(K)):
                t_curr = self._ddim_steps[i]
                t_batch = t_curr.expand(B)

                # Predict noise
                eps_pred = self._model.noise_pred(x, t_batch, obs)

                # Schedule values
                ab_curr = self._alpha_bar[t_curr]
                ab_prev = self._alpha_bar[self._ddim_steps[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

                # DDIM mean: predict x0, then compute mean of x_{t-1}
                x0_pred = (x - torch.sqrt(1 - ab_curr) * eps_pred) / torch.sqrt(ab_curr)
                x0_pred = x0_pred.clamp(-1, 1)

                # DDIM sigma for stochastic sampling
                sigma = self._get_ddim_sigma(ab_curr, ab_prev)
                sigma = sigma.clamp(min=self.min_sampling_std)

                # DDIM mean
                mu = torch.sqrt(ab_prev) * x0_pred + torch.sqrt((1 - ab_prev - sigma**2).clamp(min=0)) * eps_pred

                # Sample next step
                noise = torch.randn_like(x)
                x_next = mu + sigma * noise

                # Log-prob of this step
                log_prob_step = self._flat_log_prob(x_next, mu, sigma)

                # Discount log-prob by denoising step
                total_log_prob = total_log_prob * self.gamma_denoising + log_prob_step

                x = x_next
                chain[:, K - i] = x

            value = self._model.value_fn(obs)

        # For action chunks, return only the first action for env stepping
        action = x[:, 0, :] if self.horizon > 1 else x

        return {
            "action": action,
            "log_prob": total_log_prob,
            "value": value,
            "denoising_chain": chain,
        }

    def _recompute_log_probs(self, obs, chains):
        """Recompute log-probs under current weights using stored chains.

        The chain is stored, so we deterministically recompute what mu would be
        under the *current* weights, giving us the importance ratio for PPO.
        """
        device = obs.device
        K = len(self._ddim_steps)
        N = obs.shape[0]

        total_log_prob = torch.zeros(N, device=device)

        for i in reversed(range(K)):
            t_curr = self._ddim_steps[i]
            t_batch = t_curr.expand(N)

            # Get stored a_{k} and a_{k-1}
            x = chains[:, K - 1 - i]
            x_next = chains[:, K - i]

            # Current model prediction
            eps_pred = self._model.noise_pred(x, t_batch, obs)

            # Schedule values
            ab_curr = self._alpha_bar[t_curr]
            ab_prev = self._alpha_bar[self._ddim_steps[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

            # Same DDIM computation as predict()
            x0_pred = (x - torch.sqrt(1 - ab_curr) * eps_pred) / torch.sqrt(ab_curr)
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = self._get_ddim_sigma(ab_curr, ab_prev)
            sigma = sigma.clamp(min=self.min_logprob_std)

            mu = torch.sqrt(ab_prev) * x0_pred + torch.sqrt((1 - ab_prev - sigma**2).clamp(min=0)) * eps_pred

            # Log-prob of stored x_next under new mu
            log_prob_step = self._flat_log_prob(x_next, mu, sigma)

            total_log_prob = total_log_prob * self.gamma_denoising + log_prob_step

        return total_log_prob

    def update(self, batch: Batch) -> Dict[str, float]:
        """PPO update with chain-based log_prob recomputation."""
        device = next(self._model.parameters()).device
        self._to_device(device)

        obs = batch.obs
        rewards = batch.rewards
        values = batch.values
        dones = batch.dones
        old_log_probs = batch.log_probs
        chains = batch.extras["denoising_chain"]

        # Bootstrap value for GAE
        v_bootstrap = self._model.value_fn(obs[-1]).detach() * (1 - dones[-1])

        advantages = get_gae_vectorized(rewards, values, dones, self.gamma, self.gae_disc, v_bootstrap)
        returns = (advantages + values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten (num_steps, num_envs) -> N
        num_steps, num_envs = obs.shape[:2]
        obs_flat = obs.flatten(0, 1)
        old_log_probs_flat = old_log_probs.flatten()
        advantages_flat = advantages.flatten()
        returns_flat = returns.flatten()
        chains_flat = chains.flatten(0, 1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.grad_epochs):
            # Recompute log-probs under current weights
            new_log_probs = self._recompute_log_probs(obs_flat, chains_flat)

            # PPO clipped surrogate
            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            actor_loss = -torch.min(ratio * advantages_flat, clipped_ratio * advantages_flat).mean()

            # Value loss
            new_values = self._model.value_fn(obs_flat)
            critic_loss = F.mse_loss(new_values, returns_flat)

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / self.grad_epochs,
            "critic_loss": total_critic_loss / self.grad_epochs,
            "total_loss": (total_actor_loss + total_critic_loss) / self.grad_epochs,
        }
