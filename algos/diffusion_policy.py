"""Diffusion Policy for behavior cloning with DDPM training and DDIM inference.

Training (DDPM):
  1. Sample action chunk from expert data
  2. Sample random timestep t ~ Uniform(0, T)
  3. Add noise: a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1 - alpha_bar_t) * eps
  4. Predict noise: eps_pred = UNet(a_t, t, obs)
  5. Loss = MSE(eps_pred, eps)

Inference (DDIM):
  1. Start with pure noise a_T ~ N(0, I)
  2. For each step in [T, T-k, T-2k, ...]:
     eps_pred = UNet(a_t, t, obs)
     Deterministically denoise using DDIM update rule
  3. Return denoised action chunk a_0

Action chunking:
  T_o = observation horizon (how many past obs frames to condition on)
  T_p = prediction horizon (how many future actions to predict)
  T_a = action horizon (how many of those to actually execute)
  Typically T_a < T_p — predict 16 steps, execute 8, re-predict.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from algos.base import BaseAlgo
from algos.diffusion_nets import TemporalUNet
from core.registry import ALGO_REGISTRY


def cosine_beta_schedule(num_diffusion_steps, s=0.008):
    """Cosine noise schedule from Nichol & Dhariwal (2021).

    Produces a smoother noise schedule than the linear one from the
    original DDPM paper. Better for smaller models and shorter sequences.

    Args:
        num_diffusion_steps: Total number of diffusion timesteps T.
        s: Small offset to prevent beta from being too small at t=0.

    Returns:
        betas: (T,) tensor of noise amounts per timestep.
    """
    steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / num_diffusion_steps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(max=0.999).float()


@ALGO_REGISTRY.register("diffusion_policy")
class DiffusionPolicy(BaseAlgo):
    """Diffusion Policy with DDPM training and DDIM inference.

    The denoising network is a 1D Temporal U-Net conditioned on observations.
    Trains via behavior cloning (noise prediction MSE loss on expert data).
    Infers via DDIM deterministic sampling for fast action generation.
    """

    def __init__(self, obs_dim, action_dim,
                 T_p=16, T_a=8,
                 num_diffusion_steps=100, num_ddim_steps=10,
                 base_channels=128, channel_mults=(1, 2, 4), cond_dim=256,
                 lr=1e-4):
        """
        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            T_p: Prediction horizon (action chunk length to predict).
            T_a: Action horizon (how many actions to execute from chunk).
            num_diffusion_steps: Total DDPM diffusion steps T.
            num_ddim_steps: Number of DDIM sampling steps (< T for speed).
            base_channels: U-Net base channel width.
            channel_mults: U-Net channel multipliers per level.
            cond_dim: Conditioning embedding dimension.
            lr: Learning rate.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.T_p = T_p
        self.T_a = T_a
        self.num_diffusion_steps = num_diffusion_steps
        self.num_ddim_steps = num_ddim_steps

        # Build the denoising network
        self.net = TemporalUNet(
            action_dim=action_dim,
            obs_dim=obs_dim,
            T_p=T_p,
            base_channels=base_channels,
            channel_mults=channel_mults,
            cond_dim=cond_dim,
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)

        # Precompute noise schedule quantities
        betas = cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers (not parameters, but move with .to(device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # Precompute DDIM step indices (evenly spaced subset of [0, T])
        self.ddim_steps = torch.linspace(0, num_diffusion_steps - 1, num_ddim_steps).long()

    def register_buffer(self, name, tensor):
        """Store schedule tensors as attributes. They'll move with model.to(device)
        when we call algo.model.to(device) — but these live on the algo, not the model.
        We handle device transfer manually in predict() and update().
        """
        setattr(self, name, tensor)

    @property
    def model(self) -> nn.Module:
        """The U-Net denoising network — used for weight sync."""
        return self.net

    def _to_device(self, device):
        """Move schedule tensors to match the model's device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.ddim_steps = self.ddim_steps.to(device)

    def predict(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate actions via DDIM sampling conditioned on observations.

        Starts from pure noise and iteratively denoises to produce an
        action chunk. Returns the first T_a actions for execution.

        Args:
            obs: (B, obs_dim) current observation, already on device.

        Returns:
            Dict with 'action' (B, T_a, action_dim) — actions to execute.
        """
        device = obs.device
        self._to_device(device)
        B = obs.shape[0]

        with torch.no_grad():
            # Start from pure Gaussian noise
            x = torch.randn(B, self.T_p, self.action_dim, device=device)

            # DDIM deterministic denoising (iterate backwards through selected steps)
            for i in reversed(range(len(self.ddim_steps))):
                t_curr = self.ddim_steps[i]
                t_batch = t_curr.expand(B)

                # Predict noise at current timestep
                eps_pred = self.net(x, t_batch, obs)

                # Current and previous alpha_bar values
                ab_curr = self.alpha_bar[t_curr]
                ab_prev = self.alpha_bar[self.ddim_steps[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

                # DDIM update: deterministic (no noise added, eta=0)
                # x_0 prediction from current noisy sample
                x0_pred = (x - torch.sqrt(1 - ab_curr) * eps_pred) / torch.sqrt(ab_curr)
                x0_pred = x0_pred.clamp(-1, 1)  # clip for stability

                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - ab_prev) * eps_pred

                # DDIM step
                x = torch.sqrt(ab_prev) * x0_pred + dir_xt

        # Return first T_a actions from the predicted chunk
        return {"action": x[:, :self.T_a, :]}

    def update(self, batch) -> Dict[str, float]:
        """DDPM training step: predict noise on randomly noised expert actions.

        For BC, the batch comes from a dataset (not from env rollouts).
        batch.obs: (B, obs_dim) observations
        batch.actions: (B, T_p, action_dim) expert action chunks

        Args:
            batch: Must have .obs and .actions attributes.

        Returns:
            Dict with 'loss' (MSE noise prediction loss).
        """
        obs = batch.obs
        actions = batch.actions
        device = obs.device
        self._to_device(device)
        B = obs.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Forward diffusion: add noise to expert actions
        ab_t = self.alpha_bar[t][:, None, None]  # (B, 1, 1) for broadcasting
        noisy_actions = torch.sqrt(ab_t) * actions + torch.sqrt(1 - ab_t) * noise

        # Predict the noise
        noise_pred = self.net(noisy_actions, t, obs)

        # MSE loss between predicted and actual noise
        loss = nn.functional.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
