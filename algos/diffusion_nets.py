"""1D Temporal U-Net for diffusion policy noise prediction.

Architecture follows Chi et al. (2023) "Diffusion Policy":
  - 1D convolutions over the action sequence (temporal) dimension
  - FiLM conditioning on diffusion timestep + observation embedding
  - Encoder-decoder with skip connections

Input:  noisy action sequence (B, T_p, action_dim) + timestep + obs
Output: predicted noise (B, T_p, action_dim), same shape as input
"""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep.

    Maps scalar timestep t → vector of dimension dim, using sin/cos at
    different frequencies. Same idea as transformer positional encoding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps.
        Returns:
            (B, dim) embedding.
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditionalResBlock1d(nn.Module):
    """Residual block with 1D convolution + FiLM conditioning.

    FiLM (Feature-wise Linear Modulation): the conditioning vector
    produces a scale and bias that are applied to the conv output.
    This is how the network knows which timestep it's denoising at
    and what observation it's conditioned on.

    x → Conv1d → GroupNorm → FiLM(cond) → Mish → Conv1d → GroupNorm → Mish → + x
    """

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()

        # FiLM: conditioning → scale + bias for each channel
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)

        # Residual projection if channel dims don't match
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, T) feature map.
            cond: (B, cond_dim) conditioning vector (timestep + obs).
        Returns:
            (B, C_out, T) feature map.
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM conditioning: scale and shift the normalized features
        scale, bias = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + bias[:, :, None]

        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.residual_proj(x)


class TemporalUNet(nn.Module):
    """1D Temporal U-Net for diffusion noise prediction.

    Processes action sequences with 1D convolutions over the temporal dimension.
    Conditioned on diffusion timestep and observation via FiLM.

    Architecture:
        Input (B, T_p, action_dim) → project to channels
        → Encoder: [ResBlock + Downsample] × n_layers
        → Middle: ResBlock
        → Decoder: [ResBlock + Upsample + skip connection] × n_layers
        → project back to (B, T_p, action_dim)
    """

    def __init__(self, action_dim, obs_dim, T_p,
                 base_channels=128, channel_mults=(1, 2, 4), cond_dim=256):
        """
        Args:
            action_dim: Dimension of action space (2 for PushT).
            obs_dim: Dimension of observation space (2 for PushT).
            T_p: Prediction horizon (action sequence length).
            base_channels: Base channel width for the U-Net.
            channel_mults: Channel multiplier at each encoder level.
            cond_dim: Dimension of the conditioning embedding.
        """
        super().__init__()
        self.T_p = T_p

        # Timestep embedding: scalar t → cond_dim vector
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Observation embedding: obs → cond_dim vector
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection: action_dim → base_channels
        self.input_proj = nn.Conv1d(action_dim, base_channels, 1)

        # Build encoder
        channels = [base_channels * m for m in channel_mults]
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for out_ch in channels:
            self.encoder_blocks.append(ConditionalResBlock1d(in_ch, out_ch, cond_dim))
            self.downsamples.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch

        # Middle block
        self.mid_block = ConditionalResBlock1d(channels[-1], channels[-1], cond_dim)

        # Build decoder (reverse order, with skip connections doubling input channels)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            skip_ch = out_ch  # skip connection from encoder
            decoder_in = (channels[-1] if i == 0 else channels[len(channels) - i]) + skip_ch
            self.upsamples.append(nn.ConvTranspose1d(decoder_in, decoder_in, kernel_size=4, stride=2, padding=1))
            self.decoder_blocks.append(ConditionalResBlock1d(decoder_in, out_ch, cond_dim))

        # Output projection: base_channels → action_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], 3, padding=1),
            nn.Mish(),
            nn.Conv1d(channels[0], action_dim, 1),
        )

    def forward(self, noisy_actions, timestep, obs):
        """Predict noise given noisy actions, diffusion timestep, and observation.

        Args:
            noisy_actions: (B, T_p, action_dim) noisy action sequence.
            timestep: (B,) diffusion timestep integers.
            obs: (B, obs_dim) current observation.

        Returns:
            (B, T_p, action_dim) predicted noise.
        """
        # Conditioning: combine timestep and observation embeddings
        cond = self.time_emb(timestep) + self.obs_emb(obs)

        # (B, T_p, action_dim) → (B, action_dim, T_p) for Conv1d
        x = noisy_actions.permute(0, 2, 1)
        x = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for res_block, downsample in zip(self.encoder_blocks, self.downsamples):
            x = res_block(x, cond)
            skips.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block(x, cond)

        # Decoder with skip connections
        for res_block, upsample, skip in zip(self.decoder_blocks, self.upsamples, reversed(skips)):
            x = torch.cat([x, skip[:, :, :x.shape[-1]]], dim=1)  # handle size mismatch from stride
            x = upsample(x)
            x = x[:, :, :skip.shape[-1]]  # trim to match skip spatial size
            x = res_block(x, cond)

        # (B, base_channels, T_p) → (B, T_p, action_dim)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        return x
