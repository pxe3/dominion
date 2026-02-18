"""Behavior Cloning trainer for diffusion policy on PushT.

Trains a DiffusionPolicy via supervised learning on expert demonstrations.
Supports configurable training optimizations:
  - Mixed precision (bf16/fp16 via torch.autocast)
  - Gradient accumulation (effective batch = batch_size * accum_steps)
  - Activation checkpointing (recompute activations in backward pass)

Logs to TensorBoard: loss curves, learning rate, throughput, GPU memory.

Usage:
  python runners/train_bc.py                           # baseline (fp32)
  python runners/train_bc.py mixed_precision=true       # bf16
  python runners/train_bc.py activation_checkpointing=true
  python runners/train_bc.py grad_accum_steps=4
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import time
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint

from data.pusht_dataset import PushTDataset
from algos.diffusion_policy import DiffusionPolicy
from core.registry import ALGO_REGISTRY, auto_register

auto_register("algos")


@dataclass
class BCBatch:
    """Minimal batch for BC training. Just obs and action chunks."""
    obs: torch.Tensor
    actions: torch.Tensor


def collate_bc(samples):
    """Custom collate for BC dataset → BCBatch."""
    return BCBatch(
        obs=torch.stack([s["obs"] for s in samples]),
        actions=torch.stack([s["actions"] for s in samples]),
    )


@hydra.main(config_path="../configs", config_name="config_bc", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    print("Loading PushT dataset...")
    dataset = PushTDataset(T_o=cfg.T_o, T_p=cfg.T_p)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_bc,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Algo
    algo = DiffusionPolicy(
        obs_dim=2,
        action_dim=2,
        T_p=cfg.T_p,
        T_a=cfg.T_a,
        num_diffusion_steps=cfg.num_diffusion_steps,
        num_ddim_steps=cfg.num_ddim_steps,
        base_channels=cfg.base_channels,
        channel_mults=tuple(cfg.channel_mults),
        cond_dim=cfg.cond_dim,
        lr=cfg.lr,
    )
    algo.model.to(device)

    # Activation checkpointing: wrap U-Net encoder/decoder blocks
    if cfg.activation_checkpointing:
        _enable_activation_checkpointing(algo.model)
        print("Activation checkpointing: ENABLED")

    # Count parameters
    num_params = sum(p.numel() for p in algo.model.parameters())
    print(f"Model parameters: {num_params:,}")

    # TensorBoard
    run_name = _build_run_name(cfg)
    log_dir = os.path.join(cfg.log_dir, run_name)
    writer = SummaryWriter(log_dir)
    writer.add_text("config", f"```\n{OmegaConf.to_yaml(cfg)}\n```")
    print(f"TensorBoard: {log_dir}")

    # Training loop
    global_step = 0
    best_loss = float("inf")
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.precision == "fp16")

    # Determine autocast dtype
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    if use_amp:
        print(f"Mixed precision: ENABLED ({cfg.precision})")
    if cfg.grad_accum_steps > 1:
        print(f"Gradient accumulation: {cfg.grad_accum_steps} steps (effective batch: {cfg.batch_size * cfg.grad_accum_steps})")

    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    epoch_start = time.time()

    for epoch in range(cfg.num_epochs):
        epoch_losses = []
        algo.model.train()

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = BCBatch(
                obs=batch.obs.to(device),
                actions=batch.actions.to(device),
            )

            # Forward + backward with optional mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Manually compute loss for autocast compatibility
                obs = batch.obs
                actions = batch.actions
                B = obs.shape[0]
                t = torch.randint(0, algo.num_diffusion_steps, (B,), device=device)
                noise = torch.randn_like(actions)
                algo._to_device(device)
                ab_t = algo.alpha_bar[t][:, None, None]
                noisy_actions = torch.sqrt(ab_t) * actions + torch.sqrt(1 - ab_t) * noise
                noise_pred = algo.net(noisy_actions, t, obs)
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Scale for gradient accumulation
                loss = loss / cfg.grad_accum_steps

            # Backward
            if use_amp and cfg.precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (every grad_accum_steps)
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                if use_amp and cfg.precision == "fp16":
                    scaler.step(algo.optimizer)
                    scaler.update()
                else:
                    algo.optimizer.step()
                algo.optimizer.zero_grad()
                global_step += 1

            actual_loss = loss.item() * cfg.grad_accum_steps
            epoch_losses.append(actual_loss)

            # Log per-step metrics
            if global_step % cfg.log_interval == 0 and global_step > 0:
                writer.add_scalar("train/loss", actual_loss, global_step)
                if device.type == "cuda":
                    mem_mb = torch.cuda.max_memory_allocated() / 1e6
                    writer.add_scalar("system/gpu_memory_mb", mem_mb, global_step)

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - epoch_start
        throughput = len(dataset) / elapsed

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("system/throughput_samples_sec", throughput, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(log_dir, "best_model.pt")
            torch.save(algo.model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {avg_loss:.6f} | "
              f"Best: {best_loss:.6f} | Throughput: {throughput:.0f} samples/s | "
              f"Time: {elapsed:.1f}s")
        epoch_start = time.time()

    # Save final model
    final_path = os.path.join(log_dir, "final_model.pt")
    torch.save(algo.model.state_dict(), final_path)
    print(f"\nTraining complete. Model saved to {final_path}")

    writer.close()


def _enable_activation_checkpointing(model):
    """Wrap encoder and decoder blocks with activation checkpointing.

    torch.utils.checkpoint recomputes forward activations during backward
    instead of storing them. Saves memory at the cost of ~30% more compute.
    """
    from algos.diffusion_nets import ConditionalResBlock1d

    for name, module in model.named_modules():
        if isinstance(module, ConditionalResBlock1d):
            # Replace the forward with a checkpointed version
            original_forward = module.forward

            def make_ckpt_forward(orig_fn):
                def ckpt_forward(x, cond):
                    return checkpoint(orig_fn, x, cond, use_reentrant=False)
                return ckpt_forward

            module.forward = make_ckpt_forward(original_forward)


def _build_run_name(cfg):
    """Build a descriptive run name for TensorBoard."""
    parts = ["pusht"]
    if cfg.mixed_precision:
        parts.append(cfg.precision)
    else:
        parts.append("fp32")
    if cfg.activation_checkpointing:
        parts.append("ckpt")
    if cfg.grad_accum_steps > 1:
        parts.append(f"accum{cfg.grad_accum_steps}")
    parts.append(f"bs{cfg.batch_size}")
    parts.append(f"ch{cfg.base_channels}")
    return "_".join(parts)


if __name__ == "__main__":
    main()
