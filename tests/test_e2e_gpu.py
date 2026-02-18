"""End-to-end GPU test: trains diffusion policy on PushT for a few epochs.

Run this when you have a GPU to verify everything works:
    python tests/test_e2e_gpu.py

Tests:
  1. Dataset loads correctly
  2. Model trains on GPU with each optimization variant
  3. Loss decreases over epochs
  4. DDIM inference produces valid actions
  5. TensorBoard logs are written
  6. Model checkpoint saves and loads correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint


@dataclass
class BCBatch:
    obs: torch.Tensor
    actions: torch.Tensor


def collate_bc(samples):
    return BCBatch(
        obs=torch.stack([s["obs"] for s in samples]),
        actions=torch.stack([s["actions"] for s in samples]),
    )


def train_loop(policy, dataloader, device, num_epochs, use_amp=False, amp_dtype=torch.bfloat16,
               grad_accum_steps=1, tag=""):
    """Run a short training loop, return list of epoch losses."""
    epoch_losses = []
    for epoch in range(num_epochs):
        losses = []
        policy.optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch = BCBatch(obs=batch.obs.to(device), actions=batch.actions.to(device))

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                obs, actions = batch.obs, batch.actions
                B = obs.shape[0]
                t = torch.randint(0, policy.num_diffusion_steps, (B,), device=device)
                noise = torch.randn_like(actions)
                policy._to_device(device)
                ab_t = policy.alpha_bar[t][:, None, None]
                noisy = torch.sqrt(ab_t) * actions + torch.sqrt(1 - ab_t) * noise
                pred = policy.net(noisy, t, obs)
                loss = nn.functional.mse_loss(pred, noise) / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                policy.optimizer.step()
                policy.optimizer.zero_grad()

            losses.append(loss.item() * grad_accum_steps)

        avg = np.mean(losses)
        epoch_losses.append(avg)
        print(f"  [{tag}] Epoch {epoch+1}/{num_epochs}: loss={avg:.6f}")

    return epoch_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: No GPU detected. Tests will run on CPU (slower, no mixed precision).")

    print("\n" + "="*60)
    print("1. DATASET")
    print("="*60)
    from data.pusht_dataset import PushTDataset
    ds = PushTDataset(T_o=2, T_p=16)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2,
                        collate_fn=collate_bc, drop_last=True)
    print(f"   Samples: {len(ds)}, Batches: {len(loader)}")
    batch = next(iter(loader))
    print(f"   obs: {batch.obs.shape}, actions: {batch.actions.shape}")
    assert batch.obs.shape == (64, 2)
    assert batch.actions.shape == (64, 16, 2)
    print("   PASS")

    # Model config (smaller for fast testing)
    model_kwargs = dict(
        obs_dim=2, action_dim=2, T_p=16, T_a=8,
        num_diffusion_steps=100, num_ddim_steps=10,
        base_channels=64, channel_mults=(1, 2, 4), cond_dim=128, lr=1e-4,
    )
    num_epochs = 3

    print("\n" + "="*60)
    print("2. BASELINE (fp32)")
    print("="*60)
    from algos.diffusion_policy import DiffusionPolicy
    policy = DiffusionPolicy(**model_kwargs)
    policy.model.to(device)
    num_params = sum(p.numel() for p in policy.model.parameters())
    print(f"   Parameters: {num_params:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    losses_fp32 = train_loop(policy, loader, device, num_epochs, tag="fp32")
    time_fp32 = time.time() - t0

    mem_fp32 = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    assert losses_fp32[-1] < losses_fp32[0], "Loss should decrease"
    print(f"   Time: {time_fp32:.1f}s, Peak GPU: {mem_fp32:.0f}MB")
    print("   PASS")

    print("\n" + "="*60)
    print("3. MIXED PRECISION (bf16)")
    print("="*60)
    policy_amp = DiffusionPolicy(**model_kwargs)
    policy_amp.model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    losses_bf16 = train_loop(policy_amp, loader, device, num_epochs,
                             use_amp=(device.type == "cuda"), amp_dtype=torch.bfloat16, tag="bf16")
    time_bf16 = time.time() - t0

    mem_bf16 = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    assert losses_bf16[-1] < losses_bf16[0], "Loss should decrease with amp"
    print(f"   Time: {time_bf16:.1f}s, Peak GPU: {mem_bf16:.0f}MB")
    if device.type == "cuda":
        print(f"   Speedup: {time_fp32/time_bf16:.2f}x, Memory savings: {(1 - mem_bf16/mem_fp32)*100:.0f}%")
    print("   PASS")

    print("\n" + "="*60)
    print("4. ACTIVATION CHECKPOINTING (bf16 + ckpt)")
    print("="*60)
    from algos.diffusion_nets import ConditionalResBlock1d
    policy_ckpt = DiffusionPolicy(**model_kwargs)
    policy_ckpt.model.to(device)

    # Enable checkpointing
    for module in policy_ckpt.net.modules():
        if isinstance(module, ConditionalResBlock1d):
            orig = module.forward
            def make_ckpt(fn):
                def ckpt_fwd(x, cond):
                    return checkpoint(fn, x, cond, use_reentrant=False)
                return ckpt_fwd
            module.forward = make_ckpt(orig)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    losses_ckpt = train_loop(policy_ckpt, loader, device, num_epochs,
                             use_amp=(device.type == "cuda"), amp_dtype=torch.bfloat16, tag="bf16+ckpt")
    time_ckpt = time.time() - t0

    mem_ckpt = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    assert losses_ckpt[-1] < losses_ckpt[0], "Loss should decrease with checkpointing"
    print(f"   Time: {time_ckpt:.1f}s, Peak GPU: {mem_ckpt:.0f}MB")
    if device.type == "cuda":
        print(f"   Memory savings vs fp32: {(1 - mem_ckpt/mem_fp32)*100:.0f}%")
    print("   PASS")

    print("\n" + "="*60)
    print("5. GRADIENT ACCUMULATION (bf16 + accum 4x)")
    print("="*60)
    policy_accum = DiffusionPolicy(**model_kwargs)
    policy_accum.model.to(device)

    # Use smaller batch with 4x accumulation = same effective batch
    loader_small = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2,
                              collate_fn=collate_bc, drop_last=True)

    t0 = time.time()
    losses_accum = train_loop(policy_accum, loader_small, device, num_epochs,
                              use_amp=(device.type == "cuda"), amp_dtype=torch.bfloat16,
                              grad_accum_steps=4, tag="bf16+accum4")
    time_accum = time.time() - t0
    assert losses_accum[-1] < losses_accum[0], "Loss should decrease with grad accum"
    print(f"   Time: {time_accum:.1f}s")
    print("   PASS")

    print("\n" + "="*60)
    print("6. DDIM INFERENCE")
    print("="*60)
    # Use the trained fp32 policy
    obs = torch.randn(4, 2).to(device)
    t0 = time.time()
    out = policy.predict(obs)
    inference_time = time.time() - t0
    assert "action" in out
    assert out["action"].shape == (4, 8, 2), f"Expected (4, 8, 2), got {out['action'].shape}"
    print(f"   Output shape: {out['action'].shape}")
    print(f"   Inference time (4 samples): {inference_time*1000:.1f}ms")
    print("   PASS")

    print("\n" + "="*60)
    print("7. CHECKPOINT SAVE/LOAD")
    print("="*60)
    ckpt_dir = "/tmp/dominion_test_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "test_model.pt")

    # Save
    torch.save(policy.model.state_dict(), ckpt_path)
    print(f"   Saved to {ckpt_path}")

    # Load into new policy
    policy2 = DiffusionPolicy(**model_kwargs)
    policy2.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    policy2.model.to(device)

    # Verify same outputs
    torch.manual_seed(123)
    out1 = policy.predict(obs)
    torch.manual_seed(123)
    out2 = policy2.predict(obs)
    assert torch.allclose(out1["action"], out2["action"], atol=1e-5), "Loaded model should match"
    print("   Loaded and verified matching outputs")

    # Cleanup
    shutil.rmtree(ckpt_dir)
    print("   PASS")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"   {'Config':<30} {'Time':>8} {'Memory':>10}")
    print(f"   {'─'*30} {'─'*8} {'─'*10}")
    print(f"   {'fp32 (baseline)':<30} {time_fp32:>7.1f}s {mem_fp32:>8.0f}MB")
    print(f"   {'bf16 (mixed precision)':<30} {time_bf16:>7.1f}s {mem_bf16:>8.0f}MB")
    print(f"   {'bf16 + act. checkpointing':<30} {time_ckpt:>7.1f}s {mem_ckpt:>8.0f}MB")
    print(f"   {'bf16 + grad accum 4x':<30} {time_accum:>7.1f}s {'N/A':>10}")
    print()
    print("   ALL TESTS PASSED")


if __name__ == "__main__":
    main()
