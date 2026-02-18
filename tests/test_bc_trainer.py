"""Tests for BC trainer components.

Run: python -m pytest tests/test_bc_trainer.py -v
"""

import torch
import pytest
from dataclasses import dataclass


@dataclass
class BCBatch:
    obs: torch.Tensor
    actions: torch.Tensor


class TestBCTrainingLoop:
    """Test that the core training loop works end-to-end on CPU."""

    def setup_method(self):
        from algos.diffusion_policy import DiffusionPolicy
        self.policy = DiffusionPolicy(
            obs_dim=2, action_dim=2, T_p=16, T_a=8,
            num_diffusion_steps=50, num_ddim_steps=5,
            base_channels=32, channel_mults=(1, 2), cond_dim=64, lr=1e-4,
        )

    def test_single_update_step(self):
        """One forward+backward should run without error."""
        batch = BCBatch(obs=torch.randn(8, 2), actions=torch.randn(8, 16, 2))
        metrics = self.policy.update(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_gradient_accumulation_manual(self):
        """Gradient accumulation: multiple backward passes before step."""
        accum_steps = 4
        self.policy.optimizer.zero_grad()
        total_loss = 0
        for _ in range(accum_steps):
            batch = BCBatch(obs=torch.randn(4, 2), actions=torch.randn(4, 16, 2))
            # Manual forward/backward (like train_bc.py does)
            t = torch.randint(0, 50, (4,))
            noise = torch.randn(4, 16, 2)
            self.policy._to_device(torch.device("cpu"))
            ab_t = self.policy.alpha_bar[t][:, None, None]
            noisy = torch.sqrt(ab_t) * batch.actions + torch.sqrt(1 - ab_t) * noise
            pred = self.policy.net(noisy, t, batch.obs)
            loss = torch.nn.functional.mse_loss(pred, noise) / accum_steps
            loss.backward()
            total_loss += loss.item()
        self.policy.optimizer.step()
        assert total_loss > 0

    def test_activation_checkpointing_doesnt_break(self):
        """Model should still train correctly with activation checkpointing."""
        from torch.utils.checkpoint import checkpoint
        from algos.diffusion_nets import ConditionalResBlock1d

        # Enable checkpointing
        for module in self.policy.net.modules():
            if isinstance(module, ConditionalResBlock1d):
                orig = module.forward
                def make_ckpt(fn):
                    def ckpt_fwd(x, cond):
                        return checkpoint(fn, x, cond, use_reentrant=False)
                    return ckpt_fwd
                module.forward = make_ckpt(orig)

        # Training should still work
        batch = BCBatch(obs=torch.randn(8, 2), actions=torch.randn(8, 16, 2))
        metrics = self.policy.update(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_loss_converges_on_fixed_data(self):
        """Training on fixed data for 100 steps should reduce loss."""
        batch = BCBatch(obs=torch.randn(32, 2), actions=torch.randn(32, 16, 2))
        initial_loss = self.policy.update(batch)["loss"]
        for _ in range(99):
            self.policy.update(batch)
        final_loss = self.policy.update(batch)["loss"]
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"


class TestDataLoaderIntegration:
    """Test that dataset + model work together end-to-end."""

    def test_dataset_to_model(self):
        """Load real PushT data, feed through model, get valid loss."""
        from data.pusht_dataset import PushTDataset
        from torch.utils.data import DataLoader
        from algos.diffusion_policy import DiffusionPolicy

        ds = PushTDataset(T_o=2, T_p=16)
        loader = DataLoader(ds, batch_size=8, collate_fn=lambda s: BCBatch(
            obs=torch.stack([x["obs"] for x in s]),
            actions=torch.stack([x["actions"] for x in s]),
        ))

        policy = DiffusionPolicy(
            obs_dim=2, action_dim=2, T_p=16, T_a=8,
            num_diffusion_steps=50, num_ddim_steps=5,
            base_channels=32, channel_mults=(1, 2), cond_dim=64, lr=1e-4,
        )

        batch = next(iter(loader))
        metrics = policy.update(batch)
        assert metrics["loss"] > 0
        assert not torch.isnan(torch.tensor(metrics["loss"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
