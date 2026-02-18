"""Tests for diffusion policy components.

Run: python -m pytest tests/test_diffusion.py -v
"""

import torch
import pytest
from dataclasses import dataclass


# === U-Net Tests ===

class TestTemporalUNet:
    """Test the denoising network produces correct shapes and is trainable."""

    def setup_method(self):
        from algos.diffusion_nets import TemporalUNet
        self.action_dim = 2
        self.obs_dim = 2
        self.T_p = 16
        self.B = 4
        self.net = TemporalUNet(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            T_p=self.T_p,
            base_channels=32,       # small for fast tests
            channel_mults=(1, 2),
            cond_dim=64,
        )

    def test_output_shape(self):
        """Output shape must match input: (B, T_p, action_dim)."""
        x = torch.randn(self.B, self.T_p, self.action_dim)
        t = torch.randint(0, 100, (self.B,))
        obs = torch.randn(self.B, self.obs_dim)
        out = self.net(x, t, obs)
        assert out.shape == (self.B, self.T_p, self.action_dim), f"Expected {(self.B, self.T_p, self.action_dim)}, got {out.shape}"

    def test_different_batch_sizes(self):
        """Should handle batch sizes 1, 8, 32."""
        for B in [1, 8, 32]:
            x = torch.randn(B, self.T_p, self.action_dim)
            t = torch.randint(0, 100, (B,))
            obs = torch.randn(B, self.obs_dim)
            out = self.net(x, t, obs)
            assert out.shape == (B, self.T_p, self.action_dim)

    def test_gradient_flows(self):
        """Gradients must flow through the entire network (no dead paths)."""
        x = torch.randn(self.B, self.T_p, self.action_dim)
        t = torch.randint(0, 100, (self.B,))
        obs = torch.randn(self.B, self.obs_dim)
        out = self.net(x, t, obs)
        loss = out.sum()
        loss.backward()
        for name, param in self.net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_different_timesteps_give_different_outputs(self):
        """Conditioning on different timesteps should produce different outputs."""
        x = torch.randn(1, self.T_p, self.action_dim)
        obs = torch.randn(1, self.obs_dim)
        out_t0 = self.net(x, torch.tensor([0]), obs)
        out_t99 = self.net(x, torch.tensor([99]), obs)
        assert not torch.allclose(out_t0, out_t99), "Different timesteps should give different outputs"


# === Noise Schedule Tests ===

class TestNoiseSchedule:
    """Test cosine beta schedule properties."""

    def test_cosine_schedule_bounds(self):
        from algos.diffusion_policy import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)
        assert (betas > 0).all(), "Betas must be positive"
        assert (betas < 1).all(), "Betas must be < 1"

    def test_alpha_bar_monotonically_decreasing(self):
        """alpha_bar should decrease: more noise at higher timesteps."""
        from algos.diffusion_policy import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        diffs = alpha_bar[1:] - alpha_bar[:-1]
        assert (diffs <= 0).all(), "alpha_bar must be monotonically decreasing"

    def test_alpha_bar_endpoints(self):
        """alpha_bar should be ~1 at t=0 (no noise) and ~0 at t=T (pure noise)."""
        from algos.diffusion_policy import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        assert alpha_bar[0] > 0.95, f"alpha_bar[0] should be ~1, got {alpha_bar[0]}"
        assert alpha_bar[-1] < 0.1, f"alpha_bar[-1] should be ~0, got {alpha_bar[-1]}"


# === DiffusionPolicy Tests ===

class TestDiffusionPolicy:
    """Test the full DiffusionPolicy algo."""

    def setup_method(self):
        from algos.diffusion_policy import DiffusionPolicy
        self.policy = DiffusionPolicy(
            obs_dim=2,
            action_dim=2,
            T_p=16,
            T_a=8,
            num_diffusion_steps=50,    # fewer steps for fast tests
            num_ddim_steps=5,
            base_channels=32,
            channel_mults=(1, 2),
            cond_dim=64,
            lr=1e-4,
        )

    def test_predict_shape(self):
        """predict() must return actions of shape (B, T_a, action_dim)."""
        obs = torch.randn(4, 2)
        out = self.policy.predict(obs)
        assert "action" in out, "predict() must return dict with 'action' key"
        assert out["action"].shape == (4, 8, 2), f"Expected (4, 8, 2), got {out['action'].shape}"

    def test_predict_single_obs(self):
        """Should handle single observation (B=1)."""
        obs = torch.randn(1, 2)
        out = self.policy.predict(obs)
        assert out["action"].shape == (1, 8, 2)

    def test_update_returns_loss(self):
        """update() must return a dict with 'loss' key."""
        @dataclass
        class FakeBatch:
            obs: torch.Tensor
            actions: torch.Tensor
        batch = FakeBatch(
            obs=torch.randn(8, 2),
            actions=torch.randn(8, 16, 2),
        )
        metrics = self.policy.update(batch)
        assert "loss" in metrics, "update() must return dict with 'loss'"
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0, "Loss should be positive"

    def test_update_decreases_loss(self):
        """Loss should decrease over multiple updates on the same batch (overfitting test)."""
        @dataclass
        class FakeBatch:
            obs: torch.Tensor
            actions: torch.Tensor
        batch = FakeBatch(
            obs=torch.randn(16, 2),
            actions=torch.randn(16, 16, 2),
        )
        # Run several updates on the same batch
        losses = []
        for _ in range(50):
            metrics = self.policy.update(batch)
            losses.append(metrics["loss"])
        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_model_property(self):
        """model property should return the U-Net."""
        from algos.diffusion_nets import TemporalUNet
        assert isinstance(self.policy.model, TemporalUNet)

    def test_weight_sync_roundtrip(self):
        """Simulate weight sync: save state_dict, load into new instance, check same outputs."""
        from algos.diffusion_policy import DiffusionPolicy
        obs = torch.randn(2, 2)
        torch.manual_seed(42)
        out1 = self.policy.predict(obs)

        # Save and load weights into a new policy
        state_dict = self.policy.model.state_dict()
        policy2 = DiffusionPolicy(
            obs_dim=2, action_dim=2, T_p=16, T_a=8,
            num_diffusion_steps=50, num_ddim_steps=5,
            base_channels=32, channel_mults=(1, 2), cond_dim=64, lr=1e-4,
        )
        policy2.model.load_state_dict(state_dict)

        torch.manual_seed(42)
        out2 = policy2.predict(obs)
        assert torch.allclose(out1["action"], out2["action"]), "Weight sync should produce identical outputs"


# === Integration: BaseAlgo Contract ===

class TestBaseAlgoContract:
    """Verify DiffusionPolicy satisfies the BaseAlgo interface."""

    def test_registered_in_registry(self):
        """Should be discoverable via ALGO_REGISTRY."""
        from core.registry import ALGO_REGISTRY, auto_register
        auto_register("algos")
        algo = ALGO_REGISTRY.make(
            "diffusion_policy",
            obs_dim=2, action_dim=2,
            T_p=16, T_a=8,
            num_diffusion_steps=50, num_ddim_steps=5,
            base_channels=32, channel_mults=(1, 2), cond_dim=64, lr=1e-4,
        )
        assert hasattr(algo, "predict")
        assert hasattr(algo, "update")
        assert hasattr(algo, "model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
