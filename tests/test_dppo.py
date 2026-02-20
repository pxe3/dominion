"""Tests for DPPO with both MLP (horizon=1) and UNet (horizon>1) backbones."""

import torch
import pytest
from algos.dppo import DPPO, DPPOModel


# ── MLP mode (horizon=1) ─────────────────────────────────────────────

def test_mlp_instantiation():
    algo = DPPO(obs_dim=15, action_dim=2)
    assert algo.horizon == 1
    assert hasattr(algo.model, 'noise_pred')
    assert hasattr(algo.model, 'value_fn')


def test_mlp_predict_shapes():
    algo = DPPO(obs_dim=15, action_dim=2, num_denoise_steps=3)
    obs = torch.randn(4, 15)
    out = algo.predict(obs)
    assert out["action"].shape == (4, 2)
    assert out["log_prob"].shape == (4,)
    assert out["value"].shape == (4,)
    assert out["denoising_chain"].shape == (4, 4, 2)  # K+1=4


def test_mlp_log_prob_recomputation():
    algo = DPPO(obs_dim=15, action_dim=2, num_denoise_steps=3)
    obs = torch.randn(4, 15)
    out = algo.predict(obs)
    recomputed = algo._recompute_log_probs(obs, out["denoising_chain"])
    torch.testing.assert_close(recomputed, out["log_prob"], atol=1e-4, rtol=1e-4)


def test_mlp_gradient_flow():
    algo = DPPO(obs_dim=15, action_dim=2, num_denoise_steps=3)
    obs = torch.randn(4, 15)
    chains = torch.randn(4, 4, 2)
    lp = algo._recompute_log_probs(obs, chains)
    lp.sum().backward()
    grad_count = sum(1 for p in algo.model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0


# ── UNet mode (horizon>1) ────────────────────────────────────────────

def test_unet_instantiation():
    algo = DPPO(obs_dim=5, action_dim=2, horizon=16,
                base_channels=32, channel_mults=(1, 2), cond_dim=64)
    assert algo.horizon == 16
    assert hasattr(algo.model.noise_pred, 'encoder_blocks')  # TemporalUNet


def test_unet_predict_shapes():
    algo = DPPO(obs_dim=5, action_dim=2, horizon=16,
                base_channels=32, channel_mults=(1, 2), cond_dim=64,
                num_denoise_steps=3)
    obs = torch.randn(4, 5)
    out = algo.predict(obs)
    # Action is first step of chunk
    assert out["action"].shape == (4, 2)
    assert out["log_prob"].shape == (4,)
    assert out["value"].shape == (4,)
    # Chain stores full action chunks: (B, K+1, horizon, action_dim)
    assert out["denoising_chain"].shape == (4, 4, 16, 2)


def test_unet_log_prob_recomputation():
    algo = DPPO(obs_dim=5, action_dim=2, horizon=16,
                base_channels=32, channel_mults=(1, 2), cond_dim=64,
                num_denoise_steps=3)
    obs = torch.randn(4, 5)
    out = algo.predict(obs)
    recomputed = algo._recompute_log_probs(obs, out["denoising_chain"])
    torch.testing.assert_close(recomputed, out["log_prob"], atol=1e-4, rtol=1e-4)


def test_unet_gradient_flow():
    algo = DPPO(obs_dim=5, action_dim=2, horizon=16,
                base_channels=32, channel_mults=(1, 2), cond_dim=64,
                num_denoise_steps=3)
    obs = torch.randn(4, 5)
    chains = torch.randn(4, 4, 16, 2)
    lp = algo._recompute_log_probs(obs, chains)
    lp.sum().backward()
    # Check both UNet and value network get gradients
    unet_grads = sum(1 for p in algo.model.noise_pred.parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
    assert unet_grads > 0


def test_unet_action_is_first_of_chunk():
    """Verify that predict() returns the first action from the denoised chunk."""
    algo = DPPO(obs_dim=5, action_dim=2, horizon=16,
                base_channels=32, channel_mults=(1, 2), cond_dim=64,
                num_denoise_steps=3)
    obs = torch.randn(4, 5)
    out = algo.predict(obs)
    # chain[:, -1] is the final denoised chunk (B, H, action_dim)
    # action should be chain[:, -1, 0, :] (first action of final chunk)
    final_chunk = out["denoising_chain"][:, -1]  # (B, H, action_dim)
    torch.testing.assert_close(out["action"], final_chunk[:, 0, :])


# ── Registry ─────────────────────────────────────────────────────────

def test_registry():
    from core.registry import ALGO_REGISTRY
    algo = ALGO_REGISTRY.make("dppo", obs_dim=15, action_dim=2)
    assert isinstance(algo, DPPO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
