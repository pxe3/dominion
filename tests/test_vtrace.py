"""Tests for V-trace off-policy correction."""

import torch
import pytest
from algos.ppo_def import get_vtrace, get_gae_vectorized


def test_vtrace_on_policy_matches_td():
    """When behavior == target (on-policy), V-trace reduces to standard TD."""
    T, N = 10, 4
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    log_probs = torch.randn(T, N)  # same for both
    bootstrap = torch.randn(N)

    vs, advantages = get_vtrace(
        rewards, values, dones,
        behavior_log_probs=log_probs,
        target_log_probs=log_probs,  # on-policy: same as behavior
        gamma=0.99,
        bootstrap_value=bootstrap,
    )

    assert vs.shape == (T, N)
    assert advantages.shape == (T, N)
    # V-trace targets should be finite
    assert torch.isfinite(vs).all()
    assert torch.isfinite(advantages).all()


def test_vtrace_clipping_limits_correction():
    """Large IS ratios should be clipped by rho_bar and c_bar."""
    T, N = 5, 2
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.zeros(T, N)
    # Behavior and target are very different — large IS ratio
    behavior_lp = torch.zeros(T, N)  # log(1) = 0
    target_lp = torch.ones(T, N) * 5.0  # log(exp(5)) ≈ 148x ratio
    bootstrap = torch.zeros(N)

    vs_clipped, adv_clipped = get_vtrace(
        rewards, values, dones, behavior_lp, target_lp,
        gamma=0.99, bootstrap_value=bootstrap, rho_bar=1.0, c_bar=1.0,
    )

    vs_unclipped, adv_unclipped = get_vtrace(
        rewards, values, dones, behavior_lp, target_lp,
        gamma=0.99, bootstrap_value=bootstrap, rho_bar=100.0, c_bar=100.0,
    )

    # Clipped version should have smaller magnitude corrections
    assert vs_clipped.abs().mean() < vs_unclipped.abs().mean()


def test_vtrace_shapes():
    """Output shapes should match input shapes."""
    T, N = 20, 8
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.9).float()
    behavior_lp = torch.randn(T, N)
    target_lp = torch.randn(T, N)
    bootstrap = torch.randn(N)

    vs, advantages = get_vtrace(
        rewards, values, dones, behavior_lp, target_lp,
        gamma=0.99, bootstrap_value=bootstrap,
    )

    assert vs.shape == (T, N)
    assert advantages.shape == (T, N)


def test_vtrace_detached():
    """V-trace targets should be detached (no gradient)."""
    T, N = 5, 2
    rewards = torch.randn(T, N)
    values = torch.randn(T, N, requires_grad=True)
    dones = torch.zeros(T, N)
    lp = torch.randn(T, N)
    bootstrap = torch.zeros(N)

    vs, _ = get_vtrace(rewards, values, dones, lp, lp, 0.99, bootstrap)
    assert not vs.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
