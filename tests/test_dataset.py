"""Tests for PushT dataset loader.

Run: python -m pytest tests/test_dataset.py -v
"""

import torch
import pytest


class TestPushTDataset:
    """Test the PushT dataset loader for diffusion policy BC training."""

    def setup_method(self):
        from data.pusht_dataset import PushTDataset
        self.ds = PushTDataset(T_o=2, T_p=16)

    def test_dataset_length(self):
        """Should have samples (fewer than raw dataset due to windowing)."""
        assert len(self.ds) > 0, "Dataset should not be empty"

    def test_sample_keys(self):
        """Each sample should have 'obs' and 'actions'."""
        sample = self.ds[0]
        assert "obs" in sample, "Sample must have 'obs'"
        assert "actions" in sample, "Sample must have 'actions'"

    def test_obs_shape(self):
        """obs shape should be (obs_dim,) — current observation for conditioning."""
        sample = self.ds[0]
        assert sample["obs"].shape == (2,), f"Expected (2,), got {sample['obs'].shape}"

    def test_actions_shape(self):
        """actions shape should be (T_p, action_dim) — the action chunk to predict."""
        sample = self.ds[0]
        assert sample["actions"].shape == (16, 2), f"Expected (16, 2), got {sample['actions'].shape}"

    def test_data_normalized(self):
        """Data should be roughly normalized to [-1, 1] range."""
        sample = self.ds[0]
        assert sample["obs"].abs().max() <= 2.0, "Obs should be normalized"
        assert sample["actions"].abs().max() <= 2.0, "Actions should be normalized"

    def test_dataloader_works(self):
        """Should work with PyTorch DataLoader for batching."""
        from torch.utils.data import DataLoader
        loader = DataLoader(self.ds, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        assert batch["obs"].shape == (8, 2)
        assert batch["actions"].shape == (8, 16, 2)

    def test_no_nans(self):
        """No NaN values in the dataset."""
        for i in range(min(100, len(self.ds))):
            sample = self.ds[i]
            assert not torch.isnan(sample["obs"]).any(), f"NaN in obs at index {i}"
            assert not torch.isnan(sample["actions"]).any(), f"NaN in actions at index {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
