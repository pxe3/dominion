"""PushT dataset loader for diffusion policy BC training.

Loads the lerobot/pusht HuggingFace dataset and converts it into
(obs, action_chunk) pairs for training a diffusion policy.

The raw dataset is a flat sequence of (obs, action) per timestep.
We need to window it into action chunks of length T_p for the
diffusion policy to predict. Each sample becomes:
  obs:     (obs_dim,)           — current observation
  actions: (T_p, action_dim)    — next T_p actions to predict

Data is normalized to [-1, 1] for stable diffusion training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset


class PushTDataset(Dataset):
    """PushT expert demonstration dataset for behavior cloning.

    Downloads lerobot/pusht from HuggingFace, windows into action chunks,
    and normalizes to [-1, 1].
    """

    def __init__(self, T_o=2, T_p=16):
        """
        Args:
            T_o: Observation horizon (unused for now — we use current obs only).
            T_p: Prediction horizon (length of action chunk to predict).
        """
        self.T_p = T_p

        # Load full dataset
        raw = load_dataset("lerobot/pusht", split="train")

        # Extract arrays
        all_obs = np.array(raw["observation.state"], dtype=np.float32)
        all_actions = np.array(raw["action"], dtype=np.float32)
        all_episodes = np.array(raw["episode_index"])

        # Compute normalization stats (min-max → [-1, 1])
        self.obs_min = all_obs.min(axis=0)
        self.obs_max = all_obs.max(axis=0)
        self.act_min = all_actions.min(axis=0)
        self.act_max = all_actions.max(axis=0)

        all_obs = self._normalize(all_obs, self.obs_min, self.obs_max)
        all_actions = self._normalize(all_actions, self.act_min, self.act_max)

        # Build windowed samples: for each valid starting index within an episode,
        # create (obs_t, actions[t:t+T_p]) pairs
        self.samples = []
        episode_ids = np.unique(all_episodes)

        for ep_id in episode_ids:
            mask = all_episodes == ep_id
            ep_obs = all_obs[mask]
            ep_actions = all_actions[mask]

            # We need T_p consecutive actions starting from each timestep
            for t in range(len(ep_actions) - T_p):
                self.samples.append({
                    "obs": torch.from_numpy(ep_obs[t]),
                    "actions": torch.from_numpy(ep_actions[t:t + T_p]),
                })

    @staticmethod
    def _normalize(x, x_min, x_max):
        """Min-max normalize to [-1, 1]."""
        return 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0

    def get_unnorm_stats(self):
        """Return stats needed to unnormalize actions back to original scale."""
        return {
            "obs_min": self.obs_min,
            "obs_max": self.obs_max,
            "act_min": self.act_min,
            "act_max": self.act_max,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
