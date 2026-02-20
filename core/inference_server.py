"""Batched InferenceServer with multi-worker support.

Architecture (SEED RL pattern):
    N Workers each send (worker_id, obs) to a shared obs_queue.
    The server collects all available observations, concatenates them
    into one mega-batch, runs ONE forward pass, then splits the outputs
    and routes them back to each worker's dedicated action_queue.

Why batching matters:
    GPU forward passes have fixed overhead (kernel launch, memory transfer).
    Running 1 batch of 64 is much faster than 8 batches of 8, because
    you pay that overhead only once. This is the same reason SEED RL
    outperforms IMPALA — centralized batched inference on GPU.
"""

import torch
import numpy as np


class InferenceServer:
    """Runs algo.predict() in a dedicated process, serving batched inference.

    Sits between Workers and Learner:
    - Workers send (worker_id, obs_np) through shared obs_queue
    - Server batches observations, runs one predict(), routes results to per-worker action_queues
    - Learner sends updated weights through weight_queue after each training step

    The server handles numpy<->tensor conversion at the boundary.
    """

    def __init__(self, algo, device, obs_queue, action_queues, weight_queue):
        """
        Args:
            algo: A BaseAlgo instance. Must implement predict() and model property.
            device: Device string (e.g. 'cuda:0', 'cpu') to run inference on.
            obs_queue: Shared Queue — workers put (worker_id, obs_np).
            action_queues: Dict[int, Queue] — per-worker queues for routing results back.
            weight_queue: Queue[state_dict] — weight updates from Learner.
        """
        self.algo = algo
        self.device = torch.device(device)
        self.algo.model.to(self.device)
        self.obs_queue = obs_queue
        self.action_queues = action_queues
        self.weight_queue = weight_queue

    def _collect_batch(self):
        """Collect observations from all ready workers.

        Blocks for at least one observation (so we don't spin-wait),
        then drains any others that are already queued. This naturally
        adapts: when workers are fast, we get big batches; when they're
        slow, we still serve promptly.

        Returns:
            List of (worker_id, obs_np) tuples.
        """
        collected = []

        # Block until at least one worker has data
        worker_id, obs = self.obs_queue.get()
        collected.append((worker_id, obs))

        # Non-blocking drain of any more ready observations
        while not self.obs_queue.empty():
            try:
                worker_id, obs = self.obs_queue.get_nowait()
                collected.append((worker_id, obs))
            except Exception:
                break

        return collected

    def run(self):
        """Main loop: load initial weights, then serve batched inference forever.

        Each iteration:
        1. Check for weight updates (non-blocking drain)
        2. Collect observations from all ready workers
        3. Concatenate into one mega-batch
        4. Run one forward pass
        5. Split outputs and route to per-worker action_queues
        """
        # Block until Learner sends initial weights
        initial_weights = self.weight_queue.get()
        self.algo.model.load_state_dict(initial_weights)
        print(f"[InferenceServer] Loaded initial weights, serving {len(self.action_queues)} workers on {self.device}")

        serve_count = 0
        while True:
            # Drain any pending weight updates (non-blocking)
            while not self.weight_queue.empty():
                new_weights = self.weight_queue.get()
                self.algo.model.load_state_dict(new_weights)

            # Collect and batch observations from all ready workers
            collected = self._collect_batch()
            worker_ids = [c[0] for c in collected]
            obs_list = [c[1] for c in collected]
            sizes = [obs.shape[0] for obs in obs_list]  # num_envs per worker

            # Concatenate into one mega-batch: (total_envs, obs_dim)
            all_obs = np.concatenate(obs_list, axis=0)

            # One batched forward pass on GPU
            obs_tensor = torch.FloatTensor(all_obs).to(self.device)
            outputs = self.algo.predict(obs_tensor)
            outputs_np = {k: v.cpu().numpy() for k, v in outputs.items()}

            # Split outputs and route to each worker's action_queue
            offset = 0
            for wid, size in zip(worker_ids, sizes):
                worker_outputs = {k: v[offset:offset + size] for k, v in outputs_np.items()}
                self.action_queues[wid].put(worker_outputs)
                offset += size

            serve_count += len(collected)
            if serve_count % 500 == 0:
                print(f"[InferenceServer] Served {serve_count} requests (last batch: {len(collected)} workers)")
