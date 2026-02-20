import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Learner:
    """Receives rollouts from Worker, runs algo.update(), broadcasts new weights.

    Data flow:
        trajectory_queue.get() -> (Batch, episode_returns)
        -> batch.to(device) -> algo.update(batch) -> metrics
        -> weight_queue.put(state_dict) -> InferenceServer picks up new weights

    Has a stop condition (max_updates) so training terminates cleanly.
    Logs episode returns and training metrics at regular intervals.
    """

    def __init__(self, algo, trajectory_queue, weight_queue,
                 device="cuda:0", max_updates=None, log_interval=5,
                 log_dir=None):
        """
        Args:
            algo: A BaseAlgo instance (e.g. PPO). Must implement update() and model property.
            trajectory_queue: Queue[(Batch, List[float])] — rollouts from Worker.
            weight_queue: Queue[state_dict] — sends updated weights to InferenceServer.
            device: Device string to train on (e.g. 'cuda:0', 'cpu').
            max_updates: Stop after this many updates. None = run forever.
            log_interval: Print metrics every N updates.
            log_dir: Directory for TensorBoard logs. None = no TensorBoard.
        """
        self.algo = algo
        self.device = torch.device(device)
        self.algo.model.to(self.device)
        self.trajectory_queue = trajectory_queue
        self.weight_queue = weight_queue
        self.max_updates = max_updates
        self.log_interval = log_interval
        self.update_count = 0
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def run(self):
        """Main loop: send initial weights, then train until max_updates.

        Each iteration:
        1. Receive (batch, episode_returns) from Worker
        2. Move batch to training device
        3. Run algo.update() to get metrics
        4. Broadcast new weights to InferenceServer
        5. Log metrics periodically
        """
        # InferenceServer blocks until it gets initial weights
        self.weight_queue.put(self.algo.model.state_dict())
        print(f"[Learner] Sent initial weights, training on {self.device}")

        while self.max_updates is None or self.update_count < self.max_updates:
            # Block until Worker sends a rollout
            batch, episode_returns = self.trajectory_queue.get()

            # Data arrives on CPU (came through a Queue), move to training device
            batch = batch.to(self.device)

            # Run training step
            metrics = self.algo.update(batch)
            self.update_count += 1

            # Broadcast updated weights to InferenceServer
            self.weight_queue.put(self.algo.model.state_dict())

            # Log periodically
            if self.update_count % self.log_interval == 0:
                log_str = f"[Learner] Update {self.update_count}"
                if episode_returns:
                    log_str += f" | Avg Return: {np.mean(episode_returns):.2f}"
                if metrics:
                    for k, v in metrics.items():
                        log_str += f" | {k}: {v:.4f}"
                print(log_str)

            # TensorBoard logging (every update, not just log_interval)
            if self.writer:
                if episode_returns:
                    self.writer.add_scalar("train/avg_episode_return",
                                           np.mean(episode_returns), self.update_count)
                if metrics:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.update_count)

        if self.writer:
            self.writer.close()
        print(f"[Learner] Finished {self.update_count} updates")
