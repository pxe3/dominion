"""Distributed trainer: spawns Worker, InferenceServer, and Learner in separate processes.

Architecture:
    Worker (CPU)  <--obs/action queues-->  InferenceServer (GPU)
                                                  ^
                                                  | weight queue
                                                  v
    Worker  --trajectory queue-->  Learner (GPU)

Each process creates its own algo instance (separate memory spaces).
The Learner owns the stop condition (max_updates) — when it finishes,
the orchestrator terminates the other processes.
"""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from multiprocessing import Process, Queue

from core.registry import ENV_REGISTRY, ALGO_REGISTRY, auto_register


auto_register("envs")
auto_register("algos")


def _make_algo(cfg_algo, obs_dim, action_dim):
    """Create an algo instance from config. Generic — no PPO-specific kwargs."""
    return ALGO_REGISTRY.make(
        cfg_algo.name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        **cfg_algo.args,
    )


def _run_inference_server(cfg_algo, obs_dim, action_dim, device,
                          obs_queue, action_queue, weight_queue):
    """Process target for InferenceServer."""
    auto_register("algos")
    from core.inference_server import InferenceServer

    algo = _make_algo(cfg_algo, obs_dim, action_dim)
    server = InferenceServer(algo, device, obs_queue, action_queue, weight_queue)
    server.run()


def _run_worker(cfg_env, num_steps, num_envs,
                obs_queue, action_queue, trajectory_queue):
    """Process target for RolloutWorker."""
    auto_register("envs")
    from core.worker import RolloutWorker

    def env_fn():
        return ENV_REGISTRY.make(cfg_env.name, **cfg_env.args)

    worker = RolloutWorker(
        env_fn=env_fn,
        num_steps=num_steps,
        num_envs=num_envs,
        obs_queue=obs_queue,
        action_queue=action_queue,
        trajectory_queue=trajectory_queue,
    )
    worker.run()


def _run_learner(cfg_algo, obs_dim, action_dim, device,
                 trajectory_queue, weight_queue, max_updates, log_interval,
                 log_dir=None):
    """Process target for Learner."""
    auto_register("algos")
    from core.learner import Learner

    algo = _make_algo(cfg_algo, obs_dim, action_dim)
    learner = Learner(
        algo=algo,
        trajectory_queue=trajectory_queue,
        weight_queue=weight_queue,
        device=device,
        max_updates=max_updates,
        log_interval=log_interval,
        log_dir=log_dir,
    )
    learner.run()


class DistributedTrainer:
    """Thin orchestrator: creates queues, spawns processes, waits for completion.

    Does not contain any algo-specific logic. All it knows is:
    - How to read config
    - How to create queues and spawn processes
    - How to shut down when the Learner finishes
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

        # Get env dimensions from a throwaway env instance
        dummy_env = ENV_REGISTRY.make(cfg.env.name, **cfg.env.args)
        self.obs_dim = dummy_env.observation_shape[0]
        self.action_dim = dummy_env.action_shape[0]

        # Inter-process communication
        self.obs_queue = Queue()           # Worker -> InferenceServer
        self.action_queue = Queue()        # InferenceServer -> Worker
        self.trajectory_queue = Queue()    # Worker -> Learner
        self.weight_queue = Queue()        # Learner -> InferenceServer

    def start(self):
        """Spawn all processes and wait for the Learner to finish."""
        inference_proc = Process(
            target=_run_inference_server,
            args=(self.cfg.algo, self.obs_dim, self.action_dim, self.device,
                  self.obs_queue, self.action_queue, self.weight_queue),
        )
        worker_proc = Process(
            target=_run_worker,
            args=(self.cfg.env, self.cfg.num_steps, self.cfg.num_envs,
                  self.obs_queue, self.action_queue, self.trajectory_queue),
        )
        log_dir = self.cfg.get("log_dir", None)
        learner_proc = Process(
            target=_run_learner,
            args=(self.cfg.algo, self.obs_dim, self.action_dim, self.device,
                  self.trajectory_queue, self.weight_queue,
                  self.cfg.max_updates, self.cfg.log_interval, log_dir),
        )

        # Start all processes
        inference_proc.start()
        learner_proc.start()
        worker_proc.start()

        # Learner has the stop condition — wait for it to finish
        learner_proc.join()
        print("[Trainer] Learner finished, shutting down...")

        # Terminate the infinite-loop processes
        worker_proc.terminate()
        inference_proc.terminate()
        worker_proc.join()
        inference_proc.join()
        print("[Trainer] All processes stopped")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = DistributedTrainer(cfg)
    trainer.start()


if __name__ == "__main__":
    main()
