"""Distributed trainer: spawns N Workers, InferenceServer, and Learner in separate processes.

Architecture (SEED RL pattern):
    Worker 0 (CPU)  ──┐
    Worker 1 (CPU)  ──┤── shared obs_queue ──> InferenceServer (GPU) ──> per-worker action_queues
    Worker N (CPU)  ──┘                              ^
                                                     | weight queue
                                                     v
    Workers  ── shared trajectory_queue ──>  Learner (GPU)

Key design choices:
    - Shared obs_queue: workers tag messages with (worker_id, obs) so the
      InferenceServer can batch observations from all workers into one forward pass.
    - Per-worker action_queues: results are routed back to the correct worker.
    - Shared trajectory_queue: Learner processes rollouts from any worker,
      first-come-first-served. This is truly asynchronous.
    - V-trace corrects for the staleness from async data collection.
"""

import hydra
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
import numpy as np

from core.registry import ENV_REGISTRY, ALGO_REGISTRY, auto_register


auto_register("envs")
auto_register("algos")


def _make_algo(cfg_algo, obs_dim, action_dim):
    """Create an algo instance from config. Generic — no algo-specific kwargs."""
    return ALGO_REGISTRY.make(
        cfg_algo.name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        **cfg_algo.args,
    )


def _run_inference_server(cfg_algo, obs_dim, action_dim, device,
                          obs_queue, action_queues, weight_queue):
    """Process target for InferenceServer."""
    auto_register("algos")
    from core.inference_server import InferenceServer

    algo = _make_algo(cfg_algo, obs_dim, action_dim)
    server = InferenceServer(algo, device, obs_queue, action_queues, weight_queue)
    server.run()


def _run_worker(cfg_env, num_steps, num_envs,
                obs_queue, action_queue, trajectory_queue, worker_id):
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
        worker_id=worker_id,
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
    """Orchestrator: creates queues, spawns N workers + inference + learner.

    Does not contain any algo-specific logic. All it knows is:
    - How to read config
    - How to create queues and spawn processes
    - How to shut down when the Learner finishes
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = cfg.get("num_workers", 1)

        # Get env dimensions from a throwaway env instance
        dummy_env = ENV_REGISTRY.make(cfg.env.name, **cfg.env.args)
        self.obs_dim = dummy_env.observation_shape[0]
        self.action_dim = dummy_env.action_shape[0]

        # Inter-process communication
        self.obs_queue = mp.Queue()            # All Workers -> InferenceServer (shared)
        self.trajectory_queue = mp.Queue()    # All Workers -> Learner (shared)
        self.weight_queue = mp.Queue()        # Learner -> InferenceServer

        # Per-worker action queues (InferenceServer routes results back)
        self.action_queues = {}
        for i in range(self.num_workers):
            self.action_queues[i] = mp.Queue()

    def start(self):
        """Spawn all processes and wait for the Learner to finish."""
        log_dir = self.cfg.get("log_dir", None)

        # InferenceServer (1 process, serves all workers)
        inference_proc = mp.Process(
            target=_run_inference_server,
            args=(self.cfg.algo, self.obs_dim, self.action_dim, self.device,
                  self.obs_queue, self.action_queues, self.weight_queue),
        )

        # Workers (N processes, each with own action_queue)
        worker_procs = []
        for i in range(self.num_workers):
            proc = mp.Process(
                target=_run_worker,
                args=(self.cfg.env, self.cfg.num_steps, self.cfg.num_envs,
                      self.obs_queue, self.action_queues[i],
                      self.trajectory_queue, i),
            )
            worker_procs.append(proc)

        # Learner (1 process)
        learner_proc = mp.Process(
            target=_run_learner,
            args=(self.cfg.algo, self.obs_dim, self.action_dim, self.device,
                  self.trajectory_queue, self.weight_queue,
                  self.cfg.max_updates, self.cfg.log_interval, log_dir),
        )

        # Start all processes
        inference_proc.start()
        learner_proc.start()
        for proc in worker_procs:
            proc.start()

        print(f"[Trainer] Started {self.num_workers} workers, 1 inference server, 1 learner")

        # Learner has the stop condition — wait for it to finish
        learner_proc.join()
        print("[Trainer] Learner finished, shutting down...")

        # Terminate the infinite-loop processes
        for proc in worker_procs:
            proc.terminate()
        inference_proc.terminate()
        for proc in worker_procs:
            proc.join()
        inference_proc.join()
        print("[Trainer] All processes stopped")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = DistributedTrainer(cfg)
    trainer.start()


if __name__ == "__main__":
    main()
