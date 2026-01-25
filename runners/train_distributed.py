import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os

from core.buffer import RolloutBuffer
from multiprocessing import Process, Queue
from core.worker import RolloutWorker
from core.learner import Learner

from core.registry import ENV_REGISTRY, ALGO_REGISTRY, auto_register


auto_register("envs")
auto_register("algos")


class DistributedTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_steps = cfg.num_steps
        self.num_envs = cfg.num_envs
        self.log_interval = cfg.log_interval
        self.seed = cfg.seed

        self.trajectory_queue = Queue()
        self.weight_queue = Queue()

        # Get dims from appropriate env
        dummy_env = ENV_REGISTRY.make(cfg.env.name, **cfg.env.args)
        self.obs_dim = dummy_env.observation_shape[0]
        self.action_dim = dummy_env.action_shape[0]

        self.worker_process, self.learner_process = None, None
    
    @staticmethod
    def _run_workers(env_cfg, obs_dim, action_dim, num_steps, num_envs, cfg_algo, trajectory_queue, weight_queue):

        auto_register("envs")
        auto_register("algos")

        algo = ALGO_REGISTRY.make(
            cfg_algo.name,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg_algo.hidden_dim,
            lr=cfg_algo.lr,
            gamma=cfg_algo.gamma,
            gae_disc=cfg_algo.gae_lambda,
            eps_clip=cfg_algo.eps_clip,
            grad_epochs=cfg_algo.grad_epochs
        )


        def env_fn():
            return ENV_REGISTRY.make(env_cfg.name, **env_cfg.args)

        rolloutWorker = RolloutWorker(
            env_fn = env_fn,
            policy=algo,
            num_steps=num_steps,
            num_envs=num_envs,
            trajectory_queue=trajectory_queue,
            weight_queue=weight_queue
        )


        initial_weights = weight_queue.get()
        algo.ac.load_state_dict(initial_weights)
        rolloutWorker.run()
    
    @staticmethod
    def _run_learner(obs_dim, action_dim, cfg_algo, trajectory_queue, weight_queue):
        auto_register("algos")

        algo_m = ALGO_REGISTRY.make(
            cfg_algo.name,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg_algo.hidden_dim,
            lr=cfg_algo.lr,
            gamma=cfg_algo.gamma,
            gae_disc=cfg_algo.gae_lambda,
            eps_clip=cfg_algo.eps_clip,
            grad_epochs=cfg_algo.grad_epochs
        )
        
        weight_queue.put(algo_m.ac.state_dict())
    
        learner = Learner(
            policy=algo_m,
            trajectory_queue=trajectory_queue,
            weight_queue=weight_queue
        )
        learner.run()

    def start(self):

        self.learner_process = Process(
            target=DistributedTrainer._run_learner,
            args=(self.obs_dim, self.action_dim, self.cfg.algo, 
            self.trajectory_queue, self.weight_queue)
        )

        self.worker_process = Process(
            target=DistributedTrainer._run_workers,
            args=(self.cfg.env, self.obs_dim, self.action_dim, self.num_steps,
            self.num_envs, self.cfg.algo, self.trajectory_queue, self.weight_queue)
        )

        self.learner_process.start()
        self.worker_process.start()
        self.learner_process.join()
        self.worker_process.join()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = DistributedTrainer(cfg)
    trainer.start()

if __name__ == "__main__":
    main()