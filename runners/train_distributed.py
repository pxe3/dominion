import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os

from envs.car_env import CarEnv
from envs.vec_env import VecEnv
from envs.mjx_env import MJXEnv
from algos.ppo_def import PPO
from core.buffer import RolloutBuffer
from multiprocessing import Process, Queue
from core.worker import RolloutWorker
from core.learner import Learner





class DistributedTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_steps = cfg.num_steps
        self.num_envs = cfg.num_envs
        self.log_interval = cfg.log_interval
        self.seed = cfg.seed
        self.backend = getattr(cfg.env, 'backend', 'cpu')  # 'cpu' or 'mjx'

        self.trajectory_queue = Queue()
        self.weight_queue = Queue()

        # Get dims from appropriate env
        if self.backend == 'mjx':
            dummy_env = MJXEnv(env_name=cfg.env.name, num_envs=1, seed=0)
        else:
            dummy_env = CarEnv(max_steps=cfg.env.max_steps, goal=cfg.env.goal, render=False)
        self.obs_dim = dummy_env.observation_shape[0]
        self.action_dim = dummy_env.action_shape[0]

        self.worker_process, self.learner_process = None, None
    
    @staticmethod
    def _run_workers(env_cfg, obs_dim, action_dim, num_steps, num_envs, cfg_algo, trajectory_queue, weight_queue, backend='cpu'):
        from envs.mjx_env import MJXEnv  # import here to avoid JAX init in main process

        ppo = PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg_algo.hidden_dim,
            lr=cfg_algo.lr,
            gamma=cfg_algo.gamma,
            gae_disc=cfg_algo.gae_lambda,
            eps_clip=cfg_algo.eps_clip,
            grad_epochs=cfg_algo.grad_epochs
        )

        # Create env based on backend
        if backend == 'mjx':
            env = MJXEnv(env_name=env_cfg.name, num_envs=num_envs, seed=0)
        else:
            def env_fn():
                return CarEnv(max_steps=env_cfg.max_steps, goal=env_cfg.goal, render=False)
            env = VecEnv(env_fn, num_envs)

        rolloutWorker = RolloutWorker(
            env_fn=lambda: None,  # not used, we pass env directly below
            policy=ppo,
            num_steps=num_steps,
            num_envs=num_envs,
            trajectory_queue=trajectory_queue,
            weight_queue=weight_queue
        )
        rolloutWorker.env = env  # override with our env
        rolloutWorker.obs_shape = env.obs_shape
        rolloutWorker.action_shape = env.action_shape
        rolloutWorker.states = env.reset()

        initial_weights = weight_queue.get()
        ppo.ac.load_state_dict(initial_weights)
        rolloutWorker.run()
    
    @staticmethod
    def _run_learner(obs_dim, action_dim, cfg_algo, trajectory_queue, weight_queue):

        ppo_m = PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg_algo.hidden_dim,
            lr=cfg_algo.lr,
            gamma=cfg_algo.gamma,
            gae_disc=cfg_algo.gae_lambda,
            eps_clip=cfg_algo.eps_clip,
            grad_epochs=cfg_algo.grad_epochs
        )
        
        weight_queue.put(ppo_m.ac.state_dict())
    
        learner = Learner(
            policy=ppo_m,
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
            self.num_envs, self.cfg.algo, self.trajectory_queue, self.weight_queue, self.backend)
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