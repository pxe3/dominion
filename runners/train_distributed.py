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

        # Queues for communication
        self.trajectory_queue = Queue()  # worker -> learner
        self.weight_queue = Queue()      # learner -> inference server
        self.obs_queue = Queue()         # worker -> inference server
        self.action_queue = Queue()      # inference server -> worker

        # Get dims from appropriate env
        dummy_env = ENV_REGISTRY.make(cfg.env.name, **cfg.env.args)
        self.obs_dim = dummy_env.observation_shape[0]
        self.action_dim = dummy_env.action_shape[0]

        self.worker_process = None
        self.learner_process = None
        self.inference_process = None

    @staticmethod
    def _run_inference_server(obs_dim, action_dim, cfg_algo, obs_queue, action_queue, weight_queue):
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
        model = algo.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Wait for initial weights from learner
        print("[Inference] Waiting for initial weights...")
        initial_weights = weight_queue.get()
        model.load_state_dict(initial_weights)
        print("[Inference] Ready")

        serve_count = 0
        while True:
            # Check for weight updates (non-blocking)
            while not weight_queue.empty():
                new_weights = weight_queue.get()
                model.load_state_dict(new_weights)
                print("[Inference] Loaded new weights")

            # Get obs from worker, run inference, send back
            obs = obs_queue.get()
            serve_count += 1
            if serve_count % 100 == 0:
                print(f"[Inference] Served {serve_count} requests")
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                outputs = model.forward_all(obs_tensor)
            outputs_np = {k: v.cpu().numpy() for k, v in outputs.items()}
            action_queue.put(outputs_np)

    @staticmethod
    def _run_worker(env_cfg, num_steps, num_envs, trajectory_queue, obs_queue, action_queue):
        auto_register("envs")

        def env_fn():
            return ENV_REGISTRY.make(env_cfg.name, **env_cfg.args)

        worker = RolloutWorker(
            env_fn=env_fn,
            num_steps=num_steps,
            num_envs=num_envs,
            trajectory_queue=trajectory_queue,
            obs_queue=obs_queue,
            action_queue=action_queue
        )
        worker.run()

    @staticmethod
    def _run_learner(obs_dim, action_dim, cfg_algo, trajectory_queue, weight_queue):
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

        # Send initial weights to inference server
        weight_queue.put(algo.model.state_dict())

        learner = Learner(
            policy=algo,
            trajectory_queue=trajectory_queue,
            weight_queue=weight_queue
        )
        learner.run()

    def start(self):
        # Start inference server first (needs to receive initial weights)
        self.inference_process = Process(
            target=DistributedTrainer._run_inference_server,
            args=(self.obs_dim, self.action_dim, self.cfg.algo,
                  self.obs_queue, self.action_queue, self.weight_queue)
        )

        self.learner_process = Process(
            target=DistributedTrainer._run_learner,
            args=(self.obs_dim, self.action_dim, self.cfg.algo,
                  self.trajectory_queue, self.weight_queue)
        )

        self.worker_process = Process(
            target=DistributedTrainer._run_worker,
            args=(self.cfg.env, self.num_steps, self.num_envs,
                  self.trajectory_queue, self.obs_queue, self.action_queue)
        )

        self.inference_process.start()
        self.learner_process.start()
        self.worker_process.start()

        self.inference_process.join()
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