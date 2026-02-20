import numpy as np
from core.buffer import RolloutBuffer
from envs.vec_env import SubprocVecEnv


class RolloutWorker:
    """Collects experience by stepping environments and requesting actions from InferenceServer.

    Data flow each step:
        obs (numpy) -> obs_queue -> InferenceServer -> action_queue -> env.step()

    After num_steps, sends (Batch, episode_returns) to Learner via trajectory_queue.
    The worker is algo-agnostic: it uses outputs["action"] for env stepping and
    passes any extra keys (log_prob, value, etc.) through to the buffer.

    In multi-worker mode, each worker has a unique worker_id and its own action_queue.
    All workers share the same obs_queue (tagged with worker_id) and trajectory_queue.
    """

    def __init__(self, env_fn, num_steps, num_envs, obs_queue, action_queue,
                 trajectory_queue, worker_id=0):
        """
        Args:
            env_fn: Callable that returns a single env instance.
            num_steps: Steps per rollout before sending to Learner.
            num_envs: Number of parallel environments in VecEnv.
            obs_queue: Shared Queue to send (worker_id, obs) to InferenceServer.
            action_queue: Per-worker Queue to receive action dicts from InferenceServer.
            trajectory_queue: Shared Queue to send (Batch, episode_returns) to Learner.
            worker_id: Unique integer ID for this worker (used for routing in batched inference).
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.trajectory_queue = trajectory_queue
        self.worker_id = worker_id

        self.env = SubprocVecEnv(env_fn, num_envs)
        self.obs_shape = self.env.obs_shape
        self.action_shape = self.env.action_shape

        self.obs = self.env.reset()
        self.buffer = RolloutBuffer(num_steps, num_envs, self.obs_shape[0], self.action_shape[0])

    def collect_rollout(self):
        """Run num_steps of env interaction, return (Batch, episode_returns).

        Sends (worker_id, obs) to InferenceServer, receives dict of outputs.
        Only 'action' is required; 'log_prob' and 'value' are optional.
        """
        episode_returns = []

        for step in range(self.num_steps):
            # Tag obs with worker_id so InferenceServer can route the response
            self.obs_queue.put((self.worker_id, self.obs))
            outputs = self.action_queue.get()

            # 'action' is required; other keys are algo-specific
            actions = outputs["action"]
            log_probs = outputs.get("log_prob")
            values = outputs.get("value")

            # Collect any extra keys (e.g. denoising_chain for DPPO)
            known_keys = {"action", "log_prob", "value"}
            extras = {k: v for k, v in outputs.items() if k not in known_keys}
            extras = extras if extras else None

            next_obs, rewards, dones, infos = self.env.step(actions)
            self.buffer.add(self.obs, actions, rewards, values, dones, log_probs, extras=extras)
            self.obs = next_obs

            # Collect completed episode returns for logging
            for info in infos:
                if "episode" in info:
                    episode_returns.append(info["episode"]["r"])

        return self.buffer.get(), episode_returns

    def run(self):
        """Main loop: collect rollouts and send to Learner forever."""
        rollout_count = 0
        while True:
            batch, episode_returns = self.collect_rollout()
            self.buffer.clear()
            self.trajectory_queue.put((batch, episode_returns))
            rollout_count += 1
            if rollout_count % 10 == 0:
                print(f"[Worker {self.worker_id}] Completed {rollout_count} rollouts")
