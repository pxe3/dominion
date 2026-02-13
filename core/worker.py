import numpy as np
from core.buffer import RolloutBuffer
from envs.vec_env import VecEnv


class RolloutWorker:
    """Collects experience by stepping environments and requesting actions from InferenceServer.

    Data flow each step:
        obs (numpy) -> obs_queue -> InferenceServer -> action_queue -> env.step()

    After num_steps, sends (Batch, episode_returns) to Learner via trajectory_queue.
    The worker is algo-agnostic: it uses outputs["action"] for env stepping and
    passes any extra keys (log_prob, value, etc.) through to the buffer.
    """

    def __init__(self, env_fn, num_steps, num_envs, obs_queue, action_queue, trajectory_queue):
        """
        Args:
            env_fn: Callable that returns a single env instance.
            num_steps: Steps per rollout before sending to Learner.
            num_envs: Number of parallel environments in VecEnv.
            obs_queue: Queue to send observations to InferenceServer.
            action_queue: Queue to receive action dicts from InferenceServer.
            trajectory_queue: Queue to send (Batch, episode_returns) to Learner.
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.trajectory_queue = trajectory_queue

        self.env = VecEnv(env_fn, num_envs)
        self.obs_shape = self.env.obs_shape
        self.action_shape = self.env.action_shape

        self.states = self.env.reset()
        self.buffer = RolloutBuffer(num_steps, num_envs, self.obs_shape[0], self.action_shape[0])

    def collect_rollout(self):
        """Run num_steps of env interaction, return (Batch, episode_returns).

        Sends obs to InferenceServer, receives dict of outputs. Only 'action'
        is required; 'log_prob' and 'value' are optional (for PPO, not for BC).
        """
        episode_returns = []

        for step in range(self.num_steps):
            # Request inference from InferenceServer
            self.obs_queue.put(self.states)
            outputs = self.action_queue.get()

            # 'action' is required; other keys are algo-specific
            actions = outputs["action"]
            log_probs = outputs.get("log_prob")
            values = outputs.get("value")

            next_states, rewards, dones, infos = self.env.step(actions)
            self.buffer.add(self.states, actions, rewards, values, dones, log_probs)
            self.states = next_states

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
                print(f"[Worker] Completed {rollout_count} rollouts")
