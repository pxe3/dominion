import numpy as np
from core.buffer import RolloutBuffer
from envs.vec_env import VecEnv

class RolloutWorker:
    def __init__(self, env_fn, num_steps, num_envs, trajectory_queue, obs_queue, action_queue):

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.trajectory_queue = trajectory_queue
        self.obs_queue = obs_queue
        self.action_queue = action_queue

        self.env = VecEnv(env_fn, num_envs)
        self.obs_shape = self.env.obs_shape
        self.action_shape = self.env.action_shape

        self.states = self.env.reset()
        self.buffer = RolloutBuffer(num_steps, num_envs, self.obs_shape[0], self.action_shape[0])

    def get_actions(self, obs):
        self.obs_queue.put(obs)
        outputs = self.action_queue.get()
        return outputs['action'], outputs['log_prob'], outputs['value']

    def collect_rollout(self):
        for step in range(self.num_steps):
            actions, log_probs, values = self.get_actions(self.states)
            next_states, rewards, dones, infos = self.env.step(actions)
            self.buffer.add(self.states, actions, rewards, values, dones, log_probs)
            self.states = next_states
        return self.buffer.get()

    def run(self):
        while True:
            batch = self.collect_rollout()
            self.buffer.clear()
            self.trajectory_queue.put(batch)

    

    


