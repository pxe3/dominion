import numpy as np
from core.buffer import RolloutBuffer
from envs.vec_env import VecEnv

class RolloutWorker:
    def __init__(self, env_fn, policy, num_steps, num_envs, trajectory_queue, weight_queue):
        
        
        self.policy = policy
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.trajectory_queue = trajectory_queue
        self.weight_queue = weight_queue
        

        self.env = VecEnv(env_fn, num_envs)
        self.obs_shape = self.env.obs_shape
        self.action_shape = self.env.action_shape

        self.states = self.env.reset()
        self.buffer = RolloutBuffer(num_steps, num_envs, self.obs_shape[0], self.action_shape[0])
    
    def collect_rollout(self):
        for step in range(self.num_steps):
            actions, log_probs, values = self.policy.select_action(self.states)
            next_states, rewards, dones, infos = self.env.step(actions)
            self.buffer.add(self.states, actions, rewards, values, dones, log_probs)
            self.states = next_states
        return self.buffer.get()

    def sync_weights(self):
        if self.weight_queue.empty():
            pass
        else:
            self.policy.load_state_dict(self.weight_queue.get())

    def run(self):
        while True:
            batch = self.collect_rollout()
            self.trajectory_queue.put(batch)
            self.sync_weights()

    

    


