import torch
import numpy as np



class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, act_dim):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.ptr = 0

        self.states = np.zeros((num_steps, num_envs, obs_dim))
        self.actions = np.zeros((num_steps, num_envs, act_dim))
        self.rewards = np.zeros((num_steps, num_envs))
        self.values = np.zeros((num_steps, num_envs))
        self.dones = np.zeros((num_steps, num_envs))
        self.log_probs = np.zeros((num_steps, num_envs))
    
    def add(self, states, actions, rewards, values, dones, log_probs):

        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.ptr += 1

    def get(self):
        return(
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs)
        )
    
    def clear(self):
        self.ptr = 0
