import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from core.buffer import Batch
from algos.base import BaseAlgo


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std)
        return (mean, std)
    
    def get_action(self, state):
        mean, stdev = self.forward(state)
        dist = torch.distributions.Normal(mean, stdev)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1) 
        return action, log_prob
    
    def get_log_prob(self, state, action):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # critic only predicts value
        )
    
    def forward(self, state):
        return self.net(state)


class PPO(BaseAlgo):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, lr=1e-4, gamma=0.99, gae_disc=0.95, eps_clip=0.2, grad_epochs=10):

        self.actor = Actor(obs_dim, action_dim, hidden_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(obs_dim)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr)

        self.gamma = gamma
        self.gae_disc = gae_disc
        self.eps_clip= eps_clip
        self.grad_epochs = grad_epochs

    def select_action(self, states):
        states = torch.FloatTensor(states)
        action, log_prob = self.actor.get_action(states)
        value = self.critic(states).squeeze()
        return action.detach().numpy(), log_prob.detach().numpy(), value.detach().numpy()
    
    def update(self, buffer: Batch):

        states = buffer.states
        actions = buffer.actions
        rewards = buffer.rewards
        values = buffer.values
        dones = buffer.dones
        log_probs = buffer.log_probs
        
        v_bootstrap = self.critic(states[-1]).squeeze().detach() * (1-dones[-1]) # 
        advantages = get_gae_vectorized(rewards, values, dones, self.gamma, self.gae_disc, v_bootstrap)
        returns = (advantages + values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = states.flatten(0,1)
        actions = actions.flatten(0,1)
        log_probs = log_probs.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()
        

        for grad_step in range(self.grad_epochs):

            new_values = self.critic(states).squeeze()
            critic_loss = ((returns - new_values)**2).mean()

            new_log_probs = self.actor.get_log_prob(states, actions)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clip(ratio, 1-self.eps_clip, 1+self.eps_clip)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            full_loss = actor_loss + critic_loss

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            full_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        buffer.clear()

def get_gae(rewards, values, dones, gamma, gae_disc, v_bootstrap):
    # 1 td errors : reward + gamma * next_val(1-dones[t]) - values[t]
    # 2 gae --> value + gamma * gae_disc * gae ( 1 - done[t])
    # 3 insert
    # 4 increment value
    # after this loop, return
    advantages = []
    gae = 0
    next_val = v_bootstrap
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + gamma * next_val * (1-dones[i]) - values[i]
        gae = td_error + gamma * gae_disc * gae * (1 - dones[i])
        advantages.insert(0, gae)
        next_val = values[i]
    return torch.stack(advantages)


def get_gae_vectorized(rewards, values, dones, gamma, gae_disc, v_bootstrap):
    num_steps = rewards.shape[0] # 256 since rewards --> 256 x 8
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1]) # 8 cus 8 envs
    next_val = v_bootstrap # v_bootstrap is a vector if we call the v_boostrap = ... line on a state vector

    for t in reversed(range(num_steps)):
        td_error = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
        gae = td_error + gamma * gae_disc * gae * (1 - dones[t])
        advantages[t] = gae
        next_val = values[t]
    return advantages
