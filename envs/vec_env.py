import numpy as np
from typing import Callable, Any, List, Dict, Tuple
from envs.base import BaseEnv

class VecEnv:

    def __init__(self, env_fn: Callable[[], BaseEnv], num_envs: int):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        self.obs_shape = self.envs[0].observation_shape
        self.action_shape = self.envs[0].action_shape

    def reset(self) -> np.ndarray:
        obs_stack = np.stack([env.reset() for env in self.envs])
        return obs_stack
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for env, action in zip(self.envs, actions):

            obs, reward, done, info = env.step(action)

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        obs_batch = np.stack(obs_list)
        reward_batch = np.asarray(reward_list)
        done_batch = np.asarray(done_list)

        return obs_batch, reward_batch, done_batch, info_list







    

