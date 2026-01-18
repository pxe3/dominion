import mujoco
import mujoco.viewer
import numpy as np
import os
from envs.base import BaseEnv


class CarEnv(BaseEnv):
    def __init__(self, max_steps=1500, seed=0, render=False, goal=None):
         xml_path = os.path.join(os.path.dirname(__file__), 'mujoco_assets', 'car.xml')
         self.model = mujoco.MjModel.from_xml_path(xml_path)
         self.data = mujoco.MjData(self.model)
         self.duration = int(max_steps//500)
         self._observation_shape = (15,)
         self._action_shape = (2,)
         self.viewer = None
         self.goal = np.array(goal) if goal is not None else np.array([-1, 4])
         self.reset()
         if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def action_shape(self):
        return self._action_shape

    def set_goal(self, goal):
        self.goal = np.array(goal)
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.model.body_pos[goal_body_id][:2] = self.goal

    def _get_state(self):
        car_pos = self.data.body('car').xpos[:2]
        goal_vec = self.goal - car_pos
        return np.hstack((
            self.data.body('car').xpos[:3],
            self.data.body('car').cvel,
            self.data.body('car').xquat,
            goal_vec
        ))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.episodic_return = 0
        state = self._get_state()
        if self.viewer is not None:
            self.viewer.sync()
        return state
    
    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()
    
    def close(self):
        self.viewer.close()

    def step(self, action):
        old_dist = np.linalg.norm(self.goal - self.data.body('car').xpos[:2])
        self.data.ctrl = np.tanh(action)
        mujoco.mj_step(self.model, self.data)

        state = self._get_state()

        new_dist = np.linalg.norm(self.goal - state[:2])    
        reward = (old_dist - new_dist) * 10

        self.episodic_return += reward
    
        done = False
        info = {}

        if new_dist < 0.1: #10 cm
            done = True
            reward += 10.0
        elif self.data.time >= self.duration: #timeout
            done = True
        
        if done:
            info['episode'] = {'r': self.episodic_return, 'l': self.data.time}
            info['terminal_obs'] = state.copy()
            state = self.reset()
        return state, reward, done, info