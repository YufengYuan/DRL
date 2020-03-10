import gym
import numpy as np

from gym.envs.classic_control import MountainCarEnv

class ModifiedMoutainCar(MountainCarEnv):

	def __init__(self):
		super(ModifiedMoutainCar, self).__init__()
		self.pos = -0.5

	def step(self, action):
		#obs = super(ModifiedMoutainCar, self).state
		obs = self.state
		next_obs, reward, done, info = super(ModifiedMoutainCar, self).step(action)
		reward = 1000 * ((np.sin(3 * next_obs[0]) * 0.0025 + 0.5 * next_obs[1] * next_obs[1]) -
		                (np.sin(3 * obs[0]) * 0.0025 + 0.5 * obs[1] * obs[1]))
		#reward = np.abs(obs[0] + 0.5)
		return obs, reward, done, info

	def reset(self):
		obs = super(ModifiedMoutainCar, self).reset()
		self.pos = obs[0]
		return obs