import copy
import numpy as np
import torch
from common import DeterministicActor, QvalueCritic, ReplayBuffer
import torch.nn.functional as F
from algs.base import BaseAgent


class Random(BaseAgent):
	def __init__(self, env, buffer_size=int(1e6), device=None):

		super(Random, self).__init__(env, device)
		self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

	def act(self, obs):
		return self.env.action_space.sample()


	def step(self, t):

		self.episode_timesteps += 1

		# Select action randomly or according to policy
		action = self.env.action_space.sample()

		# Perform action
		next_obs, reward, done, _ = self.env.step(action)
		done_bool = float(done)# if self.episode_timesteps < self.env._max_episode_steps else 0
		# Store data in replay buffer
		self.replay_buffer.add(copy.deepcopy(self.obs), action, next_obs, reward, done_bool)
		self.obs = next_obs
		self.episode_reward += reward
		# Train agent after collecting sufficient data
		if done:
			self.episode_end_handle(t)

