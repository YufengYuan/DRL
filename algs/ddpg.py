import copy
import numpy as np
import torch
from common import DeterministicActor, QvalueCritic, ReplayBuffer
import torch.nn.functional as F
import gym
from algs.base_agent import BaseAgent


class DDPG(BaseAgent):
	def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, buffer_size=int(1e6),
	             start_timesteps=5000, expl_noise=0.1, eval_freq=5000, batch_size=256,
				 device=None):

		super(DDPG, self).__init__(env, device)

		self.actor = DeterministicActor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = QvalueCritic(self.obs_dim, self.act_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.replay_buffer = ReplayBuffer(buffer_size)

		self.start_timesteps = start_timesteps
		self.expl_noise = expl_noise
		self.batch_size = batch_size
		self.lr = lr
		self.gamma = gamma
		self.tau = tau
		self.eval_freq = eval_freq

	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		return self.actor(obs).cpu().data.numpy().flatten()

	def train(self, state, action, next_state, reward, done):
		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (1 - done) * self.gamma * target_Q.detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def step(self, t):

		self.episode_timesteps += 1

		# Select action randomly or according to policy
		if t < self.start_timesteps:
			action = self.env.action_space.sample()
		else:
			action = (
					self.actor.act(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
					+ np.random.normal(0, self.act_limit * self.expl_noise, size=self.act_dim)
			).clip(-self.act_limit, self.act_limit)

		# Perform action
		next_obs, reward, done, _ = self.env.step(action)
		done_bool = float(done) if self.episode_timesteps < self.env._max_episode_steps else 0
		# Store data in replay buffer
		self.replay_buffer.add(copy.deepcopy(self.obs), action, next_obs, reward, done_bool)
		self.obs = next_obs

		self.episode_reward += reward
		# Train agent after collecting sufficient data
		if t > self.start_timesteps:
			batch = self.replay_buffer.sample(self.batch_size)
			self.train(*batch)

		# TODO: use logger to record online information
		if done:
			self.episode_end_handle(t)

