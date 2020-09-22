from algs.base import BaseAgent
import numpy as np
import copy
import torch
import torch.nn.functional as F
from common import DeterministicActor, DoubleQvalueCritic, ReplayBuffer
from common.models import GaussianActor

class Custom(BaseAgent):

	def __init__(self, env, lr=1e-3, gamma=0.99, tau=0.005, buffer_size=int(1e6),
	             start_timesteps=5000, expl_noise=0.1, batch_size=128,
	             policy_noise=0.2, noise_clip=0.5, policy_freq=2, device=None, **kwargs):

		super(Custom, self).__init__(env, device)


		self.actor = GaussianActor(self.obs_dim, self.act_dim, self.act_limit, **kwargs).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.behavior = GaussianActor(self.obs_dim, self.act_dim, self.act_limit, **kwargs).to(self.device)
		self.behavior_optimizer = torch.optim.Adam(self.behavior.parameters(), lr=lr)

		self.critic = DoubleQvalueCritic(self.obs_dim, self.act_dim, **kwargs).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

		self.start_timesteps = start_timesteps
		self.expl_noise = expl_noise
		self.batch_size = batch_size
		self.lr = lr
		self.gamma = gamma
		self.tau = tau
		self.policy_noise=policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0

		self.c_loss, self.a_loss = [], []

	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		return self.actor(obs, True).cpu().data.numpy().flatten()

	def behavior_init(self, iteration=1000):
		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)



	def train(self):

		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)

		self.total_it += 1

		cur_action = self.actor(obs)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_obs) + noise
			).clamp(-self.act_limit, self.act_limit)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (1 - done) * self.gamma * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()
		a_loss = 0
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			for param in self.critic.parameters():
				param.requires_grad = False

			# Compute actor losse
			actor_loss = -self.critic.Q1(obs, cur_action).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			a_loss = actor_loss.item()
			self.actor_optimizer.step()

			for param in self.critic.parameters():
				param.requires_grad = True

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return c_loss, a_loss

	def step(self, t):
		c, a = self.train()
		self.c_loss.append(c); self.a_loss.append(a)
		if t % 100 == 0:
			#self.evaluate(self.env)
			print(f'Iteration {t}: Critic Loss: {np.mean(self.c_loss)}, Actor Loss: {np.mean(self.a_loss)*2}')
			self.c_loss, self.a_loss = [], []
		self.episode_timesteps += 1

