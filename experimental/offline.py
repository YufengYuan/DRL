from algs.base import BaseAgent
import numpy as np
import copy
import torch
import torch.nn.functional as F
from common import DeterministicActor
from common import ValueCritic
from experimental.replay_buffer import ReplayBuffer

class TD3(BaseAgent):

	def __init__(self, env, lr=1e-3, gamma=0.99, tau=0.005, buffer_size=int(1e6),
	             start_timesteps=5000, expl_noise=0.1, batch_size=128,
	             policy_noise=0.2, noise_clip=0.5, policy_freq=1, device=None, **kwargs):

		super(TD3, self).__init__(env, device)

		self.actor = DeterministicActor(self.obs_dim, self.act_dim, self.act_limit, **kwargs).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = ValueCritic(self.obs_dim).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

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

		self.c_loss, self.a_loss, self.b_value = [], [], []
		self.priority = np.ones(buffer_size)
		self.stable_priority = np.ones(buffer_size)
		self.total_idxes = np.arange(buffer_size)
		self.partial_idxes = np.arange(buffer_size)

		self.values  = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
		self.advs    = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
		self.rewards = torch.tensor(self.replay_buffer.reward, dtype=torch.float32, device=self.device)
		self.dones   = torch.tensor(self.replay_buffer.not_done, dtype=torch.float32, device=self.device)



	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		return self.actor(obs).cpu().data.numpy().flatten()

	def train(self):

		if self.total_it == 0:
			# Initialize Monte-Carlo return
			self.returns = np.zeros_like(self.replay_buffer.reward, dtype=np.float32)
			last_return = 0
			self.replay_buffer.reward /= 10
			for i in reversed(range(self.replay_buffer.size)):
				reward = self.replay_buffer.reward[i]
				done = self.replay_buffer.not_done[i]
				self.returns[i] = last_return = reward + (1 - done) * self.gamma * last_return
			self.returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)

		normal_idxes = np.random.choice(self.total_idxes, self.batch_size)
		priori_idxes = np.random.choice(self.total_idxes, self.batch_size, p=self.priority/np.sum(self.priority))

		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=priori_idxes)
		# Update actor
		actor_loss = torch.mean((self.actor(obs) - action)**2)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		a_loss = actor_loss.item()
		self.actor_optimizer.step()

		# Update value estimate
		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=normal_idxes)
		target = reward + (1-done) * self.gamma * self.critic_target(next_obs)
		#target = self.returns[normal_idxes]
		critic_loss = torch.mean((self.critic(obs) - target)**2)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()

		# Update priority
		advs = reward + (1 - done) * self.gamma * self.critic_target(next_obs) - self.critic_target(obs)
		#advs = self.returns[normal_idxes] - self.critic(obs)
		advs = torch.exp((advs / 1).clamp_max(4)).detach().cpu().numpy().flatten()
		self.priority[normal_idxes] = advs
		self.total_it += 1

		return c_loss, a_loss, 0



	def step(self, t):
		c, a, b = self.train()
		self.c_loss.append(c); self.a_loss.append(a); self.b_value.append(b)
		if t % 5000 == 0:
			print(f'Iteration {t}: '
				  f'Critic Loss: {np.mean(self.c_loss):.3f}, '
				  f'Actor Loss: {np.mean(self.a_loss):.3f}, '
				  f'Beta: {np.mean(self.b_value):.3f}')
			self.c_loss, self.a_loss, self.b_value = [], [], []
		self.episode_timesteps += 1

