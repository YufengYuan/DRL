from algs.base import BaseAgent
import numpy as np
import copy
import torch
import torch.nn.functional as F
from common import DeterministicActor, DoubleQvalueCritic, GaussianActor, REMCritic
from common import ValueCritic
from torch.autograd import grad
from experimental.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
import time

class TD3(BaseAgent):

	def __init__(self, env, lr=1e-3, gamma=0.99, tau=0.005, buffer_size=int(1e6),
	             start_timesteps=5000, expl_noise=0.1, batch_size=128,
	             policy_noise=0.2, noise_clip=0.5, policy_freq=1, device=None, **kwargs):

		super(TD3, self).__init__(env, device)

		self.actor = DeterministicActor(self.obs_dim, self.act_dim, self.act_limit, **kwargs).to(self.device)
		#self.actor = GaussianActor(self.obs_dim, self.act_dim, self.act_limit, **kwargs).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		#self.critic = DoubleQvalueCritic(self.obs_dim, self.act_dim, **kwargs).to(self.device)
		self.critic = ValueCritic(self.obs_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		#self.log_beta = torch.tensor(np.log(0.5), requires_grad=True, device=self.device)
		#self.target_mse = []
		#self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=1e-4, betas=(0.5, 0.999))
		#self.beta = self.log_beta.exp()

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

		self.lam     = 0.95


	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		#print(f'True Action: {self.actor(obs).cpu().data.numpy().flatten()}')
		return self.actor(obs).cpu().data.numpy().flatten()

	def xtrain(self):

		if self.total_it == 0:
			# Initialize Monte-Carlo return
			self.returns = np.zeros_like(self.replay_buffer.reward, dtype=np.float32)
			last_return = 0
			for i in reversed(range(self.replay_buffer.size)):
				reward = self.replay_buffer.reward[i]
				done = self.replay_buffer.not_done[i]
				self.returns[i] = last_return = reward + (1 - done) * self.gamma * last_return
			self.returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)

		normal_idxes = np.random.choice(self.total_idxes, self.batch_size)
		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=normal_idxes)

		# Update value estimate
		with torch.no_grad():
			target = reward + (1 - done) * self.gamma * self.critic(next_obs)
			adv = target - self.critic(obs)

		#critic_loss = torch.mean((self.critic(obs) - self.returns[normal_idxes]) ** 2)
		scale = torch.exp((adv / 1).clamp_max(3))
		scale = scale / torch.sum(scale) * self.batch_size
		critic_loss = torch.mean(scale * (self.critic(obs) - self.returns[normal_idxes])**2)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()

		# Update actor
		scale = torch.exp((adv / 1).clamp_max(3))
		scale = scale / torch.sum(scale) * self.batch_size
		#print(scale.mean(), scale.std(), scale.max(), scale.min())
		actor_loss = torch.mean(scale * torch.sum((self.actor(obs) - action)**2, dim=1, keepdim=True))
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		a_loss = actor_loss.item()
		self.actor_optimizer.step()

		self.total_it += 1

		return c_loss, a_loss, 0



	def __train(self):
		normal_idxes = np.random.choice(self.total_idxes, self.batch_size)

		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=normal_idxes)

		# Update value estimate
		with torch.no_grad():
			target = reward + (1 - done) * self.gamma * self.critic_target(next_obs)
			adv = target - self.critic_target(obs)

		critic_loss = torch.mean((self.critic(obs) - target) ** 2)
		#scale = torch.exp((adv / 1).clamp_max(3))
		#scale = scale / torch.sum(scale) * self.batch_size
		#critic_loss = torch.mean(scale * (self.critic(obs) - target)**2)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()

		# Update actor
		scale = torch.exp((adv / 1).clamp_max(3))
		scale = scale / torch.sum(scale) * self.batch_size
		actor_loss = torch.mean(scale * torch.sum((self.actor(obs) - action)**2, dim=1, keepdim=True))
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		a_loss = actor_loss.item()
		self.actor_optimizer.step()

		self.total_it += 1
		if self.total_it % 1000 == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(param.data)
		#for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
		#	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return c_loss, a_loss, 0


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
		#advs = reward + (1 - done) * self.gamma * self.critic(next_obs) - self.critic(obs)
		#advs = self.returns[normal_idxes] - self.critic(obs)
		#print(advs)
		advs = torch.exp((advs / 1).clamp_max(4)).detach().cpu().numpy().flatten()
		#tau = 1.
		self.priority[normal_idxes] = advs# * tau + self.priority[normal_idxes] * (1 - tau)
		#print(f'max: {self.priority.max():.4f}, '
		#      f'min: {self.priority.min():.4f}, '
		#      f'mean: {self.priority.mean():.4f}, '
		#      f'std: {self.priority.std():.4f}, ')

		self.total_it += 1
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return c_loss, a_loss, 0


	def __train(self):

		if self.total_it % 10000 == 0:
			# Update value and advantage estimation
			batch_size = 2048
			for start in range(0, self.replay_buffer.size, batch_size):
				idx = self.total_idxes[start: start+batch_size]
				obs, _ = self.replay_buffer.sample_state_action(idx)
				self.values[idx] = self.critic(obs).flatten()
			# Update returns
			lastgaelam = 0
			for t in reversed(range(len(self.rewards))):
				if t == len(self.rewards) - 1:
					last_obs = torch.tensor(self.replay_buffer.next_state[-1], dtype=torch.float32, device=self.device)
					nextvalues = self.critic(last_obs).flatten()
				else:
					nextvalues = self.values[t + 1]
				nextnonterminal = 1.0 - self.dones[t]
				delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
				self.advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
			print(self.advs.mean(), self.advs.std())

		idx = np.random.choice(self.total_idxes, self.batch_size)
		obs, action = self.replay_buffer.sample_state_action(idx)
		# Update value estimate
		critic_loss = torch.mean(((self.values[idx] + self.advs[idx]).detach() - self.critic(obs))**2)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()
		# Update policy
		scale = torch.exp((self.advs[idx].detach() / 0.02).clamp_max(3))
		diff = torch.mean((self.actor(obs) - action)**2, dim=1)
		actor_loss = torch.mean(scale * diff)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		a_loss = actor_loss.item()
		self.actor_optimizer.step()

		self.total_it += 1

		return c_loss, a_loss, 0

	def _train(self):

		if self.total_it % 100 == 0:
			self.prob = self.priority / np.sum(self.priority)
		critic_idxes = np.random.choice(self.total_idxes, self.batch_size)
		actor_idxes  = np.random.choice(self.total_idxes, self.batch_size, p=self.prob)

		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=critic_idxes)

		self.total_it += 1

		with torch.no_grad():
			# Compute the target Q value
			_next_action = self.actor_target(next_obs)
			raw_Q = self.critic_target(next_obs, _next_action)
			prob = torch.rand_like(raw_Q)
			prob = prob / torch.sum(prob, dim=1, keepdim=True)
			target_Q = torch.sum(prob * raw_Q, dim=1, keepdim=True)
			target_Q = reward + (1 - done) * self.gamma * target_Q

			advs = reward + (1 - done) * self.gamma * torch.mean(raw_Q, dim=1, keepdim=True) - self.critic_target(obs, self.actor_target(obs)).mean(dim=1, keepdim=True)
			#advs = torch.mean(advs, dim=1, keepdim=True)
			self.priority[critic_idxes] = torch.exp(advs / 1).cpu().numpy().flatten()

		# Double Q learning update
		#current_Q1, current_Q2 = self.critic(obs, action)
		#critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # REM Q learning update
		current_Q = self.critic(obs, action)
		critic_loss = F.mse_loss(target_Q, torch.sum(prob * current_Q, dim=1, keepdim=True))

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		c_loss = critic_loss.item()
		self.critic_optimizer.step()
		# Delayed policy updates

		obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, with_idxes=actor_idxes)
		cur_action = self.actor(obs)
		#advs = target_Q - self.critic_target.Q1(obs, self.actor_target(obs))
		#advs = self.critic_target.Q1(obs, action) - self.critic_target.Q1(obs, self.actor_target(obs))
		#advs = torch.exp((advs / 1.).clamp_max(4))
		#scale = advs / torch.sum(advs) * self.batch_size
		#scale.detach_()
		#actor_loss = torch.mean(advs * (cur_action - action)**2)
		actor_loss = torch.mean((cur_action - action)**2)
		#actor_loss = -self.actor.logprob(obs, action).mean()
		## Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		a_loss = torch.mean((cur_action - action)**2).item()
		self.actor_optimizer.step()

		b_value = 0 #self.beta.item()
		# Update the frozen target models
		if self.total_it % 100 == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(param.data)

		return c_loss, a_loss, b_value

	def step(self, t):
		#cur_time = time.time()
		c, a, b = self.train()
		#print(time.time() - cur_time)
		self.c_loss.append(c); self.a_loss.append(a); self.b_value.append(b)
		if t % 5000 == 0:
			#self.evaluate(self.env)
			print(f'Iteration {t}: '
				  f'Critic Loss: {np.mean(self.c_loss):.10f}, '
				  f'Actor Loss: {np.mean(self.a_loss):.3f}, '
				  f'Beta: {np.mean(self.b_value):.3f}')
			self.c_loss, self.a_loss, self.b_value = [], [], []
		self.episode_timesteps += 1

