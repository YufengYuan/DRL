import torch
from common import ReplayBuffer, SquashedGaussianActor, DoubleQvalueCritic
import gym
import copy
import torch.nn.functional as F
from algs.base_agent import BaseAgent


class SAC(BaseAgent):

	def __init__(self, env, buffer_size=int(1e6), gamma=0.99, tau=0.005,
	             eval_freq=5000, lr=3e-4, start_timesteps=5000, train_freq=10, batch_size=256,
	             alpha=0.2, device=None):

		super(SAC, self).__init__(env, device)

		self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = DoubleQvalueCritic(self.obs_dim, self.act_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.replay_buffer = ReplayBuffer(buffer_size)

		self.start_timesteps = start_timesteps
		self.tau = tau
		self.gamma = gamma
		self.alpha = alpha
		self.eval_freq = eval_freq
		self.train_freq = train_freq
		self.batch_size = batch_size

	def train(self, state, action, next_state, reward, done):

		with torch.no_grad():

			next_action, logprob = self.actor(next_state)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (1 - done) * self.gamma * (target_Q - self.alpha * logprob)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		cur_action, logprob = self.actor(state)

		# TODO: freeze critic parameters here to prevent unnecessary backpropagation
		current_Q1, current_Q2 = self.critic(state, cur_action)
		current_Q = torch.min(current_Q1, current_Q2)

		actor_loss = (self.alpha * logprob - current_Q).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		obs = obs.reshape(1, -1)
		return self.actor.act(obs, deterministic=True)

	def step(self, t):

		self.episode_timesteps += 1

		# Select action randomly or according to policy
		if t < self.start_timesteps:
			action = self.env.action_space.sample()
		else:
			action = self.actor.act(torch.tensor(self.obs, dtype=torch.float32, device=self.device))

		# Perform action
		next_obs, reward, done, _ = self.env.step(action)
		done_bool = float(done) if self.episode_timesteps < self.env._max_episode_steps else 0
		# Store data in replay buffer
		self.replay_buffer.add(copy.deepcopy(self.obs), action, next_obs, reward, done_bool)
		self.obs = next_obs
		self.episode_reward += reward

		# Train agent after collecting sufficient data
		if t > self.start_timesteps and t % self.train_freq == 0:
			for _ in range(self.train_freq):
				batch = self.replay_buffer.sample(self.batch_size)
				self.train(*batch)

		if done:
			self.episode_end_handle(t)
