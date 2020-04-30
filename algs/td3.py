from algs.ddpg import DDPG
from common import DoubleQvalueCritic
import copy
import torch
import torch.nn.functional as F


# TD3 is very similar to DDPG

class TD3(DDPG):

	def __init__(self, env, policy_noise=0.2, noise_clip=0.5, policy_freq=2, **kwargs):
		super(TD3, self).__init__(env, **kwargs)

		self.critic = DoubleQvalueCritic(self.obs_dim, self.act_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

		self.policy_noise=policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0

	def train(self, state, action, next_state, reward, done):

		self.total_it += 1

		cur_action = self.actor(state)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state) + noise
			).clamp(-self.act_limit, self.act_limit)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (1 - done) * self.gamma * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, cur_action).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
