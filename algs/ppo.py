import torch
from common.models import StochasticActor, ValueCritic
from torch import nn
from torch.optim import Adam
import numpy as np
from algs.base_agent import BaseAgent
from common import BatchBuffer
import copy


class PPO(BaseAgent):

	def __init__(self, env, lr=3e-4, batch_size=2048, num_minibatch=32, gamma=0.99,
	             lam=0.95, vf_coef=1, ent_coef=0, clip_range=0.2, n_epochs=10, max_grad_norm=10,
			     device=None):

		super(PPO, self).__init__(env, device)


		self.actor = StochasticActor(self.obs_dim, self.act_dim).to(self.device)
		self.actor_optimizer = Adam(self.actor.parameters(), lr)

		self.critic = ValueCritic(self.obs_dim).to(self.device)
		self.critic_optimizer = Adam(self.critic.parameters(), lr)

		self.batch_buffer = BatchBuffer(batch_size, self.obs_dim, self.act_dim, self.device)

		self.gamma = gamma
		self.lam = lam
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.n_minibatch = num_minibatch
		self.n_epoch = n_epochs
		self.clip_range = clip_range
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size


	def train(self, obses, actions, logprobs, returns, values):

		batch_size = len(returns)
		advs = returns - values

		advs -= advs.mean()
		if advs.std() != 0 and not torch.isnan(advs.std()):
			advs /= advs.std()

		idxes = np.arange(batch_size)
		minibatch_size = batch_size // self.n_minibatch
		for i in range(self.n_epoch):
			np.random.shuffle(idxes)
			for start in range(0, batch_size, minibatch_size):
				end = start + minibatch_size
				idx = idxes[start : end]
				# Compute value loss with mean square error
				cur_values = self.critic(obses[idx])
				value_loss = torch.mean((returns[idx] - cur_values) ** 2)
				# Compute policy loss with clipped objective
				cur_pi = self.actor(obses[idx])
				grad_pi = torch.exp(cur_pi.log_prob(actions[idx]).unsqueeze_(-1) - logprobs[idx])
				policy_loss = -grad_pi * advs[idx]
				policy_loss_clipped = -torch.clamp(grad_pi, 1 - self.clip_range, 1 + self.clip_range) * advs[idx]
				policy_loss = torch.mean(torch.max(policy_loss, policy_loss_clipped))

				# Compute entropy loss
				#entropy_loss = -torch.mean(cur_pi.entropy())
				#policy_loss += self.ent_coef * entropy_loss

				# Update the critic
				self.critic_optimizer.zero_grad()
				value_loss.backward()
				if self.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.critic_optimizer.step()

				# Update the actor
				self.actor_optimizer.zero_grad()
				policy_loss.backward()
				if self.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				self.actor_optimizer.step()
		return None

	def compute_return(self, rewards, values, dones, last_value):
		advs = np.zeros_like(rewards)
		batch_size = len(rewards)
		lastgaelam = 0
		for t in reversed(range(batch_size)):
			if t == batch_size - 1:
				nextvalues = last_value
			else:
				nextvalues = values[t + 1]
			nextnonterminal = 1.0 - dones[t]
			delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
			advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
		return advs + values

	def act(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
		return self.actor.act(obs)[0]

	def step(self, t):

		self.episode_timesteps += 1

		action, logprob = self.actor.act(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
		value = self.critic.value(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
		last_obs, reward, done, _ = self.env.step(action)

		self.batch_buffer.add(copy.deepcopy(self.obs), action, logprob, reward, done, value)

		self.obs = last_obs
		self.episode_reward += reward
		# done = next_done
		if t % self.batch_size == 0:
			last_value = self.critic.value(torch.tensor(last_obs, dtype=torch.float32, device=self.device))
			# self.batch_buffer.compute_return(next_value, next_done, 0.99, 0.95)
			#return_batch = self.compute_return(reward_batch, value_batch, done_batch, last_value)
			self.batch_buffer.compute_return(last_value, self.gamma, self.lam)

			#batch = self.batch_buffer.get_batch()
			batch = self.batch_buffer.get_batch()
			self.train(*batch)

		if done:
			self.episode_end_handle(t)

if __name__ == '__main__':
	import gym
	try:
		import pybullet_envs
	except ImportError:
		print('Fail to import pybullet_envs')
	import envs

	env = gym.make('VisualReacherBulletEnv-v0')
	obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
	agent = PPO(obs_dim, act_dim)
	agent.run(env, 10000)
