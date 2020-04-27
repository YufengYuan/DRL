import sys
sys.path.append('../')

import torch
from torch.distributions import Normal, MultivariateNormal
from common.models import StochasticActor, ValueCritic
from torch import nn
from torch.optim import Adam
import numpy as np
from cnn_ppo.cnn_model import CNN


class PPO:

	def __init__(self, obs_dim, act_dim, lr=3e-4, batch_size=2048, num_minibatch=32, gamma=0.99,
	             lam=0.95, vf_coef=1, ent_coef=0, clip_range=0.2, n_epochs=10, max_grad_norm=10,
			     load_path=None, save_model=False, device=None, **network_kwargs):

		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.gamma = gamma
		self.lam = lam
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.n_minibatch = num_minibatch
		self.n_epoch = n_epochs
		self.clip_range = clip_range
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size
		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

		self.actor = StochasticActor(obs_dim, act_dim).to(self.device)
		self.actor_optimizer = Adam(self.actor.parameters(), lr)

		self.critic = ValueCritic(obs_dim).to(self.device)
		self.critic_optimizer = Adam(self.critic.parameters(), lr)

		self.ep_returns = []
		self.ep_lengths = []

		# Load the pretrained CNN model
		self.cnn = CNN(110, 110, 2).to(self.device)
		self.cnn.load_state_dict(torch.load("pretrained_cnn.pt"))



	def train(self, obses, actions, logprobs, returns, values):

		obses = torch.tensor(obses, dtype=torch.float32, device=self.device)
		actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
		logprobs = torch.tensor(logprobs, dtype=torch.float32, device=self.device)
		returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
		values = torch.tensor(values, dtype=torch.float32, device=self.device)
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
				entropy_loss = -torch.mean(cur_pi.entropy())
				policy_loss += self.ent_coef * entropy_loss

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


	def run(self, env, total_timesteps):
		obs = env.reset()
		#done = False
		ep_return = 0
		ep_length = 0

		obs_batch = np.zeros([self.batch_size, self.obs_dim], dtype=np.float32)
		action_batch = np.zeros([self.batch_size, self.act_dim], dtype=np.float32)
		logprob_batch = np.zeros([self.batch_size, 1], dtype=np.float32)
		reward_batch = np.zeros([self.batch_size, 1], dtype=np.float32)
		done_batch = np.zeros([self.batch_size, 1], dtype=np.int)
		#return_batch = np.zeros([self.batch_size, 1], dtype=np.float32)
		value_batch = np.zeros([self.batch_size, 1], dtype=np.float32)

		for i in range(1, total_timesteps + 1):

			# Recover the original observation with the CNN prediction
			obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
			img_tensor = torch.tensor(env.get_image(), dtype=torch.float32, device=self.device).unsqueeze_(0)
			img_tensor = img_tensor.permute(0, 3, 1, 2)
			obs_tensor[:2] -= self.cnn(img_tensor).squeeze_(0)

			action, logprob = self.actor.act(obs_tensor)
			value = self.critic.value(obs_tensor)
			last_obs, reward, done, _ = env.step(action)

			ep_return += reward
			ep_length += 1

			idx = i % self.batch_size
			obs_batch[idx] = obs
			action_batch[idx] = action
			logprob_batch[idx] = logprob
			reward_batch[idx] = reward
			done_batch[idx] = done
			value_batch[idx] = value

			obs = last_obs
			# done = next_done
			if i % self.batch_size == 0:

				# Recover the original observation space with the CNN prediction
				obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
				img_tensor = torch.tensor(env.get_image(), dtype=torch.float32, device=self.device).unsqueeze_(0)
				img_tensor = img_tensor.permute(0, 3, 1, 2)
				obs_tensor[:2] -= self.cnn(img_tensor).squeeze_(0)

				last_value = self.critic.value(obs_tensor)

				# self.batch_buffer.compute_return(next_value, next_done, 0.99, 0.95)
				return_batch = self.compute_return(reward_batch, value_batch, done_batch, last_value)
				#batch = self.batch_buffer.get_batch()
				loss_info = self.train(obs_batch, action_batch, logprob_batch, return_batch, value_batch)

			if done:
				obs = env.reset()
				self.ep_returns.append(ep_return)
				self.ep_lengths.append(ep_length)
				ep_return, ep_length = 0, 0
		return self.ep_returns#, self.ep_lengths


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
	agent.run(env, int(1e6))