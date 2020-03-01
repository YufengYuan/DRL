import numpy as np
import torch
import gym


class BatchBuffer:
	"""
	Batch Buffer used in batch policy-gradient methods like PPO or TRPO
	"""
	def __init__(self,
	             batch_size,
	             obs_space,
	             act_space,
	             device=None):
		self.batch_size = batch_size
		self.obs_dim = obs_space.shape[0]
		if isinstance(act_space, gym.spaces.Discrete):
			self.act_dim = 1
		elif isinstance(act_space, gym.spaces.Box):
			self.act_dim = act_space.shape[0]
		else:
			raise NotImplementedError
		if device is not None:
			self.device = torch.device(device)
		else:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.cur_idx = 0

		self.obses = np.zeros([batch_size, self.obs_dim], dtype=np.float32)
		self.actions = np.zeros([batch_size, self.act_dim], dtype=np.float32)
		self.logprobs = np.zeros([batch_size, 1], dtype=np.float32)
		self.rewards = np.zeros([batch_size, 1], dtype=np.float32)
		self.dones = np.zeros([batch_size, 1], dtype=np.int)
		self.returns = np.zeros([batch_size, 1], dtype=np.float32)
		self.values = np.zeros([batch_size, 1], dtype=np.float32)


	def __len__(self):
		return self.cur_idx

	def add(self, obs, act, logp, rew, done, value):
		if self.cur_idx == self.batch_size:
			self.cur_idx = 0
		self.obses[self.cur_idx] = obs
		self.actions[self.cur_idx] = act
		self.logprobs[self.cur_idx] = logp
		self.rewards[self.cur_idx] = rew
		self.dones[self.cur_idx] = done
		self.values[self.cur_idx] = value
		self.cur_idx += 1

	def get_batch(self, batch_size=None):
		size = self.batch_size if batch_size is None else batch_size
		obses = torch.tensor(self.obses, dtype=torch.float32, device=self.device)
		actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
		logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=self.device)
		returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
		values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
		return obses, actions, logprobs, returns, values

	def compute_return(self, next_value, next_done, gamma=0.99, lam=0.95):
		advs = np.zeros_like(self.rewards)
		lastgaelam = 0
		for t in reversed(range(self.batch_size)):
			if t == self.batch_size - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - self.dones[t + 1]
				nextvalues = self.values[t + 1]
			delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
			advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
		self.returns[:] = self.values + advs

















