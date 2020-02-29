import numpy as np
import torch
import gym


class ReplayBuffer:
	"""
	Batch Buffer used in batch policy-gradient methods like PPO or TRPO
	"""
	def __init__(self,
	             buffer_size,
	             batch_size,
	             obs_space,
	             act_space,
	             device=None):
		self.buffer_size = buffer_size
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
		self.cur_size = 0
		self.cur_idx = 0


		# handle the case where the observation space is a dictionary
		#if isinstance(obs_dim, dict):
		#	self.obses = {}
		#	for name, dim in obs_dim.items():
		#		self.obses[name] = np.zeros([buffer_size, *dim], dtype=np.float32)
		#else:
		self.obses = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
		self.actions = np.zeros([buffer_size, self.act_dim], dtype=np.float32)
		self.logprobs = np.zeros([buffer_size, 1], dtype=np.float32)
		self.rewards = np.zeros([buffer_size, 1], dtype=np.float32)
		self.dones = np.zeros([buffer_size, 1], dtype=np.int)
		self.returns = np.zeros([buffer_size, 1], dtype=np.float32)
		self.values = np.zeros([buffer_size, 1], dtype=np.float32)


	def __len__(self):
		return self.cur_size

	def add(self, obs, act, logp, rew, done, value):
		if self.cur_idx == self.buffer_size:
			self.cur_idx = 0
		#if isinstance(self.obs_dim, dict):
		#	for name, dim in self.obs_dim.items():
		#		self.obses[name][self.cur_idx] = obs[name]
		#else:
		self.obses[self.cur_idx] = obs
		self.actions[self.cur_idx] = act
		self.logprobs[self.cur_idx] = logp
		self.rewards[self.cur_idx] = rew
		self.dones[self.cur_idx] = done
		self.values[self.cur_idx] = value
		self.cur_size = min(self.cur_size + 1, self.buffer_size)
		self.cur_idx += 1

	def get_batch(self, batch_size=None):
		size = self.batch_size if batch_size is None else batch_size
		assert self.cur_size - size >= 0, 'Batch size is larger than current buffer size!'
		obses = torch.tensor(self.obses[:self.cur_size], dtype=torch.float32, device=self.device)
		actions = torch.tensor(self.actions[:self.cur_size], dtype=torch.float32, device=self.device)
		logprobs = torch.tensor(self.logprobs[:self.cur_size], dtype=torch.float32, device=self.device)
		returns = torch.tensor(self.returns[:self.cur_size], dtype=torch.float32, device=self.device)
		values = torch.tensor(self.values[:self.cur_size], dtype=torch.float32, device=self.device)
		self.cur_size = 0
		self.cur_idx = 0
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

















