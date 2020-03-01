import torch
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal





class BaseModel:

	def __init__(self, obs_space, act_space):
		super(BaseModel, self).__init__()
		self.obs_dim = obs_space.shape[0]
		if isinstance(act_space, gym.spaces.Discrete):
			self.act_dim = act_space.n
			self.act_type = 'Discrete'
		elif isinstance(act_space, gym.spaces.Box):
			self.act_dim = act_space.shape[0]
			self.log_std = nn.Parameter(torch.zeros(self.act_dim, dtype=torch.float32))
			assert np.abs(act_space.low).all() == np.abs(act_space.high).all(), 'Action lower range and higher range are different!'
			self.act_lim = torch.tensor(act_space.high, dtype=torch.float32)
			self.act_type = 'Continuous'
		else:
			raise NotImplementedError
		#self.h_dim = h_dim
		#if device is None:
		#	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		#self.device = torch.device(device)


	def value(self, *args, **kwargs):
		raise NotImplementedError

	def q_value(self, *args, **kwargs):
		raise NotImplementedError

	def pi(self, *args, **kwargs):
		raise NotImplementedError

	def log_prob(self, *args, **kwargs):
		raise NotImplementedError

	def save(self, *args, **kwargs):
		raise NotImplementedError

	def load(self, *args, **kwargs):
		raise NotImplementedError
