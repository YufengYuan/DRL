import torch
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal



class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()

class FC(nn.Module):

	def __init__(self, in_dim, out_dim, h_dim):
		super(FC, self).__init__()
		assert isinstance(h_dim, list) and len(h_dim) >= 1
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.h_dim = h_dim
		self.h_layers = nn.ModuleList()
		for i in range(len(h_dim) - 1):
			self.h_layers.append(nn.Linear(h_dim[i], h_dim[i+1]))
		#self.h = nn.Linear(h_dim[0], h_dim[1])
		self.in2h = nn.Linear(in_dim, h_dim[0])
		self.h2out = nn.Linear(h_dim[-1], out_dim)
		self.h2out.weight.data.mul_(0.1)
		if hasattr(self.h2out, 'bias'):
			self.h2out.bias.data.mul_(0)
		#for params in self.parameters():
		#	print(params.shape)

	def forward(self, x):
		activ = torch.relu
		activ = torch.tanh
		h = activ(self.in2h(x))
		for layer in self.h_layers:
			h = activ(layer(h))
		#h = activ(self.h(h))
		return self.h2out(h)


class BaseModel:

	def __init__(self, obs_space, act_space, h_dim, device=None):
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
		self.h_dim = h_dim
		if device is None:
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = torch.device(device)


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
