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

	def __init__(self, in_dim, out_dim, h_dim=[4]):
		super(FC, self).__init__()
		assert isinstance(h_dim, list) and len(h_dim) >= 1
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.h_dim = h_dim
		self.h_layers = []
		for i in range(len(h_dim) - 1):
			self.h_layers.append(nn.Linear(h_dim[i], h_dim[i+1]))
		self.in2h = nn.Linear(in_dim, h_dim[0])
		self.h2out = nn.Linear(h_dim[-1], out_dim)

	def forward(self, x):
		activ = F.relu
		activ = lambda x: x
		h = activ(self.in2h(x))
		for layer in self.h_layers:
			h = activ(layer(h))
		return self.h2out(h)


class BaseModel(nn.Module):

	def __init__(self):
		super(BaseModel, self).__init__()

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

class QModel(nn.Module):

	def __init__(self):
		pass


class ActorCritic(BaseModel):

	def __init__(self, obs_space, act_space, h_dim, device=None):
		super(ActorCritic, self).__init__()
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
		# TODO: should clip action here?
		self.actor = FC(self.obs_dim, self.act_dim, h_dim)
		self.critic = FC(self.obs_dim, 1, h_dim)
		if device is None:
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(device)
		self.device = device
	def value(self, x):
		return self.critic(x)

	def pi(self, x):
		if self.act_type == 'Discrete':
			dist = Categorical(logits=self.actor(x))
		elif self.act_type == 'Continuous':
			dist = MultivariateNormal(F.tanh(self.actor(x)) * self.act_lim, torch.diag_embed(torch.exp(self.log_std)))
		return dist

	def log_prob(self, x, a):
		dist = self.pi(x)
		# Pytorch doesn't support sampling arbitrary dimension of batch from Categorical distribution
		if self.act_type == 'Discrete':
			a = a.flatten()
			return dist.log_prob(a).unsqueeze_(-1)
		else:
			return dist.log_prob(a).unsqueeze_(-1)

	def forward(self, x):
		dist = self.pi(x)
		action = dist.sample()
		logp = dist.log_prob(action)
		value = self.value(x)
		return action.data.cpu().numpy(), logp.data.cpu().numpy(), value.data.cpu().numpy()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.optimizer.state_dict(), filename + "_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))









