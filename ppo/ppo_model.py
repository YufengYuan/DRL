import torch
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from common import BaseModel, FC, CNN
from torch.optim import Adam, SGD

def net_init(net, orth=1, w_fac=0.1, b_fac=0.0):
	if orth:
		for module in net:
			if hasattr(module, 'weight'):
				nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
			if hasattr(module, 'bias'):
				nn.init.constant_(module.bias, val=0)
	else:
		net[-1].weight.data.mul_(w_fac)
		print(net[-1].weight.shape)
		if hasattr(net[-1], 'bias'):
			net[-1].bias.data.mul_(b_fac)

class PPOModel(BaseModel, nn.Module):

	def __init__(self,
	             network,
	             lr,
	             obs_space,
	             act_space,
	             device,
	             **network_kwargs):
		super(PPOModel, self).__init__(obs_space, act_space)
		self.actor = network(self.obs_dim, self.act_dim, **network_kwargs)
		self.critic = network(self.obs_dim, 1, **network_kwargs)
		self.optimizer = Adam(self.parameters(), lr)#Adam(self.parameters(), lr=lr)
		self.to(device)

	def value(self, x):
		return self.critic(x)

	def pi(self, x):
		if self.act_type == 'Discrete':
			return Categorical(logits=self.actor(x))
		elif self.act_type == 'Continuous':
			dist = MultivariateNormal(self.actor(x), torch.diag_embed(torch.exp(self.log_std)))
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

