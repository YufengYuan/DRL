import torch
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from common import BaseModel, FC, CNN

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

	def __init__(self, obs_space, act_space, h_dim=[64, 64], device=None):
		super(PPOModel, self).__init__(obs_space, act_space, h_dim, device)
		self.actor = FC(self.obs_dim, self.act_dim, h_dim)
		#net_init(self.actor.parameters())
		self.critic = FC(self.obs_dim, 1, h_dim)
		#net_init(self.critic.parameters())
		self.to(self.device)

	def value(self, x):
		return self.critic(x)

	def pi(self, x):
		if self.act_type == 'Discrete':
			return Categorical(logits=self.actor(x))
		elif self.act_type == 'Continuous':
			#return MultivariateNormal(F.tanh(self.actor(x)) * self.act_lim, torch.diag_embed(torch.exp(self.log_std)))
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

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.optimizer.state_dict(), filename + "_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
