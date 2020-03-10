import torch
import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from common import BaseModel, FC, CNN
from torch.optim import Adam

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
	             fixed=False,
	             **network_kwargs):
		super(PPOModel, self).__init__(obs_space, act_space)
		outputs = 16
		self.cnn = CNN(h=110, w=110, outputs=outputs)
		#if fixed:
		#	self.cnn.load_state_dict(torch.load('pre_trained_cnn'))
		self.actor = network(self.obs_dim + outputs, self.act_dim, **network_kwargs)
		self.critic = network(self.obs_dim + outputs, 1, **network_kwargs)
		self.optimizer = Adam(self.parameters(), lr=lr)
		self.to(device)
		self.fixed = fixed

	def value(self, x, img):
		if not self.fixed:
			return self.critic(torch.cat([x, self.cnn(img)], dim=1))
		else:
			return self.critic(torch.cat([x, self.cnn(img).detach()], dim=1))

	def pi(self, x, img):
		if self.act_type == 'Discrete':
			if not self.fixed:
				return Categorical(logits=self.actor(torch.cat([x, self.cnn(img)], dim=1)))
			else:
				return Categorical(logits=self.actor(torch.cat([x, self.cnn(img).detach()], dim=1)))
		elif self.act_type == 'Continuous':
			if not self.fixed:
				dist = MultivariateNormal(self.actor(torch.cat([x, self.cnn(img)], dim=1)),
			                          torch.diag_embed(torch.exp(self.log_std)))
			else:
				dist = MultivariateNormal(self.actor(torch.cat([x, self.cnn(img).detach()], dim=1)),
				                          torch.diag_embed(torch.exp(self.log_std)))
			return dist

	def log_prob(self, x, img, a):
		dist = self.pi(x, img)
		# Pytorch doesn't support sampling arbitrary dimension of batch from Categorical distribution
		if self.act_type == 'Discrete':
			a = a.flatten()
			return dist.log_prob(a).unsqueeze_(-1)
		else:
			return dist.log_prob(a).unsqueeze_(-1)

	def forward(self, x, img):
		dist = self.pi(x, img)
		action = dist.sample()
		logp = dist.log_prob(action)
		value = self.value(x, img)
		# TODO: this might not be compatible with vectorized environments
		return action.data.cpu().numpy().flatten(), \
		       logp.data.cpu().numpy().flatten(), \
		       value.data.cpu().numpy().flatten()

