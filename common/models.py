import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class StochasticActor(nn.Module):

	def __init__(self, obs_dim, act_dim):
		super(StochasticActor, self).__init__()

		self.l1 = nn.Linear(obs_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, act_dim)

		self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))


	def forward(self, obs):
		h = F.relu(self.l1(obs))
		h = F.relu(self.l2(h))
		h = self.l3(h)
		dist = MultivariateNormal(h, torch.diag_embed(torch.exp(self.log_std)))
		return dist

	def act(self, obs):
		dist = self.forward(obs)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()


class ValueCritic(nn.Module):

	def __init__(self, obs_dim):
		super(ValueCritic, self).__init__()
		self.l1 = nn.Linear(obs_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, obs):
		h = F.relu(self.l1(obs))
		h = F.relu(self.l2(h))
		return self.l3(h)

	def value(self, obs):
		return self.forward(obs).detach().cpu().data.numpy().flatten()


class DeterministicActor:
	pass


class QvalueFunction:
	pass
