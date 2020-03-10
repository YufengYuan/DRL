import torch
import numpy as np
from torch.optim import Adam
from torch import nn
from baselines.common import explained_variance


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPOAgent:

	def __init__(self,
	             model,
	             num_minibatch=32,
	             clip_range=0.2,
	             gamma=0.99,
	             lam=0.95,
	             n_epochs=10,
	             normalize_adv=True,
	             vf_coef=0.5,
	             ent_coef=0,
	             max_grad_norm=10
				):

		self.model = model
		self.num_minibatch = num_minibatch
		self.clip_range = clip_range
		self.gamma = gamma
		self.lam = lam
		self.n_epochs = n_epochs
		self.normalize_adv = normalize_adv
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.max_grad_norm = max_grad_norm

	def act(self, *obs):
		return self.model(*obs)

	def get_value(self, *obs):
		return self.model.value(*obs).data.cpu().numpy()

	def get_pi(self, *obs):
		return self.model.pi(*obs).data.cpu().numpy()

	def get_logprob(self, *obs, act):
		return self.model.log_prob(*obs, act).data.cpu().numpy()

	def train(self, batch):
		mean_policy_loss, mean_value_loss, mean_entropy_loss = 0, 0, 0
		clip_frac = 0; v= 0
		*obses, actions, logprobs, returns, values = batch
		batch_size = len(returns)
		#assert batch_size % self.minibatch_size == 0, 'Minibatch size not correct!'
		advs = returns - values
		if self.normalize_adv:
			advs -= advs.mean()
			if advs.std() != 0 and not torch.isnan(advs.std()):
				advs /= advs.std()
		idxes = np.arange(batch_size)
		# TODO: early stop when the approximate KL divergence > 0.01
		minibatch_size = batch_size // self.num_minibatch
		for i in range(self.n_epochs):
			np.random.shuffle(idxes)
			for start in range(0, batch_size, minibatch_size):
				end = start + minibatch_size
				idx = idxes[start : end]
				# Compute value loss with mean square error
				cur_values = self.model.value(*[obs[idx] for obs in obses])
				value_loss = torch.mean((returns[idx] - cur_values) ** 2)
				# Compute policy loss with clipped objective
				log_pi = self.model.log_prob(*[obs[idx] for obs in obses], actions[idx])
				grad_pi = torch.exp(log_pi - logprobs[idx])
				# TODO: check if this is correct
				policy_loss = -grad_pi * advs[idx]
				policy_loss_clipped = -torch.clamp(grad_pi, 1 - self.clip_range, 1 + self.clip_range) * advs[idx]
				policy_loss = torch.mean(torch.max(policy_loss, policy_loss_clipped))
				# Compute entropy loss
				# TODO: optimize implementation for entropy loss
				entropy_loss = -torch.mean(self.model.pi(*[obs[idx] for obs in obses]).entropy())
				# Compute total loss
				loss = policy_loss + self.vf_coef * value_loss# + self.ent_coef * entropy_loss
				clip_num = torch.sum(grad_pi < 1 - self.clip_range) + torch.sum(grad_pi > 1 + self.clip_range)
				clip_frac += clip_num.item() / minibatch_size
				v += explained_variance(cur_values.cpu().data.numpy().flatten(), returns[idx].data.cpu().numpy().flatten())
				mean_policy_loss += policy_loss.item()
				mean_value_loss += value_loss.item()
				mean_entropy_loss += entropy_loss.item()
				self.model.optimizer.zero_grad()
				loss.backward()
				if self.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				self.model.optimizer.step()
				# TODO: use logger to record the data here

		return {'Policy Loss': (mean_policy_loss/int(batch_size/minibatch_size)/self.n_epochs),
		        'Value Loss': (mean_value_loss/int(batch_size/minibatch_size)/self.n_epochs),
		        'Entropy Loss': (mean_entropy_loss/int(batch_size/minibatch_size)/self.n_epochs),
			    'Clip Frac': (clip_frac/int(batch_size/minibatch_size)/self.n_epochs),
		        'Explained V': (v /int(batch_size/minibatch_size)/self.n_epochs)
		        }

	def compute_return(self, rewards, values, dones, last_value):
		#last_value = self.get_value(torch.tensor(last_obs, dtype=torch.float32, device=self.device))
		advs = np.zeros_like(rewards)
		batch_size = len(advs)
		lastgaelam = 0
		for t in reversed(range(batch_size)):
			if t == batch_size - 1:
				nextvalues = last_value
			else:
				nextvalues = values[t + 1]
			nextnonterminal = 1.0 - dones[t]
			delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
			advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
		return values + advs



