import torch
import numpy as np
from torch.optim import Adam
from torch import nn
from baselines.common import explained_variance


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPOAgent:

	def __init__(self,
	             model,
	             lr=3e-4,
	             minibatch_size=256,
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
		self.lr = lr
		self.minibatch_size = minibatch_size
		self.clip_range = clip_range
		self.gamma = gamma
		self.lam = lam
		self.n_epochs = n_epochs
		self.normalize_adv = normalize_adv
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.max_grad_norm = max_grad_norm
		self.optimizer = Adam(self.model.parameters(), lr=self.lr)

	def act(self, obs):
		return self.model(obs)

	def get_value(self, obs):
		return self.model.critic(obs)

	def get_pi(self, obs):
		return self.model.pi(obs)

	def get_logprob(self, obs, action):
		return self.model.log_prob(obs, action)

	def train(self, batch):
		mean_policy_loss, mean_value_loss, mean_entropy_loss = 0, 0, 0
		clip_frac = 0; v= 0
		obses, actions, logprobs, returns, values = batch
		batch_size = len(returns)
		#assert batch_size % self.minibatch_size == 0, 'Minibatch size not correct!'
		advs = returns - values
		if self.normalize_adv:
			advs -= advs.mean()
			if advs.std() != 0 and not torch.isnan(advs.std()):
				advs /= advs.std()
		idxes = np.arange(batch_size)
		# TODO: early stop when the approximate KL divergence > 0.01
		for i in range(self.n_epochs):
			np.random.shuffle(idxes)
			for start in range(0, batch_size, self.minibatch_size):
				end = start + self.minibatch_size
				idx = idxes[start : end]
				# Compute value loss with mean square error
				cur_values = self.get_value(obses[idx])
				value_loss = torch.mean((returns[idx] - cur_values) ** 2)
				#xx = (returns[idx] - cur_values)**2
				#print(returns[idx].shape, cur_values.shape)
				# Compute policy loss with clipped objective
				log_pi = self.get_logprob(obses[idx], actions[idx])
				grad_pi = torch.exp(log_pi - logprobs[idx])
				#for x, y in zip(log_pi, logprobs[idx]):
				#	print(x, y)
				# TODO: check if this is correct
				policy_loss = -grad_pi * advs[idx]
				policy_loss_clipped = -torch.clamp(grad_pi, 1 - self.clip_range, 1 + self.clip_range) * advs[idx]
				policy_loss = torch.mean(torch.max(policy_loss, policy_loss_clipped))
				# Compute entropy loss
				#print(policy_loss)
				# TODO: optimize implementation for entropy loss
				entropy_loss = -torch.mean(self.model.pi(obses[idx]).entropy())
				# Compute total loss
				loss = policy_loss + self.vf_coef * value_loss# + self.ent_coef * entropy_loss
				clip_num = torch.sum(grad_pi < 1 - self.clip_range) + torch.sum(grad_pi > 1 + self.clip_range)
				clip_frac += clip_num.item() / self.minibatch_size
				v += explained_variance(cur_values.cpu().data.numpy().flatten(), returns[idx].data.cpu().numpy().flatten())
				mean_policy_loss += policy_loss.item()
				mean_value_loss += value_loss.item()
				mean_entropy_loss += entropy_loss.item()
				self.optimizer.zero_grad()
				loss.backward()
				#print(self.model.actor.h2out.weight)
				if self.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				#print('Actor')
				#for param in self.model.actor.parameters():
				#	print(torch.mean(param.grad))
				#print('Critic')
				#for param in self.model.critic.parameters():
				#	print(torch.mean(param.grad))

				self.optimizer.step()
		return {'Policy Loss': (mean_policy_loss/int(batch_size/self.minibatch_size)/self.n_epochs),
		        'Value Loss': (mean_value_loss/int(batch_size/self.minibatch_size)/self.n_epochs),
		        'Entropy Loss': (mean_entropy_loss/int(batch_size/self.minibatch_size)/self.n_epochs),
			    'Clip Frac': (clip_frac/int(batch_size/self.minibatch_size)/self.n_epochs),
		        'Explained V': (v /int(batch_size/self.minibatch_size)/self.n_epochs)
		        }

	def compute_return(self, rewards, values, dones, last_value, last_done):
		advs = np.zeros_like(rewards)
		batch_size = len(advs)
		lastgaelam = 0
		for t in reversed(range(batch_size)):
			if t == batch_size - 1:
				nextnonterminal = 1.0 - last_done
				nextvalues = last_value
			else:
				nextnonterminal = 1.0 - dones[t + 1]
				nextvalues = values[t + 1]
			delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
			advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
		return values + advs

