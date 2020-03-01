

import numpy as np
import gym
import torch


class BaseRunner:

	def __init__(self, env_name, alg_name, seed=0, device=None):
		self.env_name = env_name
		self.alg_name = alg_name
		self.seed = seed
		self.ep_returns = []
		self.ep_lengths = []
		self.env = gym.make(env_name)

		self.env.seed(seed)
		np.random.seed(seed)
		random_state = np.random.get_state()
		torch_seed = np.random.randint(1, 2 ** 31 - 1)
		torch.manual_seed(torch_seed)
		torch.cuda.manual_seed_all(torch_seed)
		self.file_name = f'{self.alg_name}_{self.env_name}_{self.seed}'

		if device is None:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

	def run(self):
		raise NotImplementedError

	def save_returns(self):
		np.save(f'results/{self.file_name}', self.ep_returns)
		#np.save(f'results/{file_name}_lengths', self.ep_lengths)

