

import numpy as np
import gym


class BaseRunner:

	def __init__(self, env_name, alg_name, random_seed=0):
		self.env_name = env_name
		self.alg_name = alg_name
		self.random_seed = random_seed
		self.ep_returns = []
		self.ep_lengths = []


	def run(self):
		raise NotImplementedError

	def record(self):
		file_name = f'{self.alg_name}_{self.env_name}_{self.random_seed}'
		np.save(f'results/{file_name}', self.ep_returns)
		#np.save(f'results/{file_name}_lengths', self.ep_lengths)

