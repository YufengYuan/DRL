import gym
from common import BatchBuffer
from ppo import PPOAgent, PPOModel
from tqdm import tqdm
import torch
import numpy as np
from common import BaseRunner, FC
from baselines.common import explained_variance
from baselines.common.vec_env import SubprocVecEnv

try:
	import pybullet_envs
except ImportError:
	print('PyBullet environments not found!')

'''
def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
'''

class PPORunner(BaseRunner):

	def __init__(self, model_name, env_name, total_timesteps, lr=3e-4, batch_size=2048, num_minibatch=32, gamma=0.99,
	             lam=0.95, vf_coef=1, ent_coef=0, clip_range=0.2, n_epochs=10, seed=0, max_grad_norm=0,
			     load_path=None, log_interval=10, save_interval=10, device=None,
			     **network_kwargs):

		# TODO: support gym vectorized environments
		super(PPORunner, self).__init__(env_name, 'PPO', seed, device)

		# TODO: should be able to assign specific model to use
		model_name = FC
		model = PPOModel(model_name,
		                 lr,
		                 self.env.observation_space,
		                 self.env.action_space,
		                 device=self.device,
		                 **network_kwargs)

		self.agent = PPOAgent(model, num_minibatch, clip_range, gamma, lam, n_epochs, True,
		                      vf_coef, ent_coef, max_grad_norm)

		self.batch_buffer = BatchBuffer(batch_size,
		                                 self.env.observation_space,
		                                 self.env.action_space,
		                                 device=self.device)

		self.total_timesteps = total_timesteps
		self.batch_size = batch_size
		self.model = model

		if load_path is not None:
			self.load(load_path)

		self.log_interval = log_interval
		self.save_interval = save_interval
		self.model_name = model_name

	def run(self):
		obs = self.env.reset()
		#done = False
		ep_return = 0
		ep_length = 0
		for i in tqdm(range(1, self.total_timesteps+1)):
			action, log_prob, value = self.agent.act(torch.tensor(obs, dtype=torch.float32, device=self.device))
			last_obs, reward, done, _ = self.env.step(action)
			ep_return += reward
			ep_length += 1
			self.batch_buffer.add(obs, action, log_prob, reward, done, value)
			obs = last_obs
			#done = next_done
			if i % self.batch_size == 0:
				last_value = self.agent.get_value(torch.tensor(last_obs, dtype=torch.float32, device=self.device))
				#self.batch_buffer.compute_return(next_value, next_done, 0.99, 0.95)
				self.batch_buffer.returns[:] = self.agent.compute_return(self.batch_buffer.rewards,
				                                                         self.batch_buffer.values,
				                                                         self.batch_buffer.dones,
				                                                         last_value)
				batch = self.batch_buffer.get_batch()
				loss_info = self.agent.train(batch)

				# TODO: use logger to log the information needed
				for key, value in loss_info.items():
					print(key + ':' + str(value))
				print('Episode step: ' + str(np.mean(self.ep_lengths[-100:])))
				print('Episode Return:' + str(np.mean(self.ep_returns[-100:])))
			if done:
				obs = self.env.reset()
				self.ep_returns.append(ep_return)
				self.ep_lengths.append(ep_length)
				ep_return, ep_length = 0, 0
		self.save_returns()
		self.save_model()

	def save_model(self):
		torch.save(self.model.state_dict(), 'saved_models/' + self.file_name + '_model')
		torch.save(self.model.optimizer.state_dict(), 'saved__models/' + self.file_name + '_optimizer')

	def load_model(self, filename):
		self.model.load_state_dict(torch.load('saved_models/' + filename + '_model'))
		self.model.optimizer.load_state_dict(torch.load('saved_models/' + filename + '_optimizer'))


if __name__ == '__main__':
	runner = PPORunner(2048 * 100, 2048, 'Reacher-v2')
	runner.run()



















