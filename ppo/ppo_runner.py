import gym
from common import ReplayBuffer
from ppo import PPOAgent, PPOModel
from tqdm import tqdm
import torch
import numpy as np
from common import BaseRunner
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

	def __init__(self, env_name, total_timesteps, lr=3e-4, batch_size=2048, minibatch_size=64, gamma=0.99,
	             lam=0.95, vf_coef=1, ent_coef=0, clip_range=0.2, n_epochs=10, seed=0, max_grad_norm=0,
			     load_path=None, model_name=None, log_interval=10, save_interval=10, device=None,
			     **network_kwargs):

		super(PPORunner, self).__init__(env_name, 'PPO', seed)

		if device is None:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

		# TODO: support gym vectorized environments
		env = gym.make(env_name)
		# set the random seed on all environments
		env.seed(seed)
		np.random.seed(seed)
		random_state = np.random.get_state()
		torch_seed = np.random.randint(1, 2 ** 31 - 1)
		torch.manual_seed(torch_seed)
		torch.cuda.manual_seed_all(torch_seed)
		#torch.manual_seed(seed)
		self.model = PPOModel(env.observation_space, env.action_space, device=self.device, **network_kwargs)
		self.replay_buffer = ReplayBuffer(batch_size, batch_size, env.observation_space, env.action_space, device=device)
		self.agent = PPOAgent(self.model, lr, minibatch_size, clip_range, gamma, lam, n_epochs, True,
		                      vf_coef, ent_coef, max_grad_norm)
		self.total_timesteps = total_timesteps
		self.env = env
		self.batch_size = batch_size
		if load_path is not None:
			PPOModel.load(load_path)
		self.log_interval = log_interval
		self.save_interval = save_interval
		self.model_name = model_name

	def run(self):
		obs = self.env.reset()
		done = False
		ep_return = 0
		ep_length = 0
		for i in tqdm(range(1, self.total_timesteps+1)):
			action, log_prob, value = self.agent.act(torch.tensor(obs, dtype=torch.float32, device=self.device))
			next_obs, reward, next_done, _ = self.env.step(action)
			ep_return += reward
			ep_length += 1
			self.replay_buffer.add(obs, action, log_prob, reward, done, value)
			obs = next_obs
			done = next_done
			if i % self.batch_size == 0:
				next_value = self.agent.get_value(torch.tensor(next_obs, dtype=torch.float32, device=self.device)).data.cpu().numpy()
				self.replay_buffer.compute_return(next_value, next_done, 0.99, 0.95)
				batch = self.replay_buffer.get_batch()
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
			if i % int(self.batch_size * self.save_interval) == 0:
				self.record()
		self.record()
		if self.model_name is not None:
			self.model.save(self.model_name)



if __name__ == '__main__':
	runner = PPORunner(2048 * 100, 2048, 'Reacher-v2')
	runner.run()



















