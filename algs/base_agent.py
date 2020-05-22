import torch
import copy
from common.logger import EpochLogger

class BaseAgent:

	def __init__(self, env, device=''):

		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]
		self.act_limit = float(env.action_space.high[0])
		self.env = env
		if device is '' or device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

		self.obs = self.env.reset()

		self.logger = EpochLogger()

		self.episode_timesteps = 0
		self.episode_reward = 0
		self.episode_num = 0

	def act(self, obs):
		pass

	def train(self, *batch):
		pass

	def step(self, t):
		pass

	def episode_end_handle(self, t):
		# TODO: add logger to log useful information
		print(
			f"Total T: {t + 1} Episode Num: {self.episode_num + 1} Episode T: {self.episode_timesteps} Reward: {self.episode_reward:.3f}")
		self.logger.store(EpisodeReturn=self.episode_reward)
		self.logger.store(EpisodeLength=self.episode_timesteps)

		# Reset environment
		self.obs = self.env.reset()
		self.episode_reward = 0
		self.episode_timesteps = 0
		self.episode_num += 1

	def evaluate(self, eval_env, eval_episodes=10):
		avg_reward = 0.
		avg_length = 0.
		for _ in range(eval_episodes):
			obs, done = eval_env.reset(), False
			while not done:
				action = self.act(obs)
				obs, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				avg_length += 1
			self.logger.store(EvalEpisodeReturn=avg_reward)
			self.logger.store(EvalEpisodeLength=avg_length)
		avg_reward /= eval_episodes
		avg_length /= eval_episodes
		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} length: {avg_length:.3f}")
		print("---------------------------------------")
		return avg_reward








	def save(self, filename):
		if hasattr(self, 'critic'):
			torch.save(self.critic.state_dict(), filename + "_critic")
		if hasattr(self, 'critic_optimizer'):
			torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		if hasattr(self, 'actor'):
			torch.save(self.actor.state_dict(), filename + "_actor")
		if hasattr(self, 'actor_optimizer'):
			torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		if hasattr(self, 'critic'):
			self.critic.load_state_dict(torch.load(filename + "_critic"))
			if hasattr(self, 'critic_target'):
				self.critic_target = copy.deepcopy(self.critic)
		if hasattr(self, 'critic_optimizer'):
			self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		if hasattr(self, 'actor'):
			self.actor.load_state_dict(torch.load(filename + "_actor"))
			if hasattr(self, 'actor_target'):
				self.actor_target = copy.deepcopy(self.actor)
		if hasattr(self, 'actor_optimizer'):
			self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
