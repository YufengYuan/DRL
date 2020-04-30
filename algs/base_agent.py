import torch
import copy

class BaseAgent:

	def __init__(self, env, device=None):

		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]
		self.act_limit = float(env.action_space.high[0])
		self.env = env
		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

		self.obs = self.env.reset()

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
		print(
			f"Total T: {t + 1} Episode Num: {self.episode_num + 1} Episode T: {self.episode_timesteps} Reward: {self.episode_reward:.3f}")
		# Reset environment
		self.obs = self.env.reset()
		self.episode_reward = 0
		self.episode_timesteps = 0
		self.episode_num += 1

	def save(self, filename):
		if hasattr(self, 'critic'):
			torch.save(self.critic.state_dict(), filename + "_critic")
			torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		if hasattr(self, 'actor'):
			torch.save(self.actor.state_dict(), filename + "_actor")
			torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		if hasattr(self, 'critic'):
			self.critic.load_state_dict(torch.load(filename + "_critic"))
			self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)
		if hasattr(self, 'actor'):
			self.actor.load_state_dict(torch.load(filename + "_actor"))
			self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)