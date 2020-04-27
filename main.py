import argparse
import envs
import gym
import os
import torch
import numpy as np
import algs
try:
	import pybullet_envs
except ImportError:
	print('Fail to import pybullet_envs!')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument("--network", default='mlp') # Network architecture to use (mlp, linear, CNN)
	parser.add_argument("--alg", default="PPO")  # Algorithms name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="FixedReacherBulletEnv-v0")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--total_timesteps", default=int(1e6), type=int)  # Max time steps to run environment
	#parser.add_argument("--start_timesteps", default=1e3, type=int)  # Time steps initial random policy is used
	#parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
	#parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
	parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--device", default="cpu")  # Specify the device for training
	args = parser.parse_args()

	file_name = f"{args.alg}_{args.env}_{args.seed}"

	print("-------------------------------------")
	print(f"{args.alg}_{args.env}_{args.seed}")
	print("-------------------------------------")

	if args.alg == 'PPO':
		from algs.ppo_agent import PPO as Agent
	#elif args.alg == 'CNN_PPO':
	#	from cnn_ppo_agent import PPO as Agent
	else:
		raise NotImplementedError

	file_name = f"{args.alg}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.alg}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	agent = Agent(obs_dim, act_dim)
	returns = agent.run(env, args.total_timesteps)
	print(returns)
	np.save(f"./results/{file_name}", returns)

