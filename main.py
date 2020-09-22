import argparse
import gym
import os
import torch
import numpy as np
import copy
import envs
from envs.env_builder import make_env


try:
	import pybullet_envs
except ImportError:
	print('Fail to import pybullet_envs!')

try:
	import d4rl
except ImportError:
	print('Fail to import d4rl!')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--h_dim", default='(256, 256)') # Network architecture to use (mlp, linear, CNN)
	parser.add_argument("--alg", default="PPO")  # Algorithms name (PPO, TD3, DDPG, SAC)
	parser.add_argument("--env", default="HopperBulletEnv-v0")  # OpenAI gym environment name
	parser.add_argument("--device", default="")  # Specify the device for training
	parser.add_argument("--seed", default=9, type=int)  # Seeds used for Gym, PyTorch and Numpy
	parser.add_argument("--total_timesteps", default=int(1e4), type=int)  # Total timesteps t5o train the agent
	parser.add_argument("--eval_freq", default=5e3, type=int)  # Evaluation frequency of the agent
	parser.add_argument("--save_model", action="store_true")  # Save model and optimizer or not
	parser.add_argument("--load_model", action="store_true")  # Load the saved model and optimizer or not
<<<<<<< HEAD
	parser.add_argument("--save_data", default="")  # Specify the device for training
	parser.add_argument("--load_data", default="")  # Specify the device for training

=======
	parser.add_argument("--device", default="")  # Specify the device for training
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf

	args = parser.parse_args()


	file_name = f"{args.alg}_{args.env}_{args.seed}"

	kwargs = {
		'device': None if args.device == "" else args.device,
		#'h_dim': eval(args.h_dim)
	}
	if args.alg == 'Random':
		from algs.random import Random as Agent
	elif args.alg == 'PPO':
		from algs.ppo import PPO as Agent
	elif args.alg == 'DDPG':
		from algs.ddpg import DDPG as Agent
	elif args.alg == 'TD3':
		from algs.td3 import TD3 as Agent
	elif args.alg == 'SAC':
		from algs.sac import SAC as Agent
	# Temparily for experimental code
<<<<<<< HEAD
	elif args.alg == 'CQL':
		from experimental.ctd3 import TD3 as Agent
	elif args.alg == 'OFFLINE':
		from experimental.offline import Custom as Agent
=======
	elif args.alg == 'EXP':
		from experimental.sac import SAC as Agent
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf
	else:
		raise NotImplementedError(f'Algorithm {args.alg} is not implemented.')

	file_name = f'{args.alg}_{args.env}_{args.seed}'
	print('----------------------------------------------------------')
	print(f'Algorithm: {args.alg}, Env: {args.env}, Seed: {args.seed}')
	print('----------------------------------------------------------')

	if not os.path.exists(f"./results/"):
		os.makedirs(f"./results/")

	if args.save_model and not os.path.exists(f"./models/"):
		os.makedirs(f"./models/")

	# Create env and evaluation env
	#env = gym.make(args.env)
	#eval_env = copy.deepcopy(env)
	env = make_env(args.env)
	eval_env = make_env(args.env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed + 100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# List to store evaluation result
	evaluation = []

	# Create the agent
	agent = Agent(env, **kwargs)
	if args.load_model:
		agent.save(f"./models/{file_name}")
<<<<<<< HEAD

	# Test
	if args.alg in ['CQL', 'OFFLINE', 'NAF']:
		#from experimental.replay_buffer import ReplayBuffer
		#dataset = ReplayBuffer(agent.obs_dim, agent.act_dim, max_size=int(1e6))
		#dataset.load('datasets/')
		#agent.offline_initialize(dataset)
		#agent.offline_improve(dataset)
		dataset = env.get_dataset()
		N = dataset['rewards'].shape[0]
		agent = Agent(env, buffer_size=N, **kwargs)
		print('Loading buffer!')
		for i in range(N - 1):
			obs = dataset['observations'][i]
			new_obs = dataset['observations'][i + 1]
			action = dataset['actions'][i]
			reward = dataset['rewards'][i]
			done_bool = bool(dataset['terminals'][i])
			agent.replay_buffer.add(obs, action, reward, new_obs, done_bool)
		print('Loaded buffer')
		#agent.replay_buffer.load(f'datasets/{args.env}_{args.load_data}')
=======
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf

	# Main loop
	for t in range(args.total_timesteps):
		agent.step(t)
		if t % args.eval_freq == 0:
			eval = agent.evaluate(eval_env)
			evaluation.append(eval)
			#evaluation.append(evaluate_agent(agent, eval_env))
			np.save(f"./results/{file_name}", evaluation)
<<<<<<< HEAD
	if args.save_data:
		agent.save_buffer(f"./datasets/{args.env}_{args.save_data}")

	if args.save_model:
		agent.save(f"./models/{file_name}")

=======
	if args.save_model:
		agent.save(f"./models/{file_name}")
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf

