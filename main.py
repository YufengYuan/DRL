import argparse
import gym
import os
import torch
import numpy as np
import copy
from common.utils import evaluate_agent
try:
	import pybullet_envs
except ImportError:
	print('Fail to import pybullet_envs!')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument("--network", default='mlp') # Network architecture to use (mlp, linear, CNN)
	parser.add_argument("--alg", default="PPO")  # Algorithms name (PPO, TD3, DDPG, SAC)
	parser.add_argument("--env", default="HopperBulletEnv-v0")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Seeds used for Gym, PyTorch and Numpy
	parser.add_argument("--total_timesteps", default=int(1e6), type=int)  # Total timesteps t5o train the agent
	parser.add_argument("--eval_freq", default=5e3, type=int)  # Evaluation frequency of the agent
	parser.add_argument("--save_model", action="store_true")  # Save model and optimizer or not
	parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--device", default="")  # Specify the device for training

	# Arguments for specific algorithms
	parser.add_argument("--alpha", default=0.5, type=float)

	args = parser.parse_args()


	file_name = f"{args.alg}_{args.env}_{args.seed}"

	kwargs = {
		'device': args.device
	}

	if args.alg == 'PPO':
		from algs.ppo import PPO as Agent
	elif args.alg == 'DDPG':
		from algs.ddpg import DDPG as Agent
	elif args.alg == 'TD3':
		from algs.td3 import TD3 as Agent
	elif args.alg == 'SAC':
		from algs.sac import SAC as Agent
		kwargs['alpha'] = args.alpha
	elif args.alg == 'SACE':
		from algs.sac_extra import SAC as Agent
		kwargs['alpha'] = args.alpha
	else:
		raise NotImplementedError(f'Algorithm {args.alg} is not implemented nor proposed yet.')

	file_name = f'{args.alg}_{args.env}_{args.seed}'
	print('----------------------------------------------------------')
	print(f'Algorithm: {args.alg}, Env: {args.env}, Seed: {args.seed}')
	print('----------------------------------------------------------')

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	# Create env and evaluation env
	env = gym.make(args.env)
	eval_env = copy.deepcopy(env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed + 100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# List to store evaluation result
	evaluation = []

	# Create the agent
	agent = Agent(env, **kwargs)

	# Main loop
	for t in range(args.total_timesteps):
		agent.step(t)
		if t % args.eval_freq == 0:
			evaluation.append(evaluate_agent(agent, eval_env))
			np.save(f"./results/{file_name}", evaluation)
	if args.save_model:
		agent.save(file_name)

