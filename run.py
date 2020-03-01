from ppo import PPORunner
#from vae_ppo import PPORunner
import argparse
import envs




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--alg", default="PPO")  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e3, type=int)  # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--total_timesteps", default=int(3e5), type=int)  # Max time steps to run environment
	parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
	parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.alg}_{args.env}_{args.seed}"
	print(f"{args.alg}_{args.env}_{args.seed}")
	if args.alg == 'PPO':
		from ppo import PPORunner
		runner = PPORunner('FC', args.env, args.total_timesteps, seed=args.seed, device='cpu')
	else:
		runner = None
		raise NotImplementedError

	runner.run()
