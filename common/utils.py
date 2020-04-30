import torch

def ensure2d(tensor):
	if len(tensor.shape) == 2:
		return tensor
	elif len(tensor.shape) == 1:
		return tensor.unsqueeze_(0)
	elif len(tensor.shape) == 3:
		return tensor.squeeze_(0)
	else:
		return tensor

def evaluate_agent(agent, eval_env, eval_episodes=10):
	avg_reward = 0.
	avg_length = 0.
	for _ in range(eval_episodes):
		obs, done = eval_env.reset(), False
		while not done:
			action = agent.act(obs)
			obs, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			avg_length += 1
	avg_reward /= eval_episodes
	avg_length /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} length: {avg_length:.3f}")
	print("---------------------------------------")
	return avg_reward