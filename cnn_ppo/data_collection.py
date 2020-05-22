import sys
sys.path.append('../')

from tqdm import tqdm
import gym
import envs
import numpy as np


def save_dataset(size=200):
	xs, ys = [], []
	env = gym.make('VisualReacherBulletEnv-v0')
	obs, done = env.reset(), False
	for i in tqdm(range(size)):
		# Collect data by random action
		action = env.action_space.sample()
		obs, reward, done, info = env.step(action)
		xs.append(info['image'])
		ys.append(info['target'])
		if done or i % 10 == 0:
			obs, done = env.reset(), False
	np.save('images', np.asarray(xs, dtype=np.uint8))
	np.save('labels', np.asarray(ys, dtype=np.float32))

if __name__ == '__main__':
	save_dataset()
