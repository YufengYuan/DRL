import torch
from common import CNN, FC
from torch import nn
import envs
import gym
from baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm
import numpy as np
try:
	import pybullet_envs
except ImportError:
	pass
from torch.optim import Adam

class model(nn.Module):

	def __init__(self):
		super(model, self).__init__()
		self.cnn = CNN(h=110, w=110, outputs=16)
		self.fc = FC(in_dim=16, out_dim=2)
		self.to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

	def forward(self, x):
		return self.fc(self.cnn(x))

def create_env(num_envs=1):
	def env_fn():
		return gym.make('VisualReacherBulletEnv-v0')
	env = DummyVecEnv([env_fn for _ in range(num_envs)])
	return env

def save_dataset(iters=100000, n_envs=1):
	xs, ys = [], []
	env = create_env(n_envs)
	obs = env.reset()
	for i in tqdm(range(iters)):
		action = env.action_space.sample()
		obs, reward, done, info = env.step(action)
		for j in range(n_envs):
			xs.append(info[j]['image'])
			ys.append(info[j]['target'])
	np.save('images', np.asarray(xs, dtype=np.uint8))
	np.save('labels', np.asarray(ys, dtype=np.float32))

def pre_train_cnn():
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	xs = np.load('xs.npy')
	ys = np.load('ys.npy')
	cnn = model()
	losses = []
	print(np.mean(ys ** 2))
	optimizer = Adam(cnn.parameters(), 1e-4)
	for i in range(10000):
		idx = np.random.choice(np.arange(len(ys)), 1)
		x = torch.tensor(xs[idx], dtype=torch.float32, device=device)
		x = x.permute(0, 3, 1, 2)
		y = torch.tensor(ys[idx], dtype=torch.float32, device=device)
		y_pred = cnn(x)
		loss = torch.mean((y - y_pred) ** 2)
		print(f'Iteration {i} with loss {loss.item()}')
		losses.append(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	torch.save(cnn.state_dict(), 'pre_trained_cnn')
	np.save('losses', losses)

if __name__ == '__main__':
	#save_dataset(iters=50000, n_envs=1)
	pre_train_cnn()

















