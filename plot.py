
import numpy as np
from matplotlib import pyplot as plt


segments = 201
env = 'HumanoidBulletEnv-v0'
num = 5

def one_key_plot(ax,
                 env_name='HalfCheetahBulletEnv-v0',
                 #policy_name=['TD3/TD3', 'TD4/TD4', 'TD5/TD5', 'TD6/TD6'],
                 policy_name=['SAC'],
                 policy_color=['orange', 'green', 'blue', 'red', 'black'],
                 seeds=[0,1,2,3,4]):
	num = len(seeds)
	ax.set_title(env_name)
	for policy, color in zip(policy_name, policy_color):
		results = np.zeros([num, 1000], dtype=np.float32)
		min_len = 1000
		for i in range(num):
			try:
				data = np.load(f'results/{policy}_{env_name}_{seeds[i]}.npy')
			except FileNotFoundError:
				continue
			results[i, :len(data)] = data[:1000]
			min_len = min(min_len, len(data))
			print(env_name, len(data), results[i, len(data)-2])
		if 'TD4/TD4' in policy and 'Ant' in env_name:
			mu = np.mean(results[:4, :min_len], axis=0)
			sigma = np.std(results[:4, :min_len], axis=0)
			#mu = results[4, :min_len]
			#sigma = np.zeros_like(results[4, :min_len])
		else:
			mu = np.mean(results[:, :min_len], axis=0)
			sigma = np.std(results[:, :min_len], axis=0)
		ax.plot(mu, color=color)
		ax.fill_between(np.arange(len(mu)), mu-sigma, mu+sigma, color=color, alpha=0.2)
		ax.legend(policy_name)
		ax.grid(True)
		ax.set_ylabel('average return')
		ax.set_xlabel('million steps')


fig, ax = plt.subplots(2, 3)

one_key_plot(ax[0, 0], 'ball_in_cup@catch', ['SAC', 'EXP'])
one_key_plot(ax[0, 1], 'cartpole@swingup', ['SAC', 'EXP'])
one_key_plot(ax[0, 2], 'cheetah@run', ['SAC', 'EXP'])
one_key_plot(ax[1, 0], 'finger@spin', ['SAC', 'EXP'])
one_key_plot(ax[1, 1], 'reacher@easy', ['SAC', 'EXP'])
one_key_plot(ax[1, 2], 'walker@walk', ['SAC', 'EXP'])

plt.show()
