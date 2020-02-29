import matplotlib.pyplot as plt
import numpy as np

segments = 100
env = 'Hopper'
name = f'PPO_{env}BulletEnv-v0'
num = 5

def compute(segments, name, num, truncate=False):
	result = np.zeros([num, segments], dtype=np.float32)
	for i in range(num):
		data = np.load(f'results/{name}_{i}.npy')
		l = int(len(data) // segments / 2) if truncate else len(data) // segments

		for j in range(segments):
			result[i, j] = np.mean(data[l * j : (j+1) * l])
			if j == segments - 1:
				print(data[l *j : (j+1) * l])
				print(np.mean(data[l * j: (j+1) * l]))
	return result



plt.title(name)
plt.xlabel('Every 3k time steps (300k in total)')
plt.ylabel('Episode returns')

result = compute(segments, name, num, True)
l = len(result)
mu = np.mean(result[:, :], axis=0)
sigma = np.std(result[:, :], axis=0)

plt.plot(mu, color='green')
plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='green')

name = f'PPO2_{env}BulletEnv-v0'

result = compute(segments, name, num)
mu = np.mean(result[:, :], axis=0)
sigma = np.std(result[:, :], axis=0)

plt.plot(mu, color='red')
plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='red')


plt.legend(['Yufeng\'s PPO', 'Original PPO'])




plt.show()

