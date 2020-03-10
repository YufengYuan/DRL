import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
segments = 100
env = 'VisualReacher'
#name = f'l00/PPO_{env}BulletEnv-v0'
#name = f'l09/PPO_Acrobot-v1'
num = 5

def compute(segments, name, num):
	result = np.zeros([num, segments], dtype=np.float32)
	for i in range(num):
		data = np.load(f'results/{name}_{i}.npy')
		l =  len(data) // segments
		#print(len(data))
		for j in range(segments):
			result[i, j] = np.mean(data[l * j : (j+1) * l])
			#if j == segments - 1:
			#	print(data[l *j : (j+1) * l])
			#	print(np.mean(data[l * j: (j+1) * l]))
	return result



#plt.title(env)
plt.xlabel('Every 10k time steps (1 million in total)')
plt.ylabel('Episode returns (averaged over 5 runs)')
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
env = 'Acrobot-v1'

name = f'l09/PPO_{env}'
result = compute(segments, name, num)
mu = np.mean(result[:, :], axis=0)
sigma = sem(result[:, :], axis=0)
ax.plot(mu, color='green')
ax.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='green')
#ax.text(70, 264, 'Lambda-return Actor-Critic', color='green')
ax.text(55, -78, 'Lambda-return Actor-Critic', color='green')

name = f'l10/PPO_{env}'
result = compute(segments, name, num)
mu = np.mean(result[:, :], axis=0)
sigma = sem(result[:, :], axis=0)
ax.plot(mu, color='blue')
ax.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='blue')
#ax.text(21, 264, 'REINFORCE with baseline', color='blue')
ax.text(55, -200, 'REINFORCE with baseline', color='blue')

name = f'l00/PPO_{env}'
result = compute(segments, name, num)
mu = np.mean(result[:, :], axis=0)
sigma = sem(result[:, :], axis=0)
ax.plot(mu, color='red')
ax.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='red')
#ax.text(78, 47, 'One-step Actor-Critic', color='red')
ax.text(55, -120, 'One-step Actor-Critic', color='red')


#mus, sigmas = [], []
#for i in range(1, 6):
#	name = f'l0{i*2-1}/PPO_{env}'
#	result = compute(segments, name, num)
#	mus.append(np.mean(result[:, -1], axis=0))
#	sigmas.append(sem(result[:, -1], axis=0))
#
#
#
#ax.errorbar(0.2 * np.arange(1, 6) - 0.1,
#            mus,
#            yerr=sigmas)
#







#plt.xticks(0.2 * np.arange(1, 6) - 0.1)
#plt.xlabel('Lambda values')
#plt.ylabel('Mean and standard error')





plt.show()

