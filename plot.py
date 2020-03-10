import matplotlib.pyplot as plt
import numpy as np

segments = 100
env = 'VisualReacher'
#name = f'l00/PPO_{env}BulletEnv-v0'
#name = f'l09/PPO_Acrobot-v1'
num = 3

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


plt.title('FixedReacher with partial target information')
plt.xlabel('Every 10k time steps (1 million in total)')
plt.ylabel('Episode returns (averaged over 5 runs)')

env = 'FixedReacherBulletEnv-v0'
name = f'backup_/PPO_{env}'
result = compute(segments, name, num)
l = len(result)
mu = np.mean(result[:, :], axis=0)
sigma = np.std(result[:, :], axis=0)
plt.plot(mu, color='green')
plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='green')

env = 'FixedReacherBulletEnv-v0'
name = f'noxy/PPO_{env}'
result = compute(segments, name, num)
l = len(result)
mu = np.mean(result[:, :], axis=0)
mu = result[1]
sigma = np.std(result[:, :], axis=0)
plt.plot(mu, color='blue')
plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='blue')

env = 'FixedReacherBulletEnv-v0'
name = f'notxty/PPO_{env}'
result = compute(segments, name, num)
l = len(result)
mu = np.mean(result[:, :], axis=0)
sigma = np.std(result[:, :], axis=0)
plt.plot(mu, color='red')
plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='red')

#name = f'l03/PPO_{env}'
#result = compute(segments, name, num)
#l = len(result)
#mu = np.mean(result[:, :], axis=0)
#sigma = np.std(result[:, :], axis=0)
#plt.plot(mu, color='orange')
#plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='orange')
#
#name = f'l01/PPO_{env}'
#result = compute(segments, name, num)
#l = len(result)
#mu = np.mean(result[:, :], axis=0)
#sigma = np.std(result[:, :], axis=0)
#
#plt.plot(mu, color='purple')
#plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='purple')

#name = f'PPO_FixedReacherBulletEnv-v0'
#
#result = compute(segments, name, num)
#mu = np.mean(result[:, :], axis=0)
#sigma = np.std(result[:, :], axis=0)
#
#plt.plot(mu, color='red')
#plt.fill_between(np.arange(segments), mu-sigma, mu+sigma, alpha=0.2, color='red')
#
#
#plt.legend(['lambda=0.9', 'lambda=0.7', 'lambda=0.5', 'lambda=0.3', 'lambda=0.1'])
plt.legend(['Original observation', 'No target coordinate', 'No target-tip difference'])

plt.show()

