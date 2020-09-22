import os
from envs.env_builder import make_env
from experimental.replay_buffer import ReplayBuffer
import numpy as np


env_name = 'Ant-v2'

env = make_env(env_name)
buffer_size = int(1e6)
replay_buffer = ReplayBuffer(env.observation_space.shape[0],
                             env.action_space.shape[0],
                             max_size=buffer_size)


state = env.reset()
done = False
print(env.observation_space, env.action_space)
for i in range(buffer_size):
    action = env.action_space.sample()
    #action = np.tanh(state[:env.action_space.shape[0]])
    #action = np.tanh(0.3 + 0.1 * np.random.randn(env.action_space.shape[0]))
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
    if done:
        state = env.reset()
        done = False
    else:
        state = next_state

if not os.path.exists('datasets/'):
    os.makedirs('datasets/')

replay_buffer.save(f'datasets/{env_name}_random')




