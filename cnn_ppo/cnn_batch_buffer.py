from common import BatchBuffer
import numpy as np
import torch


class CNNBatchBuffer(BatchBuffer):
	"""
	Batch Buffer for PPO experiment, images will automatically convert to c-h-w format
	"""
	def __init__(self, batch_size, obs_space, act_space, h, w, c, device=None):
		super(CNNBatchBuffer, self).__init__(batch_size, obs_space, act_space, device)
		self.images = np.zeros([batch_size, h, w, c], dtype=np.float32)

	def add(self, obs, act, logp, rew, done, value, images):
		if self.cur_idx == self.batch_size:
			self.cur_idx = 0
		self.images[self.cur_idx] = images
		super(CNNBatchBuffer, self).add(obs, act, logp, rew, done, value)

	def get_batch(self, batch_size=None):
		obses, actions, logprobs, returns, values = super(CNNBatchBuffer, self).get_batch()
		images = torch.tensor(self.images, dtype=torch.float32, device=self.device)
		images =images.permute(0,3,1,2)

		return obses, images, actions, logprobs, returns, values

