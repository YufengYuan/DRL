import torch
from torch import nn



class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()

class FC(nn.Module):

	def __init__(self, in_dim, out_dim, h_dim=[64, 64]):
		super(FC, self).__init__()
		assert isinstance(h_dim, list) and len(h_dim) >= 1
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.h_dim = h_dim
		self.h_layers = nn.ModuleList()
		for i in range(len(h_dim) - 1):
			self.h_layers.append(nn.Linear(h_dim[i], h_dim[i+1]))
		self.in2h = nn.Linear(in_dim, h_dim[0])
		self.h2out = nn.Linear(h_dim[-1], out_dim)
		self.h2out.weight.data.mul_(0.1)
		if hasattr(self.h2out, 'bias'):
			self.h2out.bias.data.mul_(0)

	def forward(self, x):
		activ = torch.relu
		activ = torch.tanh
		h = activ(self.in2h(x))
		for layer in self.h_layers:
			h = activ(layer(h))
		return self.h2out(h)
