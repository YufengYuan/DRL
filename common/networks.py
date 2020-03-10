import torch
from torch import nn

def get_network(name):
	if name == 'mlp':
		return FC
	elif name == 'linear':
		return Linear
	elif name == 'CNN':
		return CNN
	else:
		raise NotImplementedError





class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 3):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return self.head(x.contiguous().view(x.size(0), -1))

#class CNN(nn.Module):
#
#	def __init__(self):
#		super(CNN, self).__init__()

class Linear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.layer = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		return self.layer(x)


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
