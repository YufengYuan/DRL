import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=4)
        #self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        #self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(16)
        #self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 3):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 5, 4), 3, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 5, 4), 3, 2), 3, 1)
        linear_input_size = convw * convh * 128
        self.l1 = nn.Linear(linear_input_size, 256)
        self.l2 = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.l1(x.contiguous().view(x.size(0), -1)))
        return self.l2(x)
        #return self.head(x.contiguous().view(x.size(0), -1))