import sys
sys.path.append('../')

import torch
from cnn_model import CNN
from torch.optim import Adam
import numpy as np


def pre_train_cnn():
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	xs = np.load('images.npy') / 255.
	ys = np.load('labels.npy')
	n_epochs = 100
	# segment training set and test set
	train_xs = xs[:45000, :, :, :]
	train_ys = ys[:45000]
	test_xs = xs[45000:, :, :, :]
	test_ys = ys[45000:]
	cnn = CNN(110, 110, 2).to(device)
	losses = []
	optimizer = Adam(cnn.parameters(), 1e-4)
	idxes  = np.arange(len(train_ys))
	for i in range(n_epochs):
		np.random.shuffle(idxes)
		#idxes = np.random.shuffle(np.arange(len(train_ys)))
		training_loss = []
		for j in range(1000):
			idx = idxes[i * 45 : (i + 1) * 45]
			x = torch.tensor(xs[idx], dtype=torch.float32, device=device)
			x = x.permute(0, 3, 1, 2)
			y = torch.tensor(ys[idx], dtype=torch.float32, device=device)
			y_pred = cnn(x)
			loss = torch.mean(torch.sum((y - y_pred) ** 2, dim=1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			training_loss.append(loss.item())
		# Test the loss every epoch
		x = torch.FloatTensor(test_xs).to(device)
		x = x.permute(0, 3, 1, 2)
		y_pred = cnn(x)
		y = torch.FloatTensor(test_ys).to(device)
		#print(y[0, :], y_pred[0, :])
		test_loss = torch.mean(torch.sum((y - y_pred    )**2, dim=1))
		print(f'Iteration {i}: Testing loss: {test_loss.item()}, training loss {np.mean(training_loss)}')
		if len(losses) > 0 and test_loss.item() < losses[-1]:
			torch.save(cnn.state_dict(), 'pretrained_cnn.pt')
		losses.append(test_loss.item())
		np.save('losses', losses)

if __name__ == '__main__':
	pre_train_cnn()