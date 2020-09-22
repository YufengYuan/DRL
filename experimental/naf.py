import sys
import os
from algs.base import BaseAgent
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from common.replay_buffer import ReplayBuffer
import numpy as np

def MSELoss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)
        self.tril_mask = Variable(torch.tril(torch.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0)).to(device)
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs))).unsqueeze(0)).to(device)

    def forward(self, inputs):
        x, u = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        mu = F.tanh(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF(BaseAgent):

    def __init__(self, env, gamma=0.99, tau=0.005, hidden_size=256, device=None):
        super(NAF, self).__init__(env, device=None)
        self.action_space = self.act_dim
        self.num_inputs = self.obs_dim
        num_inputs = self.obs_dim
        action_space = self.act_dim
        self.model = Policy(hidden_size, num_inputs, action_space).to(self.device)
        self.target_model = Policy(hidden_size, num_inputs, action_space).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.c_loss, self.a_loss = [], []
        self.gamma = gamma
        self.tau = tau

        hard_update(self.target_model, self.model)

    def act(self, state, action_noise=None, param_noise=None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = state.reshape(1, -1)
        self.model.eval()
        mu, _, _ = self.model((Variable(state), None))
        self.model.train()
        mu = mu.data
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu.clamp(-1, 1).cpu().data.numpy().flatten()

    def train(self):
        #state_batch = Variable(torch.cat(batch.state))
        #action_batch = Variable(torch.cat(batch.action))
        #reward_batch = Variable(torch.cat(batch.reward))
        #mask_batch = Variable(torch.cat(batch.mask))
        #next_state_batch = Variable(torch.cat(batch.next_state))

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.sample(128)
        _, _, next_state_values = self.target_model((next_state_batch, None))

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_values = reward_batch + (self.gamma * (1 - mask_batch) * next_state_values)

        _, state_action_values, _ = self.model((state_batch, action_batch))

        loss = MSELoss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        soft_update(self.target_model, self.model, self.tau)

        return loss.item(), 0

    def step(self, t):
        c, a = self.train()
        self.c_loss.append(c);
        self.a_loss.append(a)
        if t % 5000 == 0:
            # self.evaluate(self.env)
            print(f'Iteration {t}: Critic Loss: {np.mean(self.c_loss)}, Actor Loss: {np.mean(self.a_loss) * 2}')
            self.c_loss, self.a_loss = [], []
        self.episode_timesteps += 1

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/naf_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(model_path))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))