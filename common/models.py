import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import numpy as np


def mlp(sizes, activation, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        # To be compatible with torch_gpu 1.0.0
        if act is not None:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)

class StochasticActor(nn.Module):

    def __init__(self, obs_dim, act_dim, h_dim=(256,256), activation=nn.ReLU):
        super(StochasticActor, self).__init__()
        self.net = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mu = self.net(obs)
        pi_dist = Normal(mu, self.log_std.exp())
        #logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        #logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        #logp_pi.unsqueeze_(-1)
        #dist = MultivariateNormal(out, torch.diag_embed(torch.exp(self.log_std)))
        return pi_dist

    def act(self, obs):
        pi_dist = self.forward(obs)
        action = pi_dist.sample()
        log_prob = pi_dist.log_prob(action).sum(dim=-1).unsqueeze_(-1)
        return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()

class ValueCritic(nn.Module):

    def __init__(self, obs_dim, h_dim=(256,256), activation=nn.ReLU):
        super(ValueCritic, self).__init__()
        self.net = mlp([obs_dim] + list(h_dim) + [1], activation)

    def forward(self, obs):
        return self.net(obs)

    def value(self, obs):
        return self.forward(obs).detach().cpu().data.numpy().flatten()


class DeterministicActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256, 256), activation=nn.ReLU):
        super(DeterministicActor, self).__init__()
        self.net = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.act_limit = act_limit

    def forward(self, obs):
        out = self.net(obs)
        return self.act_limit * torch.tanh(out)

    def unnormlized_action(self, obs):
        return self.net(obs)

    def act(self, obs):
        out = self.net(obs)
        action = self.act_limit * torch.tanh(out)
        return action.cpu().data.numpy().flatten()

import math
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        #self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(1))
        #self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        #self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(1))
        #self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        #self.reset_noise()

    def forward(self, x, expl=False):
        if expl:
            #print(torch.mean(self.weight_sigma), torch.mean(self.bias_sigma))
            #weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            #bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
            weight = Normal(self.weight_mu.detach(), self.weight_sigma).rsample()
            bias = Normal(self.bias_mu.detach(), self.bias_sigma).rsample()
            #print(self.weight_sigma.item())
            return F.linear(x, weight, bias)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init)# / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init)# / math.sqrt(self.bias_sigma.size(0)))

    #def reset_noise(self):
    #    epsilon_in = self._scale_noise(self.in_features)
    #    epsilon_out = self._scale_noise(self.out_features)

    #    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    #    self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    #def _scale_noise(self, size):
    #    x = torch.randn(size)
    #    x = x.sign().mul(x.abs().sqrt())
    #    return x


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state):
        z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def get_action(self, state):
        u, _, _ = self.forward(state)
        return u.cpu().data.numpy().flatten()

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

class NoisyActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256, 256), activation=nn.ReLU):
        super(NoisyActor, self).__init__()
        self.l1 = nn.Linear(obs_dim, h_dim[0])
        self.l2 = NoisyLinear(h_dim[0], h_dim[1])
        self.l3 = NoisyLinear(h_dim[1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, expl=False):
        out = torch.relu(self.l1(obs))
        out = torch.relu(self.l2(out, expl))
        action = self.l3(out, expl)
        return self.act_limit * torch.tanh(action)

    def unnormlized_action(self, obs):
        return self.net(obs)

    def act(self, obs, expl=False):
        action = self.forward(obs, expl)
        return action.cpu().data.numpy().flatten()


class DoubleQvalueCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, h_dim=(256,256), activation=nn.ReLU):
        super(DoubleQvalueCritic, self).__init__()
        self.net1 = mlp([obs_dim + act_dim] + list(h_dim) + [1], activation)
        self.net2 = mlp([obs_dim + act_dim] + list(h_dim) + [1], activation)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        return q1, q2

    def Q1(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        return self.net1(sa)

    def Q2(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        return self.net2(sa)


class QvalueCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, h_dim=(256,256), activation=nn.ReLU):
        super(QvalueCritic, self).__init__()
        self.net = mlp([obs_dim + act_dim] + list(h_dim) + [1], activation)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        q = self.net(sa)
        return q

LOG_STD_MAX = 2
LOG_STD_MIN = -10

class SquashedGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256,256), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim] + list(h_dim), activation, activation)
        self.mu = nn.Linear(h_dim[-1], act_dim)
        self.log_std = nn.Linear(h_dim[-1], act_dim)
        self.act_limit = act_limit

    def logprob(self, obs, pi_action):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        #print(torch.mean(std, dim=0))
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        logp_pi.unsqueeze_(-1)
        return logp_pi

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        #print(torch.mean(std, dim=0))
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(dim=1)
            logp_pi.unsqueeze_(-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, deterministic=False, with_logprob=False):
        with torch.no_grad():
            a, logprob = self.forward(obs, deterministic, with_logprob)
            if with_logprob:
                return a.cpu().data.numpy().flatten(), logprob.cpu().data.numpy().flatten()
            else:
                return a.cpu().data.numpy().flatten()


