import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from copy import deepcopy
import numpy as np



def mlp(sizes, activation, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        # Original setting is not compatible with torch_gpu 1.0.0
        if act is not None:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)


# TODO: update it as GaussianActor with mlp function
class StochasticActor(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(StochasticActor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, act_dim)

        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        h = F.relu(self.l1(obs))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        dist = MultivariateNormal(h, torch.diag_embed(torch.exp(self.log_std)))
        return dist

    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()

# TODO: update with mlp function
class ValueCritic(nn.Module):

    def __init__(self, obs_dim):
        super(ValueCritic, self).__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, obs):
        h = F.relu(self.l1(obs))
        h = F.relu(self.l2(h))
        return self.l3(h)

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

    def mu_(self, obs):
        net_out = self.net(obs)
        mu = self.mu(net_out).detach()
        return mu

    def logprob(self, obs, pi_action):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        print(torch.mean(std, dim=0))
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


class SquashedGaussianActor2(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.mu = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.log_std = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        mu = self.mu(obs)
        log_std = self.log_std(obs)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # print(torch.mean(std, dim=0))
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu.detach(), std)
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
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
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


class Explorer(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256,256), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim] + list(h_dim), activation, activation)
        self.log_std = nn.Linear(h_dim[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return log_std
        #std = torch.exp(log_std)
        # Pre-squash distribution and sample
        #return std

   # def act(self, obs, mu, deterministic=False):
   #     with torch.no_grad():
   #         a, _ = self.forward(obs, mu, deterministic)
   #         return a.cpu().data.numpy().flatten()