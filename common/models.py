import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
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

def cnn():
    pass



class StochasticActor(nn.Module):

    def __init__(self, obs_dim, act_dim, h_dim=(256,256), activation=nn.ReLU):
        super(StochasticActor, self).__init__()
        self.net = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mu = self.net(obs)
        pi_dist = Normal(mu, self.log_std.exp())
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

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(400, 300), activation=nn.ReLU):
        super(DeterministicActor, self).__init__()
        self.net = mlp([obs_dim] + list(h_dim) + [act_dim], activation)
        self.act_limit = act_limit

    def forward(self, obs, tanh=True):
        out = self.net(obs)
<<<<<<< HEAD
        if tanh:
            return self.act_limit * torch.tanh(out)
        else:
            return out
=======
        return self.act_limit * torch.tanh(out)
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf

    def act(self, obs):
        out = self.net(obs)
        action = self.act_limit * torch.tanh(out)
        return action.cpu().data.numpy().flatten()


class DoubleQvalueCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, h_dim=(400,300), activation=nn.ReLU):
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
        sa = torch.cat([obs, action], dim=-1)
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

    def mu_std(self, obs, use_numpy=True):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
<<<<<<< HEAD
        if use_numpy:
            return mu.cpu().data.numpy(), std.cpu().data.numpy()
        else:
            return mu, std

    def forward(self, obs, deterministic=False, with_logprob=True, detach=False):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out) if not detach else self.log_std(net_out.detach())
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
=======
>>>>>>> 1add20232aa895ad3c57ef6e5facaccef5d39bdf
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std) if not detach else Normal(mu.detach(), std)
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

    def __init__(self, obs_dim, act_dim, act_limit=1, h_dim=(256,256), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim] + list(h_dim), activation, activation)
        self.mu = nn.Linear(h_dim[-1], act_dim)
        self.log_std = nn.Linear(h_dim[-1], act_dim)
        self.act_limit = act_limit
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.act_dim = act_dim

    def logprob(self, obs, pi_action):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        #log_std = self.log_std(net_out)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        #std = self.std
        #std = self.std.detach()
        #print(torch.mean(std, dim=0))
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        #logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        #logp_pi.unsqueeze_(-1)
        return logp_pi

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        #log_std = self.log_std(net_out)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        #std = self.std
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        #pi_action = torch.tanh(pi_action)
        #pi_action = self.act_limit * pi_action
        pi_action = pi_action.clamp(-self.act_limit, self.act_limit)
        #print(std.mean())
        return pi_action

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return  self.forward(obs, deterministic).cpu().data.numpy().flatten()





class GaussianActor(nn.Module):

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
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        return logp_pi

    def mu_std(self, obs, use_numpy=True):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if use_numpy:
            return mu.cpu().data.numpy(), std.cpu().data.numpy()
        else:
            return mu, std

    def dist(self, obs):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return  self.forward(obs, deterministic).cpu().data.numpy().flatten()



class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action=1, device=None):
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
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

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

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def act(self, obs, numpy=True):
        if numpy:
            return self.decode(obs).cpu().data.numpy().flatten()
        else:
            return self.decode(obs)
