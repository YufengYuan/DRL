from algs.base import BaseAgent
import torch
from common import ReplayBuffer, SquashedGaussianActor, DoubleQvalueCritic
import copy
import torch.nn.functional as F
import numpy as np
from scipy.stats import truncnorm
from common.models import SquashedGaussianActor2


class SAC(BaseAgent):

    def __init__(self, env, buffer_size=int(1e6), gamma=0.99, tau=0.005,
                 lr=1e-3, start_timesteps=1000, actor_train_freq=2, batch_size=128,
                 init_temperature=0.1, device=None):

        super(SAC, self).__init__(env, device)
        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = DoubleQvalueCritic(self.obs_dim, self.act_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Adjustable alpha
        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True, device=self.device)
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.5, 0.999))

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.start_timesteps = start_timesteps
        self.tau = tau
        self.gamma = gamma
        self.alpha = self.log_alpha.exp()
        self.actor_train_freq = actor_train_freq
        self.batch_size = batch_size

    def offline_initialize(self, replay_buffer, epoch=1):
        conf = 2
        # PPO-style mini-batch training
        critic_losses, actor_losses = [], []
        idxes = np.arange(replay_buffer.size - 1)
        print(replay_buffer.size)
        for i in range(epoch):
            np.random.shuffle(idxes)
            for j in range(replay_buffer.size // self.batch_size):
                idx = idxes[i * self.batch_size : (i+1) * self.batch_size]
                obs, action, reward, next_obs, done, next_action =  replay_buffer.sample(self.batch_size, True, idx)
                # SARSA-style policy evaluation
                #with torch.no_grad():
                #    # Compute the target Q value
                #    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                #    target_Q = torch.min(target_Q1, target_Q2)
                #    target_Q = reward + (1 - done) * self.gamma * target_Q
                ## Get current Q estimates
                #current_Q1, current_Q2 = self.critic(obs, action)
                ## Compute critic loss
                #critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                #critic_losses.append(critic_loss.item())
                ## Optimize the critic
                #self.critic_optimizer.zero_grad()
                #critic_loss.backward()
                #self.critic_optimizer.step()
                # Behavior cloning under entropy-regularization
                _, logprob = self.actor(obs)
                _action = 0.5 * torch.log((1 + action) / (1 - action))
                #actor_loss = (self.alpha * logprob - self.actor.logprob(obs, _action)).mean()
                actor_loss = - self.actor.logprob(obs, _action).mean()
                #print(action, _action)
                actor_losses.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                #alpha_loss = (self.log_alpha * (-logprob - self.target_entropy).detach()).mean()
                #self.alpha_optimizer.zero_grad()
                #alpha_loss.backward()
                #self.alpha_optimizer.step()
                #self.alpha = self.log_alpha.exp()
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            print(f'Epoch {i} Critic Loss: {np.mean(critic_losses)}, Actor Loss: {np.mean(actor_losses)}')
            critic_losses, actor_losses = [], []

        # Approximate support with the learn policy
        self.lower_bound = np.zeros((replay_buffer.size, self.act_dim))
        self.upper_bound = np.zeros((replay_buffer.size, self.act_dim))
        idxes = np.arange(replay_buffer.size)
        for _ in range(epoch):
            for i in range(int(np.ceil(replay_buffer.size / self.batch_size))):
                idx = idxes[i * self.batch_size: (i + 1) * self.batch_size]
                obs, action, reward, next_obs, done = replay_buffer.sample(self.batch_size, with_idxes=idx)
                mu, std = self.actor.mu_std(obs)
                self.lower_bound[i * self.batch_size: (i + 1) * self.batch_size] = mu - conf * std
                self.upper_bound[i * self.batch_size: (i + 1) * self.batch_size] = mu + conf * std


    def offline_improve(self, replay_buffer, epoch=10):

        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.actor_target = copy.deepcopy(self.actor)
        # Adjustable alpha
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.5, 0.999))

        actor_losses, critic_losses = [], []
        idxes = np.arange(replay_buffer.size - 1)
        for i in range(epoch):
            np.random.shuffle(idxes)
            for j in range(replay_buffer.size // self.batch_size):
                idx = idxes[i * self.batch_size: (i + 1) * self.batch_size]
                obs, action, reward, next_obs, done = replay_buffer.sample(self.batch_size, with_idxes=idx)
                if j % 100 == 0:
                    self.evaluate(self.env)
                # SARSA-style policy evaluation
                with torch.no_grad():
                    # No constrain
                    #next_action, logprob = self.actor(next_obs)
                    #target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                    #target_Q = torch.min(target_Q1, target_Q2)
                    #target_Q = reward + (1 - done) * self.gamma * (target_Q - self.alpha * logprob)

                    # Probablistically constrain
                    #mu, std = self.actor.mu_std(next_obs)
                    #a, b = (self.lower_bound[idx+1] - mu)/std, (self.upper_bound[idx+1] - mu)/std
                    #dist = truncnorm(a, b, loc=mu, scale=std)
                    #next_action = torch.tensor(dist.rvs(), dtype=torch.float32, device=self.device)
                    #logprob = self.actor.logprob(next_obs, next_action)
                    #target_Q1, target_Q2 = self.critic_target(next_obs, torch.tanh(next_action))
                    #target_Q = torch.min(target_Q1, target_Q2)
                    #target_Q = reward + (1 - done) * self.gamma * (target_Q - self.alpha * logprob)

                    # Q-learning constrain
                    mu, std = self.actor_target.mu_std(next_obs)
                    next_action =  mu  #np.clip(mu, self.lower_bound[idx+1], self.upper_bound[idx+1])
                    next_action = torch.tensor(self.act_limit * np.tanh(next_action), dtype=torch.float32, device=self.device)
                    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + (1 - done) * self.gamma * target_Q#(target_Q - self.alpha * logprob)
                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(obs, action)
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                critic_losses.append(critic_loss.item())
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                # Behavior cloning under entropy-regularization

                # TODO: freeze critic parameters here to prevent unnecessary backpropagation
                for param in self.critic.parameters():
                    param.requires_grad = False

                cur_action, _ = self.actor.mu_std(obs, False)
                cur_action = torch.tanh(cur_action)
                current_Q1, current_Q2 = self.critic(obs, cur_action)
                current_Q = torch.min(current_Q1, current_Q2)
                actor_loss = -current_Q.mean()
                #cur_action, logprob = self.actor(obs, detach=True)
                #current_Q1, current_Q2 = self.critic(obs, cur_action)
                #current_Q = torch.min(current_Q1, current_Q2)
                #actor_std_loss = (self.alpha * logprob - current_Q).mean()
                #actor_loss = actor_std_loss + actor_mu_loss

                #cur_action, logprob = self.actor(obs, detach=True)
                #current_Q1, current_Q2 = self.critic(obs, cur_action)
                #current_Q = torch.min(current_Q1, current_Q2)
                #actor_loss = (self.alpha * logprob - current_Q).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_losses.append(-current_Q.mean().item())
                self.actor_optimizer.step()

                for param in self.critic.parameters():
                    param.requires_grad = True

                #alpha_loss = (self.log_alpha * (-logprob - 3 * self.target_entropy).detach()).mean()
                #self.alpha_optimizer.zero_grad()
                #alpha_loss.backward()
                #self.alpha_optimizer.step()
                #self.alpha = self.log_alpha.exp()
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            print(f'Epoch {i} Critic Loss: {np.mean(critic_losses)}, Actor Loss: {np.mean(actor_losses)}')
            critic_losses, actor_losses = [], []

    def train(self, obs, action, next_obs, reward, done):

        with torch.no_grad():

            next_action, logprob = self.actor(next_obs)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * (target_Q - self.alpha * logprob)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        cur_action, logprob = self.actor(obs)

        # TODO: freeze critic parameters here to prevent unnecessary backpropagation
        for param in self.critic.parameters():
            param.requires_grad = False

        current_Q1, current_Q2 = self.critic(obs, cur_action)
        current_Q = torch.min(current_Q1, current_Q2)

        actor_loss = (self.alpha * logprob - current_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        alpha_loss = (self.log_alpha * (-logprob - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.reshape(1, -1)
        return self.actor.act(obs, deterministic=True)

    def step(self, t):

        self.episode_timesteps += 1

        # Select action randomly or according to policy
        #if t < self.start_timesteps:# or t > self.start_timesteps:
        #    action = self.env.action_space.sample()
        #else:
        #    action = self.actor.act(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
        action = self.actor.act(torch.tensor(self.obs, dtype=torch.float32, device=self.device))

        # Perform action
        next_obs, reward, done, _ = self.env.step(action)
        #done_bool = float(done) if self.episode_timesteps < self.env._max_episode_steps else 0
        done_bool = float(done)# if self.episode_timesteps < self.env._max_episode_steps else 0
        # Store data in replay buffer
        self.replay_buffer.add(copy.deepcopy(self.obs), action, next_obs, reward, done_bool)
        self.obs = next_obs
        self.episode_reward += reward

        # Train agent after collecting sufficient data, extra training iterations added when first reached start_timesteps
        if t == self.start_timesteps:
            for _ in range(self.start_timesteps):
                batch = self.replay_buffer.sample(self.batch_size)
                self.train(*batch)
        elif t > self.start_timesteps:
            batch = self.replay_buffer.sample(self.batch_size)
            self.train(*batch)

        if done:
            self.episode_end_handle(t)

