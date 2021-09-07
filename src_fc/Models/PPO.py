import math
import random
import pickle

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable

class ActorNet(nn.Module):

    def __init__(self, num_inputs, hidden1_size, hidden2_size, num_outputs):
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden2_size, num_outputs)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        probs = self.actor(x)
        return probs


class CriticNet(nn.Module):

    def __init__(self, num_inputs, hidden1_size, hidden2_size):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden2_size, 1)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic(x)
        return value


class PPO():
    def __init__(self, env, num_inputs, num_outputs, hidden1_size, hidden2_size, std=0.0, window_size=50,
                 learning_rate=1e-1, gamma=0.99, batch_size=20):
        super(PPO, self).__init__()
        self.actor_net = ActorNet(
            num_inputs, hidden1_size, hidden2_size, num_outputs)
        self.critic_net = CriticNet(
            num_inputs, hidden1_size, hidden2_size)

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), learning_rate)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), learning_rate)

        self.batch_size = batch_size
        self.gamma = gamma
        self.rewards = []
        self.states = []
        self.action_probs = []
        self.ppo_update_time = 10
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.training_step = 0

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value

    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor_net(state)
            value = self.critic_net(state)
        return probs, value

    def remember(self, probs, value, reward, done, device, action, state, next_state, action_p, obs):
        dist = Categorical(torch.softmax(probs, dim=-1))
        log_prob = dist.log_prob(torch.tensor(action))
        self.rewards.append(torch.FloatTensor(
            [reward]).unsqueeze(-1).to(device))

        self.states.append(state)
        self.action_probs.append(action_p)

    def train(self):
        if len(self.states) < self.batch_size:
            return
        old_action_log_prob = torch.stack(self.action_probs)
        R = 0
        Gt = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.states))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = torch.tensor([self.critic_net(self.states[ind])
                                  for ind in index], dtype=torch.float).view(-1, 1)
                delta = Gt_index - V
                advantage = delta.detach()
                action_prob = torch.softmax(torch.stack([self.actor_net(self.states[ind])
                                                         for ind in index]), dim=1)

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                    1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss = Variable(action_loss, requires_grad=True)
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss = Variable(value_loss, requires_grad=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.rewards = []
        self.states = []
        self.action_probs = []

    def save_using_model_name(self, model_name_path):
        torch.save(self.actor_net.state_dict(), model_name_path + "_actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   model_name_path + "_critic.pkl")

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(model_name_path + "_actor.pkl"))
        self.critic_net.load_state_dict(
            torch.load(model_name_path + "_critic.pkl"))
