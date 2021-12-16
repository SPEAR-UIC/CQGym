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


class A2C(nn.Module):
    def __init__(self, env, num_inputs, num_outputs, std=0.0, window_size=50,
                 learning_rate=1e-2, gamma=0.99, batch_size=20, layer_size=[]):
        super(A2C, self).__init__()
        self.hidden1_size = layer_size[0]
        self.hidden2_size = layer_size[1]
        self.critic = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, self.hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden1_size, self.hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden2_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, self.hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden1_size, self.hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden2_size, num_outputs)
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.window_size = window_size
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.rewards_seq = []
        self.entropy = 0

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value

    def remember(self, probs, value, reward, done, device, action):
        dist = Categorical(torch.softmax(probs, dim=-1))
        log_prob = dist.log_prob(torch.tensor(action))
        self.entropy += dist.entropy().mean()

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.FloatTensor(
            [reward]).unsqueeze(-1).to(device))
        self.rewards_seq.append(reward)

    def train(self, next_value, optimizer):
        if len(self.values) < self.batch_size:
            return

        returns = self.compute_returns(next_value)

        self.log_probs = torch.tensor(self.log_probs)
        returns = torch.cat(returns).detach()
        self.values = torch.cat(self.values)

        advantage = returns - self.values

        actor_loss = -(self.log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropy = 0

    def compute_returns(self, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step][0] + self.gamma * R
            returns.insert(0, R)
        return returns

    def save_using_model_name(self, model_name_path):
        torch.save(self.state_dict(), model_name_path + ".pkl")

    def load_using_model_name(self, model_name_path):
        self.load_state_dict(
            torch.load(model_name_path + ".pkl"))
