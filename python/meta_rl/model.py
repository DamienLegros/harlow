import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):

    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3,
                                out_channels=10,
                                kernel_size=2,
                                stride=2)
        self.max_pool_1 = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2d(in_channels=10,
                                out_channels=20,
                                kernel_size=2,
                                stride=2)
        self.max_pool_2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=500, out_features=256)
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=num_hidden_units,
                            num_layers=1)

        self.action_layer = nn.Linear(num_hidden_units, num_actions)
        self.value_layer = nn.Linear(num_hidden_units, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state, hidden_state, cell_state):
        state = torch.from_numpy(np.transpose(
            np.array(state))).float().unsqueeze(0)
        #print(state.size())
        state = F.relu(self.conv_1(state))
        #print(state.size())
        state = self.max_pool_1(state)
        #print(state.size())
        state = F.relu(self.conv_2(state))
        #print(state.size())
        state = self.max_pool_2(state)
        #print(state.size())
        state = self.flatten(state)
        #print(state.size())
        state = F.relu(self.linear_1(state)).unsqueeze(0)
        #print(state.size())
        #print(hidden.size())
        state, (hidden_state,
                cell_state) = self.lstm(state, (hidden_state, cell_state))
        #print(state.size())
        #print(hidden[0][0, 0, :].size())
        state_value = self.value_layer(hidden_state[0, 0, :])
        action_probs = F.softmax(self.action_layer(hidden_state[0, 0, :]),
                                 dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item(), hidden_state, cell_state

    def calculateLoss(self, gamma=0.99):

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0

        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values,
                                          rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            reward.resize_(1)
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
