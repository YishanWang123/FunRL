import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.q_head = torch.nn.Linear(256, 1)

    def forward(self, state, action):      #should flatten to dim = 1(if action is chunk)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_head(x)
        return q_value