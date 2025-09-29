import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mean_head = torch.nn.Linear(128, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x, deterministic=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if deterministic:
            mean = self.mean_head(x)
            action = torch.tanh(mean)
            return action
        else:
            mean = self.mean_head(x)
            mean = torch.tanh(mean)
            log_std = self.log_std_head.expand_as(mean)
            std = torch.exp(log_std)
            std = torch.clamp(std, min=1e-6)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return mean, std, log_prob