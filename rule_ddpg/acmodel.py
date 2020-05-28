import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        hide = 8#500
        hide2 = 4#128
        self.FC1 = nn.Linear(dim_observation, hide)
        self.FC2 = nn.Linear(hide, hide2)
        self.FC3 = nn.Linear(hide2, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        dist = Categorical(F.softmax(result, dim=-1))
        return dist
    def printprob(self,obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return F.softmax(result, dim=-1)
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        hide = 32#1024
        hide2 = 8#512
        hide3 = 4#300
        self.FC1 = nn.Linear(obs_dim, hide)
        self.FC2 = nn.Linear(hide+act_dim, hide2)
        self.FC3 = nn.Linear(hide2, hide3)
        self.FC4 = nn.Linear(hide3, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(1, 2).to(device)
    #trainIters(actor, critic, n_iters=4)
    state = [0]
    state = torch.FloatTensor(state).to(device)
    dist = actor(state)
    action = dist.sample()
    log_prob = dist.log_prob(action).unsqueeze(0)
    print("....log..",log_prob)