import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from envs import matrixgame
from ActorCritic import ActorCritic
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = matrixgame.MatrixGame() #gym.make("CartPole-v0").unwrapped

state_size = 1#env.observation_space.shape[0]
action_size = 2#env.action_space.n
lr = 0.0006

def transTensor(next_state,acts):
    next_state = [ [x[0] for x in next_state  ] ]
    next_state = np.stack(next_state)
    next_state = torch.FloatTensor(next_state).to(device)
    if acts!=[]:
        thact = Variable(torch.Tensor([acts]))
    else:
        thact = []
    return next_state,thact


def trainIters(n_iters):
    AC = ActorCritic(2, state_size, 1,2, 32,device)
    step_n = 50
    for iter in range(n_iters):
        state = env.reset()
        state = np.stack(state)
        done = False
        for i in range(step_n):
            state = torch.FloatTensor(state).to(device)
            #action = dist.sample()
            dist,action,log_prob,act_prob = AC.select_action(state)
            acts = [action.detach() for ag in range(2)]
            obs_n,reward_n,_, _ = env.step(acts)

            if i == step_n-1:
                done = True
            next_state = obs_n
            #entropy += dist.entropy().mean()
            AC.storeSample(state,log_prob,reward_n,1-done,acts)
            state = next_state

            if done:
                if iter%20==0:
                    print('Iteration: {}, Score: {}'.format(iter, np.sum(np.array(AC.rewards)) ))
                break
        
        next_state,thact = transTensor(next_state,acts)
        for ag in range(AC.n_agents):
            next_value = AC.critics[ag](next_state, thact)
            AC.next_value.append(next_value)

        actloss = AC.update()
        if iter%20 == 0:
            print("action_loss  ",actloss)
        ifsave = False
        if iter%250 == 0 and ifsave:
            torch.save(AC.actor, 'model/actor_v2.pkl')
            for j in range(AC.n_agents):
                torch.save(AC.critics[j], 'model/criticv2'+ str(j) +'.pkl')
    return AC

if __name__ == '__main__':
    trainIters(n_iters=201)