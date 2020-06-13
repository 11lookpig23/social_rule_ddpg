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
'''
env = matrixgame.MatrixGame()


state_size = 1
action_size = 2
lr = 0.0006
'''
def transTensor(next_state,acts,n_agents):
    #next_state = [ [x[0] for x in next_state  ] ]
    next_state = np.stack(next_state)
    next_state0 = next_state
    sta = []
    for j in range(n_agents):
        sta = sta+list(next_state0[0])
    next_state = torch.FloatTensor([sta]).to(device)
    if acts!=[]:
        thact = Variable(torch.Tensor([acts]))
    else:
        thact = []
    return next_state,thact


def trainAC(env,state_size,action_size,lr,n_agents,dim_act,dim_actprob,batch_size,setting):
    #n_agents, dim_obs, dim_act,dim_actprob, batch_size,device
    #ifload = setting["ifload"]#False
    n_iters = setting["iter"]
    AC = ActorCritic(n_agents, state_size, dim_act,dim_actprob, batch_size,device,setting)
    step_n = 10
    for iter in range(n_iters):
        state = env.reset()
        state = np.stack(state)
        done = False
        for i in range(step_n):
            state = torch.FloatTensor(state).to(device)
            #action = dist.sample()
            dist,actions,log_probs,act_prob = AC.select_action(state)
            acts = [act.detach() for act in actions]
            obs_n,reward_n,_, _ = env.step(acts)
            #print(obs_n,"]]]]]]")
            if i == step_n-1:
                done = True
            next_state = obs_n
            #entropy += dist.entropy().mean()
            AC.storeSample(state,log_probs,reward_n,1-done,acts)
            state = next_state

            if done:
                if iter%20==0:
                    print('Iteration: {}, Score: {}'.format(iter, np.sum(np.array(AC.rewards)) ))
                break
        
        next_state,thact = transTensor(next_state,acts,n_agents)
        for ag in range(AC.n_agents):
            next_value = AC.critics[ag](next_state, thact)
            AC.next_value.append(next_value)

        actloss = AC.update()
        if iter%20 == 0:
            print("action_loss  ",actloss)
        ifsave = setting["ifsave"]#True
        if iter%250 == 0 and ifsave:
            torch.save(AC.actor, "model_rule/actor_"+ setting["actor_name"]+".pkl")
            for j in range(AC.n_agents):
                torch.save(AC.critics[j], "model_rule/critic_"+ setting["critic_name"]+str(j) +'.pkl')
    return AC

if __name__ == '__main__':
    trainAC(n_iters=201)