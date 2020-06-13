import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
#from acmodel import Actor,Critic
from model import Critic, Social
from torch.autograd import Variable

class ActorCritic:
    def __init__(self, n_agents, dim_obs, dim_act,dim_actprob, batch_size,device,setting):
        ifload = setting["ifload"]
        self.critics = []
        if ifload:
            self.actor = torch.load("model_rule/actor_"+ setting["actor_name"]+".pkl",map_location = torch.device('cpu'))
            for j in range(n_agents):
                self.critics.append(torch.load("model_rule/critic_"+ setting["critic_name"]+str(j) +'.pkl',map_location = torch.device('cpu')))
        else:
            self.actor = Social(dim_obs, dim_actprob).to(device)
            self.critics = [Critic(n_agents, dim_obs,
                                dim_act) for i in range(n_agents)]
        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerCs = [optim.Adam(self.critics[ag].parameters()) for ag in range(n_agents)]
        self.n_agents = n_agents
        self.emptyBuffer()
        self.device = device
    def update(self):
        ### update Critic  
        for j in range(len(self.acts)):
            tem = []
            for ag in range(self.n_agents):
                state, thact = self.transTensor(self.obss[j],self.acts[j])
                tem.append(self.critics[ag](state, thact).squeeze().unsqueeze(0) )
            self.Qvalues.append(tem)
        self.Qvalues = np.array(self.Qvalues)
        self.Qvalues = self.Qvalues.T
        self.rewards = np.array(self.rewards)
        #print(self.Qvalues)
        for ag in range(self.n_agents):
            returns = self.compute_returns(self.next_value[ag], self.rewards[:,ag], self.masks)
            returns = torch.cat(returns).detach()
            values = torch.cat(list(self.Qvalues[ag,:]))
            advantage = returns - values
            critic_loss = advantage.pow(2).mean()
            self.optimizerCs[ag].zero_grad()
            critic_loss.backward()
            self.optimizerCs[ag].step()
        ### update actor
        for ag in range(self.n_agents):
            sumR = np.array([sum(rew) for rew in self.rewards])
            R_ep = self.compute_returns(sum(self.next_value),sumR,self.masks)
            advantage = R_ep - np.array([ sum(x) for x in self.Qvalues.T ])
            advantage = np.array([ x.squeeze().detach() for x in advantage])
            advantage = Variable(torch.Tensor(advantage))
            log_prob_a = torch.cat(list(np.array(self.log_probs)[:,ag]))
            ### ????
            actor_loss = -(log_prob_a * advantage).mean()
            self.optimizerA.zero_grad()
            actor_loss.backward()
            self.optimizerA.step()
        self.emptyBuffer()
        return actor_loss
    def getPos(self,state,hei):
        busy = state[-1]
        mypos = state[:hei*2]
        mypos = mypos.reshape((hei,2))
        finalpos = np.where(mypos==1)
        return busy,finalpos
    def select_action(self,state):
        actions = []
        log_probags = []
        for ag in range(self.n_agents):
            state0 = state[ag]
            #print("...state... ",state0)
            busy,pos = self.getPos(state0,8)
            #print("agnumber...",ag," busy...",int(busy),"coordination...", pos[1][0])
            
            dist = self.actor(state0)
            action = dist.sample()
            #print("action....  ",action)#self.actor.printprob(state0))
            #print("==="*6)
            log_prob = dist.log_prob(action).unsqueeze(0)
            actions.append(action)
            log_probags.append(log_prob)
        return dist,actions,log_probags,dist

    def storeSample(self,state,log_prob,reward_n,mask,act_n):
        self.log_probs.append(log_prob)
        self.rewards.append(reward_n)
        self.masks.append(mask)
        self.acts.append(act_n)
        self.obss.append(state.detach().numpy())

    def emptyBuffer(self):
        self.log_probs = []
        self.Qvalues = []
        self.rewards = []
        self.masks = []
        self.acts = []
        self.obss = []
        self.next_value = []
    def compute_returns(self,next_value, rewards, masks, gamma=1):
        R = next_value
        returns = []
        #print("...",len(rewards),len(masks))
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def transTensor(self,state,acts):
        #state = [ [x[0] for x in state ] ]
        state = np.stack(state)
        state0 = state
        sta = []
        for j in range(self.n_agents):
            sta = sta+list(state0[0])
        #state = torch.FloatTensor([sta]).to(device)
        state = torch.FloatTensor([sta]).to(self.device)
        if acts!=[]:
            thact = Variable(torch.Tensor([acts]))
        else:
            thact = []
        return state,thact