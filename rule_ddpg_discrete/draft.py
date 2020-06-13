from model import Critic, Actor, Constrain
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
from torch.autograd import Variable
from scipy.special import softmax


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):

        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        ifload = False
        if ifload:
            for i in range(2):
                name1 = "parameter/actor_v3"+str(i)+".pth"
                name2 = "parameter/critic_v3"+str(i)+".pth"
                #print(name1)
                self.actors[i].load_state_dict(th.load(name1,map_location = th.device('cpu')))
                self.critics[i].load_state_dict(th.load(name2,map_location = th.device('cpu')))
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        ## Constrain........
        self.constrain = Constrain(dim_obs,2)
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.0008) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0002) for x in self.actors]

        self.constrain_optimizer = Adam(self.constrain.parameters(),lr = 0.0006)

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()
            self.constrain.cuda()
        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            #print("whole_action",whole_action)
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 30 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
                if self.steps_done % 300 == 0: 
                    th.save(self.critics[i].state_dict(),"parameter/critic_v3"+str(i)+".pth")
                    th.save(self.actors[i].state_dict(),"parameter/actor_v3"+str(i)+".pth")
        return c_loss, a_loss

    def update_rule(self):
        if self.episode_done <= self.episodes_before_train:
            return None
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = th.stack(batch.states).type(FloatTensor)
        action_batch = th.stack(batch.actions).type(FloatTensor)

        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)        

        #for ag in range(self.n_agents):
        true_act,rules = self.select_rule_action(state_batch)
        
        if self.steps_done%2==0:
            id = 0
        else:
            id = 1

        Q = []

        for ag  in range(self.n_agents):
            Q.append( self.critics[ag](whole_state, Variable(th.Tensor( true_act))) )
        Qsum = sum(Q)
        if self.steps_done%600==0:
            print("true_act..",true_act[15])
            print("rule..",rules[id][15])
            print("Qsum..",Qsum[15])
        loss_r = -rules[id]*Qsum
        loss_r = loss_r.mean()
        loss_r.backward()
        self.constrain_optimizer.step()
        return loss_r

    def rule_act(self,state_batch):
        #true_act = []
        rules = []

        #obs = state_batch[:,0,:]
        #rule = self.constrain(obs)
        rules.append(self.constrain(state_batch[:,0,:]))
        rules.append(self.constrain(state_batch[:,1,:]))
        rule = rules[1].detach().numpy()
        #print(rule)
        
        action = [np.random.choice(2,1,p = softmax(x))  for x in rule ] #[ [0] if x[0]>x[1] else [1] for x in rule ]
        #true_act.append(list(action))
        #true_act.append(list(action))
        
        true_act = [  [x[0],x[0]] for x in action]#action#np.array(true_act).reshape(self.batch_size,2)
        #print(true_act)
        #print(true_act[1])
        return true_act, rules
#
    def select_rule_action(self,state_batch):
        true_act = []
        rules = []
        for id in range(2):
            obs = state_batch[:,id,:]
            act = self.actors[id](obs)
            act = th.clamp(act, 0.0, 1.0)  ## ??
            act = act.detach().numpy()
            act_prob = [ [1-x[0],x[0] ]   for x in act  ] #[ 1-act[0], act[0]]
            #act_prob = Variable(th.Tensor( act_prob))
            self.constrain_optimizer.zero_grad()
            rule = self.constrain(obs)
            rules.append(rule)
            rule0 = rule.detach().numpy()

            scale_act = [ softmax(np.array(rule0[i])*np.array(act_prob[i])) for i in range(self.batch_size) ]
            #scale_act = softmax(scale_act)
            action = [ np.random.choice(2,1,p = x) for x in scale_act ]
            true_act.append(action)
        true_act = np.array(true_act).reshape(self.batch_size,2)
        return true_act,rules

    def select_rule_action2(self,state_batch):
        true_act = []
        rules = []

        obs = state_batch[:,0,:]
        rule = self.constrain(obs)
        rules.append(rule)
        rules.append(self.constrain(state_batch[:,1,:]))
        rule = rule.detach().numpy()
        action = [ np.random.choice(2,1,p = x) for x in rule ]
        #scale_act = [ softmax(np.array(rule[i])*np.array(act_prob[i])) for i in range(self.batch_size) ]
        #scale_act = softmax(scale_act)
        #action = [ np.random.choice(2,1,p = x) for x in scale_act ]
        true_act.append(action)
        true_act.append(action)
        true_act = np.array(true_act).reshape(self.batch_size,2)
        return true_act,rules

    def getLaw(self,rule_prob,action_prob):
        forbidden_prob = [rule_prob[1],rule_prob[0]]
        for k in range(len(action_prob)):
            if action_prob[k] < forbidden_prob[k]:
                action_prob[k] = 0
        return action_prob
    def select_action(self,state_batch,rule_prob):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0))#.squeeze()

            act += th.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.05:
                self.var[i] *= 0.999998

            act = th.clamp(act, 0.0, 1.0)
            #print("act...",act)
            actProb = [1-act[0][0],act[0][0]]
            action_prob = self.getLaw(rule_prob,actProb)

            at = np.argmax(np.array(action_prob))
            #print("at...",at)
            act = Variable(th.Tensor([[at]]))
            actions[i, :] = act
        self.steps_done += 1

        return actions

    def select_action2(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0))#.squeeze()

            act += th.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, 0.0, 1.0)

            actions[i, :] = act

        return actions

    def select_eval_action(self, state_batch,rule_prob,rule):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        #FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sta = Variable(th.Tensor( [[0]]))
            act = self.actors[i](sta)#.squeeze()
            act = th.clamp(act, 0.0, 1.0)
            if rule:
                actProb = [1-act[0][0],act[0][0]]
                action_prob = self.getLaw(rule_prob,actProb)
                #if law:#act[0][0]>0.88:
                at = np.argmax(np.array(action_prob))
                #print("at... ",at)
                act = Variable(th.Tensor([[at]]))
            actions[i, :] = act
        self.steps_done += 1
        return actions