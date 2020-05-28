from MADDPG import MADDPG
import numpy as np
import torch as th
from params import scale_reward
from envs import matrixgame
from scipy.special import softmax
from torch.autograd import Variable
from ActorCritic import ActorCritic
from ac2 import trainIters

e_render = False

world = matrixgame.MatrixGame()
n_coop = 2

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
n_agents = 2#world.n_pursuers
n_states = 1
n_actions = 1
capacity = 1000000
batch_size = 32

n_episode = 101#1
n_ep_eval = 60
max_steps = 60#60
episodes_before_train = 10


win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
ifeval = False
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def ruleLearner():
    print(" social rule learning .....   ")
    #AC = ActorCritic(2, n_states, 1,2, 32,device)
    AC = trainIters(4001)
    rule_prob = AC.actor.printprob(th.FloatTensor([[0],[0]]).to(device)[0])
    print("rule_prob..",rule_prob)
    return AC

def maddpgTrainer(AC):
    print(" MADDPG --- trainning --- ")
    for i_episode in range(n_episode):
        if ifeval:
            break
        obs = world.reset()
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        rr = np.zeros((n_agents,))
        for t in range(max_steps):
            rule_prob = AC.actor.printprob(th.FloatTensor([[0],[0]]).to(device)[0])
            obs = obs.type(FloatTensor)
            
            action = maddpg.select_action(obs,rule_prob).data.cpu()

            actlist = [ 1 if x[0]>0.5 else 0 for x in action.numpy()]
            #print("actli..",actli)
            obs_, reward, done, _ = world.step(actlist)
            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None

            total_reward += reward.sum()
            rr += reward.cpu().numpy()
            maddpg.memory.push(obs.data, action, next_obs, reward)
            obs = next_obs

            c_loss, a_loss = maddpg.update_policy()
            r_loss = maddpg.update_rule()

        if (i_episode)%100==0 and i_episode!=0 and i_episode!=99:
            #for ag in range(2):
            for k in range(2):
                for j in range(2):
                    sta = Variable(th.Tensor( [[0,0]]))
                    acts =  Variable(th.Tensor( [[k,j]]))
                    Qsum = maddpg.critics[0](sta,acts)+maddpg.critics[1](sta,acts)
                    print("..i..j..",k,j,"..Q..",Qsum)
            print("closs..",[c_loss[0].detach().numpy(),c_loss[1].detach().numpy()],"..aloss..",[a_loss[0].detach().numpy(),a_loss[1].detach().numpy()])

        if (i_episode)%10 == 0 and i_episode!=10 and i_episode!=0:
            rules = maddpg.constrain(obs_[0])
            print("f_rule...",rules.detach().numpy())
            print("r_loss...",r_loss.detach().numpy())
        maddpg.episode_done += 1
        print('Episode: %d, reward = %f' % (i_episode, total_reward))
        reward_record.append(total_reward)



def evaluate(AC):
    for i_ev in range(n_ep_eval):
        obs = world.reset()
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        rr = np.zeros((n_agents,))
        obeyrule = True
        if i_ev>30:
            obeyrule = False
        for t in range(max_steps):
            obs = obs.type(FloatTensor)
            rule_prob = AC.actor.printprob(th.FloatTensor([[0],[0]]).to(device)[0])
            action = maddpg.select_eval_action(obs,rule_prob,obeyrule).data.cpu()
            actli = [ 1 if x[0]>0.5 else 0 for x in action.numpy()]
            obs_, reward, done, _ = world.step(actli)
            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None

            total_reward += reward.sum()
            rr += reward.cpu().numpy()
            obs = next_obs
        maddpg.episode_done += 1
        print('Episode: %d, reward = %f' % (i_ev, total_reward))
        reward_record.append(total_reward)


if __name__ == "__main__":
    AC = ruleLearner()
    maddpgTrainer(AC)
    print("==========================================")
    print(" testing .....   ")
    evaluate(AC)