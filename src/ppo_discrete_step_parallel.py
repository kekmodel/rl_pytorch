import pickle
from collections import deque
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from running_mean_std import RunningMeanStd

N_PROCESS = 4
ROLL_LEN = 2048 * N_PROCESS
BATCH_SIZE = 2048
P_LR = 3e-4
V_LR = 1e-3
ITER = 80
CLIP = 0.2
GAMMA = 0.999
LAMBDA = 0.97
# BETA = 3.0
# ENT_COEF = 0.0
GRAD_NORM = False
OBS_NORM = True

# set device
use_cuda = torch.cuda.is_available()
print('cuda:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')
writer = SummaryWriter()

# random seed
torch.manual_seed(5)
if use_cuda:
    torch.cuda.manual_seed_all(5)

# make an environment
# env = gym.make('CartPole-v0')
# env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')
# env = gym.make('LunarLander-v2')


class ActorCriticNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        h = 32
        self.pol = nn.Sequential(
            nn.Linear(obs_space, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, action_space)
        )
        self.val = nn.Sequential(
            nn.Linear(obs_space, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        logit = self.pol(x).reshape(x.shape[0], -1)
        log_p = self.log_softmax(logit)
        v = self.val(x).reshape(x.shape[0], 1)
        return log_p, v


# +
def learn(net, optimizer, train_memory):
    global steps
    old_net = deepcopy(net)
    net.train()
    old_net.train()
    dataloader = DataLoader(
            train_memory,
            shuffle=False,
            batch_size=BATCH_SIZE,
            pin_memory=use_cuda
        )
    advs = []
    for data in dataloader.dataset:
        advs.append(data[3])
    advs = torch.tensor(advs, dtype=torch.float32).to(device)
    adv_mean = advs.mean()
    adv_std = advs.std()
    for _ in range(ITER):
        for (s, a, ret, adv) in dataloader:
            s_batch = s.to(device).float()
            a_batch = a.to(device).long()
            ret_batch = ret.to(device).float()
            adv_batch = adv.to(device).float()
            adv_batch = (adv_batch - adv_mean) / adv_std
            with torch.no_grad():
                log_p_batch_old, v_batch_old = old_net(s_batch)
                log_p_acting_old = log_p_batch_old[range(BATCH_SIZE), a_batch]

            log_p_batch, v_batch = net(s_batch)
            log_p_acting = log_p_batch[range(BATCH_SIZE), a_batch]
            p_ratio = (log_p_acting - log_p_acting_old).exp()
            p_ratio_clip = torch.clamp(p_ratio, 1 - CLIP, 1 + CLIP)
            p_loss = -(torch.min(p_ratio * adv_batch, p_ratio_clip * adv_batch).mean())
            v_loss = (ret_batch - v_batch).pow(2).mean()
            
            kl_div = ((log_p_batch.exp() * (log_p_batch - log_p_batch_old)).sum(dim=-1).mean()).detach().item()

#             log_p, _ = net(s_batch)
#             entropy = -(log_p.exp() * log_p).sum(dim=1).mean()

            # loss
            loss = p_loss + v_loss

            if kl_div <= 0.01 * 1.5:
                optimizer[0].zero_grad()
                p_loss.backward()
                if GRAD_NORM:
                    nn.utils.clip_grad_norm_(net.parameters() , max_norm=1.0)
                optimizer[0].step()
            else:
                print("Pass the Pi update!")
            optimizer[1].zero_grad()
            v_loss.backward()
            if GRAD_NORM:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer[1].step()
            if rank == 0:
                writer.add_scalar('data/p_loss', p_loss.item(), steps)
                writer.add_scalar('data/v_loss', v_loss.item(), steps)
                writer.add_scalar('data/kl_div',kl_div, steps)
            steps += 1
    train_memory.clear()
    return net.state_dict()


# -

def get_action_and_value(obs, old_net):
    old_net.eval()
    with torch.no_grad():
        state = torch.tensor([obs]).to(device).float()
        log_p, v = old_net(state)
        m = Categorical(log_p.exp())
        action = m.sample()
    return action.item(), v.item()


def compute_adv_with_gae(rewards, values, roll_memory):
    rew = np.array(rewards, 'float')
    val = np.array(values[:-1], 'float')
    _val = np.array(values[1:], 'float')
    ret = rew + GAMMA * _val
    delta = ret - val
    gae_dt = np.array([(GAMMA * LAMBDA)**(i) * dt for i, dt in enumerate(delta.tolist())],  'float')
    for i, data in enumerate(roll_memory):
        data.append(ret[i])
        data.append(sum(gae_dt[i:] / (GAMMA * LAMBDA)**(i)))

    rewards.clear()
    values.clear()
    return roll_memory


def roll_out(env, length, rank, child):
    env.seed(rank)

    # hyperparameter
    roll_len = length

    # for play
    episodes = 0
    ep_steps = 0
    ep_rewards = []

    # memories
    train_memory = []
    roll_memory = []
    rewards = []
    values = []

    # recieve
    old_net, norm_obs  = child.recv()

    # Play!
    while True:
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # env.render()
            if OBS_NORM:
                norm_obs.update(np.array([obs]))
                obs_norm = np.clip((obs - norm_obs.mean) / np.sqrt(norm_obs.var), -10, 10)
                action, value = get_action_and_value(obs_norm, old_net)
            else:
                action, value = get_action_and_value(obs, old_net)

            # step
            _obs, reward, done, _ = env.step(action)

            # store
            values.append(value)

            if OBS_NORM:
                roll_memory.append([obs_norm, action])
            else:
                roll_memory.append([obs, action])

            rewards.append(reward)
            obs = _obs
            ep_reward += reward
            ep_steps += 1

            if done or ep_steps % roll_len == 0:
                if OBS_NORM:
                    norm_obs.update(np.array([_obs]))
                if done:
                    _value = 0.
                else:
                    if OBS_NORM:
                        _obs_norm = np.clip((_obs - norm_obs.mean) / np.sqrt(norm_obs.var), -10, 10)
                        _, _value = get_action_and_value(_obs_norm, old_net)
                    else:
                        _, _value = get_action_and_value(_obs, old_net)

                values.append(_value)
                train_memory.extend(compute_adv_with_gae(rewards, values, roll_memory))
                roll_memory.clear()

            if ep_steps % roll_len == 0:
                child.send(((train_memory, ep_rewards), 'train', rank))
                train_memory.clear()
                ep_rewards.clear()
                state_dict, norm_obs = child.recv()
                old_net.load_state_dict(state_dict)
                break

        if done:
            episodes += 1
            ep_rewards.append(ep_reward)
            print('{:3} Episode in {:4} steps, reward {:4} [Process-{}]'.format(episodes, ep_steps, ep_reward, rank))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    n_eval = 100
    net = ActorCriticNet(obs_space, action_space).to(device)
    net.share_memory()
    param_p = [p for n, p in net.named_parameters() if 'pol' in n]
    param_v = [p for n, p in net.named_parameters() if 'val' in n]
    optim_p = torch.optim.AdamW(param_p, lr=P_LR, eps=1e-6)
    optim_v = torch.optim.AdamW(param_v, lr=V_LR, eps=1e-6)
    optimizer = [optim_p, optim_v]
    norm_obs = RunningMeanStd(shape=env.observation_space.shape)

    jobs = []
    pipes = []
    trajectory = []
    rewards = deque(maxlen=n_eval)
    update = 0
    steps = 0
    for i in range(N_PROCESS):
        parent, child = mp.Pipe()
        p = mp.Process(target=roll_out, args=(env, ROLL_LEN//N_PROCESS, i, child), daemon=True)
        jobs.append(p)
        pipes.append(parent)

    for i in range(N_PROCESS):
        pipes[i].send((net, norm_obs))
        jobs[i].start()

    while True:
        for i in range(N_PROCESS):
            data, msg, rank = pipes[i].recv()
            if msg == 'train':
                traj, ep_rews = data
                trajectory.extend(traj)
                rewards.extend(ep_rews)

        if len(rewards) == n_eval:
            writer.add_scalar('data/reward', np.mean(rewards), update)
            if np.mean(rewards) >= env.spec.reward_threshold:
                print('\n{} is sloved! [{} Update]'.format(env.spec.id, update))
                torch.save(net.state_dict(), f'./test/saved_models/{env.spec.id}_up{update}_clear_model_ppo_st.pt')
                with open(f'./test/saved_models/{env.spec.id}_up{update}_clear_norm_obs.pkl', 'wb') as f:
                    pickle.dump(norm_obs, f, pickle.HIGHEST_PROTOCOL)
                break

        if len(trajectory) == ROLL_LEN:
            state_dict = learn(net, optimizer, trajectory)
            update += 1
            print(f'Update: {update}')
            for i in range(N_PROCESS):
                pipes[i].send((state_dict, norm_obs))
    env.close()
