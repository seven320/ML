#encoding utf-8

import copy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import wrappers
import random

MEMORY_SIZE = 200 # メモリサイズいくつで学習を開始するか
BATCH_SIZE = 50 # バッチサイズ
EPSILON = 1.0 # ε-greedy法
EPSILON_DECREASE = 0.001 # εの減少値
EPSILON_MIN = 0.1 # εの下限
START_REDUCE_EPSILON = 200 # εを減少させるステップ数
TRAIN_FREQ = 10 # Q関数の学習間隔
UPDATE_TARGET_Q_FREQ = 20
GAMMA = 0.97 # 割引率


env = gym.make("CartPole-v0")

obs_num = env.observation_space.shape[0]
acts_num = env.action_space.n
HIDDEN_SIZE = 100

#数式モデル

class NN(nn.Module):
    def __init__(self):

        super(NN,self).__init__()
        self.fc1 = nn.Linear(obs_num,HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE,acts_num)

    def __call__(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y


class DQN():
    def __init__(self):
        self.Q = NN()
        self.Q_ast = copy.deepcopy(self.Q)
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=0.00015, alpha=0.95, eps=0.01)
        self.memory = []
        self.pobs = None
        self.pact = None
        self.epsilon = 1
        self.pact = None
        self.total_step = 0

    def decide_action(self,pobs):#pobs:predict_observation
        self.pobs = pobs
         # 行動選択
        pact = random.choice(list(range(acts_num)))
        # ε-greedy法
        # print(self.epsilon)
        if np.random.rand() > self.epsilon:
            pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))
            pobs_ = torch.from_numpy(pobs_)
            pact = self.Q(pobs_)
            maxs, indices = torch.max(pact.data, 1)
            pact = indices.numpy()[0]

        self.pact = pact

        return self.pact

    def train(self,reward,obs,done):
        # メモリに蓄積
        self.memory.append((self.pobs, self.pact, reward, obs, done)) # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
        if len(self.memory) > MEMORY_SIZE: # メモリサイズを超えていれば消していく
            self.memory.pop(0)

        #学習
        if len(self.memory) == MEMORY_SIZE: # メモリサイズ分溜まっていれば学習
            # 経験リプレイ
            if self.total_step % TRAIN_FREQ == 0:
                memory_ = np.random.permutation(self.memory)
                memory_idx = range(len(memory_))
                for i in memory_idx[::BATCH_SIZE]:
                    batch = np.array(memory_[i:i+BATCH_SIZE]) # 経験ミニバッチ
                    pobss = np.array(batch[:,0].tolist(),dtype="float32").reshape((BATCH_SIZE, obs_num))
                    pacts = np.array(batch[:,1].tolist(), dtype="int32")
                    rewards = np.array(batch[:,2].tolist(), dtype="int32")
                    obss = np.array(batch[:,3].tolist(), dtype="float32").reshape((BATCH_SIZE, obs_num))
                    dones = np.array(batch[:,4].tolist(), dtype="bool")
                    # set y
                    pobss_ = torch.from_numpy(pobss)
                    q = self.Q(pobss_)
                    obss_ = torch.from_numpy(obss)
                    maxs, indices = torch.max(self.Q_ast(obss_).data, 1)
                    maxq = maxs.numpy() # maxQ
                    target = copy.deepcopy(q.data.numpy())
                    for j in range(BATCH_SIZE):
                        target[j, pacts[j]] = rewards[j]+GAMMA*maxq[j]*(not dones[j]) # 教師信号
                    # Perform a gradient descent step
                    self.optimizer.zero_grad()
                    loss = nn.MSELoss()(q, torch.from_numpy(target))
                    loss.backward()
                    self.optimizer.step()

            # Q関数の更新
            if self.total_step % UPDATE_TARGET_Q_FREQ == 0:
                self.Q_ast = copy.deepcopy(self.Q)

         # εの減少
        if self.epsilon > EPSILON_MIN and self.total_step > START_REDUCE_EPSILON:
            self.epsilon -= EPSILON_DECREASE

        self.total_step += 1
