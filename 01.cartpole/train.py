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

import dqn
# 定数
EPOCH_NUM = 3000 # エポック数
STEP_MAX = 200 # 最高ステップ数
LOG_FREQ = 500 # ログ出力の間隔


MONITER = False
#環境確認
env = gym.make("CartPole-v0")
# print("observasionj space num:",env.observation_space.shape[0])
# print("action space num:", env.action_space.n)


done = False

# 学習開始
print("\t".join(["epoch", "epsilon", "reward", "total_step", "elapsed_time"]))
start = time.time()

dqn = dqn.DQN()

total_step = 0
total_rewards = []

for epoch in range(EPOCH_NUM):
    pobs = env.reset()#環境初期化
    step = 0#ステップ数
    done = False#ゲーム終了フラグ
    total_reward = 0#累積報酬

    while not done and step < STEP_MAX:
        if MONITER:
            env.render()
        #行動選択
        pact = dqn.decide_action(pobs)


        #行動
        obs, reward, done, _ = env.step(pact)

        if done:
            reward = -1
        else:
            reward = 1

        dqn.train(reward,obs,done)

        total_reward += reward
        step += 1
        total_step += 1
        pobs = obs

    total_rewards.append(total_reward)

    if (epoch+1) % LOG_FREQ == 0:
        r = sum(total_rewards[((epoch+1)-LOG_FREQ):(epoch+1)])/LOG_FREQ # ログ出力間隔での平均累積報酬
        elapsed_time = time.time()-start
        print("\t".join(map(str,[epoch+1, dqn.epsilon, r, total_step, str(elapsed_time)+"[sec]"]))) # ログ出力
        start = time.time()

plt.figure(figsize=(10,5))
resize = (len(total_rewards)//10, 10)
tmp = np.array(total_rewards, dtype="float32").reshape(resize)
tmp = np.average(tmp, axis=1)
plt.plot(tmp)
plt.show()
