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

#
# class NN(nn.Module):
#     def __init__(self):
#
#         super(NN,self).__init__()
#         self.fc1 = nn.Linear(obs_num,HIDDEN_SIZE)
#         self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
#         self.fc3 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
#         self.fc4 = nn.Linear(HIDDEN_SIZE,acts_num)
#
#     def __call__(self,x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         h = F.relu(self.fc3(h))
#         y = F.relu(self.fc4(h))
#         return y
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Conv2d(1,10,5)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(10,20,5)
#         self.fc1 = nn.Linear(320,50)
#         self.fc2 = nn.Linear(50,10)
#
#     def __call__(self,input):
#         x = self.pool(F.relu(self.conv1(input)))
#         y = self.pool(F.relu(self.conv2(x)))
#
#         return x
#
# #hook
# def printnorm(self,input,output):
#     # input is a tuple of packed inputs
#     # output is a Tensor. output.data is the Tensor we are interested
#     print('Inside ' + self.__class__.__name__ + ' forward')
#     print('')
#     print('input: ', type(input))
#     print('input[0]: ', type(input[0]))
#     print('output: ', type(output))
#     print('')
#     print('input size:', input[0].size())
#     print('output size:', output.data.size())
#     print('output norm:', output.data.norm())
#
# model = CNN()
# model.conv1.register_forward_hook(printnorm)
# model.conv2.register_forward_hook(printnorm)
#
# print(model)
# input = torch.randn(1,1,28,28)
# out = model(input)
# print(out.size())


class RNN(nn.Module):
    def __init__(self,data_size,hidden_size,output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)

    def __call__(self,data,last_hidden):
        input = torch.cat((data,last_hidden),1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)

        return hidden,output


rnn_model = RNN(50,20,10)

loss_fn = nn.MSELoss()
batch_size = 10
TIMESTEPS = 5

# some fake data
batch = torch.randn(batch_size,50)
hidden = torch.zeros(batch_size,20)
targe = torch.zeros(batch_size,10)

loss = 0

for i in range(TIMESTEPS):
    hidden,output = rnn_model(batch,hidden)
    loss += loss_fn(output,target)

loss.backward()
