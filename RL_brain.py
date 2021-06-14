import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Models import Deepnet
from Modules import *
np.random.seed(1)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.loss_func = nn.MSELoss()
        # total learn step
        self.learn_step_counter = 0

        # initialize zero memory
        self.memory = np.zeros((self.memory_size, n_features*2 + 2))
        self.eval_net = Net(self.n_features, self.n_actions)
        self.target_net = Net(self.n_features, self.n_actions)
        # consist of [target_net,evaluate_net]

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.writer = SummaryWriter('logs/')
        self.cost_his = []
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=learning_rate)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net.forward(observation)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch_memory[:, :self.n_features])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_s)
        q_next = self.target_net(b_s_).detach()

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features+1]

        #print("batch_index", batch_index)
        # print("eval_act_index",eval_act_index)
        #print("reward, {}".format(torch.FloatTensor(reward).shape))
        #print("q_next: ",q_next.max(1)[0].view(self.batch_size, 1).shape)
        q_target[batch_index, eval_act_index] = torch.FloatTensor(
            reward)+self.gamma*q_next.max(1)[0].view(self.batch_size)

        # train eval network

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        self.writer.add_graph(self.eval_net, b_s)
        self.writer.add_graph(self.target_net, b_s_)
        self.writer.close()
        # increasing epsilon
        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def write(self):
        temp = self.memory[0]
        b_s = torch.FloatTensor(temp[:, :self.n_features])
        b_s_ = torch.FloatTensor(temp[:, -self.n_features:])
        self.writer.add_graph(self.eval_net, b_s)
        self.writer.add_graph(self.target_net, b_s_)
        self.writer.close()
