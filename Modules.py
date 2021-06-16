# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:19:42 2021

@author: A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Embedding
from collections import OrderedDict

class FMLayer(nn.Module):
    def __init__(self, n=10, k=5):
        """
        :param n: 特征维度
        :param k: 隐向量维度
        """
        super(FMLayer, self).__init__()
        self.dtype = torch.float
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)   # 前两项线性层
        '''
        torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor的区别
        就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
        注意到，nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torth.Tensor对象的默认值相反。
        在nn.Module类中，pytorch也是使用nn.Parameter来对每一个module的参数进行初始化的。
        '''
        self.v = nn.Parameter(torch.randn(self.n, self.k))   # 交互矩阵
        nn.init.uniform_(self.v, -0.1, 0.1)

    def fm_layer(self, x):
        # x 属于 R^{batch*n}
        linear_part = self.linear(x)
        #print("linear_part",linear_part.shape)
        # linear_part = torch.unsqueeze(linear_part, 1)
        # print(linear_part.shape)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v)  # out_size = (batch, k) # 矩阵a和b矩阵相乘。 vi,f * xi
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))  # out_size = (batch, k)
        # 这里torch求和一定要用sum
        inter = 0.5 * torch.sum(torch.sub(torch.pow(inter_part1, 2), inter_part2),1,keepdim=True)
        #print("inter",inter.shape)
        output = linear_part + inter
        output = torch.sigmoid(output)
        #print(output.shape) # out_size = (batch, 1)
        return output
    def forward(self, x):
        return self.fm_layer(x)


class SelfAttention_Layer(nn.Module):
    def __init__(self, dim):
        super(SelfAttention_Layer, self).__init__()
        self.dim = dim
        self.Weight = nn.Parameter(nn.init.uniform_(torch.empty(self.dim, self.dim)))

    def forward(self, inputs, **kwargs):
        q, k, v, mask = inputs
        # pos encoding
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        # Nonlinear transformation
        q = F.relu(torch.matmul(q, self.Weight))  # (None, seq_len, dim)
        k = F.relu(torch.matmul(k, self.Weight))  # (None, seq_len, dim)
        mat_qk = torch.matmul(q, k.transpose(1, 2))  # (None, seq_len, seq_len)

        dk = torch.FloatTensor([self.dim])
        # Scaled
        scaled_att_logits = mat_qk / torch.sqrt(dk)
        # Mask
        mask = torch.unsqueeze(mask, 1).repeat(1, q.shape[1], 1)  # (None, seq_len, seq_len)
        paddings = torch.ones(scaled_att_logits.shape) * (-2 ** 32 + 1)
        outputs = torch.where(mask.eq(0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = F.softmax(outputs, dim=-1)  # (None, seq_len, seq_len)
        # output
        outputs = torch.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = torch.mean(outputs, dim=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                     np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.FloatTensor(pos_encoding)

class DNN(nn.Module):
    """
    Deep part
    """
    def __init__(self, input_dim, hidden_units, dnn_dropout=0.):
        """
        DNN part
        :param hidden_units: A list like `[unit1, unit2,...,]`. List of hidden layer units's numbers
        :param activation: A string. Activation function.
        :param dnn_dropout: A scalar. dropout number.
        """
        super(DNN, self).__init__()
        orderdict=[]
        for i in range(len(hidden_units)):
            if i==0:
                orderdict.append(('fc1', nn.Linear(input_dim, hidden_units[0])))
                orderdict.append(('relu1', nn.ReLU()))
            else:
                orderdict.append(('fc'+str(i+1), nn.Linear(hidden_units[i-1], hidden_units[i])))
                orderdict.append(('relu' + str(i + 1), nn.ReLU()))

        self.dnn_network = nn.Sequential(OrderedDict(orderdict))
        self.dropout = nn.Dropout(dnn_dropout)

    def forward(self, inputs, **kwargs):
        x = inputs
        x = self.dnn_network(x)
        x = self.dropout(x)
        return x