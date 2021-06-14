import pandas as pd

'''
虚拟环境env
主要功能：对action做出反应给出reward和下一个状态observation_
'''


class virtual_env(object):
    def __init__(self, feature_columns, train_X):
        self.data = train_X

    def next_step(action):

        return observation_, reward
