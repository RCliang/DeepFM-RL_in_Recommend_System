import pandas as pd
import numpy as np
'''
虚拟环境env
主要功能：对action做出反应给出reward和下一个状态observation_
'''
class virtual_env(object):
    def __init__(self, user_id, train_X, data_df):
        self.user_df = data_df[(data_df.user_id==user_id)]
        self.user_id = user_id
        index = np.argwhere(train_X[0] == user_id)
        index = np.squeeze(index)
        self.need_seq = train_X[1][index]
        self.need_feats = np.array(train_X[2])[index]
        self.all_movies = self.need_seq[:,-1]
    def next_step(self, observation, action):
        if action in self.all_movies:
            idx = np.argwhere(self.all_movies == action)
            observation_=np.concatenate((np.squeeze(self.need_seq[idx, :],0), np.squeeze(self.need_feats[idx,:],0)), axis=1)
            temp = self.user_df[self.user_df['item_id'] == action]
            reward = temp['label'].values[0]
            print("reward is {}".format(reward))
        else:
            observation_ = observation
            reward = 0
        return observation_, reward
