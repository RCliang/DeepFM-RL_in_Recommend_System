# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:18:42 2021

@author: A
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_implicit_ml_1m_dataset(file1, file2, trans_score=2, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    # rating_data
    data_df = pd.read_csv(file1, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # user_data
    user_df = pd.read_csv(file2, sep="::", engine='python',
                          names=['user_id', 'gender', 'occupation', 'zip_code'])
    # implicit dataset
    data_df = data_df[data_df.label >= trans_score]

    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    user_df = user_df.sort_values(by=['user_id'])
    enc = OneHotEncoder()
    user_feat = enc.fit_transform(
        user_df[['gender', 'age', 'occupation']]).toarray()

    train_data, val_data, test_data = [], [], []

    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()
        temp_feat = user_feat[user_id-1].tolist()
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data.append([user_id, hist_i, temp_feat])
            elif i == len(pos_list) - 2:
                val_data.append([user_id, hist_i, temp_feat])
            else:
                train_data.append([user_id, hist_i, temp_feat])

    # feature columns
    user_num, item_num = data_df['user_id'].max(
    ) + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'user_feat'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'user_feat'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'user_feat'])
    print('==================Padding===================')

    # create dataset
    def df_to_list(data):
        return [data['user_id'].values, pad_sequences(data['hist'], maxlen=maxlen),
                [np.array(x) for x in data['user_feat']]]

    train_X = df_to_list(train)
    val_X = df_to_list(val)
    test_X = df_to_list(test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, val_X, test_X
