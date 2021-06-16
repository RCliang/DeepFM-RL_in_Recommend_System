import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from Modules import FMLayer, SelfAttention_Layer, DNN
# 电影数量：3953
# 最终全连接输出： 3953


class Deepnet(nn.Module):
    def __init__(self, feature_columns, all_dim=40, gamma=0.5, w=0.5, max_len=40):
        super(Deepnet, self).__init__()
        self.user_fea_col, self.item_fea_col = feature_columns
        self.all_dim = all_dim
        self.gamma = gamma
        self.w = w
        self.max_len = max_len
        # self.user_embedding = Embedding(num_embeddings =self.user_fea_col['feat_num'],
        #                                embedding_dim =self.user_fea_col['embed_dim'])
        self.item_embedding = Embedding(num_embeddings=self.item_fea_col['feat_num'],
                                        embedding_dim=self.item_fea_col['embed_dim'])
        self.self_attention = SelfAttention_Layer(
            dim=self.item_fea_col['embed_dim'])
        self.FM = FMLayer(30, 50)
        self.dnn = DNN(30, [56, 128, 50])
        self.fc1 = nn.Linear(self.all_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 3953)

    def forward(self, inputs):
        seq_inputs, feat_inputs = inputs[:,:self.max_len], inputs[:,self.max_len:]
        mask = torch.tensor(seq_inputs).ne(0).type(
            torch.FloatTensor)  # (batch, maxlen)
        # user_embed = self.user_embedding(torch.LongTensor(user_inputs)) # (batch, dim)
        seq_embed = self.item_embedding(
            torch.LongTensor(seq_inputs))  # (None, maxlen, dim)
        short_interest = self.self_attention(
            [seq_embed, seq_embed, seq_embed, mask])  # (batch, dim)
        deep_feat = self.dnn(torch.FloatTensor(feat_inputs))
        wide_feat = self.FM(torch.FloatTensor(feat_inputs))
        wide_deep = self.w*deep_feat+(1-self.w)*wide_feat
        all_feat = torch.cat((wide_deep, short_interest), 1)
        res = F.relu(self.fc1(all_feat))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)
        return res
