from recommend_env import virtual_env
from Models import Deepnet
from Modules import *
from utils import *
from RL_brain import DeepQNetwork
import time

'''
for eoch in epoches:
    for user_id in users:
    step=0
        for k in K_times:
            init(s)
            action = RL_brain.choose_Action(observation)
            observation_, reward, done = virtual_env.next_step(action)
            RL_brain.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_
'''
# prepare datasets
file1 = './ml-1m/ratings.dat'
file2='./ml-1m/users.dat'
epoches = 5
K_times = 40
data_df = pd.read_csv(file1, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
PATH = './checkpoints'
launchTimestamp = str(time.time())
#device = torch.device('cuda:0')
feature_columns, train_X, val_X, test_X = create_implicit_ml_1m_dataset(file1, file2, embed_dim=100, maxlen=K_times)
users, seqs, feats = train_X
users_list = np.unique(users)


RL = DeepQNetwork(feature_columns, 150, 3953, n_features=K_times+30, memory_size=2000)
#RL.eval_net.to(device)
#RL.target_net.to(device)
for epoch in range(epoches):
    for user_id in users_list:
        env = virtual_env(user_id, train_X, data_df)
        index = np.argwhere(users == user_id)
        index = np.squeeze(index)
        need_seq = seqs[index]
        need_feats = np.squeeze(np.array(feats)[index])
        good_pools = need_seq[:, -1]
        observation = np.concatenate((np.expand_dims(need_seq[0],0), np.expand_dims(need_feats[0],0)), axis=1)
        for k in range(K_times):
            action = RL.choose_action(observation, good_pools)
            print(action)
            observation_, reward = env.next_step(observation, action)
            RL.store_transition(np.squeeze(observation), action, reward, np.squeeze(observation_))
            if (k > 20) and (k % 5 == 0):
                RL.learn()
                # swap observation
            observation = observation_
    print("epoch is {}, min loss is {}".format(epoch, min(RL.cost_his)))
    torch.save({'epoch': epoch + 1, 'state_dict': RL.eval_net.state_dict(), 'best_loss': min(RL.cost_his)},
               PATH + '/m-' + launchTimestamp + '-' + '.pth.tar')

