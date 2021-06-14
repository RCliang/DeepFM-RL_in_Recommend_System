from recommend_env import virtual_env
from Models import Deepnet
from Modules import *
from utils import *

'''
for eoch in epoches:
    for user_id in users:
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
