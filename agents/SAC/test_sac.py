import torch as T
import numpy as np
from tqdm import tqdm
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym
from sac import SAC

env = gym.make('Pendulum-v1')

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

model = SAC.load("log/sac_Pendulum-v1")

done = False
state = env.reset()
cum_reward=0
with T.no_grad():
    model.actor.eval()
    while True:
        action = model.predict(state)
        state, reward, done,_ = env.step(action)
        env.render()
        cum_reward+=reward
        if done:
            print(cum_reward)
            cum_reward=0
            state = env.reset()