import torch as T
import numpy as np
from tqdm import tqdm
from ou_noise import OUActionNoise
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym
from ddpg import DDPG

# test agent on lunar lander
env = gym.make('LunarLanderContinuous-v2')

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

model = DDPG(env,1e-4,1e-3,ActorNetwork,CriticNetwork,[400,300],
                1e6,device=device,tau=0.001,gamma=0.99,batch_size=128)
rews = model.learn(num_episodes=1000,verbose=1)    
plt.plot(rews)
plt.show()

# If networks saved load using below line
# model.load_checkpoint()
    
def predict(model, state):
    obs = T.from_numpy(state).type(T.float).to(device)
    a = model.actor(obs)
    action = a.cpu().detach().numpy()
    
    return action

done = False
state = env.reset()
with T.no_grad():
    model.actor.eval()
    while True:
        action = predict(model, state)
        state, reward, done,_ = env.step(action)
        env.render()
        if done:
            state = env.reset()