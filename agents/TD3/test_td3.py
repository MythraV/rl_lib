import torch as T
import numpy as np
from tqdm import tqdm
from noise import GaussNoise
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym
from td3 import TD3


# If test agent on lunar lander uncomment below
# env = gym.make('LunarLanderContinuous-v2')
env = gym.make('BipedalWalker-v3')

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

model = TD3(env,1e-3,1e-3,ActorNetwork,CriticNetwork,[400,300],
                1e6,device=device,tau=0.005,gamma=0.99,batch_size=100)
avg_rewards = model.learn(num_episodes=1000,verbose=1)    
plt.plot(avg_rewards)
plt.show()

# If networks are saved use load_checkpoint as below
#model.load_checkpoint()
    
def predict(model, state):
    obs = T.from_numpy(state).type(T.float).to(device)
    a = model.actor(obs)
    action = a.cpu().detach().numpy()
    
    return action

done = False
state = env.reset()
cum_reward=0
with T.no_grad():
    model.actor.eval()
    while True:
        action = predict(model, state)
        state, reward, done,_ = env.step(action)
        env.render()
        cum_reward+=reward
        if done:
            print(cum_reward)
            cum_reward=0
            state = env.reset()