import torch as T
from policies import *
from matplotlib import pyplot as plt
import gym
from sac import SAC


# test agent on Pendulum-v1
env = gym.make('Pendulum-v1')

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

model = SAC(env,3e-4,3e-4,GaussianActorNetwork,CriticNetwork,[256,256],
                1e6,device=device,tau=0.005,gamma=0.99,batch_size=256,
                alpha=0.9,tune_alpha=True,lr_alpha=3e-4)

avg_rewards = model.learn(num_episodes=250,verbose=1)   # Running average over last 100 episodes

model.save('log/sac')       #Save model (Ensure log folder exists)

plt.plot(avg_rewards)
plt.show()

# del model

# if model already exists
# model = SAC.load("log/sac_Pendulum-v1") 

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