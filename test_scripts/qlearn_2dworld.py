from tabnanny import verbose
import numpy as np
from TD_Agents.q_learning import QAgent
from TD_ObsEnv.world2d_v1 import GridWorld
import pickle
from stable_baselines3 import DQN

env_vars = {'render':True,
            'width':20,
            'length':20,
            'goal':[18,19],
            'obstacles':[[3,3,5,5],[7,3,8,6],[10,15,15,20]]}

agent_vars = {'render':False,
                'alpha':0.1,
                'epsilon':0.2,
                'gamma':1}

train = False

env = GridWorld(env_vars, scalar_states=True)

if train:
    model = QAgent(env,agent_vars)
    model.learn(10000)
    model.save('Models_log/QAgent_world2d_30ep')            

    model2 = DQN('MlpPolicy',env,gamma=1.0,verbose=1)
    model2.learn(total_timesteps=100000, log_interval=10)
    model2.save('Models_log/DQN_world2d')

    del model

model = QAgent.load('Models_log/QAgent_world2d_30ep')
done = False
env.reset()
state = env.start()

while True:
    action = model.predict(state)
    state,reward,done,_ = env.step(action)
    print(reward,done)
    env.render(100)
    if done:
        env.reset()
        state=env.start()
