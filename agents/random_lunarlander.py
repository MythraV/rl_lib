import gym
import numpy as np

def weak_heuristic_policy(state):
    print(state)
    if state[0]<0:
        action=3
    elif state[0]>0:
        action=1
    if state[1]>0:
        action=np.random.choice([2,2,action,action,action,0,0,0])
    return action

if __name__=="__main__":
    env = gym.make('LunarLander-v2')
    done = False
    state = env.reset()
    while not done:
        action = weak_heuristic_policy(state)
        state,reward,done,info = env.step(action)
        env.render()
        if done:
            state = env.reset()