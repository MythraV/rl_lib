import numpy as np
from tqdm import tqdm
import pickle
import gym

class TD0Prediction():
    def __init__(self,env, policy={}, gamma=0.99, alpha=0.1, num_states=10):
        ''' State space is 4 tuple (x pos, x vel, pole angle, angular vel)'''
        ''' check gym/envs/cartpole for ref'''
        self.policy = policy
        self.num_states = num_states
        self.state_lim = [env.observation_space.low[2],env.observation_space.high[2]] 
        self.state_step = (self.state_lim[1]-self.state_lim[0])/(2*self.num_states)
        
        self.v = np.zeros(self.num_states)
        self.env = env
        
        # Update params
        self.gamma = gamma
        self.alpha = alpha

        self.last_state = None
        self.last_action = None

    def get_state(self,state):
        if state[2]<self.state_lim[0]/2:
            return 0
        elif state[2]>self.state_lim[1]/2:
            return 9
        return int((state[2]-self.state_lim[0]/2)//self.state_step)

    def start(self, state):
        action = self.policy(state)
        self.last_action = action
        self.last_state = state

    def step(self, state, reward):
        state_t_1 = self.get_state(self.last_state)
        state_t = self.get_state(state)
        self.v[state_t_1] += self.alpha*(reward
                                    +self.gamma*self.v[state_t]
                                    -self.v[state_t_1])
        # Update state and action
        self.last_action = self.policy(state)
        self.last_state = state
        
    def end(self, reward):
        state_t_1 = self.get_state(self.last_state)
        self.v[state_t_1] += self.alpha*(reward-self.v[state_t_1])
        
    def learn(self, n_episodes=10):
        for i in tqdm(range(n_episodes),desc='Episode',total=n_episodes):
            state = self.env.reset()
            done = False
            self.start(state)
            while not done:
                state, reward, done,_=self.env.step(self.last_action)
                if not done:
                    self.step(state,reward)
                else:
                    self.end(reward)

        return self.v    
        
    def save(self, nam='TD0_values'):
        with open(nam,'wb') as f:
            pickle.dump(self.v,f) 


def policy_cartpole(state):
    # if pole angle -ve move left else right
    action = 0 if state[2]<0 else 1
    return action

if __name__=="__main__":
    env = gym.make('CartPole-v0')
    td = TD0Prediction(env,policy_cartpole)
    v = td.learn(5000)
    print(v)
    td.save('TD0_values_cartpole')