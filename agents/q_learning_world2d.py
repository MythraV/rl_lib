import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import pickle

class QAgent():
    def __init__(self,env,params={}):
        #Learning rate
        self.alpha = params['alpha'] if 'alpha' in params else 0.001
        self.epsilon = params['epsilon'] if 'epsilon' in params else 0.05
        self.gamma = params['gamma'] if 'gamma' in params else 0.9
        self.renderq = params['render'] if 'render' in params else False

        self.env = env
        self.env.start()
        self.actions = env.actions
        self.num_actions = len(self.actions)

        self.q = np.zeros([env.num_states,env.num_actions])
        self.last_state = None
        self.last_action = None

        rand_seed = params['random_seed'] if 'random_seed' in params else 5
        self.rand_gen = np.random.RandomState(rand_seed)

    def argmax(self, qv):
        maxq = float('-inf')
        ties = []
        for i in range(len(qv)):
            if qv[i]>maxq:
                maxq=qv[i]
                ties=[]
            
            if qv[i]==maxq:
                ties.append(i)
        return self.rand_gen.choice(ties)

    def start(self):
        self.last_state = self.env.get_state()
        action = self.act(self.last_state)
        self.last_action = action
        return action

    def act(self, state):
        ''' Choose actions epsilon greedy '''
        if self.rand_gen.rand()<self.epsilon:
            action = self.rand_gen.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state,:])
        
        return action

    def step(self, state, reward):
        
        # Update q
        self.q[self.last_state,self.last_action]+=self.alpha*(reward
                                                             +self.gamma*np.max(self.q[state,:])
                                                             -self.q[self.last_state,self.last_action])

        self.last_action = self.act(self.last_state)
        self.last_state = state
        return self.last_action

    def end(self, reward):      
        # Update q
        self.q[self.last_state,self.last_action]+=self.alpha*(reward
                                                             -self.q[self.last_state,self.last_action])

        
    def learn(self, n_episodes=10):
        for i in tqdm(range(n_episodes),desc='Episode',total=n_episodes):
            self.env.reset()
            action = self.start()
            done = False
            while not done:
                state, reward, done, _ = self.env.step(action)
                if not done:
                    action = self.step(state, reward)
                else:
                    self.end(reward)
                if self.renderq:
                    self.env.render()

    def predict(self, state):
        action = self.argmax(self.q[state,:])
        return action
    
    def save(self, nam='Qagent_env.npy'):
        with open(nam,'wb') as f:
            pickle.dump(self,f) 

    @classmethod
    def load(self,nam):
        self = pickle.load(open(nam,'rb'))
        return self

if __name__=="__main__":
    q_model = 1