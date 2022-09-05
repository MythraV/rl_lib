import numpy as np
from tqdm import tqdm
import pickle
import itertools
import gym
from collections import defaultdict
import dill

class MCAgent():
    def __init__(self, env, policy = None, gamma=0.99, epsilon=0.01):
        ''' State space is 3 tuple (player sum, dealer card, usable ace)'''
        ''' player sum : 4-21, dealer card: Ace-10, usable ace: 1/0'''
        self.num_actions = env.action_space.n
        
        if policy is None:
            self.policy = defaultdict(lambda: [1.0/self.num_actions]*self.num_actions)
        else:
            self.policy=policy

        self.states = {}
        for i,j in enumerate(itertools.product(range(4,22),range(1,11),range(2))):
            self.states[j]=i
        
        self.q = defaultdict(lambda: [0]*self.num_actions)
        self.ret = defaultdict(lambda: [])
        self.env = env
        #
        self.gamma = gamma
        self.epsilon = epsilon

        # Setup action space
        off = 0
        if hasattr(env.action_space,'start'):
            off = env.action_space.start
        self.actions = [off+i for i in range(self.num_actions)]
        
        self.rand_gen = np.random.RandomState()

    def argmax(self, qv):
        maxq = float('-inf')
        ties = []
        for i in range(len(qv)):
            if qv[i]>maxq:
                maxq=qv[i]
                ties=[]
            
            if qv[i]==maxq:
                ties.append(i)
        return self.actions[self.rand_gen.choice(ties)]

    def act(self, state):
        ''' Choose actions epsilon greedy '''
        action = self.rand_gen.choice(self.actions, p=self.policy[state])
        return action

    def get_episode(self, state):
        done = False
        action = self.act(state)
        seq = []
        first_visit = set()
        while not done:
            n_state, reward, done,_=self.env.step(action)
            fq = 1 # First visit
            if (state,action) in first_visit:
                fq = 0
            else:
                first_visit.add((state,action))

            seq.append((state,action,reward,fq))
            state=n_state
            action=self.act(state)
        return seq

    def learn(self, n_episodes=10):
        for i in tqdm(range(n_episodes),desc='Episode',total=n_episodes):
            state = self.env.reset()
            seq = self.get_episode(state)
            ret_g = 0
            for s,a,r,fq in reversed(seq):
                ret_g = self.gamma*ret_g+r
                
                if fq:      # Check if first visit
                    self.ret[(s,a)].append(ret_g)
                    self.q[s][a] = np.mean(self.ret[(s,a)])

                    # Update the policy
                    a_star = self.argmax(self.q[s])
                    for n_a in self.actions:
                        if n_a == a_star:
                            self.policy[s][n_a] = 1-self.epsilon+self.epsilon/self.num_actions
                        else:
                            self.policy[s][n_a] = self.epsilon/self.num_actions

    def predict(self, state):
        action = self.argmax(self.q[state])
        return action
    
    def save(self, nam='MCagent_model'):
        with open(nam,'wb') as f:
            dill.dump(self,f) 

    @classmethod
    def load(self, nam='MCagent_model'):
        with open(nam,'rb') as f:
            self = dill.load(f)
        return self

if __name__=="__main__":
    env = gym.make('Blackjack-v1')
    train = False
    if train:
        mc = MCAgent(env)
        mc.learn(200000)
        mc.save('MCAgent_blackjack')
    # Test model
    mc = MCAgent.load('MCAgent_blackjack')
    outcomes=[]
    # Eval for 1000 episodes
    for i in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = mc.predict(state)
            state, reward, done, _ = env.step(action)
            if done:
                outcomes.append(reward)
    out_arr = np.array(outcomes)
    print(len(out_arr[out_arr>0]))