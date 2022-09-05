import numpy as np
from tqdm import tqdm
import pickle
import itertools
import gym

class MCEstimator():
    def __init__(self,env, policy={}, gamma=0.99):
        ''' State space is 3 tuple (player sum, dealer card, usable ace)'''
        ''' player sum : 4-21, dealer card: Ace-10, usable ace: 1/0'''
        self.policy = policy
        self.states = {}
        for i,j in enumerate(itertools.product(range(4,22),range(1,11),range(2))):
            self.states[j]=i
        self.v = np.zeros(len(self.states))
        self.ret = [[] for i in range(len(self.states))]
        self.env = env
        #
        self.gamma = gamma

    # def start(self):
    #     self.last_state = self.env.get_state()
    #     action = self.act(self.last_state)
    #     self.last_action = action
    #     return action

    # def act(self, state):
    #     ''' Choose actions epsilon greedy '''
    #     if self.rand_gen.rand()<self.epsilon:
    #         action = self.rand_gen.randint(self.num_actions)
    #     else:
    #         action = self.argmax(self.q[state,:])
        
    #     return action

    # def step(self, state, reward):
        
    #     # Update q
    #     self.q[self.last_state,self.last_action]+=self.alpha*(reward
    #                                                          +self.gamma*np.max(self.q[state,:])
    #                                                          -self.q[self.last_state,self.last_action])

    #     self.last_action = self.act(self.last_state)
    #     self.last_state = state
    #     return self.last_action

    # def end(self, reward):      
    #     # Update q
    #     self.q[self.last_state,self.last_action]+=self.alpha*(reward
    #                                                          -self.q[self.last_state,self.last_action])

    def get_episode(self, state):
        done = False
        action = self.policy(state)
        seq = []
        first_visit = set()
        while not done:
            n_state, reward, done,_=self.env.step(action)
            fq = 1 # First visit
            if state in first_visit:
                fq = 0
            else:
                first_visit.add(state)

            seq.append((state,action,reward,fq))
            state=n_state
            action=self.policy(state)
        return seq

    def learn(self, n_episodes=10):
        for i in tqdm(range(n_episodes),desc='Episode',total=n_episodes):
            state = self.env.reset()
            seq = self.get_episode(state)
            ret_g = 0
            for s,a,r,fq in reversed(seq):
                ret_g = self.gamma*ret_g+r
                if fq:
                    self.ret[self.states[s]].append(ret_g)
                    self.v[self.states[s]] = np.mean(self.ret[self.states[s]])

    # def predict(self, state):
    #     action = self.argmax(self.q[state,:])
    #     return action
    
    # def save(self, nam='Qagent_env.npy'):
    #     with open(nam,'wb') as f:
    #         pickle.dump(self,f) 


def policy(state):
    # 1: hit 0: stick
    action = 1 if state[0]<20 else 0
    return action

if __name__=="__main__":
    env = gym.make('Blackjack-v1')
    # mc = MCEstimator(env,policy)
    # print(len(mc.v))
    # mc.learn(500000)
    # with open('mc_policy_eval','wb') as f:
    #     pickle.dump(mc,f)
    with open('mc_policy_eval','rb') as f:
        mc = pickle.load(f)
    print(mc.v[mc.states[(21,3,1)]])
    print(mc.v[mc.states[(4,1,0)]])
