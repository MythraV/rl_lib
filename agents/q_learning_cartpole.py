import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import dill
from collections import defaultdict
import gym

class QAgent():
    def __init__(self,env,params={}):
        #Learning rate
        self.alpha = params['alpha'] if 'alpha' in params else 0.001
        self.epsilon = params['epsilon'] if 'epsilon' in params else 0.05
        self.gamma = params['gamma'] if 'gamma' in params else 0.9
        self.renderq = params['render'] if 'render' in params else False

        self.env = env
        self.actions = [i for i in range(env.action_space.n)]
        self.num_actions = env.action_space.n

        self.q = defaultdict(lambda: [0]*self.num_actions)
        self.last_state = None
        self.last_action = None

        rand_seed = params['random_seed'] if 'random_seed' in params else 5
        self.rand_gen = np.random.RandomState(rand_seed)

        # State transformation fn
        self.digitize_state = params['state_tfn'] if 'state_tfn' in params else self._digitize_state

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

    def _digitize_state(self,state):
        return state

    def start(self):
        self.last_state = self.digitize_state(self.env.reset())
        action = self.act(self.last_state)
        self.last_action = action
        return action

    def act(self, state):
        ''' Choose actions epsilon greedy '''
        if self.rand_gen.rand()<self.epsilon:
            action = self.rand_gen.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state])
        
        return action

    def step(self, state, reward):
        
        # Update q
        self.q[self.last_state][self.last_action]+=self.alpha*(reward
                                                             +self.gamma*np.max(self.q[state])
                                                             -self.q[self.last_state][self.last_action])

        self.last_action = self.act(self.last_state)
        self.last_state = state
        return self.last_action

    def end(self, reward):      
        # Update q
        self.q[self.last_state][self.last_action]+=self.alpha*(reward
                                                             -self.q[self.last_state][self.last_action])


    def learn(self, n_episodes=10,verbose=0):
        avg_rewards = []
        rewards_arr=[]
        delta_eps = 2*(self.epsilon-0.01)/n_episodes
        for i in tqdm(range(n_episodes),desc='Episode',total=n_episodes):
            if i<n_episodes//2:
                self.epsilon -= delta_eps
            cum_rewards=0

            self.env.reset()
            action = self.start()
            done = False
            while not done:
                cstate, reward, done, _ = self.env.step(action)
                state = self.digitize_state(cstate)
                if not done:
                    action = self.step(state, reward)
                else:
                    self.end(reward)
                if self.renderq:
                    self.env.render()
                cum_rewards+=reward
            
            rewards_arr.append(cum_rewards)
            if verbose:
                if (i+1)%1000==0:
                    print('episode ', i,
                             'Cummulative Reward: ', cum_rewards,
                             'epsilon: ', self.epsilon)

            if (i+1)%100==0:
                avg_rewards.append(np.mean(rewards_arr))
                rewards_arr=[]
        return avg_rewards

    def predict(self, state):
        action = self.argmax(self.q[state])
        return action
    
    def save(self, nam='Qagent_env.npy'):
        with open(nam,'wb') as f:
            dill.dump(self,f) 

    @classmethod
    def load(self,nam):
        self = pickle.load(open(nam,'rb'))
        return self

def state_digitize_cartpole(state):
    nstate = [0]*4
    nstate[0] = int((max(min(state[0],2.4),-2.4)+2.4)//(2*0.24))    # States from -2.4,2.4 20 states
    nstate[1] = int((max(min(state[1],5),-5)+5)//(0.5*2))    # States from -2.4,2.4 20 states
    nstate[2] = int((max(min(state[2],0.2095),-0.2095)+0.2095)//(2*0.02095))    # States from -2.4,2.4 20 states
    nstate[3] = int((max(min(state[3],4),-4)+4)//(0.4*2))    # States from -2.4,2.4 20 states
    
    return tuple(nstate)

if __name__=="__main__":
    #test case cartpole
    env = gym.make('CartPole-v1')

  
    agent_params = {'render':False,
                    'epsilon':1,
                    'alpha':0.1,
                    'gamma':0.99,
                    'state_tfn':state_digitize_cartpole}
    qag = QAgent(env, agent_params)
    rews = qag.learn(50000,verbose=1)

    np.save('cartpole_rewards.npy',np.array(rews))
    plt.plot(rews)
    plt.show()
