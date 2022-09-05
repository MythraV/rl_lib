import numpy as np
import torch
import gym
from collections import defaultdict
from tqdm import tqdm
import dill
import torch.nn.functional as F
from matplotlib import pyplot as plt

class PolicyNet(torch.nn.Module):
    def __init__(self,input_dims,output_dims) -> None:
        super(PolicyNet,self).__init__()

        self.fc1 = torch.nn.Linear(input_dims,128)
        self.fc2 = torch.nn.Linear(128,128)
        self.fc3 = torch.nn.Linear(128,output_dims)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=0)
        return x

class REINFORCE():
    def __init__(self, env, params):
        #Learning rate
        self.alpha = params['alpha'] if 'alpha' in params else 0.001
        self.gamma = params['gamma'] if 'gamma' in params else 0.9
        self.renderq = params['render'] if 'render' in params else False

        self.env = env
        self.actions = [i for i in range(env.action_space.n)]
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0] # number of state variables

        rand_seed = params['random_seed'] if 'random_seed' in params else 5
        self.rand_gen = np.random.RandomState(rand_seed)

        #policy network
        policy_net = params['policy'] if 'policy' in params else self._random_policy
        self.policy = policy_net(self.num_states,self.num_actions)
        if 'device' in params:
            self.device = params['device']
            self.policy.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.alpha)

        # Memory buffers
        self.reward_buffer = []
        self.action_buffer = []

    def _random_policy(self,state):
        return self.rand_gen.choice(self.actions)

    def act(self, state):
        obs = torch.from_numpy(state).to(self.device)
        probs = self.policy(obs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        self.action_buffer.append(log_probs)

        return action.item()

    def start(self, state):
        action = self.act(state)
        return action

    def step(self, state, reward):
        
        # Store reward
        self.reward_buffer.append(reward)
        action = self.act(state)
        return action 

    def gen_episode(self):
        state = self.env.reset()
        done = False
        action = self.start(state)
        
        while not done:
            state, reward, done, _ = self.env.step(action)
            action = self.step(state,reward)
    
    def _compute_returns(self,):
        G = np.zeros_like(self.reward_buffer, dtype=np.float64)
        G[-1] = self.reward_buffer[-1]
        for t,r in enumerate(reversed(self.reward_buffer[:-1])):
            G[~(t+1)] = self.gamma*G[~t]+r
        return G

    def learn(self, num_episodes=10, verbose=0):
        avg_rewards = []
        rewards_arr=[]
        for i in tqdm(range(num_episodes),desc='Episode',total=num_episodes):
            self.optimizer.zero_grad()
            # Generate an episode
            self.gen_episode()
            # Get returns
            G = torch.from_numpy(self._compute_returns()).to(self.device)
            
            # Compute loss and update params
            loss = 0
            for g,logprob in zip(G,self.action_buffer):
                loss += -g*logprob
            loss.backward()
            self.optimizer.step()

            # Plot rewards
            if verbose:
                rewards_arr.append(np.sum(self.reward_buffer))
                if (i+1)%100==0:
                    avg_rewards.append(np.mean(rewards_arr))
                    rewards_arr = []

            # Reset buffers
            self.action_buffer = []
            self.reward_buffer = []
        if verbose:
            return avg_rewards


    def save(self, nam='Reinforce'):
        with open(nam,'wb') as f:
            dill.dump(self,f) 

    @classmethod
    def load(self,nam):
        self = dill.load(open(nam,'rb'))
        return self

if __name__=="__main__":
    # test agent on lunar lander
    env = gym.make('LunarLander-v2')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_params = {'render':False,
                    'alpha':0.0005,
                    'gamma':0.99,
                    'policy':PolicyNet,
                    'device':device
                    }

    model = REINFORCE(env,model_params)
    rews = model.learn(num_episodes=3000,verbose=1)
    model.save('Models/Reinforce_lunarlander')
    
    plt.plot(rews)
    plt.show()

    