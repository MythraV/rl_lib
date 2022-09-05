import numpy as np
import torch
import gym
from collections import defaultdict
from tqdm import tqdm
import dill
import torch.nn.functional as F
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self,input_dims,actor_out,critic_out) -> None:
        super(ActorCriticNetwork,self).__init__()

        self.fc1 = torch.nn.Linear(input_dims,2048)
        self.fc2 = torch.nn.Linear(2048,2048)
        self.fc3 = torch.nn.Linear(2048,actor_out)
        self.fc4 = torch.nn.Linear(2048,critic_out)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Actor output
        x_a = F.softmax(self.fc3(x),dim=0)
        #Critic output
        x_c = self.fc4(x)
        return x_a, x_c

class ActorCritic():
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
        self.policy = policy_net(self.num_states,self.num_actions,1)
        if 'device' in params:
            self.device = params['device']
            self.policy.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.alpha)

        self.log_prob = None
        self.last_valu = None
        self.last_state = None
        self.delta = None # TD error

    def _random_policy(self,state):
        return self.rand_gen.choice(self.actions)

    def act(self, state):
        obs = torch.from_numpy(state).to(self.device)
        probs,_ = self.policy(obs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(action)
        
        # Store log prob
        self.log_prob = log_probs
        return action.item()

    def start(self, state):
        action = self.act(state)
        self.last_state = state
        return action

    def step(self, state, reward_):

        # Compute
        last_obs = torch.from_numpy(self.last_state).to(self.device)
        obs = torch.from_numpy(state).to(self.device)
        _, last_valu = self.policy(last_obs)
        _, valu = self.policy(obs)
        
        reward = torch.Tensor([reward_]).to(self.device)
        
        self.delta =  reward + self.gamma*valu - last_valu
        
        # Get next action and state
        action = self.act(state)
        
        self.last_state = state

        return action 

    def gen_episode(self):
        state = self.env.reset()
        done = False
        action = self.start(state)
        
        while not done:
            state, reward, done, _ = self.env.step(action)
            action = self.step(state,reward)

    def learn(self, num_episodes=10, verbose=0):
        avg_rewards = []
        rewards_arr=[]
        
        for i in tqdm(range(num_episodes),desc='Episode',total=num_episodes):
            self.optimizer.zero_grad()
            state = self.env.reset()
            done = False
            action = self.start(state)

            nu = 1
            cum_reward = 0
            # Iterate per episode
            while not done:
                state, reward, done, _ = self.env.step(action)
                action = self.step(state,reward)

                actor_loss = -self.delta*self.log_prob*nu
                critic_loss = self.delta*self.delta

                loss = actor_loss + critic_loss
                loss.backward()
                # Update parameters
                self.optimizer.step()

                cum_reward += reward
                nu *= self.gamma

            # Plot rewards
            if verbose:
                rewards_arr.append(cum_reward)    
                if (i+1)%100==0:
                    avg_rewards.append(np.mean(rewards_arr))
                    rewards_arr = []

        if verbose:
            return avg_rewards


    def save(self, nam='ActorCritic'):
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
                    'alpha':5e-6,
                    'gamma':0.99,
                    'policy':ActorCriticNetwork,
                    'device':device
                    }

    model = ActorCritic(env,model_params)
    rews = model.learn(num_episodes=2100,verbose=1)
    model.save('Models/ActorCritic_lunarlander')
    
    plt.plot(rews)
    plt.show()

    