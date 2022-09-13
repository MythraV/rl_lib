import torch as T
import numpy as np
from tqdm import tqdm
from ou_noise import OUActionNoise
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym

class DDPG():
    def __init__(
        self,
        env,
        lr_actor,
        lr_critic,
        actor_net,
        critic_net,
        hidden_layer_dims=[400,300],
        replay_buffer_size=1e5,
        device=None,
        render=False,
        checkpoint_file='ddpg',
        noise=None,
        batch_size=128,
        gamma=0.99,
        tau=0.001 
        ) -> None:
        
        self.env = env
        self.renderq = render
        
        self.num_action_feats = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape # number of state variables

        # policy network
        self.device = T.device('cpu') if device is None else device
        optim = T.optim.Adam

        self.actor_checkpt_file='log/'+checkpoint_file+'_actor_'+str(lr_actor)+'_'
        self.actor = actor_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_actor,device=self.device,
                                checkpt_file=self.actor_checkpt_file)

        self.critic_checkpt_file='log/'+checkpoint_file+'_critic_'+str(lr_critic)
        self.critic = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, w_decay=0.01, device=self.device,
                                checkpt_file=self.critic_checkpt_file)
        
        # Initialize target networks
        self.target_actor_checkpt_file='log/'+checkpoint_file+'_tactor_'+str(lr_actor)+'_'
        self.target_actor = actor_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_actor,device=self.device,
                                checkpt_file=self.target_actor_checkpt_file)

        self.target_critic_checkpt_file='log/'+checkpoint_file+'_tcritic_'+str(lr_critic)
        self.target_critic = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, w_decay=0.01, device=self.device,
                                checkpt_file=self.target_critic_checkpt_file)

        if noise is None:
            self.noise = OUActionNoise(mu=np.zeros(self.num_action_feats),
                                        sigma=0.15,
                                        theta=0.15)
        else:
            self.noise = noise
            
        self.replay_buffer = ReplayBuffer(self.num_states,self.num_action_feats, replay_buffer_size)
        
        # Update targnet network parameters
        self.soft_update(self.target_actor, self.actor, tau=1)      # tau=1 , because full copy initially
        self.soft_update(self.target_critic, self.critic, tau=1)      # tau=1 , because full copy initially

        self.batch_size = batch_size
        self.tau=tau
        self.gamma = gamma 

        self.criterion = T.nn.MSELoss()
        self.last_state = None
        self.last_action = None
    
    def choose_action(self, state):
        self.actor.eval()       # don't want layer norm stats here

        obs = T.from_numpy(state).type(T.float).to(self.device)
        a = self.actor(obs)
        n = self.noise()
        
        action = a.cpu().detach().numpy()+n
        
        self.actor.train()
        
        return action

    def store_transition(self,state,reward,done):
        self.replay_buffer.add(self.last_state,self.last_action,reward,state,done)
    
    def start(self, state):
        action = self.choose_action(state)
        
        self.last_state = state
        self.last_action = action

        return action

    def step(self, state, reward, done):

        # Store transition in buffer
        self.store_transition(state,reward,done)

        if self.replay_buffer.buffer_len>self.batch_size:
            # Sample minibatch from buffer
            states, actions, rewards, states_, dones = self.replay_buffer.sample(self.batch_size)

            rewards = T.from_numpy(rewards).type(T.float).to(self.device)
            obs = T.from_numpy(states).type(T.float).to(self.device)
            obs_ = T.from_numpy(states_).type(T.float).to(self.device)
            actions = T.from_numpy(actions).type(T.float).to(self.device)
            done = T.tensor(dones).to(self.device)

            # Compute yi
            action_t = self.target_actor(obs_)
            q_state_t = self.target_critic(obs_,action_t)

            y = rewards + self.gamma*(q_state_t.view(-1))*(~done)
            y = y.view(self.batch_size,1)

            # Compute value from critic
            q_last_state = self.critic(obs, actions)

            # Update critic params
            self.critic.optim.zero_grad()
            loss = self.criterion(y,q_last_state)
            loss.backward()
            self.critic.optim.step()

            # Update actor params
            self.actor.optim.zero_grad()
            a_loss = -self.critic(obs, self.actor(obs))
            a_loss = T.mean(a_loss)
            a_loss.backward()
            self.actor.optim.step()

            # Soft update target network params
            self.soft_update(self.target_actor, self.actor, tau=self.tau)   
            self.soft_update(self.target_critic, self.critic, tau=self.tau) 

        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state

        return action


    def learn(self, num_episodes, verbose=0):
        avg_rewards = []
        rewards_arr = []
        base_score = self.env.reward_range[0] 

        for i in range(num_episodes):#tqdm(range(num_episodes),desc='Episode',total=num_episodes):

            self.noise.reset()
            state = self.env.reset()
            done = False
            action = self.start(state)

            cum_reward = 0
            # Iterate per episode
            while not done:
                state, reward, done, _ = self.env.step(action)
                action = self.step(state,reward,done)

                cum_reward += reward
            
            # Plot rewards
            if verbose:
                rewards_arr.append(cum_reward)
                if (i+1)%10==0:
                    print("Episode: ", i+1, " Avg Reward(last 50): ", np.mean(rewards_arr[-50:]),
                     " Reward (base =", str(base_score), "): ", cum_reward)    
                avg_rewards.append(np.mean(rewards_arr[-100:]))
            if avg_rewards[-1]>base_score:   # Save networks
                base_score=avg_rewards[-1]
                self.save_checkpoint(0)

        if verbose:
            return avg_rewards

    def save_checkpoint(self,eps=0):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_checkpoint(self,eps=0):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def soft_update(self,target,src,tau=0.001):
        with T.no_grad():
            for target_params, params in zip(target.parameters(), src.parameters()):
                target_params.data.copy_(tau*params.data+(1-tau)*target_params.data)

if __name__=="__main__":
    # test agent on lunar lander
    env = gym.make('LunarLanderContinuous-v2')

    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    model = DDPG(env,1e-4,1e-3,ActorNetwork,CriticNetwork,[400,300],
                    1e6,device=device,tau=0.001,gamma=0.99,batch_size=128)
    rews = model.learn(num_episodes=1000,verbose=1)    
    plt.plot(rews)
    plt.show()
