import torch as T
import numpy as np
from tqdm import tqdm
from noise import GaussNoise
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym

class TD3():
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
        checkpoint_file='td3',
        noise=None,
        batch_size=100,
        gamma=0.99,
        tau=0.001,
        clip=0.5,
        update_actor_interval=2,
        warmup=1000
        ) -> None:
        
        self.env = env
        self.renderq = render
        
        self.num_action_feats = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0] # number of state variables
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup

        # policy network
        self.device = T.device('cpu') if device is None else device
        optim = T.optim.Adam

        self.actor_checkpt_file='log/'+checkpoint_file+'_actor_'+str(lr_actor)+'_'
        self.actor = actor_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_actor,device=self.device,
                                checkpt_file=self.actor_checkpt_file)

        self.critic1_checkpt_file='log/'+checkpoint_file+'_critic1_'+str(lr_critic)
        self.critic1 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.critic1_checkpt_file)
        
        self.critic2_checkpt_file='log/'+checkpoint_file+'_critic2_'+str(lr_critic)
        self.critic2 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.critic2_checkpt_file)
        
        # Initialize target networks
        self.target_actor_checkpt_file='log/'+checkpoint_file+'_tactor_'+str(lr_actor)+'_'
        self.target_actor = actor_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_actor,device=self.device,
                                checkpt_file=self.target_actor_checkpt_file)

        self.target_critic1_checkpt_file='log/'+checkpoint_file+'_tcritic1_'+str(lr_critic)
        self.target_critic1 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.target_critic1_checkpt_file)
        
        self.target_critic2_checkpt_file='log/'+checkpoint_file+'_tcritic2_'+str(lr_critic)
        self.target_critic2 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.target_critic2_checkpt_file)

        if noise is None:
            self.noise = GaussNoise(mu=np.zeros(self.num_action_feats),sigma=0.1)
        else:
            self.noise = noise
            
        self.replay_buffer = ReplayBuffer(self.num_states,self.num_action_feats, replay_buffer_size)
        
        # Update targnet network parameters
        self.soft_update(self.target_actor, self.actor, tau=1)      # tau=1 , because full copy initially
        self.soft_update(self.target_critic1, self.critic1, tau=1)      # tau=1 , because full copy initially
        self.soft_update(self.target_critic2, self.critic2, tau=1)      # tau=1 , because full copy initially

        self.batch_size = batch_size
        self.tau=tau
        self.gamma = gamma 

        self.criterion = T.nn.MSELoss()
        self.last_state = None
        self.last_action = None

        # action limits
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low
        self.min_action_tensor = T.from_numpy(self.min_action).type(T.float).to(self.device)
        self.max_action_tensor = T.from_numpy(self.max_action).type(T.float).to(self.device)
        self.actor_step = 0
        self.warmup_cnt = 0
        self.clip = clip

    
    def choose_action(self, state):        
        n = self.noise()
        if self.warmup_cnt<self.warmup:
            action = n
        else:
            obs = T.from_numpy(state).type(T.float).to(self.device)
            a = self.actor(obs)
            action = a.cpu().detach().numpy()+n
        
        action = np.clip(action, self.min_action, self.max_action)
        self.warmup_cnt+=1
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

            # Compute update y
            #   get actions from target 
            actions_t = self.target_actor(obs_)
            #   add noise to sample for regularization
            #   clamp action noise to ensure limits
            exp_noise = np.random.normal(0,0.2,size=(self.batch_size,self.num_action_feats))
            exp_noise = T.from_numpy(exp_noise).type(T.float).to(self.device)
            exp_noise = T.clamp(exp_noise, -self.clip, self.clip)

            actions_t = actions_t + exp_noise
            actions_t = T.clamp(actions_t, self.min_action_tensor,self.max_action_tensor)
            #   get critic outputs
            q_t1 = self.target_critic1(obs_,actions_t)
            q_t2 = self.target_critic2(obs_,actions_t)

            min_q_t = T.min(q_t1,q_t2)
            y = rewards + self.gamma*(min_q_t.view(-1))*(~done)
            y = y.view(self.batch_size,1)

            # Compute value from critic
            q1_last_state = self.critic1(obs, actions)
            q2_last_state = self.critic2(obs, actions)

            # Update critic params
            self.critic1.optim.zero_grad()
            self.critic2.optim.zero_grad()
            
            loss1 = self.criterion(y,q1_last_state)
            loss2 = self.criterion(y,q2_last_state)

            loss = loss1 + loss2
            loss.backward()

            self.critic1.optim.step()
            self.critic2.optim.step()

            self.actor_step+=1
            # Update actor params
            if self.actor_step%self.update_actor_interval==0:
                self.actor_step=0
                self.actor.optim.zero_grad()
                a_loss = -self.critic1(obs, self.actor(obs))
                a_loss = T.mean(a_loss)
                a_loss.backward()
                self.actor.optim.step()

                # Soft update target network params
                self.soft_update(self.target_actor, self.actor, tau=self.tau)   
            
                self.soft_update(self.target_critic1, self.critic1, tau=self.tau)
                self.soft_update(self.target_critic2, self.critic2, tau=self.tau) 

        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state

        return action


    def learn(self, num_episodes, verbose=0):
        avg_rewards = []
        rewards_arr = []
        base_score = self.env.reward_range[0] 

        for i in tqdm(range(num_episodes),desc='Episode',total=num_episodes):#range(num_episodes):#

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
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()
    
    def load_checkpoint(self,eps=0):
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.target_critic2.load_checkpoint()
    
    def soft_update(self,target,src,tau=0.001):
        with T.no_grad():
            for target_params, params in zip(target.parameters(), src.parameters()):
                target_params.data.copy_(tau*params.data+(1-tau)*target_params.data)

if __name__=="__main__":
    # test agent on lunar lander
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('BipedalWalker-v3')

    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    model = TD3(env,1e-3,1e-3,ActorNetwork,CriticNetwork,[400,300],
                    1e6,device=device,tau=0.005,gamma=0.99,batch_size=100)
    rews = model.learn(num_episodes=1000,verbose=1)    
    plt.plot(rews)
    plt.show()
