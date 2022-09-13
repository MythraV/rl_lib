import torch as T
import numpy as np
from tqdm import tqdm
from replay_buffer import ReplayBuffer
from policies import *
from matplotlib import pyplot as plt
import gym
import pickle, bz2


class SAC():
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
        reparam_noise=1e-6,
        batch_size=100,
        gamma=0.99,
        tau=0.001,
        clip=0.5,
        gradient_steps=1,
        update_interval=1,
        warmup=1000,
        alpha = 0.01,
        tune_alpha = True,
        lr_alpha=0.001,
        checkpoint_file='sac',
        save_net_checkpt=False
        ) -> None:
        
        self.env = env
        self.renderq = render
        
        self.num_action_feats = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0] # number of state variables
        self.warmup = warmup


        # action limits
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low
        self.actor_step = 0
        self.warmup_cnt = 0
        self.clip = clip

        self.gradient_steps = gradient_steps
        self.update_interval = update_interval
        
        self.sac_params = SACParams(self.max_action, self.min_action, eps=reparam_noise)

        # policy network
        self.device = T.device('cpu') if device is None else device
        optim = T.optim.Adam

        self.actor_checkpt_file='log/'+checkpoint_file+'_actor_'+str(lr_actor)+'_'
        self.actor = actor_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_actor,device=self.device,
                                checkpt_file=self.actor_checkpt_file, params=self.sac_params)

        self.critic1_checkpt_file='log/'+checkpoint_file+'_critic1_'+str(lr_critic)
        self.critic1 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.critic1_checkpt_file)
        
        self.critic2_checkpt_file='log/'+checkpoint_file+'_critic2_'+str(lr_critic)
        self.critic2 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.critic2_checkpt_file)
        
        # Initialize target networks
        self.target_critic1_checkpt_file='log/'+checkpoint_file+'_tcritic1_'+str(lr_critic)
        self.target_critic1 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.target_critic1_checkpt_file)
        
        self.target_critic2_checkpt_file='log/'+checkpoint_file+'_tcritic2_'+str(lr_critic)
        self.target_critic2 = critic_net(self.num_states, hidden_layer_dims, self.num_action_feats,
                                optim=optim, lr=lr_critic, device=self.device,
                                checkpt_file=self.target_critic2_checkpt_file)

            
        self.replay_buffer = ReplayBuffer(self.num_states,self.num_action_feats, replay_buffer_size)
        
        # Update targnet network parameters
        self.soft_update(self.target_critic1, self.critic1, tau=1)      # tau=1 , because full copy initially
        self.soft_update(self.target_critic2, self.critic2, tau=1)      # tau=1 , because full copy initially

        self.batch_size = batch_size
        self.tau=tau
        self.gamma = gamma
        self.tune_alpha = tune_alpha
        self.alpha = alpha  # temperature parameter

        if tune_alpha:
            self.entropy_target = -self.num_action_feats
            # We optimize log_alpha instead of alpha becuase of nice properties of log
            #   and alpha being constrined above 0
            #  Check https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = T.optim.Adam([self.log_alpha],lr=lr_alpha)

        self.criterion = T.nn.MSELoss()
        self.last_state = None
        self.last_action = None

        # Saving
        self.save_net_checkpt = save_net_checkpt


    def choose_action(self, state):        
        obs = T.from_numpy(np.array([state])).type(T.float).to(self.device)
        a, _, _ = self.actor.sample(obs,reparametrize=False)

        action = a.cpu().detach().numpy()[0]
        
        return action

    def predict(self, state):
        obs = T.from_numpy(state).type(T.float).to(self.device)
        a,_ = self.actor(obs)
        a = a*self.actor.action_scale + self.actor.action_bias
        action = a.cpu().detach().numpy()
        return action
        
    def store_transition(self,state,reward,done):
        self.replay_buffer.add(self.last_state,self.last_action,reward,state,done)
    
    def start(self, state):
        action = self.choose_action(state)
        
        self.last_state = state
        self.last_action = action

        return action

    def train(self,):
        if self.replay_buffer.buffer_len<self.batch_size:
            return
        
        for k in range(self.gradient_steps):
            # Sample minibatch from buffer
            states, actions, rewards, states_, dones = self.replay_buffer.sample(self.batch_size)

            rewards = T.from_numpy(rewards).type(T.float).to(self.device)
            obs = T.from_numpy(states).type(T.float).to(self.device)
            obs_ = T.from_numpy(states_).type(T.float).to(self.device)
            actions = T.from_numpy(actions).type(T.float).to(self.device)
            done = T.tensor(dones).to(self.device)

            # Compute update y
            with T.no_grad():
                actions_pi_, log_probs_, _ = self.actor.sample(obs_,reparametrize=False)

                q_1_ = self.target_critic1(obs_,actions_pi_)
                q_2_ = self.target_critic2(obs_,actions_pi_)
                min_q_ = T.min(q_1_,q_2_)-self.alpha*log_probs_
                
                y = rewards + self.gamma*(~done)*(min_q_.view(-1))
                y = y.view(self.batch_size,1)

            # Compute value from critic
            q_1 = self.critic1(obs, actions)
            q_2 = self.critic2(obs, actions)

            # Update critic params
            self.critic1.optim.zero_grad()
            self.critic2.optim.zero_grad()
            
            loss1 = self.criterion(q_1,y)
            loss2 = self.criterion(q_2,y)

            loss = loss1 + loss2
            loss.backward()

            self.critic1.optim.step()
            self.critic2.optim.step()

            # Update policy
            actions_rpi, log_probs, _ = self.actor.sample(obs,reparametrize=True)
            q_1_rep = self.critic1(obs, actions_rpi)
            q_2_rep = self.critic2(obs, actions_rpi)
            min_q_rep = T.min(q_1_rep,q_2_rep)

            self.actor.optim.zero_grad()
            a_loss = min_q_rep.view(-1)-self.alpha*log_probs.view(-1)
            a_loss = -T.mean(a_loss)
            a_loss.backward()
            self.actor.optim.step()

            # Update temperature
            if self.tune_alpha:
                self.alpha_optim.zero_grad()
                alpha_loss = -self.log_alpha*(log_probs.detach()+ self.entropy_target)
                alpha_loss = T.mean(alpha_loss)
                alpha_loss.backward()
                self.alpha_optim.step()

                # Update temperature parameter
                self.alpha = T.exp(self.log_alpha).detach().item()
            else:
                alpha_loss = T.tensor([0])
            # Soft update target network params
            self.soft_update(self.target_critic1, self.critic1, tau=self.tau)
            self.soft_update(self.target_critic2, self.critic2, tau=self.tau) 

            return [loss.item(), a_loss.item(), alpha_loss.item()]

    def step(self, state, reward, done):

        # Store transition in buffer
        self.store_transition(state,reward,done)

        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state

        return action


    def learn(self, num_episodes, verbose=0):
        avg_rewards = []
        rewards_arr = []
        base_score = self.env.reward_range[0] 
        
        losses = [-1,-1,-1.]
        time_steps = 0

        for i in range(num_episodes):#tqdm(range(num_episodes),desc='Episode',total=num_episodes):#range(num_episodes):#

            state = self.env.reset()
            done = False
            action = self.start(state)

            cum_reward = 0
            episode_step = 1
            # Iterate per episode
            while not done:
                state, reward, done, _ = self.env.step(action)
    
                if done and episode_step==self.env._max_episode_steps:
                    done = ~done
                # Store transitions and sample next action 
                action = self.step(state,reward,done)
    
                if episode_step%self.update_interval==0:
                    losses=self.train()
                    time_steps += self.gradient_steps
                else:
                    time_steps += 1
                cum_reward += reward

                episode_step+=1

            # Plot rewards
            if verbose:
                rewards_arr.append(cum_reward)
                if (i+1)%20==0:
                    self.verbose_print(i+1, avg_rewards[-1], cum_reward, losses, time_steps)
                avg_rewards.append(np.mean(rewards_arr[-100:]))
            if avg_rewards[-1]>base_score and self.save_net_checkpt:   # Save best networks
                base_score=avg_rewards[-1]
                self.save_checkpoint()

        if verbose:
            return avg_rewards

    def verbose_print(self, episode, avg_reward, reward, losses=[0,0,0], time_steps=0):
        print('-'*25)
        print('|   Episode   |  {0:05d}  |'.format(episode))
        print('|  Time Steps |{0:09d}|'.format(time_steps))
        print('| Avg. Reward |  {0:5.1f} |'.format(avg_reward))
        print('|   Reward    |  {0:5.1f} |'.format(reward)) 
        print('|'+'-'*23+'|')
        print('| Critic Loss | {0:7.3f} |'.format(losses[0]))
        print('| Actor Loss  | {0:7.3f} |'.format(losses[1]))
        print('|EntCoef Loss | {0:7.3f} |'.format(losses[2]))
        print('|'+'-'*23+'|')
        print('|  Ent Coeff  | {0:5.4f}  |'.format(self.alpha))
        print('-'*25)

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()
    
    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.target_critic2.load_checkpoint()
    
    def soft_update(self,target,src,tau=0.001):
        with T.no_grad():
            for target_params, params in zip(target.parameters(), src.parameters()):
                target_params.data.copy_(tau*params.data+(1-tau)*target_params.data)

    def save(self, nam='sac'):
        if hasattr(self.env, 'spec'):
            nam+='_'+self.env.spec.id
        with bz2.BZ2File(nam,'wb') as f:
            pickle.dump(self,f) 

    @classmethod
    def load(self, nam='sac'):
        with bz2.BZ2File(nam,'rb') as f:
            self = pickle.load(f)
        return self


if __name__=="__main__":
    # test agent on Pendulum-v1
    env = gym.make('Pendulum-v1')

    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    model = SAC(env,3e-4,3e-4,GaussianActorNetwork,CriticNetwork,[256,256],
                    1e6,device=device,tau=0.005,gamma=0.99,batch_size=256,
                    alpha=0.9,tune_alpha=True,lr_alpha=3e-4)
    rews = model.learn(num_episodes=250,verbose=1)
    model.save('log/sac')    
    plt.plot(rews)
    plt.show()
