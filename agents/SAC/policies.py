import torch as T
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class SACParams():
    def __init__(self, max_action=1, min_action=-1, log_sig_min=-20, log_sig_max=2, eps=1e-6):
        self.epsilon = eps  # noise for reparametrization trick
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.max_action = max_action
        self.min_action = min_action

class CriticNetwork(T.nn.Module):
    def __init__(
                self,
                input_dims,
                hidden_dims,
                action_dims,
                optim=T.optim.Adam, lr=0.01, w_decay=0,
                device=T.device('cpu'),
                checkpt_file='critic_model',
                ):

        super(CriticNetwork,self).__init__()
        self.checkpoint_file = checkpt_file

        self.fc1 = T.nn.Linear(input_dims+action_dims,hidden_dims[0])
        self.fc2 = T.nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = T.nn.Linear(hidden_dims[1],1)   
        
        # Optimizer
        self.optim = optim(self.parameters(), lr=lr, weight_decay=w_decay)

        self.to(device)

    def forward(self,s,a):
        x = T.cat((s,a),dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        qval = self.fc3(x)
        return qval
        
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# Stochastic gaussian policy 

class GaussianActorNetwork(T.nn.Module):
    def __init__(
                self,
                input_dims,
                hidden_dims,
                action_dims,
                optim=T.optim.Adam, lr=0.01, w_decay=0,
                device=T.device('cpu'),
                checkpt_file='actor_model',
                params = SACParams()
                ):

        super(GaussianActorNetwork,self).__init__()
        self.checkpoint_file = checkpt_file

        self.fc1 = T.nn.Linear(input_dims,hidden_dims[0])
        self.fc2 = T.nn.Linear(hidden_dims[0],hidden_dims[1])

        self.mu = T.nn.Linear(hidden_dims[1],action_dims)  
        
        self.sigma = T.nn.Linear(hidden_dims[1],action_dims)  
        
        self.p = params
        # Optimizer
        self.optim = optim(self.parameters(), lr=lr, weight_decay=w_decay)
        
        self.to(device)

        # Action offsets
        action_scale = (self.p.max_action-self.p.min_action)/2
        self.action_scale = T.tensor(action_scale).type(T.float).to(device)
        action_bias = (self.p.max_action+self.p.min_action)/2
        self.action_bias = T.tensor(action_bias).type(T.float).to(device)
        

    def forward(self,s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        
        mu = self.mu(s)
        sigma = self.sigma(s)
        log_sigma = T.clamp(sigma, self.p.log_sig_min, self.p.log_sig_max)

        # compute log prob
        
        return mu, log_sigma

    def sample(self, s, reparametrize=True):
        mu ,log_sigma = self.forward(s)
        sigma = log_sigma.exp()
        probs = Normal(mu,sigma)

        if reparametrize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        bounded_actions = T.tanh(actions)
        scaled_actions = bounded_actions*self.action_scale+self.action_bias
        log_probs = probs.log_prob(actions)
        log_probs -= T.log(self.action_scale*(1-bounded_actions.pow(2)+self.p.epsilon))
        log_probs  = log_probs.sum(1, keepdim=True)

        return scaled_actions, log_probs, mu


    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))