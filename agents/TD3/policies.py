import torch as T
import torch.nn.functional as F
import numpy as np

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
    
    def _init_weights(self, module, last=False):
        if last:
            T.nn.init.uniform_(module.weight, -3e-3, 3e-3)
            T.nn.init.uniform_(module.bias, -3e-3, 3e-3)
        else:
            f,_ = T.nn.init._calculate_fan_in_and_fan_out(module.weight)
            T.nn.init.uniform_(module.weight, -1/np.sqrt(f), 1/np.sqrt(f))
            T.nn.init.uniform_(module.bias, -1/np.sqrt(f), 1/np.sqrt(f))
    
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class ActorNetwork(T.nn.Module):
    def __init__(
                self,
                input_dims,
                hidden_dims,
                action_dims,
                optim=T.optim.Adam, lr=0.01, w_decay=0,
                device=T.device('cpu'),
                checkpt_file='actor_model',
                ):

        super(ActorNetwork,self).__init__()
        self.checkpoint_file = checkpt_file

        self.fc1 = T.nn.Linear(input_dims,hidden_dims[0])
        self.fc2 = T.nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = T.nn.Linear(hidden_dims[1],action_dims)
        
        # Optimizer
        self.optim = optim(self.parameters(), lr=lr, weight_decay=w_decay)

        self.to(device)

    def forward(self,s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a = T.tanh(self.fc3(s))
        return a
    
    def _init_weights(self, module, last=False):
        if last:
            T.nn.init.uniform_(module.weight, -3e-3, 3e-3)
            T.nn.init.uniform_(module.bias, -3e-3, 3e-3)
        else:
            f,_ = T.nn.init._calculate_fan_in_and_fan_out(module.weight)
            T.nn.init.uniform_(module.weight, -1/np.sqrt(f), 1/np.sqrt(f))
            T.nn.init.uniform_(module.bias, -1/np.sqrt(f), 1/np.sqrt(f))
    
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))