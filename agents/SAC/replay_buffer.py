import numpy as np
from collections import defaultdict

class ReplayBuffer():
    def __init__(self, input_shape, num_action_feats, buf_size=1e5):
        self.buffer_size = int(buf_size)
        self.reward_buffer = np.zeros(self.buffer_size)
        self.state_buffer = np.zeros((self.buffer_size,input_shape))
        self.next_state_buffer = np.zeros_like(self.state_buffer)
        self.action_buffer = np.zeros((self.buffer_size,num_action_feats))
        self.terminal_buffer = np.zeros(self.buffer_size,dtype=np.bool) # For indicating done
        self.curr_ind = 0
        self.buffer_len = 0

    def sample(self, n=1):
        if n>=self.buffer_len:
            out = (self.state_buffer[0:self.buffer_len],
                    self.action_buffer[0:self.buffer_len],
                    self.reward_buffer[0:self.buffer_len],
                    self.next_state_buffer[0:self.buffer_len],
                    self.terminal_buffer[0:self.buffer_len])
        else:
            sample_ids = np.random.choice(self.buffer_len,n)
            out = (self.state_buffer[sample_ids],
                    self.action_buffer[sample_ids],
                    self.reward_buffer[sample_ids],
                    self.next_state_buffer[sample_ids],
                    self.terminal_buffer[sample_ids])
        return out

    def add(self, last_state, action, reward, state, done):
        if self.curr_ind >= self.buffer_size:
            self.curr_ind=0
        
        self.state_buffer[self.curr_ind] = last_state
        self.action_buffer[self.curr_ind] = action
        self.reward_buffer[self.curr_ind] = reward
        self.next_state_buffer[self.curr_ind] = state
        self.terminal_buffer[self.curr_ind] = done

        self.curr_ind+=1
        if self.curr_ind>self.buffer_len:
            self.buffer_len=self.curr_ind
