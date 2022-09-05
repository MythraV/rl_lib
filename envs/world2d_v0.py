import cv2
import numpy as np
import gym

class GridWorld():
    def __init__(self, vars={}, scalar_states=False):
        ''' vars is dictionary of init variables '''
        self.renderF = vars['render'] if 'render' in vars else False
        self.width = vars['width'] if 'width' in vars else 10
        self.length = vars['length'] if 'length' in vars else 10
        self.start_q = vars['start'] if 'start' in vars else [2,2]
        self.goal_q = vars['goal'] if 'goal' in vars else [8,8]

        self.scalr_st = scalar_states
        # Create the world
        self.gmap = np.zeros([self.width+1,self.length+1])
        self.rmap = np.zeros([self.width+5,self.length+5,3],dtype=np.uint8)  # World for rendering

        if self.renderF:
            self.roff = 2# rendering map is larger, this is the offset
            self._setup_render()
        # Add obstacles
        # vars['obstacles'] = [[xi1,yi1,xf1,yf1],[xi2,yi2,xf2,yf2]...]
        # coordinates of top left and bottom right corners

        if 'obstacles' in vars:
            for ob in vars['obstacles']:
                self.gmap[ob[0]:ob[2],ob[1]:ob[3]] = 1  #Occupied
                # Add coloring to obstacles
                if self.renderF:
                    cv2.rectangle(self.rmap, (ob[1]+self.roff,ob[0]+self.roff),
                                            (ob[3]+self.roff,ob[2]+self.roff),
                                            (250,10,10),1)
        
        self.actions = [0,1,2,3] # up, down, left, right
        self.num_states = (self.width+1)*(self.length+1)
        self.num_actions = len(self.actions)

        self.curr_q = self.start_q.copy()

    def _setup_render(self):
        self.rmap[self.goal_q[0]+self.roff,self.goal_q[1]+self.roff] = [10,10,100] # Goal display color
        # Add boundaries
        #Up
        self.rmap[1,1:-1] = [60,60,60]
        self.rmap[-2,1:-1] = [60,60,60]
        self.rmap[1:-1,1] = [60,60,60]
        self.rmap[1:-1,-2] = [60,60,60]
        
        self.mcolor = (0,240,30)

    def start(self):
        self.curr_q[0],self.curr_q[1] = self.start_q[0],self.start_q[1]
        
        if self.scalr_st:
            return self.get_scalar_state(self.curr_q)
        else:
            return self.curr_q


    def reset(self):
        self.curr_q[0],self.curr_q[1] = self.start_q[0],self.start_q[1]
    
    def render(self,wait_time=1):
        if self.renderF:
            cv2.namedWindow("World", cv2.WINDOW_NORMAL)
            self.renderF=False
        img = self.rmap.copy()
        img[self.curr_q[0]+self.roff,self.curr_q[1]+self.roff]=self.mcolor
        cv2.imshow("World",img)
        cv2.waitKey(wait_time)

    def update_state(self,action):
        collision = False
        if action==0:   #up
            if self.curr_q[0]<self.width:
                self.curr_q[0]+=1
            else:
                collision=True
        if action==1:   #down
            if self.curr_q[0]>0:
                self.curr_q[0]-=1
            else:
                collision=True
        if action==2:   #right
            if self.curr_q[1]<self.length:
                self.curr_q[1]+=1
            else:
                collision=True
        if action==3:   #left
            if self.curr_q[1]>0:
                self.curr_q[1]-=1
            else:
                collision=True
        return collision

    def get_scalar_state(self, state):
        ''' Convert 2d state to scalar '''
        return (self.length+1)*state[0] + state[1]

    def step(self,action):
        col = self.update_state(action)
        reward = -1
        done = False
        info = []
        if col:
            reward = -2
        if self.curr_q==self.goal_q:
            reward = 5
            done = True
        
        if self.scalr_st:
            state = self.get_scalar_state(self.curr_q)
        else:
            state = self.curr_q

        return state, reward, done, info

    def get_state(self):
        if self.scalr_st:
            state = self.get_scalar_state(self.curr_q)
        else:
            state = self.curr_q
        return state


if __name__=="__main__":
    # Test environment
    vars = {'render':True}
    env = GridWorld(vars)
    env.start()
    print('Beggining', env.start_q)
    env.render()
    # Test step
    for i in range(10):
        env.step(0)
        env.render(100)
    
    # Test terminal step
    env.curr_q=[7,8]
    ret=env.step(0)
    print(ret)
    env.reset()
    print('nstate = ', env.get_state())
        
        
