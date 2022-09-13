import numpy as np

# Ornstein Uhlenbeck process noise based on stable-baselines3

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.15, dt=0.01, n0=None):
        self.mu = np.array(mu)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.n0 = n0

        self.prev_noise = np.zeros_like(self.mu)
        self.reset()
        
    def __call__(self):
        noise = (self.prev_noise
                + self.theta*(self.mu - self.prev_noise)*self.dt
                + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape))
        
        self.prev_noise = noise
        return noise

    def reset(self):
        self.prev_noise = self.n0 if self.n0 is not None else np.zeros_like(self.mu)