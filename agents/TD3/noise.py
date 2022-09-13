import numpy as np

class GaussNoise():
    def __init__(self, mu, sigma=0.2, seed=None, clip=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.clip=clip
        if seed is None:
            self.rand_gen = np.random.RandomState()
        else:
            self.rand_gen = np.random.RandomState(seed=seed)

    
    def __call__(self):
        noise = self.rand_gen.normal(self.mu, self.sigma, size=self.mu.shape)
        noise = np.clip(noise,-self.clip,self.clip)
        return noise

    def reset(self):
        pass
