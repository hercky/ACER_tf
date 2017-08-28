import numpy as np 
import random

# memory here

# from memory import Memory
# alternate create own memory object

# make sure transition object is also avialble here 
class Memory(object):
    
    
    def __init__(self, 
                 max_size = 1000):
        """
        can optimize further by using a numpy array and allocating it to zero
        """
        self.max_size = max_size
        self.store = [None] * self.max_size  # is a list, other possible data structures might be a queue 
        self.count = 0 
        self.current = 0
        
        
    def add(self, transition):
        """ insert one sample at a time """
        
        self.store[self.current] = transition
        
        # for taking care of how many total frames have been inserted into the memory
        self.count = max(self.count, self.current + 1)
        
        # increase the counter
        self.current = (self.current + 1) % self.max_size
        
    def get_sample(self, index):
        # normalize index 
        index = index % self.count
        
        return self.store[index]
    
    def get_minibatch(self, batch_size = 100, offset = 0):
        """
        a minibatch of random transitions
        """
        samples = []
        
        while len(samples) < batch_size:
            index = random.randint(0, self.count)
            samples.append(self.get_sample(index))
            
        return samples
    
    def get_trajectory(self, max_steps = 50, single_episode = True):
        """
        a trajectory from random starting point
        """
        traj = []
        
        index = random.randint(0, self.count)
        
        while len(traj) < max_steps:
            transition = self.get_sample(index) 
            traj.append(self.get_sample(index))
            index += 1
            
            if single_episode and transition.done: # break when you encounter the terminal state
                    break
            
        return traj 

