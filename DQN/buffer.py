import torch
from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''
    def __init__(self, maxsize=100000):
        # TODO: Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.buffer = deque(maxlen=maxsize)
        #pass
    
    def __len__(self):
        # TODO: Return the length of the buffer (i.e. the number of transitions).
        return len(self.buffer)
        #pass
    
    def add_experience(self, state, action, reward, next_state, done):
        # TODO: Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        transit = Transition(state, action, reward, next_state, done)
        self.buffer.append(transit)
        #pass
        
    def sample(self, batch_size):
        # TODO: Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
        transit = random.sample(self.buffer, batch_size)
        #states, actions, rewards, next_states, dones = zip(*transit)
        batch = Transition(*zip(*transit))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32)
        actions = torch.tensor(np.array(batch.action), dtype=torch.int64)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        dones = torch.tensor(np.array(batch.done), dtype=torch.float32)

        return states, actions, rewards, next_states, dones
        #pass