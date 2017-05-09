import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward'))


class ReplayMemory():
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity)

  def append(self, state, action, reward):
    # Terminal states are saved with actions/rewards as None
    self.memory.append(Transition(state, action, reward))

  def sample(self, n):
    assert len(self.memory) >= n
    return random.sample(self.memory, n)

  def __len__(self):
    return len(self.memory)
