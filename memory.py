import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

# TODO: Change implementation from episodic memory (simpler logic) to single-level transition-based memory
class ReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.memory.append([])
    self.position = 0

  def append(self, state, action, reward):
    self.memory[self.position].append(Transition(state, action, reward))
    # Terminal states are saved with actions/rewards as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = (self.position + 1) % self.num_episodes

  def sample(self, n):
    assert len(self.memory[self.position]) >= n
    return random.sample(self.memory[self.position], n)

  def __len__(self):
    return len(self.memory[self.position])
