import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'prob'))

# TODO: Change implementation from episodic memory (simpler logic) to single-level transition-based memory
class ReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.memory.append([])
    self.position = 0

  def append(self, state, action, reward, prob):
    self.memory[self.position].append(Transition(state, action, reward, prob))
    # Terminal states are saved with actions/rewards as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = (self.position + 1) % self.num_episodes

  # TODO: Change implementation once away from episodic memory
  def sample(self, n):
    # Choose random episode
    while True:
      e = random.randrange(len(self.memory))
      if len(self.memory[e]) > 0:
        return self.memory[e]
    """
    t = random.sample(range(len(self.memory[e]) - 2), min(n, len(self.memory[e]) - 2))  # Choose random indices (for non-terminal s)
    transitions = [(self.memory[e][i], self.memory[e][i + 1]) for i in t]
    return [(tr[0].state, tr[0].action, tr[0].reward, tr[1].state, tr[1].action is None) for tr in transitions]
    """

  def __len__(self):
    return len(self.memory[self.position])
