import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))

class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.memory.append([])  # List for first episode
    self.position = 0

  def append(self, state, action, reward, policy):
    self.memory[self.position].append(Transition(state, action, reward, policy))  # Save s, a, r, µ(·|s)
    # Terminal states are saved with actions/rewards as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = min(self.position + 1, self.num_episodes - 1)

  # Samples random trajectory
  def sample(self):
    while True:
      e = random.randrange(len(self.memory))
      if len(self.memory[e]) > 0:
        return self.memory[e]

  def __len__(self):
    return len(self.memory)
