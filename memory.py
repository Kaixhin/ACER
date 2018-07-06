# -*- coding: utf-8 -*-
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
    self.memory[self.position].append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append([])
      self.position = (self.position + 1)%self.num_episodes

  # Samples random trajectory
  def sample(self, maxlen=0):
    # mem = self.memory[random.randrange(len(self.memory))]
    if self.length()==self.num_episodes:
      numbers = list(range(0, self.position)) + list(range(self.position+1, self.num_episodes-1))
      e = random.choice(numbers)
    else:
      e = random.randrange(len(self.memory))
    # print (len(self.memory), self.position, e)
    mem = self.memory[e]
    T = len(mem)
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    if maxlen > 0 and T > maxlen + 1:
      t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
      return mem[t:t + maxlen + 1]
    else:
      return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    minimum_size = min(len(trajectory) for trajectory in batch)
    batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)
