import torch
from torch import multiprocessing as mp
from torch.autograd import Variable


# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value


# Converts an observation from the OpenAI Gym (a numpy array) to a batch tensor
def observation_to_tensor(observation):
  return torch.from_numpy(observation).float().unsqueeze(0)


# Converts an index and size into a one-hot batch tensor
def action_to_one_hot(action_index, action_size):
  action = torch.zeros(1, action_size)
  action[0, action_index[0, 0]] = 1
  return action


# Creates an extended input (observation + previous action + reward + timestep)
def extend_input(observation, action, reward, timestep, volatile=False):
  reward = torch.Tensor([reward]).unsqueeze(0)
  timestep = torch.Tensor([timestep]).unsqueeze(0)
  return Variable(torch.cat((observation, action, reward, timestep), 1), volatile=volatile)
