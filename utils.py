import plotly
from plotly.graph_objs import Scatter, Line
import torch
from torch import multiprocessing as mp


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


# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)


# Converts an action index and action space size into a one-hot batch tensor
def action_to_one_hot(action_index, action_size):
  action = torch.zeros(1, action_size)
  action[0, action_index] = 1
  return action


# Creates an extended input (state + previous action + reward + timestep)
def extend_input(state, action, reward, timestep):
  reward = torch.Tensor([reward]).unsqueeze(0)
  timestep = torch.Tensor([timestep]).unsqueeze(0)
  return torch.cat((state, action, reward, timestep), 1)


# Plots mean and standard deviation bars of a population over time
def plot_line(xs, ys_population):
  ys = torch.Tensor(ys_population)
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', line=Line(color='rgb(0, 176, 246)'), name='Average Reward')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', line=Line(color='transparent'), showlegend=False)

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower],
    'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
  }, filename='rewards.html', auto_open=False)
