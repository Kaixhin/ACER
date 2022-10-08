# -*- coding: utf-8 -*-
import plotly
import plotly.graph_objs as go
import torch
from torch import multiprocessing as mp
import os

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


# Plots min, max and mean + standard deviation bars of a population over time
def plot_line(xs, ys_population, save_dir):
  max_colour = 'rgb(0, 132, 180)'
  mean_colour = 'rgb(0, 172, 237)'
  std_colour = 'rgba(29, 202, 255, 0.2)'

  ys = torch.tensor(ys_population)
  ys_min = ys.min(1)[0].squeeze()
  ys_max = ys.max(1)[0].squeeze()
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = go.Scatter(x=xs, y=ys_max.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Max')
  trace_upper = go.Scatter(x=xs, y=ys_upper.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), name='+1 Std. Dev.', showlegend=False)
  trace_mean = go.Scatter(x=xs, y=ys_mean.numpy(), mode='lines', line=dict(color=mean_colour), name='Mean')
  trace_lower = go.Scatter(x=xs, y=ys_lower.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), fill='tonexty', fillcolor=std_colour, name='-1 Std. Dev.', showlegend=False)
  trace_min = go.Scatter(x=xs, y=ys_min.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_mean, trace_upper, trace_lower, trace_min, trace_max],
    'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
  }, filename=os.path.join(save_dir, 'rewards.html'), auto_open=False)
