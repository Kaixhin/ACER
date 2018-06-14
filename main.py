# -*- coding: utf-8 -*-
import argparse
import os
import csv
# import platform
import gym
import torch
from torch import multiprocessing as mp

from model import ActorCritic
from optim import SharedRMSprop
from train import train
from test import test
from utils import Counter


parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-processes', type=int, default=6, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=500000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='results', help='Save folder')
parser.add_argument('--env', type=str, default='CartPole-v1',help='environment name')


if __name__ == '__main__':
  # BLAS setup
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'

  # Setup
  args = parser.parse_args()
  # Creating directories.
  save_dir = os.path.join('results', args.name)  
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
  print(' ' * 26 + 'Options')

  # Saving parameters
  with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
      f.write(k + ' : ' + str(v) + '\n')
  # args.env = 'CartPole-v1'  # TODO: Remove hardcoded environment when code is more adaptable
  # mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
  torch.manual_seed(args.seed)
  T = Counter()  # Global shared counter
  gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings

  # Create shared network
  env = gym.make(args.env)
  shared_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  shared_model.share_memory()
  if args.model and os.path.isfile(args.model):
    # Load pretrained weights
    shared_model.load_state_dict(torch.load(args.model))
  # Create average network
  shared_average_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  shared_average_model.load_state_dict(shared_model.state_dict())
  shared_average_model.share_memory()
  for param in shared_average_model.parameters():
    param.requires_grad = False
  # Create optimiser for shared network parameters with shared statistics
  optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
  optimiser.share_memory()
  env.close()

  fields = ['t', 'rewards', 'avg_steps', 'time']
  with open(os.path.join(save_dir, 'test_results.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, args, T, shared_model))
  p.start()
  processes.append(p)

  if not args.evaluate:
    # Start training agents
    for rank in range(1, args.num_processes + 1):
      p = mp.Process(target=train, args=(rank, args, T, shared_model, shared_average_model, optimiser))
      p.start()
      print('Process ' + str(rank) + ' started')
      processes.append(p)

  # Clean up
  for p in processes:
    p.join()
