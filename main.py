import argparse
import gym
import torch
from torch import multiprocessing as mp
# TODO: Consider using Visdom

from model import ActorCritic
from optim import SharedRMSprop
from train import train
from test import test
from utils import Counter


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--env', type=str, default='CartPole-v1', metavar='ENV', help='OpenAI Gym environment')
parser.add_argument('--num-processes', type=int, default=6, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=1e6, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--test-interval', type=int, default=10000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--gae-discount', type=float, default=1, metavar='λ', help='GAE discount factor')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=1e-3, metavar='η', help='Learning rate')
parser.add_argument('--no-lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--no-truncate', action='store_true', help='Disable BPTT truncation at t-max steps')
parser.add_argument('--entropy-weight', type=float, default=0.01, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--value-loss-weight', type=float, default=0.5, metavar='WEIGHT', help='Value loss weight')
parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Max value of gradient norm for gradient clipping')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')


if __name__ == '__main__':
  # Setup
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  T = Counter()  # Global shared counter

  # Create shared network
  env = gym.make(args.env)
  shared_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  shared_model.share_memory()
  # Create optimiser for shared network parameters with shared statistics
  optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
  optimiser.share_memory()
  env.close()

  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, args, T, shared_model))
  p.start()
  processes.append(p)

  # Start training agents
  for rank in range(1, args.num_processes + 1):
    p = mp.Process(target=train, args=(rank, args, T, shared_model, optimiser))
    p.start()
    processes.append(p)

  # Clean up
  for p in processes:
    p.join()
