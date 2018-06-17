# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import gym
import torch
import csv
import pickle

from model import ActorCritic
from utils import state_to_tensor, plot_line


def test(rank, args, T, shared_model):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.eval()

  save_dir = os.path.join('results', args.name)  

  can_test = True  # Test flag
  t_start = 1  # Test step counter to check against global counter
  rewards, steps = [], []  # Rewards and steps for plotting
  l = str(len(str(args.T_max)))  # Max num. of digits for logging steps
  done = True  # Start new episode

  # stores step, reward, avg_steps and time 
  results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}

  while T.value() <= args.T_max:
    if can_test:
      t_start = T.value()  # Reset counter

      # Evaluate over several episodes and average results
      avg_rewards, avg_episode_lengths = [], []
      for _ in range(args.evaluation_episodes):
        while True:
          # Reset or pass on hidden state
          if done:
            # Sync with shared model every episode
            model.load_state_dict(shared_model.state_dict())
            hx = torch.zeros(1, args.hidden_size)
            cx = torch.zeros(1, args.hidden_size)
            # Reset environment and done flag
            state = state_to_tensor(env.reset())
            done, episode_length = False, 0
            reward_sum = 0

          # Optionally render validation states
          if args.render:
            env.render()

          # Calculate policy
          with torch.no_grad():
            policy, _, _, (hx, cx) = model(state, (hx, cx))

          # Choose action greedily
          action = policy.max(1)[1][0]

          # Step
          state, reward, done, _ = env.step(action.item())
          state = state_to_tensor(state)
          reward_sum += reward
          done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
          episode_length += 1  # Increase episode counter

          # Log and reset statistics at the end of every episode
          if done:
            avg_rewards.append(reward_sum)
            avg_episode_lengths.append(episode_length)
            break

      print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            t_start,
            sum(avg_rewards) / args.evaluation_episodes,
            sum(avg_episode_lengths) / args.evaluation_episodes))
      fields = [t_start, sum(avg_rewards) / args.evaluation_episodes, sum(avg_episode_lengths) / args.evaluation_episodes, str(datetime.now())]

      # storing data in the dictionary.
      results_dict['t'].append(t_start)
      results_dict['reward'].append(sum(avg_rewards) / args.evaluation_episodes)
      results_dict['avg_steps'].append(sum(avg_episode_lengths) / args.evaluation_episodes)
      results_dict['time'].append(str(datetime.now()))

      # Dumping the results in pickle format  
      with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
        pickle.dump(results_dict, f)

      # Saving the data in csv format
      with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

      if args.evaluate:
        return

      rewards.append(avg_rewards)  # Keep all evaluations
      steps.append(t_start)
      plot_line(steps, rewards, save_dir)  # Plot rewards
      torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))  # Save model params
      can_test = False  # Finish testing
    else:
      if T.value() - t_start >= args.evaluation_interval:
        can_test = True


    time.sleep(0.001)  # Check if available to test every millisecond

  # Dumping the results in pickle format  
  with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
    pickle.dump(results_dict, f)

  env.close()
