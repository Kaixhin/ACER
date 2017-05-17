import time
from datetime import datetime
import gym
import torch
from torch.autograd import Variable

from model import ActorCritic
from utils import action_to_one_hot, extend_input, state_to_tensor, plot_line


def test(rank, args, T, shared_model):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  action_size = env.action_space.n
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.eval()

  can_test = True  # Test flag
  t_start = 1  # Test step counter to check against global counter
  rewards, steps = [], []  # Rewards and steps for plotting
  l = str(len(str(args.T_max)))  # Max num. of digits for logging steps
  done = True  # Start new episode

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
            hx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
            cx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
            # Reset environment and done flag
            state = state_to_tensor(env.reset())
            action, reward, done, episode_length = 0, 0, False, 0
            reward_sum = 0

          # Optionally render validation states
          if args.render:
            env.render()

          # Calculate policy
          input = extend_input(state, action_to_one_hot(action, action_size), reward, episode_length)
          policy, _, _, (hx, cx) = model(Variable(input, volatile=True), (hx.detach(), cx.detach()))  # Break graph for memory efficiency

          # Choose action greedily
          action = policy.max(1)[1].data[0, 0]

          # Step
          state, reward, done, _ = env.step(action)
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
      rewards.append(avg_rewards)  # Keep all evaluations
      steps.append(t_start)
      plot_line(steps, rewards)  # Plot rewards
      torch.save(model.state_dict(), 'model.pth')  # Save model params
      can_test = False  # Finish testing
      if args.evaluate:
        return
    else:
      if T.value() - t_start >= args.evaluation_interval:
        can_test = True

    time.sleep(0.001)  # Check if available to test every millisecond

  env.close()
