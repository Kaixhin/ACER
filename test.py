import time
from datetime import datetime
import gym
import torch
from torch.autograd import Variable

from model import ActorCritic
from utils import action_to_one_hot, extend_input, observation_to_tensor


def test(rank, args, T, shared_model):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  action_size = env.action_space.n
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.eval()

  can_test = True  # Test flag
  t_start = 1  # Test step counter to check against global counter
  l = str(len(str(args.T_max)))
  done = True  # Start new episode

  while T.value() <= args.T_max:
    if can_test:
      t_start = T.value()  # Reset counter

      while True:
        # Reset or pass on hidden state
        if done:
          # Sync with shared model every episode
          model.load_state_dict(shared_model.state_dict())
          hx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
          cx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
          # Reset environment and done flag
          observation = observation_to_tensor(env.reset())
          action, reward, done, episode_length = torch.LongTensor([0]).unsqueeze(0), 0, False, 0
          reward_sum = 0
        else:
          # Break graph for memory efficiency
          hx = Variable(hx.data, volatile=True)
          cx = Variable(cx.data, volatile=True)

        # Optionally render validation states
        if args.render:
          env.render()

        # Calculate policy and value
        input = extend_input(observation, action_to_one_hot(action, action_size), reward, episode_length, volatile=True)
        policy, value, (hx, cx) = model(input, (hx, cx))

        # Choose action greedily
        action = policy.max(1)[1].data

        # Step
        observation, reward, done, _ = env.step(action[0, 0])
        observation = observation_to_tensor(observation)
        reward_sum += reward

        # Increase episode counter
        episode_length += 1
        done = done or episode_length >= args.max_episode_length

        # Log and reset statistics at the end of every episode
        if done:
          print(('[{}] Step: {:<' + l + '} Reward: {:<8} Episode Length: {:<8}').format(
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            t_start,
            reward_sum,
            episode_length))
          torch.save(model.state_dict(), 'model.pth')  # Save model params
          can_test = False  # Finish testing
          break
    else:
      if T.value() - t_start >= args.test_interval:
        can_test = True

    time.sleep(1)  # Check if available to test every second

  env.close()
