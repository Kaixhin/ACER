import math
import random
import gym
import torch
from torch import nn
from torch.autograd import Variable

from memory import EpisodicReplayMemory
from model import ActorCritic
from utils import action_to_one_hot, extend_input, state_to_tensor


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return k - 1


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimiser):
  # Zero shared and local grads
  optimiser.zero_grad()
  # Calculate gradients (not losses defined as negatives of normal update rules for gradient descent)
  loss.backward()
  # Gradient (L1) norm clipping
  nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm, 1)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  # Update shared_average_model
  for shared_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
    shared_average_param = args.trust_region_decay * shared_average_param + (1 - args.trust_region_decay) * shared_param


# Computes a trust region loss based on an existing loss and two distributions
def _trust_region_loss(model, ref_model, distribution, ref_distribution, loss, threshold):
  # Compute gradients from original loss
  loss.backward(retain_variables=True)
  # Gradients should be treated as constants (not using detach as volatility can creep in when double backprop is not implemented)
  g = [Variable(param.grad.data) for param in model.parameters()]
  model.zero_grad()

  # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
  kl = (distribution * (distribution.log() - ref_distribution.log())).mean(1).mean(0)
  # Compute gradients from (negative) KL loss (increases KL divergence)
  (-kl).backward(retain_variables=True)
  k = [Variable(param.grad.data) for param in model.parameters()]
  model.zero_grad()

  # Compute dot products of gradients
  k_dot_g = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
  k_dot_k = sum(torch.sum(k_p ** 2) for k_p in k)
  # Compute trust region update
  trust_factor = k_dot_k.data[0] > 0 and (k_dot_g - threshold) / k_dot_k or Variable(torch.zeros(1))
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  z_star = [g_p - trust_factor.expand_as(k_p) * k_p for g_p, k_p in zip(g, k)]
  trust_loss = 0
  for param, z_star_p in zip(model.parameters(), z_star):
    trust_loss += (param * z_star_p).sum()
  return trust_loss


# Trains model
def _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies, old_policies=None):
  off_policy = old_policies is not None
  action_size = policies[0].size(1)
  policy_loss, value_loss = 0, 0

  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  for i in reversed(range(t)):
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    rho = off_policy and policies[i].detach() / old_policies[i] or Variable(torch.ones(1, action_size))

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]

    # Log policy log(π(a_i|s_i; θ))
    log_prob = policies[i].gather(1, actions[i]).log()
    # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
    single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=args.trace_max) * log_prob * A).mean(0)  # Average over batch
    # Off-policy bias correction
    if off_policy:
      # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
      bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
      single_step_policy_loss -= (bias_weight * policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(1).mean(0)
    if args.trust_region:
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, shared_average_model, policies[i], average_policies[i], single_step_policy_loss, args.trust_region_threshold)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss

    # Entropy regularisation dθ ← dθ - β∙∇θH(π(s_i; θ))
    policy_loss += args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(0)

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    Q = Qs[i].gather(1, actions[i])
    value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

  # Optionally normalise loss by number of time steps
  if not args.no_time_normalisation:
    policy_loss /= t
    value_loss /= t
  # Update networks
  _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss, optimiser)


# Acts and trains model
def train(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  action_size = env.action_space.n
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  memory = EpisodicReplayMemory(args.memory_capacity, args.max_episode_length)

  t = 1  # Thread step counter
  done = True  # Start new episode

  while T.value() <= args.T_max:
    # On-policy episode loop
    while True:
      # Sync with shared model at least every t_max steps
      model.load_state_dict(shared_model.state_dict())
      # Get starting timestep
      t_start = t

      # Reset or pass on hidden state
      if done:
        hx, avg_hx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        cx, avg_cx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        action, reward, done, episode_length = 0, 0, False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # Calculate policy and values
        input = extend_input(state, action_to_one_hot(action, action_size), reward, episode_length)
        policy, Q, V, (hx, cx) = model(Variable(input), (hx, cx))
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(Variable(input), (avg_hx, avg_cx))

        # Sample action
        action = policy.multinomial().data[0, 0]  # Graph broken as loss for stochastic action calculated manually

        # Step
        next_state, reward, done, _ = env.step(action)
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        # Save (beginning part of) transition for offline training
        memory.append(input, action, reward, policy.data)  # Save just tensors
        # Save outputs for online training
        [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                           (policy, Q, V, Variable(torch.LongTensor([[action]])), Variable(torch.Tensor([[reward]])), average_policy))]

        # Increment counters
        t += 1
        T.increment()

        # Update state
        state = next_state

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = Variable(torch.zeros(1, 1))

        # Save terminal state for offline training
        memory.append(extend_input(state, action_to_one_hot(action, action_size), reward, episode_length), None, None, None)
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(Variable(input), (hx, cx))
        Qret = Qret.detach()

      # Train the network on-policy
      _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = Variable(torch.zeros(args.batch_size, args.hidden_size)), Variable(torch.zeros(args.batch_size, args.hidden_size))
        cx, avg_cx = Variable(torch.zeros(args.batch_size, args.hidden_size)), Variable(torch.zeros(args.batch_size, args.hidden_size))

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          input = torch.cat((trajectory.state for trajectory in trajectories[i]), 0)
          action = Variable(torch.LongTensor([trajectory.action for trajectory in trajectories[i]])).unsqueeze(1)
          reward = Variable(torch.Tensor([trajectory.reward for trajectory in trajectories[i]])).unsqueeze(1)
          old_policy = Variable(torch.cat((trajectory.policy for trajectory in trajectories[i]), 0))

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(Variable(input), (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(Variable(input), (avg_hx, avg_cx))

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_input = torch.cat((trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = Variable(torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1))

        # Do forward pass for all transitions
        _, _, Qret, _ = model(Variable(next_input), (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)

  env.close()
