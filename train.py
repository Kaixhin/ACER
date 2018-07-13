# -*- coding: utf-8 -*-
import math
import random
import gym
import torch
from torch import nn
from torch.nn import functional as F

from memory import EpisodicReplayMemory
from model import ActorCritic
from utils import state_to_tensor


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)


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
  """
  Calculate gradients for gradient descent on loss functions
  Note that math comments follow the paper, which is formulated for gradient ascent
  """
  loss.backward()
  # Gradient L2 normalisation
  nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  # Update shared_average_model
  for shared_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
    shared_average_param = args.trust_region_decay * shared_average_param + (1 - args.trust_region_decay) * shared_param

# Computes an "efficient trust region" loss (policy head only) based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold, g, k):

  kl = - (ref_distribution * (distribution.log()-ref_distribution.log())).sum(1).mean(0)

  # Compute dot products of gradients
  k_dot_g = (k*g).sum(1).mean(0)
  k_dot_k = (k**2).sum(1).mean(0)
  # Compute trust region update
  if k_dot_k.item() > 0:
    trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0).detach()
  else:
    trust_factor = torch.zeros(1)
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  trust_loss = loss + trust_factor*kl
 
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
    if off_policy:
      rho = policies[i].detach() / old_policies[i]
    else:
      rho = torch.ones(1, action_size)

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]

    # Log policy log(π(a_i|s_i; θ))
    log_prob = policies[i].gather(1, actions[i]).log()
    # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
    single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=args.trace_max) * log_prob * A.detach()).mean(0)  # Average over batch
    # Off-policy bias correction
    if off_policy:
      # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
      bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
      single_step_policy_loss -= (bias_weight * policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(1).mean(0)
    if args.trust_region:
      # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
      k = -average_policies[i].gather(1, actions[i]) / (policies[i].gather(1, actions[i]) + 1e-10)
      if off_policy:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[i]) \
          + (bias_weight * (Qs[i] - Vs[i].expand_as(Qs[i]))/(policies[i] + 1e-10)).sum(1)).detach()
      else:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[i])).detach()
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, policies[i].gather(1, actions[i]) + 1e-10, average_policies[i].gather(1, actions[i]) + 1e-10, single_step_policy_loss, args.trust_region_threshold, g, k)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss

    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
    policy_loss -= args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(0)  # Sum over probabilities, average over batch

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    Q = Qs[i].gather(1, actions[i])
    value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

  # Update networks
  _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss, optimiser)


# Acts and trains model
def train(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  if not args.on_policy:
    # Normalise memory capacity by number of training processes
    memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)

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
        hx, avg_hx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        cx, avg_cx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        done, episode_length = False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # Calculate policy and values
        policy, Q, V, (hx, cx) = model(state, (hx, cx))
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))

        # Sample action
        action = torch.multinomial(policy, 1)[0, 0]

        # Step
        next_state, reward, done, _ = env.step(action.item())
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        if not args.on_policy:
          # Save (beginning part of) transition for offline training
          memory.append(state, action, reward, policy.detach())  # Save just tensors
        # Save outputs for online training
        [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                           (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]), average_policy))]

        # Increment counters
        t += 1
        T.increment()

        # Update state
        state = next_state

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = torch.zeros(1, 1)

        if not args.on_policy:
          # Save terminal state for offline training
          memory.append(state, None, None, None)
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(state, (hx, cx))
        Qret = Qret.detach()

      # Train the network on-policy
      _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if not args.on_policy and len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)
        cx, avg_cx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(state, (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

        # Do forward pass for all transitions
        _, _, Qret, _ = model(next_state, (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)
    done = True

  env.close()
