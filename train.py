import math
import random
import gym
import torch
from torch import nn
from torch.nn import functional as F
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


# Linearly decays learning rate
def _decay_learning_rate(optimiser, steps):
  eps = 1e-32
  for param_group in optimiser.param_groups:
    param_group['lr'] = max(param_group['lr'] - param_group['lr'] / steps, eps)


# Updates networks
def _update_networks(args, model, shared_model, loss, optimiser, on_policy=False):
  optimiser.zero_grad()  # TODO: Zero local grads too surely? When transferring, aren't shared lost anyway?
  # Note that losses were defined as negatives of normal update rules for gradient descent
  loss.backward(retain_variables=on_policy and args.no_truncate)
  # Gradient (L2) norm clipping
  nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if on_policy and args.lr_decay:
    # Decay learning rate only on on-policy updates to ensure correct amount of decay
    _decay_learning_rate(optimiser, args.T_max)  # TODO: Fix formula

  # TODO: Update shared_average_model?


# Trains model on-policy
def _train_on_policy(args, model, shared_model, optimiser, policies, Qs, Vs, actions, rewards, Qret):
  policy_loss = 0
  value_loss = 0
  
  # Calculate n-step returns in forward view, stepping backwards from the last state
  for i in reversed(range(len(rewards))):  # TODO: Consider normalising loss by number of steps?
    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]
    
    # Policy gradient loss
    log_prob = policies[i][0][actions[i]].log()
    policy_loss -= log_prob * A
    # TODO: Entropy loss

    # Value function loss
    Q = Qs[i][0][actions[i]]
    value_loss += (Qret - Q) ** 2 / 2  # Least squares loss

    # Qret ← Qret - Q(s_i, a_i; θ) + V(s_i; θ)
    Qret = Qret - Q.detach() + Vs[i].detach()

  # Update
  _update_networks(args, model, shared_model, policy_loss + value_loss, optimiser, on_policy=True)


# Trains model off-policy
def _train_off_policy(args, model, shared_model, optimiser, policies, Qs, Vs, average_policies, old_policies, actions, rewards, Qret):
  policy_loss = 0
  value_loss = 0
  
  # Calculate n-step returns in forward view, stepping backwards from the last state
  for i in reversed(range(len(rewards))):  # TODO: Consider normalising loss by number of steps?
    # Importance sampling
    rho = policies[i][0] / Variable(old_policies[i][0])  # TODO: Account for NaNs?

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]
    
    # Truncated policy gradient loss
    log_prob = policies[i][0][actions[i]].log()
    # g ← min(c, ρ_i)∙∇θ∙log(π(a_i|s_i; θ))∙A ...
    policy_loss -= min(args.trace_max, rho[actions[i]]) * log_prob * A
    # ... + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
    policy_loss -= (max(1 - args.trace_max / rho, 0) * policies[i].log() * (Qs[i] - Vs[i].expand_as(Qs[i]))).sum(1)

    # Value function loss
    Q = Qs[i][0][actions[i]]
    value_loss += (Qret - Q) ** 2 / 2  # Least squares loss

    # Truncated importance sampling
    c = min(rho[actions[i]], 1)
    # Qret ← c∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = c * (Qret - Q.detach()) + Vs[i].detach()

  # Update
  _update_networks(args, model, shared_model, policy_loss + value_loss, optimiser, on_policy=False)
  return
  """
      (max(1 - args.trace_max / 1, 0) * policies[i] * policies[i].log() * (Qs[i] - Vs[i].expand_as(Qs[i]))).sum(1)
  # k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
  k = policies[i] * (policies[i].log() - average_policies[i].log())  # TODO: Fix undefined log(0)
  """
  
  """
  # dθ ← dθ + ∂θ/∂θ(g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k - β∙∇θH(π(s_i; θ))
  print(g)
  print(k)
  print(torch.mm(k.t(), g))
  policy_loss += g - max(0, torch.mm(k.t(), g) - args.trust_region_threshold)
  # + args.entropy_weight * entropies[i]     
  quit()
  """
  # dθ ← dθ - ∂A^2/∂θ
  # value_loss += 0.5 * A ** 2  # Least squares error
  """
  if len(log_probs) > i + 1:
    # Importance weight ρ_i+1 = π(a_i+1|s_i+1) / µ(a_i+1|s_i+1)
    hx = Variable(torch.zeros(1, args.hidden_size))
    cx = Variable(torch.zeros(1, args.hidden_size))
    pi, _, _, = model(input, (hx, cx))
    pi_a = pi.data.gather(1, actions[i + 1])  # TODO: Ideally need to push model forwards through old states?
    mu_a = log_probs[i + 1].data.exp()  # Data from "old" policy
    importance_weight = pi_a / mu_a
    # Off-policy correction Qret(s_i+1, a_i+1) - Q(s_i+1, a_i+1)
    Qdiff = Qrets[i + 1] - Qs[i + 1].data
  else:
    # Handle terminal case
    importance_weight = torch.FloatTensor([args.importance_weight_truncation])
    Qdiff = torch.zeros(1, 1)  # TODO: Assume no off-policy correction on terminal s?
    Qrets[i + 1] = torch.zeros(1, 1)  # TODO: Q on terminal s is 0, correct?

  # ρ¯_i+1 = min(c, ρ_i+1)
  truncated_importance_weight = torch.clamp(importance_weight, min=args.max_trace)
  # Qret(s_i, a_i) = r_i + γρ¯_i+1[Qret(s_i+1, s_t+1) − Q(s_t+1, s_t+1)] + γV(s_t+1)
  Qrets[i] = rewards[i] + args.discount * truncated_importance_weight * \
      Qdiff + args.discount * Vs[i + 1].data
  # dθ ← dθ - (Qret(s_i, a_i) - Q(s_i, a_i))∇θ∙Q(s_i, a_i)
  value_loss += (Variable(Qrets[i]) - Qs[i]).mean()  # TODO: Should operate only a of Q?
  """
  """
  # TD residual δ = r_i + γV(s_i+1; θ) - V(s_i; θ)
  td_error = rewards[i] + args.discount * Vs[i + 1].data - Vs[i].data
  # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
  A_GAE = A_GAE * args.discount * args.trace_decay + td_error
  # dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙Ψ - β∙∇θH(π(s_i; θ))
  entropy = -(log_policy * policy).sum(1)
  policy_loss += -log_probs[i] * Variable(A_GAE) + args.entropy_weight * entropy
  """
  _update_networks(args, model, shared_model, loss, optimiser, on_policy=False)


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
        hx = Variable(torch.zeros(1, args.hidden_size))
        cx = Variable(torch.zeros(1, args.hidden_size))
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        action, reward, done, episode_length = 0, 0, False, 0
      elif not args.no_truncate:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards = [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # Calculate policy and values
        input = extend_input(state, action_to_one_hot(action, action_size), reward, episode_length)
        policy, Q, V, (hx, cx) = model(Variable(input), (hx, cx))

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
        policies.append(policy)
        Qs.append(Q)
        Vs.append(V)
        actions.append(action)
        rewards.append(reward)

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
      _train_on_policy(args, model, shared_model, optimiser, policies, Qs, Vs, actions, rewards, Qret)

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for one (truncated) episode
        trajectory = memory.sample(maxlen=args.t_max)

        # Reset hidden state
        hx = Variable(torch.zeros(1, args.hidden_size))
        cx = Variable(torch.zeros(1, args.hidden_size))
        avg_hx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
        avg_cx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        action, reward, done, episode_length = 0, 0, False, 0

        # Lists of outputs for training
        policies, Qs, Vs, average_policies, old_policies, actions, rewards = [], [], [], [], [], [], []

        # Loop over trajectory (bar last timestep)
        for i in range(len(trajectory) - 1):
          # Unpack first half of transition
          input, action, reward, old_policy = trajectory[i]

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(Variable(input), (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(Variable(input, volatile=True), (avg_hx, avg_cx))

          # Save outputs for offline training
          policies.append(policy)
          Qs.append(Q)
          Vs.append(V)
          average_policies.append(average_policy)
          old_policies.append(old_policy)
          actions.append(action)
          rewards.append(reward)

          # Unpack second half of transition
          next_input, action, _, _ = trajectory[i + 1]
          done = action is None

          # TODO: Increment counters?
          # t += 1
          # T.increment()

        if done:
          # Qret = 0 for terminal s
          Qret = Variable(torch.zeros(1, 1))
        else:
          # Qret = V(s_i; θ) for non-terminal s
          _, _, Qret, _ = model(Variable(next_input), (hx, cx))
          Qret = Qret.detach()
        
        # Train the network off-policy
        _train_off_policy(args, model, shared_model, optimiser, policies, Qs, Vs, average_policies, old_policies, actions, rewards, Qret)

  env.close()
