import gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from memory import ReplayMemory
from model import ActorCritic
from utils import action_to_one_hot, extend_input, state_to_tensor


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


def _train_off_policy(args, model, shared_model, optimiser, memory):
    pass
    """
    # Train the network off-policy
    batch = memory.sample(args.batch_size)
    batch_size = len(batch)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = Variable(torch.cat(states, 0))
    actions = Variable(torch.LongTensor(actions).unsqueeze(1))
    rewards = Variable(torch.Tensor(rewards).unsqueeze(1), volatile=True)
    next_states = Variable(torch.cat(next_states, 0), volatile=True)
    dones = Variable(torch.Tensor(dones).unsqueeze(1), volatile=True)

    hx = Variable(torch.zeros(batch_size, args.hidden_size))
    cx = Variable(torch.zeros(batch_size, args.hidden_size))
    # TODO: Move away from traditional DQN rule
    _, next_Qs, _ = shared_model(next_states, (hx, cx))  # Treat as target network for now
    max_next_Qs = next_Qs.max(1)[0]
    targets = rewards + args.discount * (1 - dones) * max_next_Qs  # r + γV(s_i+1)
    targets.volatile = False  # Once computed without retaining state, remove volatility
    _, Qs, _ = model(states, (hx, cx))
    loss = F.smooth_l1_loss(Qs.gather(1, actions), targets)
    loss.backward()

    optimiser.zero_grad()
    # Gradient (L2) norm clipping
    nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm)

    # Transfer gradients to shared model and update
    _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    """


def train(rank, args, T, shared_model, average_model, optimiser):
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
    # Sync with shared model at least every t_max steps
    model.load_state_dict(shared_model.state_dict())
    # Get starting timestep
    t_start = t

    # Reset or pass on hidden state
    if done:
      hx = Variable(torch.zeros(1, args.hidden_size))
      cx = Variable(torch.zeros(1, args.hidden_size))
      avg_hx = Variable(torch.zeros(1, args.hidden_size))
      avg_cx = Variable(torch.zeros(1, args.hidden_size))
      # Reset environment and done flag
      state = state_to_tensor(env.reset())
      action, reward, done, episode_length = Variable(torch.LongTensor([0]).unsqueeze(0)), 0, False, 0
    elif not args.no_truncate:
      # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
      hx = hx.detach()
      cx = cx.detach()
      avg_hx = avg_hx.detach()
      avg_cx = avg_cx.detach()

    # Lists of outputs for training
    Qs, Vs, policies, average_policies, log_probs, rewards, entropies = [], [], [], [], [], [], []

    while not done and t - t_start < args.t_max:
      # Calculate policy and values
      input = extend_input(state, action_to_one_hot(action, action_size), reward, episode_length)
      policy, Q, V, (hx, cx) = model(Variable(input), (hx, cx))
      average_policy, _, (avg_hx, avg_cx) = average_model(Variable(input, volatile=True), (avg_hx, avg_cx))
      log_policy = policy.log()
      entropy = -(log_policy * policy).sum(1)

      # Sample action
      action = policy.multinomial()
      # Graph broken as loss for stochastic action calculated manually
      log_prob = log_policy.gather(1, action.detach())  # Log probability of chosen action

      # Step
      next_state, reward, done, _ = env.step(action.data[0, 0])
      next_state = state_to_tensor(next_state)
      reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
      done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
      episode_length += 1  # Increase episode counter

      # Save part of transition for offline training
      memory.append(input, action.data[0, 0], reward, log_prob.data.exp()[0, 0])
      # Save outputs for online training
      Qs.append(Q)
      Vs.append(V)
      policies.append(policy)
      average_policies.append(average_policy)
      log_probs.append(log_prob)
      rewards.append(reward)
      entropies.append(entropy)

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
      # R = V(s_i; θ) for non-terminal s
      policy, Q, _ = model(input, (hx, cx))
      # action = policy.multinomial()
      # log_prob = policy.log().gather(1, action.detach())
      Qret = (policy * Q).sum(1)
      # log_probs.append(log_prob.detach())
    # Qs.append(Q.detach())

    # Train the network on-policy
    policy_loss = 0
    value_loss = 0
    Qret = Qret.detach()
    A_GAE = torch.zeros(1, 1)  # Generalised advantage estimator Ψ
    # Qrets = [None] * len(Vs)
    # Calculate n-step returns in forward view, stepping backwards from the last state
    for i in reversed(range(len(rewards))):  # TODO: Consider normalising loss by number of steps?
      # Qret ← r_i + γQret
      Qret = rewards[i] + args.discount * Qret
      # Advantage A = Qret - V(s_i; θ)
      A = Qret - Vs[i]
      # g ← min(c, ρ_i)∙∇θ∙log(π(a_i|s_i; θ))∙A + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ))
      g = min(args.max_trace, 1) * log_probs[i] * A + \
          (max(1 - args.max_trace / 1, 0) * policies[i] * policies[i].log() * (Qs[i] - Vs[i].expand_as(Qs[i]))).sum(1)
      # k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
      k = policies[i] * (policies[i].log() - average_policies[i])
      
      # dθ ← dθ + ∂θ/∂θ(g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k - β∙∇θH(π(s_i; θ))
      print(g)
      print(k)
      print(torch.mm(k.t(), g))
      policy_loss += g - max(0, torch.mm(k.t(), g) - args.trust_max)
      # + args.entropy_weight * entropies[i]     
      quit()

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

      # TD residual δ = r_i + γV(s_i+1; θ) - V(s_i; θ)
      td_error = rewards[i] + args.discount * Vs[i + 1].data - Vs[i].data
      # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
      A_GAE = A_GAE * args.discount * args.trace_decay + td_error
      # dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙Ψ - β∙∇θH(π(s_i; θ))
      policy_loss += -log_probs[i] * Variable(A_GAE) + args.entropy_weight * entropies[i]

    # TODO: Zero local grads too surely? When transferring, aren't shared lost anyway?
    optimiser.zero_grad()
    # Note that losses were defined as negatives of normal update rules for gradient descent
    (policy_loss + value_loss).backward(retain_variables=args.no_truncate)
    # Gradient (L2) norm clipping
    nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm)

    # Transfer gradients to shared model and update
    _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    if args.lr_decay:
      # Decay learning rate
      _decay_learning_rate(optimiser, args.T_max)

    # Train the network off-policy
    _train_off_policy(args, model, shared_model, optimiser, memory)

  env.close()
