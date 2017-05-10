from torch import nn
from torch.nn import init


# TODO: Wrap up network into Agent class w/ perceive, act, train methods etc.
class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.elu = nn.ELU(inplace=True)
    self.softmax = nn.Softmax()

    # Receive state + previous action, reward and timestep as input
    input_size = observation_space.shape[0] + action_space.n + 2
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, action_space.n)
    self.fc_critic = nn.Linear(hidden_size, action_space.n)

    # Xavier weight initialisation
    for name, p in self.named_parameters():
      if 'weight' in name:
        init.xavier_uniform(p)
      elif 'bias' in name:
        init.constant(p, 0)
    # Set LSTM forget gate bias to 1
    for name, p in self.lstm.named_parameters():
      if 'bias' in name:
        n = p.size(0)
        forget_start_idx, forget_end_idx = n // 4, n // 2
        init.constant(p[forget_start_idx:forget_end_idx], 1)

  def forward(self, x, h):
    hx, cx = h
    x = self.elu(self.fc1(x))
    hx, cx = self.lstm(x, (hx, cx))
    x = hx
    policy = self.softmax(self.fc_actor(x))
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1)  # V is expectation of Q under π
    return policy, Q, V, (hx, cx)
