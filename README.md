ACER
====
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Actor-critic with experience replay (ACER) [[1]](#references). The agent also receives the previous action and reward [[2]](#references). Uses batch off-policy updates to improve stability.

Run with `python main.py <options>`. To run asynchronous advantage actor-critic (A3C) [[3]](#references) (but with a Q-value head), use the `--on-policy` option.

Requirements
------------

- [Python](https://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Plotly](https://plot.ly/python/)

Acknowledgements
----------------

- [@ikostrikov](https://github.com/ikostrikov) for [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
- [@apaszke](https://github.com/apaszke) for [Reinforcement Learning (DQN) tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [@pfnet](https://github.com/pfnet) for [ChainerRL](https://github.com/pfnet/chainerrl)

References
----------

[1] [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)  
[2] [Learning to Navigate in Complex Environments](https://arxiv.org/abs/1611.03673)  
[3] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
