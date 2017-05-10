async-rl
========
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Asynchronous advantage actor-critic (A3C) [[1]](#references) with generalised advantage estimation (GAE) [[2]](#references). The agent also receives the previous action, reward and a step counter [[3]](#references).

Run with `./run.sh <options>`. When running `main.py` directly, set `OMP_NUM_THREADS=1` to prevent multiple OpenMP threads being run (and clashing) in each process.

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

[1] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
[2] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
[3] [Learning to Navigate in Complex Environments](https://arxiv.org/abs/1611.03673)  
