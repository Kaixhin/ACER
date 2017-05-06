malmo-challenge
===============
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Team Pig Catcher's entry for the Malmo Collaborative AI Challenge. The base agent is the asynchronous advantage actor-critic (A3C) [[1]](#references) with generalised advantage estimation (GAE) [[2]](#references). The agent also receives the previous action, reward and a step counter [[3]](#references).

Run with `OMP_NUM_THREADS=1 python main.py`. The environment flag prevents multiple OpenMP threads being run in each process.

Requirements
------------

- [Python 3](https://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [OpenAI Gym](https://gym.openai.com/)

Acknowledgements
----------------

- [@ikostrikov](https://github.com/ikostrikov) for [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c) (used as reference)

References
----------

[1] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
[2] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
[3] [Learning to Navigate in Complex Environments](https://arxiv.org/abs/1611.03673)  
