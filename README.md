async-rl
========
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Actor-critic with experience replay (ACER) [[1]](#references). The agent also receives the previous action, reward and a step counter [[2]](#references). Uses batch off-policy updates to improve stability.

Run with `./run.sh <options>`. When running `main.py` directly, set `OMP_NUM_THREADS=1` to prevent multiple OpenMP threads being run (and clashing) in each process.

**TODO:** [UNREAL](https://arxiv.org/abs/1611.05397) + [Reactor](https://arxiv.org/abs/1704.04651)

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
