# rl-core

A pure-NumPy reinforcement-learning framework: the agents (DQN, PPO, DDPG, TD3, SAC and more, with an optional
PyTorch backend for DDPG, TD3 and SAC), the trainer, the `BenchmarkChallenge` interface and its registry, and the
policy export to C and ONNX. Part of [andreijw/AI](https://github.com/andreijw/AI).

```bash
pip install https://github.com/andreijw/AI/releases/download/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl
```

Extras: `video` (training videos), `torch` (the PyTorch backend), `onnx` (ONNX export). Another project adds a
challenge by declaring a module in the `rl_core.challenges` entry-point group; see the repository's
`docs/external-challenges.md`. Changes: [CHANGELOG.md](CHANGELOG.md).
