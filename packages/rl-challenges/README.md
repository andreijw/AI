# rl-challenges

The benchmark challenges of [andreijw/AI](https://github.com/andreijw/AI) on `rl-core`: CartPole-v1,
Pendulum-v1, Acrobot-v1, MountainCar-v0 and MountainCarContinuous-v0, LunarLander-v3 and its continuous version,
and BipedalWalker-v3. Each registers through the `rl_core.challenges` entry-point group. The committed configs
live in the repository's `configs/`, so training runs from a checkout.

It requires `rl-core`, which is not on PyPI: install both release wheels in one command, so pip never looks for
`rl-core` on an index (the repository's `docs/releasing.md`):

```bash
R=https://github.com/andreijw/AI/releases/download
pip install $R/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl $R/rl-challenges-v0.1.0/rl_challenges-0.1.0-py3-none-any.whl
```

Extras: `box2d` (LunarLander and BipedalWalker; build Box2D first as the repository's `./setup.sh` does). Changes:
[CHANGELOG.md](CHANGELOG.md).
