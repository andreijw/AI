# rl-challenges

The benchmark challenges of [andreijw/AI](https://github.com/andreijw/AI) on `rl-core`: CartPole-v1,
Pendulum-v1, Acrobot-v1, MountainCar-v0 and MountainCarContinuous-v0, LunarLander-v3 and its continuous version,
and BipedalWalker-v3. Each registers through the `rl_core.challenges` entry-point group. The committed configs
live in the repository's `configs/`, so training runs from a checkout.

Extras: `box2d` (LunarLander and BipedalWalker; build Box2D first as the repository's `./setup.sh` does). Changes:
[CHANGELOG.md](CHANGELOG.md).
