# BipedalWalker-v3

Teach a two-legged walker to cross rough ground. The hardest challenge here: four joints to coordinate, a `Box(24,)`
observation, long episodes and a fall that ends everything. [SAC](../algorithms/sac.md) and
[TD3](../algorithms/ddpg-td3.md) are measured on it, on the PyTorch backend.

## The task

A walker with a hull and two legs, each with a hip and a knee, must move to the right across slightly uneven terrain
without its hull touching the ground. It sees its own joints and ten lidar rays ahead of it, but not its position. This
repository runs Gymnasium's [`BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
through `rl_bipedalwalker.environments.bipedalwalker_env.BipedalWalkerEnv`, built by
`rl_bipedalwalker.environments.make_env.make_env_from_config`. It needs Box2D, the `box2d` extra; see
[usage.md](../../usage.md#lunarlander-box2d).

## Observation, action and reward

- **Observation**: `Box(24,)`: the hull's angle and angular velocity, its horizontal and vertical speed, each joint's
  angle and speed, a ground-contact flag per leg, and the ten lidar readings.
- **Action**: four motor speeds in `[-1, 1]`, `Box(4,)`, one per hip and knee.
- **Reward**: points for moving forward, over `300` in all for reaching the far end; a small cost for the motors'
  torque; and `-100` if the walker falls or leaves the left edge.

## How an episode ends and the solve threshold

An episode is **terminated** when the hull touches the ground, the walker leaves the left edge, or it reaches the end of
the course, and **truncated** at `1600` steps ([terminated and truncated](../foundations.md#terminated-and-truncated)).
A run solves it when its 100-episode training average reaches 300, Gymnasium's `reward_threshold`
(`rl_bipedalwalker.benchmark.BipedalWalkerChallenge.solve_threshold`; [the BipedalWalker
report](../../benchmarks/bipedalwalker_results.md)): crossing the whole course nearly every time, with the exploration
noise of training included.

## Who solves it

SAC reaches 300 on most seeds but holds it on few; TD3 less often. The table is in
[results.md](../../results.md#which-agents-solve-which-challenge), and the run, with the random floor, in [the
BipedalWalker report](../../benchmarks/bipedalwalker_results.md).

## What was hard

- **The training average against the greedy walk.** SAC's greedy evaluation was 324.4 ± 1.9 (321 to 327) with all 5
  seeds at 300 or more, but its training average, exploration included, held 300 on 1 of 5; TD3's greedy evaluation was
  300.1 ± 12.3, 2 of 5 at 300, and held on none. SAC's evaluation beat TD3's on all 5 paired seeds (CHANGELOG.md,
  2026-11-05). The random agent's greedy evaluation is -98.8 ([the BipedalWalker
  report](../../benchmarks/bipedalwalker_results.md)).
- **Parked, not solved.** The SAC hold is parked as a limit of the training-average rule, every seed's greedy policy
  scoring 321 to 327, to revisit with gSDE, the state-dependent exploration rl-zoo uses and this repository leaves out
  (CHANGELOG.md, 2026-11-12).
- **Departures from rl-zoo.** SAC runs without gSDE, and TD3 gets 500,000 steps, half rl-zoo's 1,000,000, so the two
  share a budget; the configs' headers list every departure (CHANGELOG.md, 2026-11-05).
- **The cost.** A seed took 6,029 s for SAC and 7,458 s for TD3 with three runs sharing the Jetson's GPU (CHANGELOG.md,
  2026-11-05). Seed 42, the seed `train.py` uses, ended SAC at a 100-episode training average of 302.4, and its actor
  exports to C and ONNX (CHANGELOG.md, 2026-11-05); that run took 636 episodes, 500,412 steps and 4,752 s (TESTING.md
  `TR-31`).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
.venv-gpu/bin/python train.py --config configs/bipedalwalker_sac.yaml   # or bipedalwalker_td3.yaml
python record.py --config configs/bipedalwalker_sac.yaml --checkpoint checkpoints/bipedalwalker/sac/agent_episode_<n>.pt --episodes 3
python train.py --config configs/bipedalwalker_default.yaml             # the random baseline
```

The SAC and TD3 configs train rl-zoo's 400×300 networks on the PyTorch backend, so they need `.venv-gpu` (`./setup.sh
--profile gpu`); `record.py` replays their checkpoints with the NumPy agent (CHANGELOG.md, 2026-11-11). Every command is
in [usage.md](../../usage.md).
