# LunarLander-v3 and LunarLanderContinuous-v3

Land a craft between two flags with its engines. A larger problem than the classic-control ones: eight inputs, a shaped
reward with many parts, long episodes and a physics engine, Box2D. The discrete version is solved by
[DQN](../algorithms/dqn.md), the continuous one by [TD3](../algorithms/ddpg-td3.md) and [SAC](../algorithms/sac.md).

## The task

A lander starts near the top of the screen with a random push and must come down on the pad between the flags, upright
and slowly, using a main engine and two side engines. This repository runs Gymnasium's
[`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (the continuous one is the same id
with `continuous=True`) through `rl_lunarlander.environments.lunarlander_env.LunarLanderEnv`, built by
`rl_lunarlander.environments.make_env.make_env_from_config`. It needs Box2D, the `box2d` extra; see
[usage.md](../../usage.md#lunarlander-box2d).

## Observation, action and reward

- **Observation** (both): eight numbers, `Box(8,)`: the lander's position and velocity in `x` and `y`, its angle and
  angular velocity, and one flag per leg for touching the ground.
- **Action**: `LunarLander-v3` has `Discrete(4)`: do nothing, fire the left engine, the main engine or the right engine.
  `LunarLanderContinuous-v3` takes two numbers in `[-1, 1]`, `Box(2,)`: the main engine's throttle and the side engines'
  direction and strength.
- **Reward**, every step: more for moving closer to the pad, slower and more level, `+10` for each leg on the ground,
  and a small cost for each frame an engine fires; then `-100` for a crash or `+100` for coming to rest.

## How an episode ends and the solve threshold

An episode is **terminated** when the lander crashes, flies off the screen or comes to rest, and **truncated** at `1000`
steps ([terminated and truncated](../foundations.md#terminated-and-truncated)). The trainer's own episode cut is set to
that same limit in the configs (its default is 500) (CHANGELOG.md, 2026-10-27).

Both versions solve at a 100-episode training average of 200, Gymnasium's `reward_threshold`
(`rl_lunarlander.benchmark.LunarLanderChallenge.solve_threshold`,
`rl_lunarlander.benchmark.LunarLanderContinuousChallenge.solve_threshold`) (CHANGELOG.md, 2026-10-27): landing safely
nearly every time, exploration included.

## Who solves it

DQN on the discrete version, TD3 and SAC on the continuous one; the table is in
[results.md](../../results.md#which-agents-solve-which-challenge). TD3 has [the LunarLanderContinuous
report](../../benchmarks/lunarlander_continuous_results.md).

- **LunarLander-v3, DQN.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-10-27): "on seeds 1 to 5 the
  0.1 floor reached 200 on 5 of 5 at 152,000 to 228,000 steps and held it on 5 of 5 (final 100-episode training averages
  214 to 251, greedy evaluation 224 on average)"; and "on held-out seeds 11 to 15 it reached 200 on 5 of 5 at 166,000 to
  297,000 steps and held it on 4 of 5 (198 to 231), greedy evaluation 238 ± 25". Seed 42, which `train.py` uses,
  "reaches 200 at about episode 450 and ends at 236.9"; the random agent ends at -178 to -194.
- **LunarLanderContinuous-v3, SAC.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-11-04): "every seed
  reaches 200 at 79,587 to 244,525 steps (episodes 272 to 523) and holds it, final 100-episode averages 266 to 284,
  greedy evaluation 282 ± 5, all 5 at 200 or more".

## What was hard

- **The exploration floor.** rl-zoo's floor of 0.1 held 200 on 5 of 5 seeds against 3 of 5 for 0.02: the opposite of
  Acrobot, where the lower floor won (CHANGELOG.md, 2026-10-27).
- **rl-zoo, translated.** DQN keeps rl-zoo's settings with one hidden layer of 256 and a squared-error loss, where SB3
  uses Huber, and trains for 300,000 steps against rl-zoo's 100,000, which rl-zoo judges by a greedy evaluation after
  training (CHANGELOG.md, 2026-10-27). TD3 trains for 400,000 steps against rl-zoo's 300,000 (CHANGELOG.md, 2026-10-28).
- **Add-ons that did not help.** Prioritised replay held 200 on 1 of 5 seeds against 5 of 5 (CHANGELOG.md, 2026-11-04).
  Double DQN held on 4 against 5; dueling reached 200 sooner on all five seeds but held no more often (CHANGELOG.md,
  2026-10-29).
- **Updating once per episode.** TD3 updating at each episode's end held on 3 of 5 seeds (CHANGELOG.md, 2026-10-28);
  updating after every step, as rl-zoo does, on 5 of 5, and on held-out seeds 5 against 4, with a mean final average of
  242.8 on seeds 11 to 15 (CHANGELOG.md, 2026-10-30). On seeds 1 to 5 the final averages are 202 to 248, and a seed took
  53 to 54 minutes alone on the Jetson's GPU, against 28; the random agent ends at -214 to -228 (CHANGELOG.md,
  2026-11-01).
- **A good training average and a varied greedy policy.** TD3's greedy evaluation ranged from 58 to 251 across seeds 1
  to 5 (mean 182 ± 69), only 2 of 5 at 200 or more; SAC's was 282 ± 5 with every seed at 200 or more (CHANGELOG.md,
  2026-11-01 and 2026-11-04).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
python train.py --config configs/lunarlander_dqn.yaml                         # random baseline: lunarlander_default.yaml
.venv-gpu/bin/python train.py --config configs/lunarlander_continuous_sac.yaml   # or _td3; PyTorch backend
python record.py --config configs/lunarlander_dqn.yaml --checkpoint checkpoints/lunarlander/dqn/agent_episode_<n>.pt
```

The continuous configs train rl-zoo's 400×300 networks on the PyTorch backend, so they need `.venv-gpu` (`./setup.sh
--profile gpu`) (CHANGELOG.md, 2026-10-28 and 2026-11-04). Every command is in [usage.md](../../usage.md).
