# Pendulum-v1

Swing a pendulum up and hold it upright with a limited torque. The first continuous-action challenge: every agent for
continuous actions here was first measured on it, from the [policy-gradient](../algorithms/policy-gradient.md) and
[PPO](../algorithms/ppo.md) agents to [DDPG and TD3](../algorithms/ddpg-td3.md) and [SAC](../algorithms/sac.md).

## The task

A pendulum hangs from a motor at its pivot and starts at a random angle and speed. The motor is too weak to lift it
straight up, so the agent has to swing it back and forth to gain energy, then catch it at the top and hold it there with
small corrections. This repository runs Gymnasium's
[`Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/) through
`rl_pendulum.environments.pendulum_env.PendulumEnv`, built by `rl_pendulum.environments.make_env.make_env_from_config`;
it adds optional observation noise, action noise and domain randomization of the physics, none of which a committed
config turns on.

## Observation, action and reward

- **Observation**: three numbers, `Box(3,)`: the cosine and sine of the angle (so the angle has no jump at the bottom)
  and the angular velocity.
- **Action**: one torque in `[-2, 2]`, `Box(1,)`.
- **Reward**: `-(θ² + 0.1 · θ̇² + 0.001 · torque²)` every step, with `θ` the angle from upright in `[-π, π]`. It is zero
  only upright, still and unpushed, and every step costs something, so the return of an episode is always negative.

## How an episode ends and the solve threshold

Pendulum never terminates: every episode is **truncated** by the time limit of `200` steps ([terminated and
truncated](../foundations.md#terminated-and-truncated)). Since no episode really ends, treating that cut as a terminal
state teaches the critic that the last states are worth nothing; every agent here that has a critic bootstraps it
([returns and bootstrapping](../foundations.md#returns-and-bootstrapping)).

Gymnasium sets no `reward_threshold` for Pendulum (it is `None`). This repository counts a run as solved when its
100-episode training average reaches -200 (`rl_pendulum.benchmark.PendulumChallenge.solve_threshold`; [the Pendulum
report](../../benchmarks/pendulum_results.md)). The random agent scores about -1293 (CHANGELOG.md, 2026-10-19), so -200
asks for a real swing-up: the pendulum must be raised quickly and held, since an episode that leaves it hanging scores
far lower.

## Who solves it

DDPG, TD3, SAC and PPO; the continuous Actor-Critic plays well greedily but does not hold the threshold. The table is in
[results.md](../../results.md#which-agents-solve-which-challenge), and the published run, with the random floor, in [the
Pendulum report](../../benchmarks/pendulum_results.md).

The report has no TD3 or SAC row; for SAC that was the maintainer's decision (CHANGELOG.md, 2026-11-04):

- **TD3.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-10-25): "every run solves -200 at 32,600 to
  33,400 steps and holds it (final averages -143 to -165, greedy evaluation -157 ± 24)".
- **SAC.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-11-04): "all solve -200 at 21,800 to 23,200
  steps (TD3: 32,600 to 33,400) and hold, final 100-episode averages -135.9 to -163.8, greedy evaluation -157 ± 20 (TD3:
  -157 ± 24); about 9 minutes a seed".

## What was hard

- **PPO's five fixes, at once.** The log-probability of the clipped action (the cause of a NaN), an initial mean far
  outside the bounds, unscaled rewards, the time limit treated as terminal, and γ 0.99 (now 0.9) (CHANGELOG.md,
  2026-10-15). Then rollouts across episodes: updating on one episode at a time solved 0 of 5, rollouts of 2048 steps
  over 200,000 steps, twice rl-zoo's 100,000, solved 5 of 5 and held on 4 of 5 (CHANGELOG.md, 2026-10-18).
- **Observation normalisation hurt PPO.** It held -200 on 2 of 5 seeds against 4 (CHANGELOG.md, 2026-11-04): the
  statistics kept moving while the learning rate and clip range annealed to zero, so the late policy could not follow
  its own inputs (CHANGELOG.md, 2026-11-09).
- **Small networks for NumPy.** DDPG uses rl-zoo's settings with 64×64 networks instead of 400×300, which take about 16
  times longer in NumPy; against rl-zoo's -152 (± 94 across episodes of one run) (CHANGELOG.md, 2026-10-18), its greedy
  evaluation averages -158 (± 25 across the 5 seeds' means) in the republished benchmark (CHANGELOG.md, 2026-10-19). TD3
  and SAC use the same small networks; SB3's 256×256 would take SAC about an hour a seed in NumPy (CHANGELOG.md,
  2026-11-04).
- **The continuous Actor-Critic stays short of -200.** A tanh-bounded mean and 8-step chunks made it beat the random
  agent (CHANGELOG.md, 2026-10-18); true A2C did not close the gap (CHANGELOG.md, 2026-10-30); observation normalisation
  helped the training average (CHANGELOG.md, 2026-11-04); rl-zoo's 8 environments raised the mean final average to
  -244.1 from -282.7, better on every seed, final averages -227.6 to -268.7, with 2 of 5 reaching -200 in a 100-episode
  window and none holding it, while the greedy evaluation fell to -184.0 ± 42.7 from -170.0 ± 25.9 (CHANGELOG.md,
  2026-11-12). On one environment the final averages had been -268 to -298 (TESTING.md `TR-5`). Its greedy evaluation is
  still far above the random agent's ([the Pendulum report](../../benchmarks/pendulum_results.md)).
- **A SAC step in NumPy.** At 64×64 a SAC gradient step takes about 11 ms on the Jetson's CPU, about 9 minutes a run
  (TESTING.md `TR-28`).
- **TD3 solves later than DDPG.** Its actor updates half as often, so it solves about 1,700 steps after DDPG on this
  easy task (CHANGELOG.md, 2026-10-25).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
python train.py --config configs/pendulum_sac.yaml       # also _ddpg, _td3, _ppo, _actor_critic; random: _default
python record.py --config configs/pendulum_sac.yaml --checkpoint checkpoints/pendulum/sac/agent_episode_<n>.pt
python benchmark.py --challenge pendulum --agents random ddpg --seeds 1 --output-dir <scratch> --report <scratch>/report.md
```

Every command is in [usage.md](../../usage.md).
