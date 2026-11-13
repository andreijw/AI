# Acrobot-v1

Swing a two-link arm until its tip rises above a line. The first rung of the challenge ladder after CartPole: the reward
says only "not yet" until the very end, so the agent has to explore its way to the first success.

## The task

Two links hang from a fixed pivot, joined by a second hinge. Only the joint between the links is powered. The agent
applies a torque there, and the arm must pump energy into the swing until the free end of the lower link rises above a
line at the height of one link above the pivot. This repository runs Gymnasium's
[`Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/) through
`rl_acrobot.environments.acrobot_env.AcrobotEnv`, built by `rl_acrobot.environments.make_env.make_env_from_config`; it
adds optional observation noise only.

## Observation, action and reward

- **Observation**: six numbers, `Box(6,)`: the cosine and sine of each joint's angle, and the two angular velocities.
- **Action**: `Discrete(3)`: a torque of `-1`, `0` or `+1` on the powered joint.
- **Reward**: `-1` for every step that does not reach the line, `0` for the step that does. The return is minus the
  number of steps it took.

## How an episode ends and the solve threshold

The episode is **terminated** when the tip rises above the line, and **truncated** by the time limit of `500` steps
([terminated and truncated](../foundations.md#terminated-and-truncated)), which a random agent always reaches. A run
solves Acrobot when its 100-episode training average reaches -100, Gymnasium's `reward_threshold`, and holds it when its
last 100 episodes still do (`rl_acrobot.benchmark.AcrobotChallenge.solve_threshold`) (CHANGELOG.md, 2026-10-22): on
average the tip must reach the line within 100 steps.

## Who solves it

PPO and DQN; the table is in [results.md](../../results.md#which-agents-solve-which-challenge).

No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-10-22): "PPO solves 5 of 5 (72k to 278k steps) and holds
5 of 5 (final averages -83 to -99), evaluation -86 ± 9; DQN solves 5 of 5 (25k to 32k steps) and holds 5 of 5 (final
averages -90 to -96), evaluation -85 ± 6; random scores -500".

## What was hard

- **How much exploration stays.** DQN with rl-zoo's exploration floor of 0.1 held on 1 of 5 seeds, its training average
  ending at -95 to -103; with 0.02 it held on 5 of 5, and with 0.05 on 5 of 5 by under 1.2 points on three seeds
  (CHANGELOG.md, 2026-10-22). Exploration counts in the training average the rule judges, and a 10% random action costs
  Acrobot steps.
- **Tuned on the reporting seeds.** That floor was chosen on the same seeds 1 to 5 it is reported on, with no held-out
  seeds, so the 5 of 5 is optimistic (CHANGELOG.md, 2026-11-11).
- **The configs depart from rl-zoo.** DQN uses one hidden layer of 256 instead of two; PPO runs on one environment,
  without normalisation, for 500,000 steps instead of 1M (CHANGELOG.md, 2026-10-22).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
python train.py --config configs/acrobot_dqn.yaml       # or acrobot_ppo.yaml; the random baseline is acrobot_default.yaml
python record.py --config configs/acrobot_dqn.yaml --checkpoint checkpoints/acrobot/dqn/agent_episode_<n>.pt
python benchmark.py --challenge acrobot --agents random dqn ppo --seeds 1 --output-dir <scratch> --report <scratch>/report.md
```

Every command is in [usage.md](../../usage.md); [DQN](../algorithms/dqn.md) and [PPO](../algorithms/ppo.md) explain the
agents.
