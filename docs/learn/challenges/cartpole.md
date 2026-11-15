# CartPole-v1

Balance a pole on a cart by pushing the cart left or right. The first challenge of the [learning path](../README.md):
small and fast, and all four learning agents for discrete actions solve it, REINFORCE on 4 of 5 seeds ([the CartPole
report](../../benchmarks/cartpole_results.md)). That makes it the place to compare four ideas on one problem: REINFORCE
and the Actor-Critic ([Policy gradients](../algorithms/policy-gradient.md)), [PPO](../algorithms/ppo.md) and
[DQN](../algorithms/dqn.md).

## The task

A pole stands on a cart on a frictionless track, joined by an unpowered hinge. Each step the agent pushes the cart left
or right with a fixed force, and the pole tips according to the push and to gravity. The goal is to keep the pole up,
and the cart on the track, for as long as possible. This repository runs Gymnasium's
[`CartPole-v1`](https://gymnasium.farama.org/environments/classic_control/cart_pole/) through its own wrapper,
`rl_cartpole.environments.cartpole_env.CartPoleEnv`, built by `rl_cartpole.environments.make_env.make_env_from_config`
from a config's `environment` section; the wrapper adds optional observation noise, action noise and domain
randomization of the physics, none of which a committed config turns on.

## Observation, action and reward

- **Observation**: four numbers, `Box(4,)`: the cart's position and velocity, the pole's angle and its angular velocity.
- **Action**: `Discrete(2)`: push left (`0`) or push right (`1`).
- **Reward**: `+1` for every step, including the last, so an episode's return is its length.

## How an episode ends and the solve threshold

The episode is **terminated** when the pole leans more than `12°` from upright or the cart's centre leaves `±2.4`, and
**truncated** by Gymnasium's time limit of `500` steps ([terminated and
truncated](../foundations.md#terminated-and-truncated)). A truncated episode did not fail: the pole was still up. An
agent with a critic should bootstrap it rather than treat it as the end, and here that was worth a great deal (see What
was hard).

A run **solves** CartPole when its training reward averaged over 100 consecutive episodes reaches 475, Gymnasium's
`reward_threshold` for this id, and **holds** it when its last 100 episodes still do ([solved and
held](../foundations.md#solved-and-held); `rl_cartpole.benchmark.CartPoleChallenge.solve_threshold`; [the CartPole
report](../../benchmarks/cartpole_results.md)). With a maximum of 500 a step, that asks for nearly perfect balancing in
almost every episode, exploration included.

## Who solves it

All four solve it; REINFORCE on 4 of 5 seeds, holding on 3 ([the CartPole
report](../../benchmarks/cartpole_results.md)). The table, with each config, budget, steps to solve and how many seeds
hold, is in [results.md](../../results.md#which-agents-solve-which-challenge), and the full run in [the CartPole
report](../../benchmarks/cartpole_results.md). The random agent's floor is in the same report.

## What was hard

- **Plain SGD.** REINFORCE with SGD solved 0 of 5 seeds in 2000 episodes, with Adam 5 of 5, under the 10-episode rule
  then in use; the Actor-Critic reached 475 at episodes 490 to 839 with Adam against 906 to 1269 with SGD (CHANGELOG.md,
  2026-10-15). Under the 100-episode rule and the benchmark republished with separate random streams, REINFORCE with
  Adam solves on 4 of 5 seeds (CHANGELOG.md, 2026-10-19) and holds on 3 ([the CartPole
  report](../../benchmarks/cartpole_results.md)).
- **The time limit read as a failure.** Once discrete PPO bootstrapped the 500-step cut, its mean steps to solve fell
  from 108,325 to 79,556 (CHANGELOG.md, 2026-10-20). The Actor-Critic with `bootstrap_truncation` held 475 on 5 of 5
  seeds against 4 of 5 (CHANGELOG.md, 2026-11-12).
- **DQN's budget.** CartPole DQN trains for 300,000 steps rather than rl-zoo's 50,000: holding a 100-episode average of
  475 alone takes 47,500 steps, and rl-zoo's 50,000-step result is a greedy evaluation after training (CHANGELOG.md,
  2026-10-18). Its first exploration rule decayed epsilon once per episode; see
  [DQN](../algorithms/dqn.md#fixes-this-repo-needed).
- **Without the velocities.** With `mask_velocity` (the velocity entries zeroed) PPO solved 0 of 5 seeds, and with a
  4-frame stack still 0 of 5 (CHANGELOG.md, 2026-11-01). Velocity survives only as a change between frames about 50
  times smaller than the velocity itself; `normalize_observations` with `frame_deltas` solved and held on 5 of 5
  (CHANGELOG.md, 2026-11-02). No masked config is committed.
- **A curriculum did not help.** With observation noise and a randomized pole length, starting easy and hardening scored
  no better than training at the target from the start: better on 2 of 5 seeds, worse on 2 (CHANGELOG.md, 2026-11-03).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
python train.py --config configs/cartpole_ppo.yaml                 # PPO, seed 42; also _dqn, _actor_critic, _reinforce
python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_<n>.pt
python benchmark.py --challenge cartpole --agents random ppo --seeds 1 --output-dir <scratch> --report <scratch>/report.md
```

`train.py` uses the config's `seed: 42`, which the published report does not include, so one run can land on either side
of the report's numbers. Every command is in [usage.md](../../usage.md).
