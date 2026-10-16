# CartPole-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `CartPole-v1` (Solved criteria: average reward >= 475)
- **Budget per Run**: each agent's own config (`training.total_steps` or `num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['actor_critic', 'dqn', 'ppo', 'random', 'reinforce']`

## How a Run Counts as Solved

A run is solved when its average training reward over 100 consecutive
episodes reaches 475. **Avg Solve Episode** is the last episode of
the first such window, and **Avg Steps to Solve** the training steps up to it,
averaged over the solved runs. **Held** is the share of runs with at least
100 episodes whose last 100 average at least 475.
**Eval ≥ Threshold** is the share of runs whose final evaluation
(10 greedy episodes) averaged at least 475; it does not count as a solve.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 500000 steps | 1486 | 499.6 | ±0.9 | 100% | 1072.8 | 308,973 | 80% | 100% | 124.49s |
| **dqn** | 300000 steps | 992 | 500.0 | ±0.0 | 100% | 754.0 | 181,723 | 100% | 100% | 63.59s |
| **ppo** | 200000 steps | 693 | 500.0 | ±0.0 | 100% | 530.2 | 121,197 | 80% | 100% | 50.30s |
| **random** | 50000 steps | 2236 | 22.0 | ±2.6 | 0% | N/A | N/A | 0% | 0% | 1.41s |
| **reinforce** | 500000 steps | 1541 | 497.5 | ±5.0 | 80% | 1257.8 | 365,311 | 40% | 100% | 106.18s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
