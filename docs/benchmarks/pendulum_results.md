# Pendulum-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `Pendulum-v1` (Solved criteria: average reward >= -200)
- **Budget per Run**: each agent's own config (`training.total_steps` or `num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['actor_critic', 'ddpg', 'ppo', 'random']`

## How a Run Counts as Solved

A run is solved when its average training reward over 100 consecutive
episodes reaches -200. **Avg Solve Episode** is the last episode of
the first such window, and **Avg Steps to Solve** the training steps up to it,
averaged over the solved runs. **Held** is the share of runs with at least
100 episodes whose last 100 average at least -200.
**Eval ≥ Threshold** is the share of runs whose final evaluation
(10 greedy episodes) averaged at least -200; it does not count as a solve.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 1000000 steps | 5000 | -151.7 | ±44.8 | 0% | N/A | N/A | 0% | 80% | 257.41s |
| **ddpg** | 40000 steps | 200 | -149.9 | ±26.6 | 100% | 156.4 | 31,280 | 100% | 100% | 202.51s |
| **ppo** | 200000 steps | 1000 | -150.0 | ±27.5 | 100% | 569.8 | 113,960 | 80% | 100% | 55.13s |
| **random** | 40000 steps | 200 | -1224.5 | ±76.1 | 0% | N/A | N/A | 0% | 0% | 5.65s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/pendulum/benchmark_comparison.png)
