# Pendulum-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `Pendulum-v1` (Solved criteria: average reward >= -200)
- **Episodes per Run**: each agent's own config (`training.num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['actor_critic', 'ppo', 'random']`

## How a Run Counts as Solved

A run is solved when its average training reward over 10 consecutive
episodes reaches -200. **Avg Solve Episode** is the first episode at
which that happens, averaged over the solved runs. **Eval ≥ Threshold** is the
share of runs whose final evaluation (10 greedy episodes) averaged at least
-200; it is reported on its own and does not count as a solve.

## Performance Summary

| Algorithm | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 300 | -1382.8 | ±62.5 | 0% | N/A | 0% | 10.01s |
| **ppo** | 500 | -284.6 | ±82.2 | 100% | 259.2 | 20% | 27.76s |
| **random** | 200 | -1224.5 | ±76.1 | 0% | N/A | 0% | 5.78s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/pendulum/benchmark_comparison.png)
