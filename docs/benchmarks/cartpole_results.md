# CartPole-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `CartPole-v1` (Solved criteria: average reward >= 475)
- **Episodes per Run**: each agent's own config (`training.num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['actor_critic', 'dqn', 'ppo', 'random', 'reinforce']`

## How a Run Counts as Solved

A run is solved when its average training reward over 10 consecutive
episodes reaches 475. **Avg Solve Episode** is the first episode at
which that happens, averaged over the solved runs. **Eval ≥ Threshold** is the
share of runs whose final evaluation (10 greedy episodes) averaged at least
475; it is reported on its own and does not count as a solve.

## Performance Summary

| Algorithm | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 1500 | 494.3 | ±11.4 | 100% | 1062.0 | 80% | 105.61s |
| **dqn** | 750 | 450.0 | ±46.4 | 80% | 449.2 | 40% | 39.09s |
| **ppo** | 2000 | 495.2 | ±9.4 | 100% | 968.2 | 100% | 438.33s |
| **random** | 1000 | 20.8 | ±5.8 | 0% | N/A | 0% | 0.62s |
| **reinforce** | 2000 | 449.3 | ±59.2 | 100% | 829.6 | 40% | 148.53s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
