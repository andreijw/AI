# Pendulum-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `Pendulum-v1` (Solved criteria: average reward >= -200)
- **Episodes per Run**: 150
- **Seeds Evaluated**: `[42, 123]`
- **Algorithms**: `['random', 'actor_critic', 'ppo']`

## Performance Summary

| Algorithm | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **random** | -1322.6 | ±138.9 | 0% | N/A | 3.81s |
| **actor_critic** | -1425.7 | ±78.9 | 0% | N/A | 4.47s |
| **ppo** | -1625.8 | ±16.0 | 0% | N/A | 7.99s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/pendulum/benchmark_comparison.png)
