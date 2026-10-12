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
| **random** | -1128.5 | ±114.0 | 0% | N/A | 4.04s |
| **actor_critic** | -1038.6 | ±276.7 | 0% | N/A | 4.75s |
| **ppo** | -1643.6 | ±25.5 | 0% | N/A | 8.39s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/pendulum/benchmark_comparison.png)
