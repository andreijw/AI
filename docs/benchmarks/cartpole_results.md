# CartPole-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `CartPole-v1` (Solved criteria: average reward >= 475)
- **Episodes per Run**: 750
- **Seeds Evaluated**: `[42, 123]`
- **Algorithms**: `['random', 'reinforce', 'actor_critic', 'ppo', 'dqn']`

## Performance Summary

| Algorithm | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **random** | 24.9 | ±5.9 | 0% | N/A | 0.48s |
| **reinforce** | 120.8 | ±19.2 | 0% | N/A | 4.08s |
| **actor_critic** | 499.9 | ±0.2 | 100% | 750.0 | 34.16s |
| **ppo** | 27.4 | ±6.5 | 0% | N/A | 11.91s |
| **dqn** | 413.4 | ±79.7 | 100% | 334.5 | 41.50s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
