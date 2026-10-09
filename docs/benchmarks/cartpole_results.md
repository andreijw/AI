# CartPole-v1 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `CartPole-v1` (Solved criteria: average reward >= 475)
- **Episodes per Run**: 500
- **Seeds Evaluated**: `[42, 123]`
- **Algorithms**: `['random', 'reinforce', 'actor_critic', 'ppo', 'dqn']`

## Performance Summary

| Algorithm | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **random** | 28.7 | ±3.8 | 0% | N/A | 0.39s |
| **reinforce** | 10.0 | ±0.0 | 0% | N/A | 2.18s |
| **actor_critic** | 476.5 | ±23.5 | 50% | 500.0 | 15.29s |
| **ppo** | 11.5 | ±1.5 | 0% | N/A | 7.28s |
| **dqn** | 386.5 | ±35.5 | 50% | 379.0 | 24.87s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
