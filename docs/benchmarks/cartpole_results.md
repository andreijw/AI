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
| **random** | 19.9 | ±0.6 | 0% | N/A | 0.59s |
| **reinforce** | 51.0 | ±41.0 | 0% | N/A | 3.57s |
| **actor_critic** | 436.5 | ±63.5 | 50% | 750.0 | 35.04s |
| **ppo** | 15.5 | ±5.5 | 0% | N/A | 11.37s |
| **dqn** | 470.0 | ±30.0 | 100% | 493.0 | 41.16s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
