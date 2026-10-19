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
Evaluation runs on a separate environment and plays the same start states
every time for a seed, so it does not change the training episodes.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 500000 steps | 1485 | 497.8 | ±4.5 | 100% | 948.4 | 249,766 | 40% | 100% | 125.64s |
| **dqn** | 300000 steps | 1003 | 500.0 | ±0.0 | 100% | 803.0 | 200,881 | 100% | 100% | 63.15s |
| **ppo** | 200000 steps | 719 | 499.6 | ±0.8 | 80% | 590.8 | 138,906 | 80% | 100% | 49.95s |
| **random** | 50000 steps | 2243 | 23.2 | ±2.5 | 0% | N/A | N/A | 0% | 0% | 1.43s |
| **reinforce** | 500000 steps | 1559 | 497.0 | ±6.0 | 100% | 1222.0 | 347,498 | 40% | 100% | 103.93s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/cartpole/benchmark_comparison.png)
