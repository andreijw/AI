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
Evaluation runs on a separate environment and plays the same start states
every time for a seed, so it does not change the training episodes.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **actor_critic** | 1000000 steps | 5000 | -172.2 | ±26.8 | 0% | N/A | N/A | 0% | 80% | 257.41s |
| **ddpg** | 40000 steps | 200 | -158.3 | ±24.8 | 100% | 156.0 | 31,200 | 100% | 100% | 202.84s |
| **ppo** | 200000 steps | 1000 | -188.3 | ±34.9 | 100% | 555.8 | 111,160 | 80% | 60% | 55.65s |
| **random** | 40000 steps | 200 | -1292.7 | ±81.2 | 0% | N/A | N/A | 0% | 0% | 5.68s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/pendulum/benchmark_comparison.png)
