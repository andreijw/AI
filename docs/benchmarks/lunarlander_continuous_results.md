# LunarLanderContinuous-v3 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `LunarLanderContinuous-v3` (Solved criteria: average reward >= 200)
- **Budget per Run**: each agent's own config (`training.total_steps` or `num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['td3', 'random']`

## How a Run Counts as Solved

A run is solved when its average training reward over 100 consecutive
episodes reaches 200. **Avg Solve Episode** is the last episode of
the first such window, and **Avg Steps to Solve** the training steps up to it,
averaged over the solved runs. **Held** is the share of runs with at least
100 episodes whose last 100 average at least 200.
**Eval ≥ Threshold** is the share of runs whose final evaluation
(10 greedy episodes) averaged at least 200; it does not count as a solve.
Evaluation runs on a separate environment and plays the same start states
every time for a seed, so it does not change the training episodes.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **td3** | 400000 steps | 1195 | 182.3 | ±69.0 | 100% | 629.8 | 240,063 | 100% | 40% | 3208.99s |
| **random** | 50000 steps | 456 | -230.0 | ±26.4 | 0% | N/A | N/A | 0% | 0% | 11.41s |

## Learning Trajectories

![Benchmark Learning Curves](../../plots/benchmarks/lunarlander_continuous/benchmark_comparison.png)
