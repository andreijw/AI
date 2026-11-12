# BipedalWalker-v3 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `BipedalWalker-v3` (Solved criteria: average reward >= 300)
- **Budget per Run**: each agent's own config (`training.total_steps` or `num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['sac', 'td3', 'random']`

## How a Run Counts as Solved

A run is solved when its average training reward over 100 consecutive
episodes reaches 300. **Avg Solve Episode** is the last episode of
the first such window, and **Avg Steps to Solve** the training steps up to it,
averaged over the solved runs. **Held** is the share of runs with at least
100 episodes whose last 100 average at least 300.
**Eval ≥ Threshold** is the share of runs whose final evaluation
(10 greedy episodes) averaged at least 300; it does not count as a solve.
Evaluation runs on a separate environment and plays the same start states
every time for a seed, so it does not change the training episodes.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **sac** | 500000 steps | 645 | 324.4 | ±1.9 | 80% | 360.0 | 307,090 | 20% | 100% | 6028.71s |
| **td3** | 500000 steps | 626 | 300.1 | ±12.3 | 40% | 581.5 | 411,898 | 0% | 40% | 7458.42s |
| **random** | 50000 steps | 86 | -98.8 | ±4.2 | 0% | N/A | N/A | 0% | 0% | 34.44s |

## How These Runs Were Made

Measured 2026-11-05 on the Jetson (`backend: torch`, CUDA) with `configs/bipedalwalker_sac.yaml`,
`configs/bipedalwalker_td3.yaml` and `configs/bipedalwalker_default.yaml` (random). Each seed ran as
its own `benchmark.py --challenge bipedalwalker --agents <agent> --seeds <seed>` process, three at
a time, and the per-seed metrics were merged into this table. **Mean Time** is therefore the time per
seed with three runs sharing the GPU and the six CPU cores; one run alone is faster. SAC trains without
the gSDE exploration rl-zoo uses for this environment (the configs' headers list the departures).
