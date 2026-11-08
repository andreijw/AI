# Navigation-v0 Multi-Agent Benchmark Results

Comparative benchmark evaluation across reinforcement learning algorithms.

## Benchmark Configuration

- **Challenge / Environment**: `Navigation-v0` (Solved criteria: average reward >= 7)
- **Budget per Run**: each agent's own config (`training.total_steps` or `num_episodes`), shown per agent
- **Seeds Evaluated**: `[1, 2, 3, 4, 5]`
- **Algorithms**: `['ppo', 'dqn', 'random']`

## How a Run Counts as Solved

A run is solved when its average training reward over 100 consecutive
episodes reaches 7. **Avg Solve Episode** is the last episode of
the first such window, and **Avg Steps to Solve** the training steps up to it,
averaged over the solved runs. **Held** is the share of runs with at least
100 episodes whose last 100 average at least 7.
**Eval ≥ Threshold** is the share of runs whose final evaluation
(100 greedy episodes) averaged at least 7; it does not count as a solve.
Evaluation runs on a separate environment and plays the same start states
every time for a seed, so it does not change the training episodes.
A step budget finishes the episode that crosses it, so a run can overshoot its budget by up to one episode.

## Performance Summary

| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) | Eval Success |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ppo** | 200000 steps | 1836 | 6.8 | ±0.8 | 100% | 982.0 | 115,769 | 20% | 20% | 293.82s | 88% |
| **dqn** | 300000 steps | 1924 | 7.8 | ±0.5 | 100% | 1250.6 | 226,974 | 100% | 80% | 874.17s | 94% |
| **random** | 40000 steps | 405 | -11.0 | ±0.5 | 0% | N/A | N/A | 0% | 0% | 57.63s | 1% |
