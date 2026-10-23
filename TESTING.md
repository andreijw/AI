# AI Reinforcement Learning Framework — Manual Testing Manual (`TESTING.md`)

This document outlines manual verification and testing procedures for the RL framework.

---

## 1. Environment & Feature Demonstrations

### Gymnasium Feature Sanity Check

- **Script**: `examples/demo_env_features.py`
- **Purpose**: Verify that observation noise, action flip noise, domain randomization, and vectorized environments
  execute without crashes or numerical errors.

| Check # | Target Feature | Steps to Execute | Expected Behavior |
| --- | --- | --- | --- |
| ENV-1 | **Standard Environment** | `python examples/demo_env_features.py` | Runs demo 1 cleanly; CartPole resets and takes random steps. |
| ENV-2 | **Observation Noise** | Observe demo 2 console output. | Adds Gaussian noise to observations; state values deviate slightly from raw dynamics. |
| ENV-3 | **Action Noise** | Observe demo 3 console output. | Actions flip stochastically based on specified probability without throwing errors. |
| ENV-4 | **Domain Randomization** | Observe demo 4 console output. | Physical parameters (`gravity`, `masscart`, `masspole`, `length`) resample per episode. |
| ENV-5 | **Vectorized Execution** | Observe demo 5 console output. | Runs 4 parallel vectorized environments; step returns batched arrays with shape `(4, 4)`. |

---

## 2. Agent Training & Evaluation Verification

### Training Execution Checks

| Check # | Algorithm | Command to Execute | Expected Behavior |
| --- | --- | --- | --- |
| TR-0 | **Step budget in train.py** | `python train.py --config <copy of configs/cartpole_default.yaml with checkpoint_dir and log_dir in a scratch folder> --total-steps 2000` | The log says "Starting training for 2000 environment steps", progress lines show `steps n/2000`, and training stops after the episode that crosses 2000. `--num-episodes` together with `--total-steps` is rejected. |
| TR-1 | **REINFORCE** | `python train.py --config configs/cartpole_reinforce.yaml` | Trains for 500,000 environment steps (about 1500 episodes) with Adam and saves checkpoints. Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams): the 100-episode training average reaches 475 on 4 of 5 (255k to 376k steps) and holds it to the end on 3 of 5; final averages 373 to 499, evaluation 497 ± 5. Seed 42, which `train.py` uses, was not measured. |
| TR-2 | **Actor-Critic** | `python train.py --config configs/cartpole_actor_critic.yaml` | Trains for 500,000 environment steps (about 1500 episodes) with Adam. Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams): reaches 475 on 5 of 5 (209k to 456k steps), holds it on 4 of 5 (final averages 473 to 488), evaluation 500 ± 0. Seed 42 not measured. |
| TR-3 | **PPO** | `python train.py --config configs/cartpole_ppo.yaml` | Trains for 200,000 environment steps (about 700 episodes) of standard PPO (256-step rollouts, 20 epochs, learning rate and clip annealed over the same 200,000 steps); `clip_fraction`, `rollout_steps` and `anneal_fraction` are logged on the episodes that train. Measured 2026-10-19 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams, time-limit cuts bootstrapped): reaches 475 on 5 of 5 (72k to 90k steps) and holds it on all 5 (final averages 500); evaluation 500 ± 0. Seed 42 not measured. |
| TR-4 | **DQN** | `python train.py --config configs/cartpole_dqn.yaml` | Trains for 300,000 environment steps (about 1000 episodes) with Adam; the logged `epsilon` falls linearly to 0.04 within the first 8,000 environment steps. Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams): 5 of 5 reach 475 at 143,952 to 183,439 steps and hold it, final averages 494 to 500, evaluation 500 ± 0 (CHANGELOG.md, 2026-10-19). Seed 42 not measured. |
| TR-5 | **Continuous A2C** | `python train.py --config configs/pendulum_actor_critic.yaml` (about 4 minutes) | Trains for 1,000,000 environment steps (5000 episodes) in 8-step chunks, with finite losses, an `anneal_fraction` falling from 1 to about 0, and a `saturation` metric that should not stay near 1 (the bounded mean pinned at a bound). Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams): it does not solve (final 100-episode averages -286 to -307, short of -200), but its greedy evaluation of -172 ± 27 across seed means beats the random agent's -1293 on 5 of 5 seeds (before the bounded mean: -1422 ± 134) (CHANGELOG.md, 2026-10-19). Seed 42 not measured. |
| TR-6 | **Continuous PPO** | `python train.py --config configs/pendulum_ppo.yaml` (about a minute) | Trains for 200,000 environment steps (1000 episodes) on rollouts of 11 whole episodes (2200 transitions): `update` metrics, including `rollout_steps`, appear only every 11th episode, so a 10-episode log record holds one rollout or none, and `anneal_fraction` falls from 1 to about 0. Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, step budget, evaluation on its own environment, named random streams): 5 of 5 reach -200 at 72,400 to 156,800 steps and 4 hold it (seed 3 ends at -213); evaluation -188 ± 35 across seed means (CHANGELOG.md, 2026-10-19). Seed 42 not measured. |
| TR-7 | **Challenge selection** | `python train.py --config configs/cartpole_default.yaml --challenge pendulum`, then a copy of `configs/cartpole_default.yaml` with `name: "Taxi-v3"` | The first exits with "--challenge pendulum does not match the config's environment 'CartPole-v1'"; the second with "Unknown environment 'Taxi-v3'. Supported: Acrobot-v1 (acrobot), CartPole-v1 (cartpole), MountainCar-v0 (mountaincar), Pendulum-v1 (pendulum)". Neither creates a log folder. |
| TR-8 | **DDPG** | `python train.py --config configs/pendulum_ddpg.yaml` (about 5 minutes) | The first 50 episodes (10,000 steps) are uniform random warm-up and log no `update` metrics; then `critic_loss`, `actor_loss`, `mean_q`, `saturation` and both gradient norms appear and stay finite. Measured 2026-10-18 through the benchmark on seeds 1 to 5 (100-episode window, 40,000 steps, evaluation on its own environment, named random streams): 5 of 5 reach -200 at 31,200 steps (episode 156) and hold it, final averages -144 to -166, evaluation averaging -158 (± 25 across seed means). Seed 42 not measured since the evaluation change. A checkpoint holds the networks, both Adam states, the counters, the replay buffer and the generators. |
| TR-9 | **Evaluation apart from training** | Copy `configs/cartpole_reinforce.yaml` twice into a scratch folder with `checkpoint_dir` and `log_dir` there, `num_episodes: 100`, and `eval_frequency: 10` in one copy and `1000` in the other; run `python train.py --config <copy>` for each | The log line "Environments created (evaluation start states from seed N)" shows the same N in both runs, and the two `metrics_*.jsonl` files have identical `avg_reward` records: evaluating every 10 episodes no longer changes training. In the first run, the `evaluation` records' rewards come from the same 10 start states each time. |
| TR-10 | **Acrobot DQN** | `python train.py --config configs/acrobot_dqn.yaml` (about 3 minutes) | Trains for 100,000 environment steps (about 900 episodes) with one gradient step per transition; the logged `epsilon` falls linearly from 1.0 to 0.02 over the first 12,000 steps, and `loss` and `mean_q` stay finite. Measured 2026-10-22 through the benchmark on seeds 1 to 5: reaches -100 on 5 of 5 (25k to 32k steps) and holds it on 5 of 5 (final 100-episode averages -90 to -96); greedy evaluation -85 ± 6, at or above -100 on all 5 (CHANGELOG.md, 2026-10-22). Seed 42 not measured. |
| TR-11 | **Acrobot PPO** | `python train.py --config configs/acrobot_ppo.yaml` (about 3 minutes) | Trains for 500,000 environment steps (about 4,600 episodes) on rollouts of at least 4096 transitions; `anneal_fraction` falls from 1 to about 0. Measured 2026-10-22 through the benchmark on seeds 1 to 5: reaches -100 on 5 of 5 (72k to 278k steps) and holds it on 5 of 5 (final averages -83 to -99); greedy evaluation -86 ± 9, at or above -100 on 4 of 5 (CHANGELOG.md, 2026-10-22). Seed 42 not measured. |
| TR-12 | **MountainCar DQN** | `python train.py --config configs/mountaincar_dqn.yaml` (about 3 minutes) | Trains for 120,000 environment steps (about 750 episodes) with half a gradient step per transition; the first 1,000 transitions are warm-up, `epsilon` falls from 1.0 to 0.02 over 24,000 steps, and `loss` and `mean_q` stay finite (`mean_q` settles around -30 to -40). Expect episodes of 200 steps (reward -200, the flag never reached) for the first several hundred episodes, then shorter ones. Measured 2026-10-23 through the benchmark on seeds 1 to 5: does not solve (final 100-episode averages -130 to -146 against -110), greedy evaluation -157 ± 15; the random agent scores -200 (CHANGELOG.md, 2026-10-23). Seed 42 not measured. |

### Video Recording Checks

| Check # | Target Feature | Command to Execute | Expected Behavior |
| --- | --- | --- | --- |
| VID-1 | **Training clips** | `python train.py --config configs/cartpole_ppo.yaml --record-video --video-every 250` | Exactly `videos/cartpole/ppo/cartpole-training-episode-{250,500,750,1000}.mp4`; balances longer. Into a folder that did not exist, no Gymnasium "Overwriting existing videos" warning. |
| VID-2 | **Trained-agent video** | `python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_1000.pt` | One `videos/cartpole/ppo/cartpole-playback-full.mp4` with 5 episodes. |
| VID-3 | **Usage errors** | `python train.py --video-every 5` and `python record.py` | Exit with "--video-every requires --record-video" and "the following arguments are required: --checkpoint". |
| VID-4 | **Pendulum recording** | After `TR-6`: `python record.py --config configs/pendulum_ppo.yaml --checkpoint checkpoints/pendulum/ppo/agent_episode_1000.pt` | Generates `videos/pendulum/ppo/pendulum-playback-full.mp4` of 5 greedy episodes. Measured 2026-10-18 with seed 42, before evaluation moved to its own environment (not remeasured) (final training average -170): -208 ± 66 over the 5 episodes, so some fall just short of -200. |
| VID-5 | **DDPG recording** | After `TR-8`: `python record.py --config configs/pendulum_ddpg.yaml --checkpoint checkpoints/pendulum/ddpg/agent_episode_200.pt` | One `videos/pendulum/ddpg/pendulum-playback-full.mp4` of 5 greedy episodes, the pendulum swung up and held upright in most. Measured 2026-10-16 with seed 42, before evaluation moved to its own environment (not remeasured): -196 ± 64 over the 5 episodes, near the -200 threshold, so one episode may fall short. |
| VID-6 | **Acrobot recording** | After `TR-11`: `python record.py --config configs/acrobot_ppo.yaml --checkpoint checkpoints/acrobot/ppo/agent_episode_<n>.pt` (the highest `<n>` in that folder) | Generates `videos/acrobot/ppo/acrobot-playback-full.mp4` of greedy episodes in which the arm swings its tip above the line. Not measured: the learning runs went through the benchmark, which records no video. |
| VID-7 | **MountainCar recording** | After `TR-12`: `python record.py --config configs/mountaincar_dqn.yaml --checkpoint checkpoints/mountaincar/dqn/agent_episode_<n>.pt` (the highest `<n>` in that folder) | Generates `videos/mountaincar/dqn/mountaincar-playback-full.mp4` of greedy episodes in which the car rocks back and forth to build speed. Whether it reaches the flag varies by episode (greedy evaluation -157 ± 15 across seeds 1 to 5). Not measured with seed 42. |
| OUT-1 | **Scoped outputs** | Train `cartpole_default.yaml` and `pendulum_ppo.yaml` with `--record-video --plot` | `checkpoints/`, `videos/`, `plots/`, `logs/` contain partitioned `<challenge>/<agent>/` paths. |

### Multi-Agent Benchmark Checks

| Check # | Target Feature | Command to Execute | Expected Behavior |
| --- | --- | --- | --- |
| BM-1 | **Benchmark Runner** | `python src/benchmark.py --challenge cartpole --agents random ppo --episodes 5 --no-plot` | Runs multi-agent benchmark without plot, writes metrics JSON and summary report. |
| BM-2 | **Benchmark Plotting** | `python src/benchmark.py --challenge cartpole --agents random reinforce --episodes 5` | Generates `plots/benchmarks/cartpole/benchmark_comparison.png` and `docs/benchmarks/cartpole_results.md`. |
| BM-3 | **Continuous Benchmark** | `python src/benchmark.py --challenge pendulum --agents random ppo --episodes 5` | Benchmarks continuous agents, writes metrics and markdown report. |
| BM-4 | **Per-Agent Budgets and Solve Rule** | `python src/benchmark.py --challenge cartpole --agents random dqn --seeds 1 --no-plot --output-dir bm4 --report bm4/report.md` (about 70 seconds) | The console says each agent uses its own config. `bm4/report.md` lists each agent's budget with its unit (`50000 steps` for random, `300000 steps` for DQN), has a "How a Run Counts as Solved" section naming the 100-episode window, and shows Solve Rate, Avg Steps to Solve, Held and Eval ≥ Threshold as separate columns. A run shorter than 100 episodes can never solve. Delete `bm4/` afterwards. |
| BM-5 | **Step Budget** | `python src/benchmark.py --challenge cartpole --agents random --steps 2000 --seeds 1 --output-dir bm5 --report bm5/report.md` | Every run trains until at least 2000 environment steps (finishing the episode that crosses it); the report shows `2000 steps` as the budget and notes the overshoot; `bm5/benchmark_comparison.png` has an "Environment steps" x-axis. Delete `bm5/` afterwards. |
| BM-6 | **Acrobot Benchmark** | `python src/benchmark.py --challenge acrobot --agents random dqn ppo --seeds 1 --output-dir bm6 --report bm6/report.md --no-plot` (about 6 minutes) | Runs each agent on its own config budget (`50000 steps` random, `100000 steps` DQN, `500000 steps` PPO); the report names the -100 threshold, random scores about -500; on seed 1 DQN evaluated -94 and PPO -80 on 2026-10-22. Delete `bm6/` afterwards. |
| BM-7 | **MountainCar Benchmark** | `python src/benchmark.py --challenge mountaincar --agents random dqn --seeds 1 --output-dir bm7 --report bm7/report.md --no-plot` (about 3 minutes) | Runs each agent on its own config budget (`40000 steps` random, `120000 steps` DQN); the report names the -110 threshold, random scores -200 and DQN's final training average lands near -130 without solving. Delete `bm7/` afterwards. |

### Evaluation & Checkpoint Verification

| Check # | Target Feature | Steps to Execute | Expected Behavior |
| --- | --- | --- | --- |
| CKPT-1 | **Checkpoint Persistence** | Run training with `save_freq` set. Inspect `checkpoints/` directory. | Serialized `.npz` file contains valid weight arrays (`W1`, `b1`, `W2`, `b2`) without NaN values. |
| CKPT-2 | **Checkpoint Reloading** | Load `.npz` checkpoint into agent: `agent.load("checkpoints/model.npz")`. | Agent policy restores exact weights and reproduces deterministic actions for identical states. |
| CKPT-3 | **Resume** | Copy `configs/cartpole_ppo.yaml` twice into a scratch folder with `checkpoint_dir` and `log_dir` in two subfolders and `save_frequency: 50`. Run the first copy with `--num-episodes 200`; run the second with `--num-episodes 100`, then again with `--num-episodes 200 --resume <its checkpoint_dir>/cartpole/ppo/agent_episode_100.pt`. | The resumed run logs "Resumed from ... after 100 episodes" and leaves checkpoints 50 and 100 untouched. Its `agent_episode_200.pt` and `trainer_episode_200.npz` hold the same arrays as the uninterrupted run's (`np.array_equal` on every key), and with `--plot` both `training_metrics.png` files are byte-identical. Measured 2026-10-21 (seed 42): identical, also for `configs/pendulum_ddpg.yaml` resumed at episode 50 of 100. |
| LOG-1 | **Structured Metrics** | Train for at least 10 episodes (e.g. `TR-4`), then open `logs/<challenge>/<agent>/metrics_<timestamp>.jsonl`. | Strict JSONL. Every 10 episodes: `episode`, `avg_reward`, `avg_length`, and `update` with the mean of each metric the agent's `update()` returned (DQN: `loss`, `epsilon`, `mean_q`; none for `random`). Each evaluation adds an `evaluation` row. Until its buffer warms up, DQN reports only `epsilon`, so `loss` and `mean_q` average real updates only. |
| LOG-2 | **Non-Finite Metrics** | Copy `configs/cartpole_ppo.yaml` to `ppo_diverge.yaml`, set `learning_rate: 1000` in the copy, then `python train.py --config ppo_diverge.yaml --num-episodes 30`. | Within the first 20 episodes one `WARNING` per affected metric: `Non-finite update metric 'value_loss' = inf first appeared at episode <n>.` (the value may read `nan`). The next metrics row has `"value_loss": null` and `"non_finite": ["value_loss"]`. The trainer keeps going; the run ends with an error once the agent's own weights are NaN. |

---

## 3. Automation & CI Verification

| Check # | Target Feature | Steps to Execute | Expected Behavior |
| --- | --- | --- | --- |
| CI-1 | **Ruff Linter & Formatter** | `ruff check .` and `ruff format --check .` | Returns 0 errors, no unformatted files. |
| CI-2 | **Mypy Static Typing** | `mypy src` | Returns "Success: no issues found". |
| CI-3 | **Pytest Test Suite** | `pytest -v` | All unit tests pass with coverage summary. |
| CI-4 | **Bandit Security** | `bandit -r src/ -ll` | No high or medium severity security issues detected. |
| CI-5 | **Doc Sync Script** | `python scripts/sync_workflow_docs.py --check` | Confirms "README.md already up to date." (exit 0); CI's `Checks` job runs the same. |
| CI-6 | **CI Policy** | `pytest tests/test_ci_policy.py --no-cov` | All pass: every workflow is SHA-pinned, read-only, time-limited and cancels superseded PR runs. |
| CI-7 | **Docs-only Pull Request** | Open a PR that edits only a `.md` file; open the `Checks` and `Unit Tests` runs | Both report success; "Docs-only change: true" in the log; Python steps show as skipped; markdownlint ran. |
