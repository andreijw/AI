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
| TR-1 | **REINFORCE** | `python train.py --config configs/cartpole_reinforce.yaml` | Completes 2000 episodes with Adam and saves checkpoints. Measured 2026-10-14 through the benchmark on seeds 1 to 5: the 10-episode training average reaches 475 on 5 of 5 (episodes 645 to 1012), evaluation 449 ± 59 (lowest seed 335). It does not always hold it: one seed ends with a training average of 173. Seed 42, which `train.py` uses, was not measured. |
| TR-2 | **Actor-Critic** | `python train.py --config configs/cartpole_actor_critic.yaml` | Completes 1500 episodes with Adam. Measured 2026-10-15 on seeds 1 to 5: reaches 475 on 5 of 5 (episodes 490 to 839), evaluation 476 ± 32. Seed 42 not measured. |
| TR-3 | **PPO** | `python train.py --config configs/cartpole_ppo.yaml` | Completes 1000 episodes of standard PPO (256-step rollouts, 20 epochs, learning rate and clip annealed over 200,000 transitions); `clip_fraction`, `rollout_steps` and `anneal_fraction` are logged on the episodes that train. Measured 2026-10-15 on seeds 1 to 5: reaches 475 on 5 of 5 (episodes 323 to 427), evaluation 500 on every seed. Seed 42 not measured. |
| TR-4 | **DQN** | `python train.py --config configs/cartpole_dqn.yaml` | Completes 750 episodes; `epsilon` falls to about 0.023. Measured 2026-10-14 on seeds 1 to 5: reaches 475 on 4 of 5 (episodes 378 to 576; seed 3 ends at 404), evaluation 450 ± 46. Seed 42 not measured. |
| TR-5 | **Continuous A2C** | `python train.py --config configs/pendulum_actor_critic.yaml` | Continuous Gaussian policy updates on Pendulum-v1; critic loss converges. |
| TR-6 | **Continuous PPO** | `python train.py --config configs/pendulum_ppo.yaml` | Completes training with finite `policy_loss`, `value_loss` and `entropy` in the metrics log. It does not yet learn: as of 2026-10-13 it scores about -1600 against a solve threshold of -200 (roadmap: "Pendulum PPO Learns and Solves Pendulum-v1"). |

### Video Recording Checks

| Check # | Target Feature | Command to Execute | Expected Behavior |
| --- | --- | --- | --- |
| VID-1 | **Training clips** | `python train.py --config configs/cartpole_ppo.yaml --record-video --video-every 250` | Exactly `videos/cartpole/ppo/cartpole-training-episode-{250,500,750,1000}.mp4`; balances longer. |
| VID-2 | **Trained-agent video** | `python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_1000.pt` | One `videos/cartpole/ppo/cartpole-playback-full.mp4` with 5 episodes. |
| VID-3 | **Usage errors** | `python train.py --video-every 5` and `python record.py` | Exit with "--video-every requires --record-video" and "the following arguments are required: --checkpoint". |
| VID-4 | **Pendulum recording** | `python record.py --config configs/pendulum_ppo.yaml --checkpoint checkpoints/pendulum/ppo/agent_episode_200.pt` | Generates `videos/pendulum/ppo/pendulum-playback-full.mp4`. |
| OUT-1 | **Scoped outputs** | Train `cartpole_default.yaml` and `pendulum_ppo.yaml` with `--record-video --plot` | `checkpoints/`, `videos/`, `plots/`, `logs/` contain partitioned `<challenge>/<agent>/` paths. |

### Multi-Agent Benchmark Checks

| Check # | Target Feature | Command to Execute | Expected Behavior |
| --- | --- | --- | --- |
| BM-1 | **Benchmark Runner** | `python src/benchmark.py --challenge cartpole --agents random ppo --episodes 5 --no-plot` | Runs multi-agent benchmark without plot, writes metrics JSON and summary report. |
| BM-2 | **Benchmark Plotting** | `python src/benchmark.py --challenge cartpole --agents random reinforce --episodes 5` | Generates `plots/benchmarks/cartpole/benchmark_comparison.png` and `docs/benchmarks/cartpole_results.md`. |
| BM-3 | **Continuous Benchmark** | `python src/benchmark.py --challenge pendulum --agents random ppo --episodes 5` | Benchmarks continuous agents, writes metrics and markdown report. |
| BM-4 | **Per-Agent Budgets and Solve Rule** | `python src/benchmark.py --challenge cartpole --agents random dqn --seeds 1 --no-plot --output-dir bm4 --report bm4/report.md` (about a minute) | The console says each agent uses its own config. `bm4/report.md` lists 1000 episodes for random and 500 for DQN, has a "How a Run Counts as Solved" section, and shows Solve Rate and Eval ≥ Threshold as separate columns. Delete `bm4/` afterwards. |

### Evaluation & Checkpoint Verification

| Check # | Target Feature | Steps to Execute | Expected Behavior |
| --- | --- | --- | --- |
| CKPT-1 | **Checkpoint Persistence** | Run training with `save_freq` set. Inspect `checkpoints/` directory. | Serialized `.npz` file contains valid weight arrays (`W1`, `b1`, `W2`, `b2`) without NaN values. |
| CKPT-2 | **Checkpoint Reloading** | Load `.npz` checkpoint into agent: `agent.load("checkpoints/model.npz")`. | Agent policy restores exact weights and reproduces deterministic actions for identical states. |
| LOG-1 | **Structured Metrics** | Train for at least 10 episodes (e.g. `TR-4`), then open `logs/<challenge>/<agent>/metrics_<timestamp>.jsonl`. | Strict JSONL. Every 10 episodes: `episode`, `avg_reward`, `avg_length`, and `update` with the mean of each metric the agent's `update()` returned (DQN: `loss`, `epsilon`, `mean_q`; none for `random`). Each evaluation adds an `evaluation` row. DQN reports 0 for `loss` and `mean_q` until its buffer warms up, which lowers the first mean. |
| LOG-2 | **Non-Finite Metrics** | Copy `configs/cartpole_ppo.yaml` to `ppo_diverge.yaml`, set `learning_rate: 1000` in the copy, then `python train.py --config ppo_diverge.yaml --num-episodes 30`. | Within the first 20 episodes one `WARNING` per affected metric: `Non-finite update metric 'value_loss' = inf first appeared at episode <n>.` (the value may read `nan`). The next metrics row has `"value_loss": null` and `"non_finite": ["value_loss"]`. The trainer keeps going; the run ends with an error once the agent's own weights are NaN. |

---

## 3. Automation & CI Verification

| Check # | Target Feature | Steps to Execute | Expected Behavior |
| --- | --- | --- | --- |
| CI-1 | **Ruff Linter & Formatter** | `ruff check .` and `ruff format --check .` | Returns 0 errors, no unformatted files. |
| CI-2 | **Mypy Static Typing** | `mypy src` | Returns "Success: no issues found". |
| CI-3 | **Pytest Test Suite** | `pytest -v` | All unit tests pass with coverage summary. |
| CI-4 | **Bandit Security** | `bandit -r src/ -ll` | No high or medium severity security issues detected. |
| CI-5 | **Doc Sync Script** | `python scripts/sync_workflow_docs.py` | Confirms "README.md already up to date." |
