# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress · `[x]` Complete

---

## Completed Releases & Milestones

### Phase 0 — Framework Foundation & CI/CD Pipeline *(Completed)*

- [x] **Repository Infrastructure & Tooling**: Configured `pyproject.toml` with Ruff, Mypy, and Pytest.
- [x] **Automated CI Workflows**: Linting (`lint.yml`), testing with coverage (`test.yml`), type-checking
  (`type-check.yml`), CodeQL security scanning (`codeql.yml`), and security auditing (`security.yml`).
- [x] **Nightly Maintenance Pipeline**: Scheduled daily automated scans and doc synchronization
  (`nightly-maintenance.yml`, `scripts/sync_workflow_docs.py`).

### Phase 1 — Gymnasium Environment Wrappers & Randomization *(Completed)*

- [x] **CartPole Gymnasium Wrapper**: Standardized `CartPoleEnv` wrapper adhering to Gymnasium v0.29+
  APIs (`terminated`, `truncated`, episode tracking, stats collection).
- [x] **Noise Injections**: Gaussian observation noise (`obs_noise_std`) and stochastic action flip
  probabilities (`action_noise_prob`).
- [x] **Domain Randomization**: Configurable physical parameters (gravity, cart mass, pole mass, pole length)
  with dynamic per-episode resampling and automatic derived parameter calculation.
- [x] **Vectorized Environments**: Synchronous and asynchronous multi-environment rollout factories
  (`make_vec_env`) with independent seeds.

### Phase 2 — Policy Gradient & Baseline Agents *(Completed)*

- [x] **Abstract Agent Architecture**: `BaseAgent` abstraction with standardized `select_action`, `update`,
  `save`, and `load` interfaces.
- [x] **Baseline Random Policy**: `RandomAgent` for environment sanity checks and performance baselining.
- [x] **REINFORCE Agent**: Monte Carlo policy gradient algorithm with discounted return returns and baseline
  subtraction.
- [x] **Actor-Critic (A2C) Agent**: Actor policy with value critic estimating state-value advantages.
- [x] **Proximal Policy Optimization (PPO)**: Clipped surrogate objective policy optimization with value
  clipping, entropy bonus, and pure NumPy feedforward neural network implementations.

### Phase 3 — Training Orchestration & Metrics Pipeline *(Completed)*

- [x] **Unified Trainer**: Trajectory collection, rollout batching, periodic evaluation, and model
  checkpoint persistence (`src/rl_cartpole/training/trainer.py`).
- [x] **Configuration Management**: Modular YAML/JSON hyperparameter parsing and merging
  (`src/rl_cartpole/utils/config.py`).
- [x] **Structured Logging**: Dual-stream console and machine-readable JSONL training metrics logging
  (`src/rl_cartpole/utils/logger.py`).
- [x] **Visualization Helpers**: Episode reward and length trajectory plotting (`visualization.py`).

---

### Phase 4 — Value-Based Learning: Deep Q-Network (DQN) *(Completed)*

- [x] **`ReplayBuffer` Component**: Fixed-capacity pure NumPy ring buffer storing transition tuples with
  O(1) insertion, vectorized batch addition, and uniform random sampling (`src/rl_cartpole/agents/replay_buffer.py`).
- [x] **`DQNAgent` Implementation**: Pure NumPy DQN agent with separate online and target Q-networks, epsilon-greedy
  decay, Bellman TD loss backpropagation, and hard/soft target synchronization (`src/rl_cartpole/agents/dqn_agent.py`).
- [x] **DQN Hyperparameter Tuning & Integration**: Calibrated `configs/cartpole_dqn.yaml`, CLI integration in
  `train.py` / `record.py`, multi-step updates, and verification solving CartPole-v1.

### Training & Workflow Tooling *(Completed)*

- [x] **Training Video Interval & Trained-Agent Recording**: `--video-every N` training clip cadence, `record.py`
  playback generator with clip concatenation, and test coverage.
- [x] **Per-Agent Output Directories**
  - **Description**: Training outputs from different agents currently share `checkpoints/`, `videos/` and `plots/`
    with identical file names, so training one agent overwrites another's checkpoints and videos. Default each
    output location to a per-agent subfolder named after the agent type.
  - **Acceptance criteria**:
    - `train.py` saves checkpoints to `<training.checkpoint_dir>/<agent_type>/` (e.g. `checkpoints/ppo/`).
    - `train.py --video-dir` and `--plot-dir` default to `videos/<agent_type>/` and `plots/<agent_type>/`;
      `record.py --video-dir` defaults to `videos/<agent_type>/`.
    - The agent type is the effective one, so `--agent-type` overrides choose the folder too.
    - An explicit `--video-dir` / `--plot-dir` is used exactly as given (no agent subfolder appended).
    - Logs are unchanged (already timestamped per run in `logs/`).
    - One shared helper builds the per-agent path for both scripts; no duplicated logic.
    - README and TESTING.md updated, including the wrong `agent_episode_1500.pt` example in the README.
  - **Known edge cases**: `training.checkpoint_dir` missing from config (falls back to `./checkpoints/<agent_type>`);
    re-training the same agent overwrites that agent's previous outputs (intended; copy the folder to keep a run).
  - **Out of scope**: Per-run (timestamped) output folders; `record.py` picking the latest checkpoint
    automatically; migrating existing files in the old flat folders.

### Phase 4 — Value-Based Learning: Deep Q-Network (DQN)

Targeting value-learning baseline for direct comparison against policy-gradient methods (A2C/PPO).

- [x] **`ReplayBuffer` Component**
  - **Description**: Fixed-capacity ring buffer storing transition tuples `(state, action, reward, next_state, done)`.
  - **Acceptance criteria**:
    - Pure NumPy ring buffer implementation supporting O(1) insertion and O(batch_size) uniform random sampling.
    - Support vectorized state/action ingestion.
    - Comprehensive unit tests in `tests/test_replay_buffer.py`.
  - **Known edge cases**: Buffer sampling before reaching minimum sample size; terminal state transitions.

- [x] **`DqnAgent` Implementation**
  - **Description**: Pure NumPy DQN agent with separate online and target Q-networks, epsilon-greedy exploration,
    and Bellman TD error optimization.
  - **Acceptance criteria**:
    - Online Q-network and target Q-network parameterized with pure NumPy matrices and backward pass autodiff.
    - Decaying epsilon-greedy policy (`epsilon_start`, `epsilon_end`, `epsilon_decay`).
    - Target network weight synchronization via periodic hard copy or Polyak soft update.
    - Unit tests in `tests/test_dqn_agent.py` achieving >85% coverage.
  - **Known edge cases**: Bellman target clipping, NaN/Inf gradient explosion, terminal state value zeroing.

- [x] **DQN Hyperparameter Configuration & Integration**
  - **Description**: Add `configs/cartpole_dqn.yaml`, register `dqn` in `train.py`, and run benchmark comparisons.
  - **Acceptance criteria**:
    - `configs/cartpole_dqn.yaml` with calibrated learning rate, replay capacity, batch size, and discount factor.
    - `train.py --config configs/cartpole_dqn.yaml` trains successfully to solve CartPole-v1 (avg reward >= 475).

---

## Upcoming Milestones

### Phase 5 — Multi-Agent Benchmarking & Comparative Evaluation

- [x] **Cross-Agent Benchmark Suite & Comparative Visualizer**
  - **Description**: Add an automated benchmark framework (`src/benchmark.py`) with environment challenge adapters
    (`src/rl_cartpole/benchmark.py`) to train, evaluate, and compare all supported agents under standardized seeds,
    measuring sample efficiency, convergence speed, evaluation stability, and wall-clock execution time.
  - **Acceptance criteria**:
    - `src/benchmark.py` running multi-agent rollouts with reproducible seed sweeps across all algorithms.
    - Generates unified comparison plots (`plots/benchmarks/<challenge>/benchmark_comparison.png`) showing mean reward
      trajectories with shaded variance bands across runs.
    - Generates markdown report (`docs/benchmarks/<challenge>_results.md`) with summary metrics table (episodes to
      solve, final evaluation reward, sample efficiency).
    - CLI arguments: `--challenge`, `--agents`, `--episodes`, `--seeds`, `--output-dir`, `--report`, and `--no-plot`.
    - Unit and integration tests in `tests/test_benchmark.py` achieving >90% coverage.
  - **Known edge cases**: Interrupted benchmark runs; mismatched agent configuration files; varying episode lengths
    affecting rollout timing.
  - **Out of scope**: Distributed multi-machine training; automated Bayesian hyperparameter optimization sweeps.

---

### Phase 6 — Advanced RL & Robotics Extensions

- [ ] **Continuous Action Space Support (Gaussian Policies)**
  - **Description**: Extend policy gradient agents (`ActorCriticAgent`, `PPOAgent`) and environments to support
    continuous control benchmarks (e.g., `Pendulum-v1`, `InvertedPendulum-v4`).
  - **Acceptance criteria**:
    - Gaussian policy heads predicting state-dependent action mean $\mu(s)$ and trainable log standard deviation
      $\log \sigma$.
    - Reparameterized action sampling ($a = \mu + \sigma \odot \epsilon$) and Gaussian log-likelihood backprop.
    - Support for `gymnasium.spaces.Box` action spaces in `CartPoleEnv` / environment factory wrappers.
    - Calibrated configuration `configs/pendulum_ppo.yaml` solving Pendulum-v1.
    - Unit tests in `tests/test_continuous_agent.py` achieving >90% coverage.
  - **Known edge cases**: Action squashing/tanh bounding; standard deviation vanishing or numerical overflow.
  - **Out of scope**: Off-policy continuous algorithms (DDPG/SAC); multi-discrete action spaces.

- [ ] **Curriculum Learning & Domain Randomization Scheduling**
  - **Description**: Add dynamic curriculum pacing that scales domain randomization and noise injection difficulty
    as the agent reaches performance milestones during training.
  - **Acceptance criteria**:
    - `CurriculumScheduler` with linear, step, and adaptive reward-threshold difficulty scheduling.
    - Dynamic updates to `DomainRandomizationWrapper` and noise wrapper parameters during rollouts.
    - Robustness evaluation benchmark assessing zero-shot transfer under out-of-distribution physical perturbations.
    - Configurable curriculum specification in environment YAML files.
    - Comprehensive unit tests in `tests/test_curriculum.py`.
  - **Known edge cases**: Catastrophic forgetting across difficulty stages; oscillation near threshold boundaries.
  - **Out of scope**: Adversarial environment generation; automatic domain randomization (ADR) meta-policies.

- [ ] **Observation Preprocessing & Frame Stacking**
  - **Description**: Support temporal history and partial observability via frame stacking and custom feature
    transformations.
  - **Acceptance criteria**:
    - `FrameStackWrapper` concatenating $K$ sequential observations into a flattened feature vector.
    - Velocity masking option simulating partially observable Markov decision processes (POMDP).
    - Full compatibility with vectorized environment execution (`make_vec_env`).
    - Unit tests in `tests/test_frame_stack.py`.
  - **Known edge cases**: Padding on episode resets; memory footprint for long history buffers.
  - **Out of scope**: Convolutional neural networks for raw RGB pixels; recurrent neural network policies (LSTM/GRU).

- [ ] **Lightweight Embedded Inference Export (C Header & ONNX)**
  - **Description**: Export trained pure NumPy model parameters into zero-dependency standalone C header files
    (`policy_weights.h`) and standard ONNX formats for edge microcontroller and robotics deployment.
  - **Acceptance criteria**:
    - CLI export script `scripts/export_c_header.py` converting checkpoints into self-contained C forward pass code.
    - Standalone C verification harness compiled via GCC/Clang and checked against NumPy numerical output (tolerance
      $10^{-5}$).
    - Export utility for standard ONNX format compatible with visualizers (Netron).
    - Unit tests in `tests/test_export.py`.
  - **Known edge cases**: Floating-point precision discrepancies (FP32 vs FP64); activation clipping overflows.
  - **Out of scope**: Quantization-aware training (INT8); target-specific MCU hardware optimizations (CMSIS-NN).
