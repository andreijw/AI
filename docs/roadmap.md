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

### Workflow Tooling

- [~] **Dev Team Skills for Claude Code**
  - **Description**: Port the dev-team skills from the sibling `eternal-descent` repo to this Python and RL
    codebase (`dev-loop`, `architect`, `test-crafter`, `cleanup`, `pr-reviewer`, `doc-sync`), add the RL-specific
    skills `algorithm-crafter`, `experiment-runner` and `training-doctor`, and add the tooling they rely on.
  - **Acceptance criteria**:
    - Nine skills under `.claude/skills/`, each naming this repository's real packages, helpers, commands and gates.
    - `.claude/settings.json` pins the brooks-lint plugin and pre-approves the quality-gate commands;
      `.brooks-lint.yaml` holds project review rules C1 to C6.
    - `tests/helpers.py` provides `numerical_grad` and `assert_gradients_match`, covered by `tests/test_helpers.py`.
    - `mutation_check.py` and `check_md_width.py` ship with the skills that use them and pass Ruff.
    - CLAUDE.md, CONTRIBUTING.md §4, GEMINI.md §5 and the README describe the workflow and its setup.
  - **Known edge cases**: Machines without Node.js cannot run `markdownlint`; the skills fall back to the
    line-length script. Without the brooks-lint plugin, the skills run with the project checks only.
  - **Out of scope**: A `release` skill (there is no `main` branch, tag or changelog to release from); adding
    gradient checks to the existing agents; the gaps found while building this (the four items below).

- [ ] **Training Diagnostics in Logs**
  - **Description**: `Trainer` discards the dictionary `agent.update()` returns, so losses, entropy and epsilon never
    reach the metrics log and a failing run can only be diagnosed from a scratch script.
  - **Acceptance criteria**:
    - Per-episode update metrics are aggregated and written with the periodic reward metrics.
    - A non-finite metric is reported loudly, with the episode it first appeared in.
    - TESTING.md `LOG-1` matches what is written.
  - **Out of scope**: TensorBoard or other external logging backends.

- [ ] **Discounted-Return Tests That Can Fail**
  - **Description**: The `*_compute_returns_discounting` tests in `tests/test_ppo_agent.py`,
    `tests/test_actor_critic_agent.py` and `tests/test_reinforce_agent.py` assert only that returns decrease over five
    rewards of 1. They still pass with the discount removed, because 5, 4, 3, 2, 1 also decreases; only
    `test_compute_returns_exact_values` in `tests/test_agents_extensive.py` notices.
  - **Acceptance criteria**:
    - Each of the three tests asserts values worked out by hand, or is removed in favour of the exact-value tests,
      with PPO covered by one.
    - Replacing `self.gamma * cumulative` with `1.0 * cumulative` fails a named test in each agent's own test file
      (checked with `mutation_check.py`).
  - **Out of scope**: Changing `_compute_returns` itself.

- [ ] **Softmax Stability Tests**
  - **Description**: Removing the `logits - np.max(logits)` line from the softmax in
    `rl_core/agents/actor_critic_base.py` or `rl_core/agents/reinforce_agent.py` fails no test in the suite, so the
    guard against overflow required by CONTRIBUTING.md §1 is not locked in.
  - **Acceptance criteria**:
    - Tests drive each forward pass with weights large enough to produce logits near ±1000 and assert that the
      probabilities are finite and sum to one.
    - Removing the stabilisation line in either file fails a named test (checked with `mutation_check.py`).
  - **Known edge cases**: All logits equal; one logit far above the rest.
  - **Out of scope**: Gradient checks for the existing agents.

- [ ] **Single `_compute_returns` Implementation**
  - **Description**: `ReinforceAgent._compute_returns` and `ActorCriticBase._compute_returns` are the same
    discounted-return loop written twice, so a fix to one can miss the other.
  - **Acceptance criteria**:
    - One implementation in `src/rl_core/agents/`, used by REINFORCE, Actor-Critic and PPO.
    - Seeded runs give bit-identical weights before and after the change.
    - The duplicate-code audit reports nothing new.
  - **Known edge cases**: Empty reward array; `gamma` of 0 and of 1.
  - **Out of scope**: Sharing the softmax or the network code between REINFORCE and the actor-critic family.

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

- [x] **Continuous Action Space Support & Core Decoupling (Gaussian Policies & Pendulum-v1)**
  - **Description**: Decoupled algorithm-agnostic RL framework into `src/rl_core/` and extended policy gradient
    agents (`ContinuousActorCriticAgent`, `ContinuousPPOAgent`, `ContinuousRandomAgent`) to support continuous control
    benchmarks, introducing the `src/rl_pendulum/` challenge package.
  - **Acceptance criteria**:
    - Gaussian policy heads predicting state-dependent action mean $\mu(s)$ and trainable log standard deviation
      $\log \sigma$.
    - Reparameterized action sampling ($a = \mu + \sigma \odot \epsilon$) and Gaussian log-likelihood backprop.
    - Support for `gymnasium.spaces.Box` action spaces in `PendulumEnv` wrapper and vectorization factories.
    - Calibrated configuration `configs/pendulum_ppo.yaml` solving Pendulum-v1.
    - Unit tests in `tests/test_continuous_agent.py` and `tests/test_pendulum.py` achieving >95% coverage.
  - **Known edge cases**: Action squashing/tanh bounding; standard deviation vanishing or numerical overflow.
  - **Out of scope**: Off-policy continuous algorithms (DDPG/SAC); multi-discrete action spaces.

- [ ] **Continuous Off-Policy Learning & DDPG (Deep Deterministic Policy Gradient)**
  - **Description**: Implement a pure NumPy `DDPGAgent` utilizing a deterministic actor network $\mu_\theta(s)$,
    an action-value critic $Q_\phi(s, a)$, continuous transition experience replay via `ReplayBuffer`, exploration
    noise (Gaussian or Ornstein-Uhlenbeck), and Polyak soft target updates to solve `Pendulum-v1`
    sample-efficiently.
  - **Acceptance criteria**:
    - Pure NumPy deterministic actor and Q-value critic network architectures.
    - Continuous transition ingestion and random mini-batch sampling from `ReplayBuffer`.
    - Critic TD loss optimization and deterministic policy gradient backpropagation
      ($\nabla_\theta Q(s, \mu_\theta(s))$).
    - Polyak soft target weight synchronization ($\theta' \leftarrow \tau \theta + (1-\tau) \theta'$ with
      $\tau \approx 0.005$).
    - Reference configuration `configs/pendulum_ddpg.yaml` solving `Pendulum-v1` (mean reward $\ge -200$)
      in $\le 100$ episodes.
    - CLI dispatch support in `train.py`, `record.py`, and `src/benchmark.py`.
    - Unit and integration tests in `tests/test_ddpg_agent.py` achieving $>90\%$ coverage.
  - **Known edge cases**: Q-value overestimation; divergence from non-smooth actor gradients; torque saturation.
  - **Out of scope**: Twin Delayed DDPG (TD3) double Q-learning; stochastic entropy auto-tuning (SAC).

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
