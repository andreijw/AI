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

## Upcoming Milestones

### Training Tooling

- [x] **Training Video Interval & Trained-Agent Recording**
  - **Description**: Let `train.py --record-video` choose how often training episodes are recorded, and add
    `record.py` to record a trained agent from a checkpoint into a single MP4.
  - **Acceptance criteria**:
    - `--video-every N` (positive int) records every Nth **training** episode as its own clip. When omitted, the
      current behavior is kept (every `training.eval_frequency` episodes).
    - Evaluation episodes are never recorded during training, and clip numbers match training episode numbers
      (fixes the drift where `Trainer.evaluate()` advanced `RecordVideo`'s episode counter by 10 per evaluation).
    - `--video-every` without `--record-video` is a usage error; `--video-every` <= 0 is rejected.
    - `record.py --checkpoint PATH [--episodes N]` loads the agent from the config, plays N evaluation episodes
      (default 5) and writes them to one `cartpole-playback-full.mp4` in `--video-dir`.
    - `record.py` reuses `train.py`'s config, environment, agent and headless setup and `Trainer.evaluate()`;
      no setup code is duplicated.
    - Memory stays bounded to one episode's frames: clips are written per episode and concatenated on disk
      with ffmpeg stream copy.
    - `plots/` and `videos/` are added to `.gitignore`.
    - Tests in `tests/test_train_video.py`, `tests/test_record.py` and `tests/test_video.py`; README and
      TESTING.md updated.
  - **Known edge cases**: Missing checkpoint file; checkpoint saved by a different agent type than the config;
    `--episodes` <= 0; training interrupted (clips already written are kept).
  - **Out of scope**: Recording every training episode into one video (too slow: ~6 min instead of ~8 s for
    1000 random-agent episodes); a config-file key for the video interval; changes to `--render`.

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

- [ ] **`ReplayBuffer` Component**
  - **Description**: Fixed-capacity ring buffer storing transition tuples `(state, action, reward, next_state, done)`.
  - **Acceptance criteria**:
    - Pure NumPy ring buffer implementation supporting O(1) insertion and O(batch_size) uniform random sampling.
    - Support vectorized state/action ingestion.
    - Comprehensive unit tests in `tests/test_replay_buffer.py`.
  - **Known edge cases**: Buffer sampling before reaching minimum sample size; terminal state transitions.

- [ ] **`DqnAgent` Implementation**
  - **Description**: Pure NumPy DQN agent with separate online and target Q-networks, epsilon-greedy exploration,
    and Bellman TD error optimization.
  - **Acceptance criteria**:
    - Online Q-network and target Q-network parameterized with pure NumPy matrices and backward pass autodiff.
    - Decaying epsilon-greedy policy (`epsilon_start`, `epsilon_end`, `epsilon_decay`).
    - Target network weight synchronization via periodic hard copy or Polyak soft update.
    - Unit tests in `tests/test_dqn_agent.py` achieving >85% coverage.
  - **Known edge cases**: Bellman target clipping, NaN/Inf gradient explosion, terminal state value zeroing.

- [ ] **DQN Hyperparameter Configuration & Integration**
  - **Description**: Add `configs/cartpole_dqn.yaml`, register `dqn` in `train.py`, and run benchmark comparisons.
  - **Acceptance criteria**:
    - `configs/cartpole_dqn.yaml` with calibrated learning rate, replay capacity, batch size, and discount factor.
    - `train.py --config configs/cartpole_dqn.yaml` trains successfully to solve CartPole-v1 (avg reward >= 475).

---

### Phase 5 — Advanced RL & Robotics Extensions

- [ ] **Observation Preprocessing & Frame Stacking**: Support pixel-based Gym environments with frame concatenation.
- [ ] **Continuous Action Space Support**: Extend policy gradient agents with Gaussian policies (mean & log-std).
- [ ] **Curriculum Learning**: Staged domain randomization difficulty scheduling across training epochs.
- [ ] **Lightweight Embedded Inference Export**: Export trained policy weights to minimal ONNX/C-header formats.
