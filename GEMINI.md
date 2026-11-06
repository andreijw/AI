# Instruction Manual for Antigravity CLI (GEMINI.md)

Welcome, AI coding assistant! To operate efficiently in this codebase, minimize token costs, and maintain code quality,
you **must** adhere strictly to the guidelines and standards defined in this document and in
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## 1. Token & Quota Optimization Guidelines

To optimize token usage and context window limits:

* **Targeted File Reads**: Do not view entire files if you only need a specific section. Use `view_file` with precise
  `StartLine` and `EndLine` parameters to read only the lines of interest.
* **Precise Edits**: Never replace entire files. Always use `replace_file_content` for single contiguous edits and
  `multi_replace_file_content` for non-contiguous changes.
* **Narrow Replacement Scopes**: Make your target content blocks in replacement chunks as concise as possible while
  remaining unique.
* **No Redundant Directory Listing**: Avoid recursive or full-directory listing unless looking for a missing file. Use
  `grep_search` to target filenames or symbol occurrences directly.
* **Model Choices**:
  * For writing unit tests, running builds, debugging compile errors, or simple refactoring tasks, utilize cheaper
    models (e.g., `gemini-3.1-flash-lite` or similar).
  * Always do a planning phase first to think of the overall architecture and design before starting to implement code.
  * Only escalate to larger, more expensive reasoning models (e.g., `gemini-3.5-flash` / `gemini-3.5-pro` equivalents)
    for complex architectural refactoring or prompt engineering tasks.

---

## 2. Codebase Directory Layout

* **`src/rl_core/`**: Algorithm-agnostic reinforcement learning framework.
  * `src/rl_core/agents/`: Base agent interfaces (`BaseAgent`, `ActorCriticBase`, `ContinuousActorCriticBase`),
    discrete agents (`RandomAgent`, `ReinforceAgent`, `ActorCriticAgent`, `PPOAgent`, `DQNAgent`),
    continuous agents (`ContinuousRandomAgent`, `ContinuousActorCriticAgent`, `ContinuousPPOAgent`, `DDPGAgent`,
    `TD3Agent`, `SACAgent`) and their replay-buffer base `OffPolicyAgent` (`off_policy.py`), the optional
    PyTorch backends of the last three (`torch_ddpg_agent.py`, `torch_td3_agent.py`, `torch_sac_agent.py`, on
    `TorchOffPolicyAgent` in `torch_off_policy.py`),
    `ReplayBuffer`, and the shared `mlp.py` (multi-layer networks), `bounds.py` (action bounds), `softmax.py`
    (the discrete policies' numerically stable softmax), `exploration_noise.py` (DDPG's Gaussian and
    Ornstein-Uhlenbeck noise), `observation_normalizer.py` (running observation statistics and `standardize`)
    and `inference.py` (`InferencePolicy`, the frozen greedy network that `BaseAgent.inference_policy()` returns
    for export).
  * `src/rl_core/export/`: `export.py`'s back end: the C header writer (`c_header.py`), the ONNX model writer and
    its onnxruntime check (`onnx_model.py`, the only module importing onnx), the comparison with NumPy and the C
    compile-and-compare check (`verify.py`), and the greedy rollout they are checked on (`rollout.py`).
  * `src/rl_core/environments/`: `BaseEnvWrapper`, the base of every challenge's environment wrapper (seeding,
    observation noise, episode statistics, `wrap_env`, and the checks of the shared noise settings `obs_noise_std`,
    `action_noise_std` and `action_noise_prob`), `checked_range` for one domain randomization range, and
    `checked_randomization` with `ParameterBounds` and `POSITIVE`, which checks a whole randomization mapping against
    a table of each parameter's bounds.
  * `src/rl_core/training/`: Training pipeline orchestration (`trainer.py`), the checkpoint's trainer file
    (`checkpoints.py`), curriculum scheduling of the training environment's noise and randomization
    (`curriculum.py`) and the robustness benchmark's perturbations and greedy rows (`robustness.py`).
  * `src/rl_core/utils/`: Configuration parsing (`config.py`) and the strict readers for integer and boolean
    settings that agents, trainer and environments share (`config_values.py`), structured logging (`logger.py`),
    video recording (`video.py`), visualization helpers (`visualization.py`, imported directly: the package does
    not re-export them, and `train.py` and `src/benchmark.py` import them only inside the code that plots, so a run
    without plots never loads matplotlib), and the rolling average behind solve numbers and plots (`smoothing.py`).
* **`src/rl_cartpole/`**: CartPole-v1 discrete balance control challenge.
  * `src/rl_cartpole/environments/`: Gymnasium wrapper implementations, observation/action noise injections,
    domain randomization configurations, and vectorized environment factories.
  * `src/rl_cartpole/benchmark.py`: CartPole benchmark challenge implementation.
* **`src/rl_pendulum/`**: Pendulum-v1 continuous torque control challenge.
  * `src/rl_pendulum/environments/`: Pendulum wrappers, observation/action noise, domain
    randomization, and vectorization.
  * `src/rl_pendulum/benchmark.py`: Pendulum benchmark challenge implementation.
* **`src/rl_acrobot/`**: Acrobot-v1 discrete swing-up challenge.
  * `src/rl_acrobot/environments/`: Acrobot wrapper (observation noise only) and its factories.
  * `src/rl_acrobot/benchmark.py`: Acrobot benchmark challenge implementation.
* **`src/rl_mountaincar/`**: MountainCar-v0 (discrete) and MountainCarContinuous-v0 exploration challenges.
  * `src/rl_mountaincar/environments/`: mountain car wrapper (observation noise only), shared by both ids, and a
    factory for each.
  * `src/rl_mountaincar/benchmark.py`: both MountainCar benchmark challenge implementations.
* **`src/rl_lunarlander/`**: LunarLander-v3 (discrete) and LunarLanderContinuous-v3 challenges, which need Box2D
  (the `box2d` extra, built by `./setup.sh`; README, LunarLander).
  * `src/rl_lunarlander/environments/`: lunar lander wrapper (observation noise only), shared by both ids, and a
    factory for each.
  * `src/rl_lunarlander/benchmark.py`: both LunarLander benchmark challenge implementations.
* **`src/rl_bipedalwalker/`**: BipedalWalker-v3 four-joint walking challenge (random, SAC and TD3), which needs
  Box2D as LunarLander does.
  * `src/rl_bipedalwalker/environments/`: bipedal walker wrapper (observation noise only) and its factory.
  * `src/rl_bipedalwalker/benchmark.py`: BipedalWalker benchmark challenge implementation.
* **`src/rl_navigation/`**: Navigation-v0 (discrete) and NavigationContinuous-v0 range-sensor navigation challenges
  (Phase 8); `__init__.py` imports nothing, so each layer below can be imported without the ones above it.
  * `src/rl_navigation/sim/`: the simulator, NumPy and the standard library only (never Gymnasium or `rl_core`):
    `geometry.py` (arena, batched ray casting, swept-disc collision, occupancy grid), `kinematics.py` (unicycle),
    `sensors.py` (range sensors, `hc_sr04` and `lidar` presets), `odometry.py` (`WheelOdometry`: noisy left and
    right wheel travel), `layouts.py` (random layouts with a reachable goal), `world.py` (`World`: pose, motion
    with wheel slip, collision, goal, sensors) and `render.py` (the arena as an RGB array).
  * `src/rl_navigation/estimation/`: pose estimation on a known map for the robot's own program, NumPy and
    `rl_navigation.sim` only (never Gymnasium, `gym_env` or `rl_core`): `maps.py` (the `RangeMap` protocol and
    `SensorMap`), `kalman.py` (the Kalman update in Joseph form), `ekf.py` (`EkfLocalizer`, iterated EKF
    localization) and `evaluation.py` (`run_trial` and `summarize`: seeded trials against the true pose). No
    environment or benchmark uses it.
  * `src/rl_navigation/gym_env.py`: `NavigationGymEnv`, the Gymnasium environment; importing it registers both ids.
    It imports Gymnasium and `rl_navigation.sim` only, never `rl_core`, so it works outside this framework.
  * `src/rl_navigation/environments/`: `NavigationEnv`, the framework wrapper shared by both ids (action noise,
    domain randomization of wheel slip, sensor noise, dropout and obstacle count), and a factory for each id.
  * `src/rl_navigation/benchmark.py`: both navigation benchmark challenge implementations (random, PPO and DQN
    on `Navigation-v0`; random, PPO and TD3 on `NavigationContinuous-v0`).
* **`src/rl_core/challenges/`**: The `BenchmarkChallenge` interface, the challenge registry and `discover_challenges`.
* **`src/benchmark.py`**: The benchmark runner and its command line, for every discovered challenge.
* **`tests/`** (located at [tests/](tests/)): Pytest unit and integration test suite
  targeting environments, agents, logging, configurations, and visualization.
* **`configs/`** (located at [configs/](configs/)): YAML files specifying hyperparameter
  configs for training default, REINFORCE, Actor-Critic, and PPO algorithms.
* **`scripts/`** (located at [scripts/](scripts/)): Automation scripts
  (e.g., `sync_workflow_docs.py`), and `pytorch_timing/`, the DDPG timing scripts behind
  `docs/pytorch_jetson_timing.md` (`time_ddpg.py`; its `torch` mode needs a separate PyTorch environment).
* **`docs/`** (located at [docs/](docs/)): Feature guides, setup documents, summaries,
  and roadmaps (`roadmap.md`, `agents.md` (how each agent works), `environment_setup_summary.md`,
  `pytorch_jetson_timing.md`).
* **`examples/`** (located at [examples/](examples/)): Demonstration scripts showcasing
  features (e.g., [demo_env_features.py](examples/demo_env_features.py)).
* **`train.py`** (located at [train.py](train.py)): The main command-line entrypoint for
  initiating RL agent training runs.
* **`record.py`**: Records a trained agent from a checkpoint into one MP4
  (reuses `train.py`'s config loading and challenge selection).
* **`export.py`**: Exports a trained agent's greedy policy from a checkpoint into one standalone C header, or an
  ONNX model with `--format onnx`, and checks it by compiling it or running it in onnxruntime
  (`src/rl_core/export/`; the same config loading and challenge selection).
* **`robustness.py`**: Plays a checkpoint greedily on environments perturbed one setting at a time and reports
  each row inside or beyond its training range (`src/rl_core/training/robustness.py`).
* **`example.py`**: Minimal standalone usage example of the framework.
* **`ARCHITECTURE.md`** / **`TESTING.md`**: Component architecture reference and manual test procedures.
* **`setup.ps1`**: Windows developer setup (installs Python 3.12, creates `.venv`, runs CI checks).
* **`pyproject.toml`**: Packaging metadata, dependencies, and tool settings (Ruff, Mypy, Pytest, Bandit).
* **`.github/workflows/`**: CI/CD: `ci.yml` (`Checks` and `Unit Tests` on every PR), `codeql.yml`,
  `weekly-maintenance.yml`; rules enforced by `tests/test_ci_policy.py`.

---

## 3. Coding Standards & Best Practices

All Python and reinforcement learning coding standards, naming conventions, pure NumPy guidelines, and formatting
rules are maintained in [CONTRIBUTING.md](CONTRIBUTING.md).

Always consult [CONTRIBUTING.md](CONTRIBUTING.md) when writing, refactoring, or reviewing code:

* **Type Safety & Hinting**: Declare explicit types and annotations for variables, parameters, and return types. Use
  type annotations from `typing` and `numpy.typing`.
* **PEP 8 Compliance & Ruff**: Follow PEP 8 guidelines. Format code using Ruff and run static checking before
  submission.
* **Gymnasium API Best Practices**:
  * Adhere to Gymnasium v1.0+ signatures. Always expect `terminated` and `truncated` as separate boolean outputs from
    environment step calls.
  * Always ensure environments are correctly closed with `.close()` when done to prevent resource leaks.
* **Vectorized Training**: Use `make_vec_env` for parallelized environment steps to speed up rollouts when
  configuration allows.
* **Pure NumPy Framework**: The neural network architectures (A2C/PPO) are custom built on top of NumPy without
  external heavy DL dependencies. Maintain optimization cleanliness and verify numerical stability (e.g., stable
  softmax).
* **Deterministic Runs**: Support reproducible seeds in random number generation. Always use the specified seed in
  configs when constructing environment and agent components.

---

## 4. Build, Compile & Test Commands Reference

Always verify compilation and tests locally before finishing a task.

### Environment & Dependencies

Run these commands in the root workspace directory:

```bash
# Windows: one-shot setup (Python 3.12 + .venv + dependencies + CI checks)
.\setup.ps1

# Linux and macOS: .venv with .[dev,video,box2d,onnx], Box2D built first (--profile gpu adds PyTorch)
./setup.sh

# Any OS: install dependencies in editable development mode inside an activated venv
# (LunarLander's and BipedalWalker's Box2D needs the extra steps in README, LunarLander)
pip install -e ".[dev,video,onnx]"
```

### Code Formatting & Linting

```bash
# Run Ruff linter checks
ruff check .

# Run Ruff formatter check
ruff format --check .

# Auto-fix linting issues and format code
ruff check --fix .
ruff format .
```

### Static Type Checking

```bash
# Check static typing with mypy
mypy src
```

### Unit & Integration Testing

```bash
# Run pytest unit test suite with coverage reporting
pytest

# Run tests quietly with short tracebacks
pytest -q
```

### Security & Duplication Audits

```bash
# Run bandit security check
bandit -r src/ -ll

# Check for duplicate code
pylint src --disable=all --enable=duplicate-code --min-similarity-lines=20 --ignore-imports=yes --ignore-signatures=yes
```

### Documentation Synchronization

```bash
# Sync README workflow documentation with .github/workflows
python scripts/sync_workflow_docs.py
```

---

## 5. Feature & Bug Workflow

The 10-step feature & bug workflow (branch → roadmap → plan → TDD red/green/refactor → manual test → complete → PR)
now lives in **[CONTRIBUTING.md §4](CONTRIBUTING.md#4-feature--bug-workflow)** so humans and every AI assistant share
one source of truth. Follow it **in order** for every new feature or bug fix, starting with Step 0: create a
`feature/<name>` (or `fix/<name>`) branch off `develop`.

Each step has a detailed checklist in [`.claude/skills/<name>/SKILL.md`](.claude/skills/) (`dev-loop`, `architect`,
`algorithm-crafter`, `test-crafter`, `experiment-runner`, `training-doctor`, `cleanup`, `pr-reviewer`, `doc-sync`).
They are written for Claude Code but are plain Markdown: read the one for the step you are on and follow it.

---

## 6. Checkpoints, Logs & Outputs Management

The policy for gitignored training artifacts (`logs/`, `checkpoints/`, `test_logs/`, `test_checkpoints/`) and `.npz`
checkpoint guidance lives in **[CONTRIBUTING.md §5](CONTRIBUTING.md#5-generated-artifacts-logs--checkpoints)**. Never
commit training logs or model checkpoints.
