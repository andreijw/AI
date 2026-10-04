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

* **`src/rl_cartpole/`** (located at [src/rl_cartpole/](file:///D:/Documents/Code/AI/src/rl_cartpole/)): Core
  reinforcement learning framework source code.
  * `src/rl_cartpole/environments/` (located at
    [environments/](file:///D:/Documents/Code/AI/src/rl_cartpole/environments/)): Gymnasium wrapper implementations,
    observation/action noise injections, domain randomization configurations, and vectorized environment factories.
  * `src/rl_cartpole/agents/` (located at [agents/](file:///D:/Documents/Code/AI/src/rl_cartpole/agents/)): RL agent
    policy implementations, including abstract `BaseAgent`, baseline `RandomAgent`, the shared actor/critic network
    base `ActorCriticBase` (`actor_critic_base.py`), and deep/policy-gradient agents: `ReinforceAgent`,
    `ActorCriticAgent`, and `PPOAgent`.
  * `src/rl_cartpole/training/` (located at [training/](file:///D:/Documents/Code/AI/src/rl_cartpole/training/)):
    Training pipelines and loop management orchestration (`trainer.py`).
  * `src/rl_cartpole/utils/` (located at [utils/](file:///D:/Documents/Code/AI/src/rl_cartpole/utils/)): Common
    utilities including yaml configuration parsing (`config.py`), progress and training metrics logging (`logger.py`),
    and visualization helpers (`visualization.py`).
* **`tests/`** (located at [tests/](file:///D:/Documents/Code/AI/tests/)): Pytest unit and integration test suite
  targeting environments, agents, logging, configurations, and visualization.
* **`configs/`** (located at [configs/](file:///D:/Documents/Code/AI/configs/)): YAML files specifying hyperparameter
  configs for training default, REINFORCE, Actor-Critic, and PPO algorithms.
* **`scripts/`** (located at [scripts/](file:///D:/Documents/Code/AI/scripts/)): Automation scripts
  (e.g., `sync_workflow_docs.py`).
* **`docs/`** (located at [docs/](file:///D:/Documents/Code/AI/docs/)): Feature guides, setup documents, summaries,
  and roadmaps (`roadmap.md`, `environment_setup_summary.md`).
* **`examples/`** (located at [examples/](file:///D:/Documents/Code/AI/examples/)): Demonstration scripts showcasing
  features (e.g., [demo_env_features.py](file:///D:/Documents/Code/AI/examples/demo_env_features.py)).
* **`train.py`** (located at [train.py](file:///D:/Documents/Code/AI/train.py)): The main command-line entrypoint for
  initiating RL agent training runs.
* **`example.py`**: Minimal standalone usage example of the framework.
* **`ARCHITECTURE.md`** / **`TESTING.md`**: Component architecture reference and manual test procedures.
* **`setup.ps1`**: Windows developer setup (installs Python 3.12, creates `.venv`, runs CI checks).
* **`pyproject.toml`**: Packaging metadata, dependencies, and tool settings (Ruff, Mypy, Pytest, Bandit).
* **`.github/workflows/`**: CI/CD automation workflows (linting, tests, type checking, security, nightly maintenance).

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
  * Adhere to Gymnasium v0.29+ signatures. Always expect `terminated` and `truncated` as separate boolean outputs from
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

# Any OS: install dependencies in editable development mode inside an activated venv
pip install -e ".[dev,video]"
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

---

## 6. Checkpoints, Logs & Outputs Management

The policy for gitignored training artifacts (`logs/`, `checkpoints/`, `test_logs/`, `test_checkpoints/`) and `.npz`
checkpoint guidance lives in **[CONTRIBUTING.md §5](CONTRIBUTING.md#5-generated-artifacts-logs--checkpoints)**. Never
commit training logs or model checkpoints.
