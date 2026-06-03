# Instruction Manual for Antigravity CLI (GEMINI.md)

Welcome, AI coding assistant! To operate efficiently in this codebase, minimize token costs, and maintain code quality,
you **must** adhere strictly to the guidelines and standards defined in this document.

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
  * For writing unit tests, running builds, debugging compile errors, or simple refactoring tasks, utilize cheaper models
  (e.g., `gemini-3.1-flash-lite` or similar).
  * Always do a planning phase first to think of the overall architecture and design before starting to implement the code.
  * Only escalate to larger, more expensive reasoning models (e.g., `gemini-3.5-flash` / `gemini-3.5-pro` equivalents)
  for complex architectural refactoring or prompt engineering tasks.

---

## 2. Codebase Directory Layout

* **`src/rl_cartpole/`** (located at [src/rl_cartpole/](file:///D:/Documents/Coding/AI/src/rl_cartpole/)): Core
reinforcement learning framework source code.
  * `src/rl_cartpole/environments/` (located at [environments/](file:///D:/Documents/Coding/AI/src/rl_cartpole/environments/)):
  Gymnasium wrapper implementations, observation/action noise injections, domain randomization configurations, and
  vectorized environment factory tools.
  * `src/rl_cartpole/agents/` (located at [agents/](file:///D:/Documents/Coding/AI/src/rl_cartpole/agents/)): RL agent
  policy implementations, including the abstract `BaseAgent`, baseline `RandomAgent`, and deep/policy-gradient agents:
  `ReinforceAgent`, `ActorCriticAgent`, and `PpoAgent`.
  * `src/rl_cartpole/training/` (located at [training/](file:///D:/Documents/Coding/AI/src/rl_cartpole/training/)):
  Training pipelines and loop management orchestration (`trainer.py`).
  * `src/rl_cartpole/utils/` (located at [utils/](file:///D:/Documents/Coding/AI/src/rl_cartpole/utils/)): Common
  utilities including yaml configuration parsing (`config.py`), progress and training metrics logging (`logger.py`),
  and visualization helpers (`visualization.py`).
* **`tests/`** (located at [tests/](file:///D:/Documents/Coding/AI/tests/)): Pytest unit and integration test suite
targeting environments, agents, logging, configurations, and visualization.
* **`configs/`** (located at [configs/](file:///D:/Documents/Coding/AI/configs/)): YAML files specifying hyperparameter
configs for training default, REINFORCE, Actor-Critic, and PPO algorithms.
* **`scripts/`** (located at [scripts/](file:///D:/Documents/Coding/AI/scripts/)): Automation scripts (e.g., `sync_workflow_docs.py`).
* **`docs/`** (located at [docs/](file:///D:/Documents/Coding/AI/docs/)): Feature guides, setup documents, and summaries.
* **`examples/`** (located at [examples/](file:///D:/Documents/Coding/AI/examples/)): Demonstration scripts showcasing
features (e.g., [demo_env_features.py](file:///D:/Documents/Coding/AI/examples/demo_env_features.py)).
* **`train.py`** (located at [train.py](file:///D:/Documents/Coding/AI/train.py)): The main command-line entrypoint for
initiating RL agent training runs.

---

## 3. Coding Standards & Best Practices

### Python / Reinforcement Learning

* **Type Safety & Hinting**: Declare explicit types and annotations for variables, parameters, and return types. Use type
annotations from `typing` and `numpy.typing`.
* **PEP 8 Compliance**: Follow PEP 8 guidelines. Format code using Ruff and run static checking before submission.
* **Gymnasium API Best Practices**:
  * Adhere to Gymnasium v0.29+ signatures. Always expect `terminated` and `truncated` as separate boolean outputs from
  environment step calls.
  * Always ensure environments are correctly closed with `.close()` when done to prevent resource leaks.
* **Vectorized Training**: Use `make_vec_env` for parallelized environment steps to speed up rollouts when configuration
allows.
* **Pure NumPy Framework**: The neural network architectures (A2C/PPO) are custom built on top of NumPy. Maintain
optimization cleanliness and verify numerical stability (e.g., stable softmax).
* **Deterministic Runs**: Support reproducible seeds in random number generation. Always use the specified seed in
configs when constructing environment and agent components.

---

## 4. Build, Compile & Test Commands Reference

Always verify compilation and tests locally before finishing a task.

Run these commands in the root workspace directory:

```bash
# Install dependencies in editable development mode
pip install -e ".[dev,video]"

# Run Ruff linter checks
ruff check .

# Run Ruff formatter check
ruff format --check .

# Auto-fix linting issues and format code
ruff check --fix .
ruff format .

# Check static typing with mypy
mypy src

# Run pytest unit test suite with coverage reporting
pytest
```

---

## 5. Checkpoints, Logs & Outputs Management

To manage local storage growth and prevent checking binary artifacts or bulky tracking files into git, directory gates
and cleanup policies are set up.

* **Gitignored Artifact Directories**: Do not commit training logs or model checkpoints to the repository. The following
folders are strictly blocked in `.gitignore`:
  * `logs/` (tensorboard and raw metrics logs)
  * `checkpoints/` (serialized numpy agent weights)
  * `test_logs/` & `test_checkpoints/` (temporary test suite outputs)
* **Checkpoints Saving**: Model weights are typically persisted using NumPy `.npz` files or generic state dicts. Keep
checkpointing frequencies balanced to avoid excessive local disk write/usage.
