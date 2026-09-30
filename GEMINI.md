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
    policy implementations, including abstract `BaseAgent`, baseline `RandomAgent`, and deep/policy-gradient agents:
    `ReinforceAgent`, `ActorCriticAgent`, and `PpoAgent`.
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
# Install dependencies in editable development mode
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

Follow these steps **in order** for every new feature or bug fix. Do not skip or reorder steps.

### Step 1 — Update the roadmap

* Find the matching item in `docs/roadmap.md`. If none exists, add it under the appropriate phase.
* Flesh out the item: write a clear description, acceptance criteria, and known edge cases.
* Mark the item **In Progress**.
* **Stop here and ask the user to confirm the roadmap item** before continuing. Do not proceed until confirmed.

### Step 2 — Finalize the work item

* Incorporate any feedback from the confirmation step into `docs/roadmap.md`.
* Clarify scope, algorithmic constraints (pure NumPy, Gymnasium v0.29+ API), and explicit out-of-scope items with the
  user.
* The work item is now locked — no scope changes without repeating steps 1–2.

### Step 3 — Plan the solution

* Identify every file, class, function, and configuration that needs to change.
* **Check existing components first** — search for similar logic in `src/rl_cartpole/` (e.g. `agents/BaseAgent`,
  `environments/`, `utils/`) before writing anything new. Plan to reuse or extend rather than duplicate.
* Design clean abstractions; avoid one-off inline logic for anything with a natural architectural boundary.
* Write a short implementation plan and share it with the user before touching any code.
* Highlight any breaking changes, config schema updates, or numerical stability considerations.

### Step 4 — Write tests first (ATDD)

* Add or update unit tests in `tests/` that encode the **expected behavior** defined in step 2.
* Cover happy paths **and** edge cases (e.g., state bounds, action masking/clamping, terminal states, reproducibility).
* Tests must fail at this point — that is the goal of this step.
* Pytest tests → `tests/test_*.py`.

### Step 5 — Implement the code changes

* Write only the code needed to make the failing tests pass.
* Follow the coding standards in section 3 and [CONTRIBUTING.md](CONTRIBUTING.md).
* Run validation commands from section 4 after each logical chunk of work (`ruff check .`, `ruff format --check .`,
  `mypy src`).

### Step 6 — Fix regressions and check for duplication

* Run the full test suite: `pytest`.
* Fix any failures caused by changes by updating the **implementation**, not the tests.
* Only change a test if it was demonstrably incorrect — document why in the PR description.
* **Scan for code duplication** introduced or exposed:
  * Check for repeated patterns across agents, environment wrappers, or training loops.
  * If duplication is found, extract it into a shared utility or helper before marking the step done.

### Step 7 — Request manual testing

* Ask the user to manually verify or test the feature before it is considered done.
* Provide specific, actionable testing instructions: runnable commands (e.g. `python train.py --config ...`,
  `python examples/demo_env_features.py`), expected metrics (rewards, episode lengths, convergence thresholds), and log
  outputs.
* Reference the relevant rows in [TESTING.md](TESTING.md).
* **Wait for the user's sign-off** before proceeding to step 8.

### Step 8 — Mark the feature complete

* Once the user confirms the feature is working as expected, update `docs/roadmap.md`:
  * Move the item to its completed milestone section (or mark `[x]`).
  * Remove the **In Progress** label.
* If the feature introduced new manual test procedures, document them in [TESTING.md](TESTING.md).
* Run `python scripts/sync_workflow_docs.py` if workflows or documentation markers were modified.

### Step 9 — Output Pull Request changes

* Prepare and output a complete Pull Request description formatted exactly according to the template in
  [.github/pull_request_template.md](.github/pull_request_template.md).
* Summarize the changes clearly, verify the checklist confirmations, detail verification steps, and list all files
  modified.

---

## 6. Checkpoints, Logs & Outputs Management

To manage local storage growth and prevent checking binary artifacts or bulky tracking files into git, directory gates
and cleanup policies are set up.

* **Gitignored Artifact Directories**: Do not commit training logs or model checkpoints to the repository. The
  following folders are strictly blocked in `.gitignore`:
  * `logs/` (tensorboard and raw metrics logs)
  * `checkpoints/` (serialized numpy agent weights)
  * `test_logs/` & `test_checkpoints/` (temporary test suite outputs)
* **Checkpoints Saving**: Model weights are typically persisted using NumPy `.npz` files or generic state dicts. Keep
  checkpointing frequencies balanced to avoid excessive local disk write/usage.
