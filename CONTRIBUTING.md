# Contributing Guidelines & Coding Standards — AI CartPole Framework

Welcome to the **AI Reinforcement Learning Framework**! Follow these mandatory conventions and workflows when
contributing, writing code, refactoring components, or submitting pull requests.

---

## 1. Python & Reinforcement Learning Coding Standards

### Python Version & Type Safety

- **Python Compatibility**: Target Python 3.12+ (the version CI tests). Modern syntax such as `X | Y` unions and
  built-in generics (`list[int]`) is available natively.
- **Explicit Type Hinting**: All public classes, methods, and functions must declare explicit argument and return types
  using `typing` (`Dict`, `List`, `Tuple`, `Optional`, `Union`, `Callable`, `Any`) and `numpy.typing` (`NDArray`).
- **Type Checking**: Code must pass `mypy src/` without errors before submission.

### Gymnasium API Conventions

- **Gymnasium v0.29+ Signatures**: Always expect and process `terminated` (task completion/failure) and `truncated`
  (time limit expiration) as separate boolean outputs from environment steps:
  `obs, reward, terminated, truncated, info = env.step(action)`
- **Resource Management**: Always ensure environments are closed cleanly using `env.close()` or context managers to
  prevent memory leaks and process zombies.
- **Standardized Vectorized Interfaces**: When utilizing vectorized rollouts (`make_vec_env`), interact through standard
  `gym.vector.VectorEnv` batch semantics.

### Pure NumPy Mathematical Integrity

- **Framework Independence**: Agent neural networks and policy representations must be implemented cleanly in pure
  NumPy without external deep learning frameworks (e.g., PyTorch, TensorFlow).
- **Numerical Stability**:
  - Implement numerically stable softmax: `exp(z - np.max(z))` to prevent overflow.
  - Safeguard logarithms and divisions: use small epsilon offsets (e.g., `np.log(probs + 1e-8)`) to prevent `NaN` or
    negative infinity values.
  - Clip probability ratios and gradients where applicable (e.g., PPO surrogate clipping in `[1 - eps, 1 + eps]`).

### Determinism & Reproducibility

- **Seeding**: Support deterministic runs by propagating explicit random seeds to environments (`env.reset(seed=...)`),
  NumPy random number generators (`np.random.default_rng(seed)` or `np.random.seed(seed)`), and action sampling policies.

---

## 2. Code Style & Automated Formatting

Formatting and linting rules are strictly enforced by Ruff and configured in `pyproject.toml`.

Always run formatting and linting checks locally before committing:

```bash
# Auto-fix linting issues and reformat code
ruff check --fix .
ruff format .

# Check static typing
mypy src
```

- **Line Length**: Max 100 characters for Python source code.
- **Import Sorting**: Handled automatically by Ruff (`I` rules).
- **Documentation**: All public modules, classes, and methods must have descriptive docstrings explaining parameters,
  return types, and raised exceptions.

---

## 3. Testing Requirements & TDD Workflow

- **Test-Driven Development (red → green → refactor)**: Write failing unit/integration tests in `tests/` that encode the
  acceptance criteria *before* implementing production logic, make them pass with the minimum code, then refactor with
  the tests green. See [§4 Steps 4–6](#step-4--red-write-failing-tests-first).
- **Run the Test Suite**:

```bash
# Run pytest with coverage reporting
pytest

# Run tests quietly
pytest -q
```

- **Coverage Goal**: New features must include tests achieving >=80% branch coverage.
- All tests must pass cleanly (`0 failed`) before submitting a pull request.

---

## 4. Feature & Bug Workflow

This workflow is the single source of truth for humans and every AI assistant (see `CLAUDE.md` and `GEMINI.md`).
Follow these steps **in order** for every new feature or bug fix. Do not skip or reorder steps.

### Step 0 — Create a branch

- Start from an up-to-date `develop` and create one branch per roadmap item:

```bash
git checkout develop
git pull
git checkout -b feature/<kebab-case-name>   # or fix/<kebab-case-name> for bugs
```

- Never work directly on `develop`. All steps below happen on this branch.

### Step 1 — Update the roadmap

- Find the matching item in `docs/roadmap.md`. If none exists, add it under the appropriate phase.
- Flesh out the item: write a clear description, acceptance criteria, and known edge cases.
- Mark the item **In Progress**.
- **Stop here and ask the user to confirm the roadmap item** before continuing. Do not proceed until confirmed.

### Step 2 — Finalize the work item

- Incorporate any feedback from the confirmation step into `docs/roadmap.md`.
- Clarify scope, algorithmic constraints (pure NumPy, Gymnasium v0.29+ API), and explicit out-of-scope items.
- The work item is now locked — no scope changes without repeating steps 1–2.

### Step 3 — Plan the solution

- Identify every file, class, function, and configuration that needs to change.
- **Check existing components first** — search `src/rl_cartpole/` for similar logic before writing anything new
  (e.g. `agents/base_agent.py` `BaseAgent`, `agents/actor_critic_base.py` `ActorCriticBase`,
  `environments/make_env.py` factories, `utils/`). Plan to reuse or extend rather than duplicate.
- Design clean abstractions; avoid one-off inline logic for anything with a natural architectural boundary.
- Write a short implementation plan and share it before touching any code.
- Highlight any breaking changes, config schema updates, or numerical stability considerations.

### Step 4 — Red: write failing tests first

- Add or update tests in `tests/test_*.py` that encode the **expected behavior** defined in step 2.
- Cover happy paths **and** edge cases (e.g., state bounds, action clamping, terminal vs. truncated states,
  reproducibility under a fixed seed).
- Run only the affected tests and confirm they **fail for the right reason** (an assertion or missing feature, not an
  import or syntax error):

```bash
pytest tests/test_<area>.py
```

### Step 5 — Green: implement the minimum code

- Write only the code needed to make the failing tests pass.
- Follow the coding standards in [§1](#1-python--reinforcement-learning-coding-standards) and the formatting rules in
  [§2](#2-code-style--automated-formatting).
- Run the validate commands after each logical chunk of work:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

### Step 6 — Refactor: fix regressions and remove duplication

- Run the full test suite: `pytest`. Keep it green throughout this step.
- Fix any failures caused by your changes by updating the **implementation**, not the tests.
- Only change a test if it was demonstrably wrong to begin with — document why in the PR.
- **Scan for code duplication** introduced or exposed by this change — repeated patterns across agents, environment
  wrappers, or training loops. Extract it into a shared helper before marking the step done. Do not defer it.
- Run the security and duplication audits that CI runs nightly:

```bash
bandit -r src/ -ll
pylint src --disable=all --enable=duplicate-code --min-similarity-lines=20 --ignore-imports=yes --ignore-signatures=yes
```

### Step 7 — Request manual testing

- Ask the user to manually verify the feature before it is considered done.
- Provide specific, actionable testing instructions: runnable commands (e.g.
  `python train.py --config configs/cartpole_ppo.yaml`, `python examples/demo_env_features.py`), expected metrics
  (rewards, episode lengths, convergence thresholds), and log output to look for.
- Reference the relevant rows in [TESTING.md](TESTING.md) (e.g. `ENV-*`, `TR-*`, `CKPT-*`).
- **Wait for the user's sign-off** before proceeding to step 8.

### Step 8 — Mark the feature complete

- Once the user confirms the feature works as expected, update `docs/roadmap.md`:
  - Move the item to its completed milestone section (or mark `[x]`).
  - Remove the **In Progress** label.
- If the feature introduced new manual test procedures, add them to [TESTING.md](TESTING.md).
- If `.github/workflows/` changed, run `python scripts/sync_workflow_docs.py` to refresh the README workflow list.

### Step 9 — Open the Pull Request

- Push the branch and open a PR into `develop`.
- Write the description exactly according to [.github/pull_request_template.md](.github/pull_request_template.md):
  summarize the changes, complete the checklist, detail verification steps, and list all files modified.

---

## 5. Generated Artifacts (Logs & Checkpoints)

Training runs and the test suite write bulky outputs that must never be committed.

- **Gitignored artifact directories** — keep these out of git (already blocked in `.gitignore`):
  - `logs/` (TensorBoard and raw metrics logs)
  - `checkpoints/` (serialized NumPy agent weights)
  - `test_logs/` and `test_checkpoints/` (temporary test-suite outputs)
- **Checkpoint format** — persist model weights as NumPy `.npz` files (or plain state dicts). Keep checkpoint frequency
  (`save_freq`) balanced to avoid excessive disk writes.
