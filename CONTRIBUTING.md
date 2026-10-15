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

- **Seeding**: Support deterministic runs by propagating explicit random seeds to environments, agents and buffers.
  Every component that draws random numbers owns a generator created with `np.random.default_rng(seed)`; never call
  `np.random.seed` or the global `np.random.*` functions (`tests/test_source_conventions.py` fails if one appears).
  New components built from a seed another component already uses take their own stream with
  `rl_core.seeding.derive_rng(seed, stream)`, as the environment wrappers do. Agents and `ReplayBuffer` do not yet:
  they use `default_rng(seed)` directly (see the roadmap).
- **Seeded environments**: A wrapper applies its constructor seed on the first `reset()` and then continues the
  sequence, so a seeded run is reproducible and every episode starts from a different state. Pass `reset(seed=...)`
  only to restart that sequence deliberately.

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

**Who decides what.** When an AI assistant does the work, the maintainer makes three decisions: which work to do,
whether the plan is right (step 3, after it has been reviewed), and when to commit (after step 6's gates have
passed). The assistant makes the implementation, test and refactoring choices in between and reports them. Committed
hyperparameters and published benchmark numbers, scope, new dependencies and changes under `.github/` always go to
the maintainer.

**Skills.** In Claude Code each step is carried out by a skill in [`.claude/skills/`](.claude/skills/), started by
`dev-loop`. Setup: [README → AI Assistant Skills](README.md#ai-assistant-skills). Other assistants and human
contributors can read the same files as checklists.

| Step | Skill |
| :--- | :--- |
| 0–3 Branch, roadmap item and plan | `dev-loop`, with `algorithm-crafter` for algorithm work and the plan reviewed by `architect` |
| 4 Tests first | `test-crafter` |
| 5 Implement | `dev-loop` |
| 6 Regressions, duplication and gates | `test-crafter` (mutation checks), `cleanup`, `experiment-runner` (learning run), then `pr-reviewer` |
| 7–9 Manual testing, completion, PR | `doc-sync`, then the `dev-loop` hand-off |
| A run that does not learn | `training-doctor`, before any hyperparameter is changed |

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
- The roadmap item is confirmed together with the plan in step 3. Stop here first only if the scope or the intended
  behaviour is unclear.

### Step 2 — Finalize the work item

- Incorporate any feedback from the confirmation step into `docs/roadmap.md`.
- Clarify scope, algorithmic constraints (pure NumPy, Gymnasium v0.29+ API), and explicit out-of-scope items.
- The work item is now locked — no scope changes without repeating steps 1–2.

### Step 3 — Plan the solution

- Identify every file, class, function, and configuration that needs to change.
- **Check existing components first** — search `src/rl_core/` and the challenge packages for similar logic before
  writing anything new (e.g. `rl_core/agents/base_agent.py` `BaseAgent`, `rl_core/agents/actor_critic_base.py`
  `ActorCriticBase`, the `environments/make_env.py` factories, `rl_core/utils/`). Plan to reuse or extend rather than
  duplicate.
- Design clean abstractions; avoid one-off inline logic for anything with a natural architectural boundary.
- For a change to a hyperparameter, a schedule or an update rule, walk the numbers through a typical run before
  proposing them. For new maths, write down the array shapes and the hand-derived gradients.
- Highlight any breaking changes, config schema updates, checkpoint format changes, or numerical stability
  considerations.
- Have the plan reviewed for structure (the `architect` skill) and revise it.
- **Stop here: share the reviewed plan and the roadmap item with the user and wait for approval** before touching
  any code.

### Step 4 — Red: write failing tests first

- Add or update tests in `tests/test_*.py` that encode the **expected behavior** defined in step 2.
- Cover happy paths **and** edge cases (e.g., state bounds, action clamping, terminal vs. truncated states,
  reproducibility under a fixed seed).
- A hand-written backward pass gets a finite-difference gradient check (`assert_gradients_match` in
  `tests/helpers.py`). For a bug, the first test reproduces the bug.
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

- **Prove the tests can fail** (mutation check): break each new rule in the smallest way, confirm a named test
  fails, and undo the break (`.claude/skills/test-crafter/scripts/mutation_check.py` automates this). List the
  mutations in the PR.
- **Show that it still learns** when the change touches an update rule, a loss, a network, the trainer or a committed
  config: seeded runs through `src/benchmark.py` with `--output-dir` and `--report` pointing outside the repository,
  so saved runs and `docs/benchmarks/` are untouched. A few-episode smoke run is not evidence of learning.
- **Pass every gate** before asking for a commit: `ruff check .`, `ruff format --check .`, `mypy src`, `pytest`, the
  two audits above, and a review of the final diff by someone who did not write it (the `pr-reviewer` skill in a
  fresh agent).

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

### Pull requests and approval

- Changes reach `develop` only through pull requests. Direct pushes, force pushes and deleting `develop` are blocked,
  and every commit must be signed.
- These CI checks must pass before a pull request can merge: Unit Tests, Ruff Linting, MyPy Type Checking, Bandit
  Security Scan and Markdown Linting. CodeQL scanning and a Copilot review also run on every pull request.
- The maintainer owns every file ([.github/CODEOWNERS](.github/CODEOWNERS)), so a pull request from anyone else
  needs the maintainer's approval to merge. A new push dismisses an earlier approval.
- Contributions from forks are welcome: open the pull request against `develop` and the maintainer will review it.

---

## 5. Generated Artifacts (Logs & Checkpoints)

Training runs and the test suite write bulky outputs that must never be committed.

- **Gitignored artifact directories** — keep these out of git (already blocked in `.gitignore`):
  - `logs/` (TensorBoard and raw metrics logs)
  - `checkpoints/` (serialized NumPy agent weights)
  - `test_logs/` and `test_checkpoints/` (temporary test-suite outputs)
- **Checkpoint format** — persist model weights as NumPy `.npz` files (or plain state dicts). Keep checkpoint frequency
  (`save_freq`) balanced to avoid excessive disk writes.
