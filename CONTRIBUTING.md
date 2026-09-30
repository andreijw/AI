# Contributing Guidelines & Coding Standards — AI CartPole Framework

Welcome to the **AI Reinforcement Learning Framework**! Follow these mandatory conventions and workflows when
contributing, writing code, refactoring components, or submitting pull requests.

---

## 1. Python & Reinforcement Learning Coding Standards

### Python Version & Type Safety

- **Python Compatibility**: Target Python 3.8 through 3.11+. Use `from __future__ import annotations` when modern type
  union syntax (`X | Y`) or deferred annotations are utilized in older Python runtimes.
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

## 3. Testing Requirements & ATDD Workflow

- **Acceptance Test-Driven Development (ATDD)**: Write failing unit/integration tests in `tests/` encoding the expected
  behavior before implementing production logic.
- **Run the Test Suite**:

```bash
# Run pytest with coverage reporting
pytest

# Run tests quietly
pytest -q
```

- **Coverage Goal**: New features must include tests achieving >=80% branch coverage.
- All tests must pass cleanly (`0 failed`) before submitting a pull request.
