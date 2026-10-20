---
name: cleanup
description: >
  Finds and removes duplication and leftovers on the RL framework: new code that repeats an existing helper
  or base-class method, near-copies that differ only in values, restated hyperparameters, repeated test
  setups, dead code and stray training outputs. Use this after implementing any feature or fix and before
  review, whenever code was generated quickly or by another assistant, and when asked to dedupe, tidy,
  simplify, reuse, or "check we did not duplicate anything". It checks the change against the whole
  codebase, not just against itself, and it fixes what it finds.
---

# Cleanup

Two copies of a rule will eventually disagree: one gets fixed and the other does not. This skill hunts for
copies the change introduced, including copies of code that already existed elsewhere, merges them, and
removes what the work left behind.

Its scope is the current change measured against the whole codebase. For a wider survey of old debt, run
`brooks-lint:brooks-debt` and bring its findings back here to act on. Do not use `brooks-sweep`: fixes go
through this project's tests, one at a time.

Commands are written `python -m …`, meaning the virtual environment's interpreter: `.venv/Scripts/python` on
Windows, `.venv/bin/python` elsewhere.

## 1. List what is new

```bash
git diff develop...HEAD --stat
git status --short
```

From the diff, list every new function, method, loop, constant, config key and test fixture. Those are the
candidates.

## 2. Search by what it does, not what it is called

A duplicate rarely shares a name with the original. For each candidate, search for the mechanism:

| The new code does this | Search for | Use instead |
| :--- | :--- | :--- |
| Picks a discrete action from probabilities | `rng.choice(`, `np.argmax(probs` | `BaseAgent._select_action_from_probs` |
| Works out a checkpoint path or extension | `.npz`, `os.path.splitext` | `BaseAgent._resolve_save_path` / `_resolve_load_path` |
| Rejects bad dimensions | `<= 0` near `raise ValueError` | `BaseAgent._validate_dims` |
| Reads an integer setting from a config | `int(config.get(`, `isinstance(value, bool)` | `rl_core.agents.config_values` (`positive_int`, `optional_positive_int`, `non_negative_int`; `check_positive_int` for list entries) |
| Computes discounted returns | `gamma *` in a reversed loop | `rl_core.agents.returns.discounted_returns` |
| Turns logits into action probabilities | `np.exp(` after `np.max(` | `rl_core.agents.softmax.stable_softmax` |
| Unpacks the trainer's batch | `batch["observations"]` | `ActorCriticBase._parse_trajectory` |
| Accumulates or applies actor-critic gradients | `dW`, `grads[` | `_zero_ac_gradients`, `_accumulate_ac_gradients`, `_apply_ac_gradients` |
| Gaussian log-probability or entropy | `np.log(2 * np.pi` | `ContinuousActorCriticBase._log_prob`, `_entropy` |
| Takes an Adam step | `beta1`, `m_hat` | `ContinuousActorCriticBase._adam_step` |
| Stores and samples transitions | a list of tuples, `random.sample` | `ReplayBuffer` |
| Wraps a Gymnasium environment with seeding, noise or episode statistics | `gym.make(` in a class, `info["episode"]` | subclass `rl_core.environments.BaseEnvWrapper`; `checked_range` for randomization ranges |
| Reads or merges a config | `yaml.safe_load`, `json.load` | `load_config`, `merge_configs`, `train.load_run_config` |
| Builds an environment or agent from config | `gym.make(`, a class lookup by name | `train.select_run_challenge`, then the challenge's `make_env` / `make_agent`; a package's `make_env_from_config` |
| Builds a per-challenge, per-agent output path | `os.path.join(` with the agent type | `train.agent_output_dir` |
| Smooths a reward curve, finds the solve episode | `np.convolve`, a rolling mean | `visualization._rolling_average`, `benchmark.find_solved_episode` |
| Reports progress | `print(` under `src/` | `Logger` / `setup_logger` |
| Runs a script inside a test | `sys.argv`, a config written inline | `tests/helpers.run_main`, `minimal_config` |

The table goes stale; the habit does not. When a candidate is not listed, list what exists and read the
names:

```bash
ls src/rl_core/agents src/rl_core/utils
```

then `Grep` for `def ` in the two or three files whose names fit, underscore-prefixed methods on the base
classes included: most of the reusable maths lives there. For tests, read `tests/helpers.py`.

Then let the tool the nightly CI job uses look for long copies:

```bash
python -m pylint src --disable=all --enable=duplicate-code --min-similarity-lines=20 --ignore-imports=yes --ignore-signatures=yes
```

It only sees blocks of twenty similar lines or more. A five-line rule written twice is still a duplicate,
and only reading finds it.

## 3. Near-copies

Look for two blocks with the same shape and different values: two forward passes, two save methods, two
argument-validation ladders, the same wrapper written for CartPole and for Pendulum. Merge them into one
function that takes the differing values. In tests, two functions that differ only in inputs become one
with `@pytest.mark.parametrize`.

When a number appears in code, a config, a test and a doc: the config owns it, the code holds only the
default, the test reads the agent's attribute or sets its own value explicitly, and the doc either points
at the config or is checked by the `doc-sync` skill.

## 4. What to merge and what to leave

- **Same rule twice**: always merge. This is the case that causes bugs.
- **Same mechanism twice with a helper available**: use the helper.
- **Similar-looking code that encodes different decisions**: leave it. The discrete and continuous agents
  look alike and differ where it matters (a softmax over actions against a Gaussian with a learned spread).
  Merging coincidental likeness produces a function with flags that nobody can change safely. Two
  occurrences are a note; the third is the signal to extract.
- **Known duplication is not licence for more.** The name-to-class maps live only in each challenge since
  2026-10-16 (they used to be repeated in `train.py`). Do not add a further copy; if the change needs one,
  that is the moment to extract. The discounted-returns loop was once written three times and is now
  `discounted_returns`; the stable softmax was written three times and is now `stable_softmax` (2026-10-19).
- Shared maths with no state becomes a module-level function under `src/rl_core/` with its own unit test.
  Shared state and behaviour for a family of agents goes on the family's base class.

## 5. Leftovers

- Imports, variables, private methods, parameters and config keys that the change orphaned. Ruff reports
  unused imports, except in `__init__.py`, where they are exempt: check exports there by hand
- A registry entry, `__init__.py` export or `CONFIG_MAP` row pointing at something that no longer exists
- A YAML key that no code reads, or a `config.get` for a key no YAML sets
- Commented-out code, debug `print` calls, and `# noqa`, `# type: ignore` or `# nosec` added to silence a
  finding instead of fixing it
- Training outputs outside the ignored folders: a stray `.npz`, `.pt`, `.mp4`, `.png` or `.jsonl` in the
  repository root or under `src/`; and an unintended change to `docs/benchmarks/*.md`, which the benchmark
  rewrites by default
- Throwaway scripts and scratch configs in the working tree
- `TODO` notes with no roadmap item behind them: either do it, or add the item
- Docstrings whose `Args`, `Returns` or `Raises` describe the previous behaviour

Do not tidy away anything under `checkpoints/`, `logs/`, `videos/` or `plots/`. They are the maintainer's
training runs, they are not in git, and nothing brings them back.

## 6. Prove nothing changed

Cleanup must not change behaviour. After the edits:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
```

For numerical code "nothing changed" means bit-for-bit: the same seed gives the same weights after the same
updates. A merged helper that draws from the generator in a different order, or sums floats in a different
order, changes every seeded run while every test still passes. If the determinism tests do not cover the
code you moved, capture the weights after a few seeded updates before the refactor and compare after.

A test that fails after a cleanup found a real difference between the "duplicates". Stop and look at which
behaviour was intended before choosing one.

## Report

Say what was merged and into what, what was removed, and what was deliberately left alone and why. If the
change was clean, say so in one line.
