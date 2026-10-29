---
name: test-crafter
description: >
  Writes the tests that lock in behaviour on the RL framework: pytest unit tests for agent maths, buffers,
  wrappers and config, finite-difference gradient checks for hand-written backward passes, integration tests
  through train.py, record.py and the benchmark, and a mutation check proving each test can actually fail.
  Use this for every feature, tuning change or bug fix before writing production code, whenever tests are
  added or changed, when asked whether something is covered or to "lock in" behaviour, and when reproducing
  a bug, a NaN or a shape error. Reviewing an existing suite's structure is a different job: that is
  `brooks-lint:brooks-test`.
---

# Test crafter

A test earns its place by failing when the behaviour breaks. Everything here serves that: put the rule where
a test can reach it, write the test before the code, then break the code on purpose to see the test fail.

Commands are written `python -m …`, meaning the virtual environment's interpreter: `.venv/Scripts/python` on
Windows, `.venv/bin/python` elsewhere.

## Three layers, and which one a check belongs in

| | Unit tests | Integration tests | Learning runs |
| :--- | :--- | :--- | :--- |
| Where | `tests/test_<area>.py` | `tests/test_trainer.py`, `test_train_video.py`, `test_record.py`, `test_benchmark.py` | Not in pytest: `TESTING.md` rows `TR-*` and `BM-*` |
| Drives | one class or function with a hand-built input | `Trainer`, or a script's `main()` through `tests/helpers.run_main` | `train.py` or `src/benchmark.py` with a committed config |
| Costs | milliseconds | a few episodes, under a second each | minutes, and varies by seed |
| Checks | the rule: a value, a gradient, a shape, a bound | that the pieces are wired together and files land where they should | that the agent actually learns |

Prefer the unit layer. Agents in `rl_core/agents/` need no environment: build the batch by hand and call
`update`. If a rule is buried where only a training run reaches it, move it into a function first.

The layers guard different things. `tests/test_agent_harness.py` runs every agent in `ALL_AGENTS` through
the same contract (action type, metrics from `update`, save and load, dimension validation), so it notices
an agent that breaks the interface, and it cannot notice a gradient with the wrong sign. Ask which layer
would actually notice the break you care about.

While iterating, run one file without the coverage report, then the whole suite before handing back:

```bash
python -m pytest tests/test_<area>.py --no-cov -q
python -m pytest
```

## What to write for a change

1. **The rule itself**, named for the behaviour in the suite's existing style:
   `test_ppo_agent_update_empty_batch`, `test_merge_configs_does_not_mutate_base`.
2. **A value worked out by hand.** For maths, pick an input small enough to compute on paper (three rewards
   and a discount; two actions and known logits) and assert the exact result with
   `np.testing.assert_allclose`. Put the arithmetic in a comment.
3. **A gradient check** for every hand-derived backward pass. See below; nothing else catches a wrong sign.
4. **Edges**: an empty batch, a one-step episode, a batch of one, a terminal first step, a buffer holding
   fewer items than the batch size, a buffer wrapping past capacity, an action on the bounds, returns that
   are all equal (zero standard deviation).
5. **Numerical stability**: logits of ±1000, a reward of 1e6, a log standard deviation at its floor. After
   the call, every weight and every returned metric is finite (`np.isfinite(...).all()`).
6. **Determinism**: the same seed gives identical actions and identical weights after an update; a different
   seed does not. Save, load into a fresh agent, and get the same greedy actions.
7. **Invariants**: probabilities sum to one; a continuous action stays inside its bounds; a target network
   is unchanged between syncs and equal after one; evaluation (`training=False`) changes no state.
8. **The contract tables.** A new agent is added to `ALL_AGENTS` in `tests/test_agent_harness.py`. Where
   content is a list (the `AGENT_CLASSES` maps, a challenge's `CONFIG_MAP`, the files in `configs/`), a test
   should fail when an entry is added without the rest: every registered agent builds from its config.
9. **An integration test** when a script, the trainer or the benchmark changed: `run_main([...], monkeypatch)`
   with a config from `minimal_config(tmp_path)`, asserting on the files and messages produced.

For a bug: first write the test that reproduces it and watch it fail. Fix the cause, not the symptom the
test happens to check.

When a change replaces a rule, existing tests that pinned the old rule are rewritten, and the PR description
says which and why. Never edit a test just to make it pass.

## Gradient checks

Backpropagation here is written by hand in NumPy, and a network with a wrong gradient often still trains,
only worse. Compare the analytic gradient with a central finite difference on a tiny network.
`tests/helpers.py` has the tools: `numerical_grad(loss_fn, param)` for one array, and
`assert_gradients_match(loss_fn, params, analytic_grads)`, which checks every named parameter and names the
one that disagrees.

```python
from tests.agent_internals import gradient_check_parameters, loss_and_grads
from tests.helpers import assert_gradients_match


def test_reinforce_gradients_match_finite_differences():
    agent = ReinforceAgent(observation_dim=4, action_dim=3, config={"hidden_dim": 6, "seed": 11})
    observations, actions = _batch()  # hand-built, no environment, no sampling
    return_weights = np.array([0.7, -1.2, 0.4, 1.5, -0.9, 0.3])  # held constant
    _, grads = loss_and_grads(agent, observations, actions, return_weights)  # nothing applied
    parameters, _ = gradient_check_parameters(agent)  # the live arrays, perturbed in place

    assert_gradients_match(
        lambda: loss_and_grads(agent, observations, actions, return_weights)[0],
        parameters,
        grads,
    )
```

`tests/test_policy_gradient_maths.py` has the working versions for REINFORCE, Actor-Critic and PPO, with
the fixtures and kink assertions to copy.

- Use a fixed batch and a fixed seed; `loss_fn` recomputes the loss from the agent's live weights, which
  the helper perturbs in place and restores.
- The weights must be float64 (the helper refuses anything else): a 1e-5 nudge vanishes in float32.
- Pass every parameter array, biases included. A parameter left out is a parameter not checked.
- Keep the network small (hidden size 3 to 6) so the check runs in milliseconds.
- Avoid inputs that sit on a kink: a ReLU unit at exactly zero, or a ratio exactly on a PPO clip bound,
  has no single gradient and will fail the comparison for no reason. Assert the margin in the test.
- Choose the fixture so that every term matters: a mix of active and inactive hidden units (or a missing
  ReLU mask passes), `value_coef` away from 1 and `entropy_coef` well above its tiny default (or dropping
  either from a gradient passes), and for PPO both clipped and unclipped steps.
- Whatever the algorithm holds constant when differentiating (returns, advantages, old log-probabilities,
  TD targets) must be an argument of the method, so the numerical gradient holds it constant too.
- A parameter stored as a Python float (`_b_v`) cannot be perturbed in place: box it in a one-element
  array and have the loss closure copy it back onto the agent (`gradient_check_parameters` in
  `tests/agent_internals.py` returns the box and a `sync` function for the closure to call).
- This needs the loss and its gradients to be computable without also applying the optimizer step. If the
  code under test does both in one method, split it; that is a design improvement, not a test hack.

A gradient check proves the gradients match the loss, not that the loss is the right one. Pair it with a
**loss worked out by hand**: zero the trunk so the policy and value are the same at every step, pick uneven
rewards, and write the expected number in the test. That is what catches a return paired with the wrong
step.

Every learning agent has both: `tests/test_policy_gradient_maths.py` (REINFORCE, Actor-Critic, PPO),
`tests/test_continuous_policy_gradient_maths.py` (the two continuous agents) and `tests/test_dqn_maths.py`.
A new agent gets the same before it is trained.

Three things those files learned the hard way: pin how `update` turns the gradients into a step (a flipped
sign there passes every gradient check); after one Adam step every weight has moved by about the learning
rate whatever the gradient, so compare inputs or take several steps; and a done flag only at the end of a
batch cannot show a missing mask, so include a batch that ends an episode in the middle. For anything with
a target network, end a test between syncs and assert the target shares no memory with the online arrays:
an aliased target only shows on the update after a sync.

## Rules that keep tests honest

- **Ask what break would make each check fail.** "The weights changed after `update`" passes for a gradient
  with the wrong sign, the wrong scale or the wrong layer. It shows the code ran. Replace it with a check that
  could fail: the loss on a fixed batch falls over several updates, or the gradient matches the numerical one.
  For an on-policy policy gradient, sample each update's batch from the current policy and assert the rewarded
  action's probability: replaying a fixed batch keeps an action the policy has dropped, whose -A·log π term
  has no lower bound, and that is what collapsed the discrete Actor-Critic in `test_policy_gradient_maths.py`
  (CHANGELOG.md, 2026-10-29). Two real cases from this suite: a test that "returns decrease over time with
  γ < 1", on five rewards of 1, still passed with the discount removed, because 5, 4, 3, 2, 1 also decreases; only
  the test asserting exact values noticed. And a softmax stability test that scaled the weights by 1e4 could
  not fail, because that seed's logits were negative and never overflow; setting the head's bias to ±1000 can.
- **Every file goes under `tmp_path`.** Never `checkpoints/`, `logs/`, `videos/` or `plots/`: those hold
  the maintainer's training runs, and `train.py` overwrites an agent's earlier files.
- **Pin every seed.** Agent config, `ReplayBuffer`, `env.reset(seed=...)`. An unseeded test is a flaky test
  that has not failed yet.
- **Exact values for maths, relationships for sampling.** A discounted return has one right answer. "Action
  0 is drawn more often than action 1" over a thousand seeded draws holds; "it is drawn 713 times" pins the
  generator's internals.
- **Bit-for-bit only within one machine.** Comparing two code paths in one run with `assert_array_equal` is
  fine. Weights recorded on one machine and asserted exactly on another are not: this ARM board and the x86 CI
  runner round matrix products differently in the last bit (a recorded-weights test failed CI by 2e-16 relative,
  2026-10-18). Pin recorded values with `rtol=1e-12`; that still catches any real change.
- **No training to convergence in pytest.** The suite must stay fast. A test may run a handful of updates
  on a fixed batch; whether CartPole is solved belongs to a learning run.
- **Close what you open.** `env.close()` in a `finally` or a fixture, including when the assertion fails.
- **Parametrize rather than copy.** Two tests that differ only in inputs become one
  `@pytest.mark.parametrize`.
- **Patch at the seam, not the internals.** The script tests monkeypatch `sys.argv` and optional imports;
  a test that reaches into private attributes to make an assertion breaks on the next refactor. Agent
  tests assert through `select_action`, the metrics `update` returns and `save`/`load`
  (`tests.helpers.assert_same_checkpoint`, loading into a differently seeded agent that has trained at least
  once, so a load that skips a zero-initialised bias still fails). Maths tests that need weights, the forward
  pass or a loss method go through `tests/agent_internals.py`, the one module allowed to name an agent's
  private members; `tests/trainer_internals.py` is the same for `Trainer`, whose tests otherwise go through
  `train()`, `evaluate()` and the files and log lines they produce.
  `test_migrated_agent_tests_use_only_the_public_interface` holds every `tests/test_*.py` to that except the
  non-agent files in `PRIVATE_ACCESS_EXEMPT`; a new test file is checked automatically.
- **Every assertion message says what should have happened**, in words a person reading CI output
  understands.

## Mutation check

After the tests pass, prove each one can fail. For every new rule:

1. Break the production code in the smallest way that violates the rule: flip a gradient's sign, drop the
   discount, remove the `(1 - done)` mask, swap `terminated` for `truncated`, widen a clip bound, skip the
   target sync, shift a buffer index by one, delete an epsilon.
2. Run the tests that should notice (`--no-cov -q`). A **named** test must fail, for the reason you expect.
3. Undo the break with the inverse edit and confirm `git diff` shows only your real changes. Do not use
   `git checkout --` on a file that has uncommitted work in it.

For more than one or two, use the script that ships with this skill. Write the breaks as a JSON list in
your scratch directory and run it from the repository root:

```json
[
  {
    "file": "src/rl_core/agents/returns.py",
    "old": "gamma * cumulative",
    "new": "1.0 * cumulative",
    "tests": ["tests/test_returns.py"],
    "why": "discount dropped from returns"
  }
]
```

```bash
python .claude/skills/test-crafter/scripts/mutation_check.py <scratch>/mutations.json
```

`old` must occur exactly once in the file. The script checks that the tests pass first, applies each break
in turn, prints `KILLED` with the tests that failed or `SURVIVED`, and writes the original bytes back even
if interrupted. It exits 1 if anything survived. Afterwards, confirm with `git status --short` that only
your real changes remain.

If nothing fails, the behaviour is not locked in: the test is in the wrong layer, asserts too little, or
adapts to the break. Fix the test, not the mutation.

List the mutations and the tests that caught them in the PR description. Three or four well-chosen breaks
are enough for a typical change; aim them at what would silently make an agent learn worse.

## Before handing back

- `python -m pytest` passes, and passes again on a second run if a new test involves sampling.
- `python -m ruff check .` and `python -m ruff format --check .` are clean; tests are linted too.
- Nothing was written outside `tmp_path`: `git status --short` shows only files you meant to change.
- For a structural second opinion on the tests you wrote, the dev loop runs `brooks-lint:brooks-test`.
