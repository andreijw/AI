---
name: training-doctor
description: >
  Diagnoses a reinforcement learning agent that misbehaves on this framework: reward stuck at the random
  level, learning that collapses after improving, NaN or inf in weights or losses, exploding Q-values, a
  policy that turns deterministic too early, results that differ wildly by seed, or an agent that scores
  well in training and badly in evaluation. Use this whenever the user says an agent "is not learning",
  "stopped learning", "diverges", "is unstable", "gives NaN", "got worse", or asks why a training run or
  benchmark looks wrong, and before changing any hyperparameter to fix a bad run. It finds the cause; the
  fix then goes through the dev loop.
---

# Training doctor

When an agent does not learn, the tempting move is to change a learning rate and run again. That costs
minutes per try and usually hides a bug. Most failures here are bugs in hand-written maths, and they can be
found in seconds with the right check. Work down this ladder: each rung is cheaper than the one below it,
and each one rules something out.

Commands are written `python …`, meaning the virtual environment's interpreter: `.venv/Scripts/python` on
Windows, `.venv/bin/python` elsewhere. Diagnostic scripts live in your scratch directory, never in `src/`.

## 0. Pin it down

Reproduce the problem with a fixed seed and the smallest run that still shows it. Write down the symptom
as numbers: which config, which seed, reward at which episode, against what was expected. Check the same
seeds on `develop` (the `experiment-runner` skill shows how, with a worktree): if `develop` is fine, the
diff holds the cause and `git diff develop...HEAD` is the first thing to read.

## 1. Look inside the update

`Trainer` discards what `agent.update` returns: the logs carry only the ten-episode reward and length
averages. To see losses, entropy or epsilon, drive the agent from a scratch script instead of adding
prints to `src/`:

```python
import numpy as np
from train import create_agent, create_env, load_run_config

np.seterr(all="raise")  # turn the first overflow, divide-by-zero or invalid value into a traceback
config, agent_type = load_run_config("configs/cartpole_dqn.yaml")
env = create_env(config, render_mode=None)
agent = create_agent(agent_type, config, env)
# roll out one episode by hand, build the batch the Trainer builds, then:
#     metrics = agent.update(batch); print(episode, metrics)
env.close()
```

Run it from the repository root with the root on the import path (`PYTHONPATH=. python <scratch>/diag.py`;
in PowerShell, `$env:PYTHONPATH = "."` first). `np.seterr(all="raise")` is the fastest way to find where a NaN is born:
the traceback points at the line, where the log file would only show the damage many steps later.

## 2. Is the maths right?

These need no environment and run in milliseconds. Write each as a pytest test (the `test-crafter` skill),
because the one that fails is the regression test for the fix.

- **Gradient check.** `assert_gradients_match` in `tests/helpers.py` compares the hand-written gradients
  with finite differences, parameter by parameter. A mismatch in one array names the layer with the bug.
- **Direction check.** Build a batch in which one action was followed by a high return. After one update
  the probability (or Q-value) of that action must rise. If it falls, a sign is wrong.
- **Overfit a fixed batch.** Call `update` on the same small batch many times. The loss must fall towards
  zero. If it cannot fit thirty transitions, it will not fit an environment.
- **Shapes.** Print the shape of every intermediate array once. `(n,)` against `(n, 1)` broadcasts to
  `(n, n)` without complaint and turns a loss into noise.

## 3. Is the data right?

Print one real batch and read it:

- **`dones`**: true on the last step only. The trainer sets it for `terminated` *or* `truncated`, so a
  bootstrapping agent treats a time-limit cut as a true end. Pendulum only ever ends that way.
- **Rewards**: their scale. CartPole pays 1 per step; Pendulum pays between about -16 and 0. A learning
  rate tuned for one is wrong by an order of magnitude for the other.
- **Actions**: the dtype and range the environment expects; a continuous action inside its bounds.
- **Observations**: finite, and of the scale the network was initialised for.
- **Wrappers**: observation noise, action noise and domain randomization are off unless the config asks
  for them, and are the first suspects when training and evaluation disagree.

## 4. Is the schedule right?

Walk the numbers with the config's real values:

- How many updates actually happen? An off-policy agent does nothing until its buffer holds
  `min_buffer_size`, and `max_updates_per_call` caps the work per episode.
- When does exploration end, compared with when the agent starts seeing long episodes?
- How often does the target network sync, measured in episodes?
- Is the learning rate times the gradient scale a small step or a leap? Print the update's norm against
  the weight's norm for one step.

## 5. Match the symptom

| Symptom | Look first at |
| :--- | :--- |
| Reward stays at the random level from the start | Gradient sign; updates never running (buffer not warm, empty batch returns early); learning rate far too small; acting greedily during training |
| Improves, then collapses | Step too large; no clipping; target network aliased to the online one (`=` without `.copy()`) or never synced; PPO ratio using recomputed, not stored, log-probabilities |
| NaN or inf | `log` of zero, division by a zero standard deviation (one-step episode, equal returns), `exp` of a large logit or log standard deviation, an unmasked bootstrap feeding on itself |
| Q-values grow without bound | Missing terminal mask on the bootstrap; reward scale; target computed from the network being updated |
| Policy turns deterministic early | Entropy bonus missing or with the wrong sign; continuous log standard deviation driven to its floor |
| Good on one seed, bad on another | Possibly nothing wrong: run more seeds before believing either. Then initialisation scale and exploration |
| Good in training, bad in evaluation (or the reverse) | The `training=False` path; exploration or wrapper noise carrying the policy; evaluation mutating agent state |
| Fine before a refactor, worse after | Order of random draws or of floating-point sums changed; compare weights after a few seeded updates on both versions |
| Loaded checkpoint behaves like a fresh agent | An array not saved or not restored; counters, epsilon or optimizer moments reset on load |

## 6. Only then, tuning

If the maths, the data and the schedule are right, the problem is a hyperparameter, and that is an
experiment: one change, several seeds, a baseline (`experiment-runner`). A change to a committed
`configs/*.yaml` is the maintainer's decision; bring them the evidence and a recommendation.

## The fix

A diagnosis is not a fix. Hand the cause to the `dev-loop` skill: the failing test that reproduces it comes
first, then the smallest change that makes it pass, then a learning run to confirm the symptom is gone.

## Report

- **Symptom**, in numbers, with the command that reproduces it.
- **What was ruled out**, and by which check.
- **Cause**: say whether it is confirmed (a check fails, and passes with the fix) or suspected.
- **Proposed fix**, and what it will be tested by.
- **Not known**: anything you did not check. Do not present a suspicion as a finding.
