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

Start with the log. Every ten episodes `Trainer` writes a record to
`logs/<challenge>/<agent>/metrics_<timestamp>.jsonl` with `avg_reward`, `avg_length` and, under `update`,
the mean of each metric `agent.update()` returned (losses, entropy, epsilon, mean Q-value). The first
non-finite value of each metric is logged as a `WARNING` naming the episode, the value becomes `null` in
the file, and its name is listed under `non_finite`. Read those before anything else: a loss that climbs,
an entropy that hits zero, or an epsilon that never moves usually names the rung to start on. That file
comes from `train.py`; `src/benchmark.py` passes a plain logger, so there the same record is only an INFO
line and the warning is what reaches the console.

The log is a ten-episode average, and a run shorter than ten episodes writes no record. To see single
updates, or to find the exact operation that produces a NaN, drive the agent from a scratch script instead
of adding prints to `src/`:

```python
import numpy as np
from rl_core.challenges import discover_challenges
from train import load_run_config, select_run_challenge

np.seterr(all="raise")  # turn the first overflow, divide-by-zero or invalid value into a traceback
discover_challenges()
config, agent_type = load_run_config("configs/cartpole_dqn.yaml")
challenge = select_run_challenge(config, agent_type)  # from environment.name
env = challenge.make_env(config["environment"], render_mode=None)
agent = challenge.make_agent(agent_type, config["agent"]["config"], env)
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

- **`dones` and `terminated`**: true on the last step only. `dones` is `terminated` *or* `truncated`; a
  bootstrapping agent that reads `dones` treats a time-limit cut as a true end (DQN reads `terminated`).
  Pendulum only ever ends by truncation; Acrobot and the two MountainCars end by termination at the goal or truncation at their limit (500, 200, 999).
- **Rewards**: their scale. CartPole pays 1 per step; Pendulum pays between about -16 and 0; Acrobot pays -1 per
  step until the goal, so a run that never reaches it sits flat at -500, and MountainCar the same at -200, where a
  random agent stays: flat at the floor there means exploration never found the flag, not a broken update.
  MountainCarContinuous charges -0.1·force² per step and pays +100 at the flag, so a run stuck just below 0 has
  learned to stand still (the local optimum), and its exploration noise alone costs about -50 per 999-step
  episode at OU sigma 0.5. A
  learning rate tuned for one challenge is wrong by an order of magnitude for another.
- **Actions**: the dtype and range the environment expects; a continuous action inside its bounds.
- **Observations**: finite, and of the scale the network was initialised for.
- **Wrappers**: observation noise, action noise and domain randomization are off unless the config asks
  for them, and are the first suspects when training and evaluation disagree.

## 4. Is the schedule right?

Walk the numbers with the config's real values:

- How many updates actually happen? An off-policy agent does nothing until its buffer holds
  `min_buffer_size`, and `max_updates_per_call` caps the work per episode.
- When does exploration end, compared with when the agent starts seeing long episodes?
- How often does the target network sync, measured in episodes? Stable-Baselines3 and rl-zoo count
  `target_update_interval` in environment steps; ours counts gradient steps, so rl-zoo's value times
  gradient steps per transition is ours (MountainCar: 600 × 0.5 = 300; the untranslated 600 cost about
  15 points of training average, 2026-10-23).
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
