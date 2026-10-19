---
name: pr-reviewer
description: >
  Reviews a finished change on the RL framework the way a careful second developer would, before the
  maintainer is asked to commit: does it meet the roadmap item, is the NumPy and Gymnasium code correct, is
  the reinforcement learning sound, are the tests real, will anything unintended be committed, are the docs
  true. Use this as the last gate on every feature or fix, whenever asked to review a diff, branch or pull
  request in this repository, and when asked whether a change is safe or whether a result can be trusted.
  It runs brooks-lint's PR review for the general rubric and adds the checks specific to this project.
---

# PR reviewer

The maintainer commits on the strength of this review, so its job is to find what would embarrass the
change after it merges. It is read-only: it reports, and the author fixes.

## How to run it

Run it in a fresh agent. Whoever wrote the code knows what it was meant to do and reads it that way; a
reviewer who starts from the diff sees what it does. Give the agent the branch, the base (`develop`), the
roadmap item's name, and this file's path.

1. Read the roadmap item in `docs/roadmap.md` and the diff (`git diff <base>...HEAD`, plus anything
   unstaged).
2. Load `brooks-lint:brooks-review` and apply it to the diff. It brings the general rubric and reads
   `.brooks-lint.yaml`, so the project risks C1 to C6 (framework leak, unowned randomness, behaviour by
   identity, reinvented helper, unlocked behaviour, silent numerical failure) are judged there. If the
   plugin is not installed, say so and use `/code-review` for that part.
3. Make the passes below. These are what the plugin cannot know.
4. Report in the format at the end.

Read the code around each hunk, not only the changed lines. Most real findings come from how a change meets
what was already there. A base-class edit reaches every subclass; a `Trainer` edit reaches every agent on
both challenges.

## Pass 1: does it do what was asked

Take the acceptance criteria one at a time and find the code and the test for each. A criterion with no
test, or a behaviour with no criterion, is a finding. Scope that crept in without a roadmap line is also a
finding, however good the code.

## Pass 2: correctness in NumPy and Gymnasium

- **Shapes.** Follow each array through the change. `(n,)` minus `(n, 1)` does not fail: it broadcasts to
  `(n, n)` and the mean of that is a plausible-looking wrong number. Check the axis of every `sum`, `mean`
  and `max`.
- **Gradients.** Sign (ascent on the objective is descent on its negative), scale (a loss averaged over the
  batch against one summed, with the same learning rate), and what is held constant: TD targets,
  advantages and old log-probabilities are data, not functions of the weights being updated.
- **Episode ends.** The bootstrap term is masked on a terminal step. `terminated` and `truncated` are not
  the same thing for a method that bootstraps; the trainer's batch only carries a combined `dones`.
- **Off-by-one.** Returns and advantages line up with the step they belong to; the last step's next value
  is handled deliberately.
- **Aliasing.** A target network made with `=` instead of `.copy()` is the online network. A buffer that
  hands out views, an `update` that mutates the batch it was given, and `+=` on an array someone else holds
  are the same bug.
- **Dtypes.** Integer arrays in arithmetic, float32 mixed with float64, a discrete action returned as a
  NumPy scalar where the environment wants an `int`, a continuous action outside the `Box` bounds or in the
  wrong dtype.
- **Numerics.** Every `log`, division, `exp` and square root has an answer for zero, for a huge value and
  for a standard deviation of zero.
- **Randomness.** Draws come from the object's own generators, each a named stream of the config seed
  (`rl_core.seeding.derive_rng`). No `np.random.seed`, no bare `np.random.*`, and no `default_rng(seed)` in an
  agent.
- **Gymnasium.** `reset` returns `(obs, info)`; `step` returns five values. The environment is closed on
  every path, including the one where training raises. Vector environments return batches and reset
  themselves. Check signatures in Context7, not from memory.
- **Python.** Mutable default arguments, a config dict mutated in place, a bare `except` that swallows the
  real error, and a `config.get("lerning_rate", 0.001)` whose default hides the typo.

## Pass 3: is the learning sound

The questions a training run, not an interpreter, would raise:

- **Walk the numbers.** Play the schedule through with the committed values. How many updates until
  exploration ends, the buffer is warm, the target first syncs? Does that fit inside `num_episodes`?
- **Every agent on the shared path.** A change to `BaseAgent`, an actor-critic base, `Trainer` or a
  wrapper is a change to every agent that uses it, discrete and continuous, CartPole and Pendulum. Were
  the others run, or only the one the change was written for?
- **On-policy and off-policy discipline.** PPO's ratio uses the log-probabilities recorded when the data
  was collected. On-policy agents do not learn from stale data. Off-policy agents do not learn before the
  buffer can fill a batch.
- **Training against evaluation.** With `training=False` the agent acts greedily (or takes the mean
  action), and evaluation episodes do not update weights, fill a buffer, decay epsilon or advance a
  schedule.
- **Interruptions.** A one-step episode. An episode cut by `max_steps_per_episode`. A run resumed from a
  checkpoint: are the step counters, epsilon and optimizer moments restored, or does it silently start
  cold with trained weights?
- **A fair comparison.** A benchmark compares agents on the same seeds, episode budget and evaluation
  rule. The reported reward is the environment's, not a shaped or noised one. A threshold is met by the
  rolling average the challenge defines, not by one good episode.
- **Visibility.** Every quantity that can go wrong is in the metrics `update` returns (loss, entropy,
  epsilon, a gradient norm): `Trainer` averages those into the metrics log every ten episodes and warns
  at the first non-finite value. A NaN is loud: it is not clipped into a number that looks fine.
- **Stated intent.** A committed hyperparameter, a `solve_threshold` or a published benchmark figure that
  moved without a roadmap line needs the maintainer, not a silent approval.
- **Evidence.** For each claim that something learns, or learns better: which config, how many episodes,
  how many seeds, training or evaluation reward. A claim with no run behind it is a finding. If the change
  touches an update rule, a loss, a network, the trainer or a committed config and no learning run is
  reported, ask for one (the `experiment-runner` skill) or list it under **Not checked**.

## Pass 4: tests

- Would a test fail if each new rule were broken? The PR description should list the mutations tried;
  spot-check one yourself if it does not.
- A hand-written backward pass has a gradient check (`assert_gradients_match` in `tests/helpers.py`), or
  a value worked out by hand. "The weights changed" is not a test of a gradient.
- Rules are tested in the unit layer and wiring through `Trainer` or `run_main`, not the other way round.
- Seeds are pinned, files go under `tmp_path`, environments are closed, and nothing trains to convergence.
- A new agent is in `ALL_AGENTS` in `tests/test_agent_harness.py`.
- A changed or deleted existing test comes with a reason.

## Pass 5: what gets committed

- No training output: nothing from `logs/`, `checkpoints/`, `videos/`, `plots/`, `test_logs/` or
  `test_checkpoints/`, and no stray `.npz`, `.pt`, `.mp4` or `.png` elsewhere.
- `docs/benchmarks/*.md` changed only if the roadmap item says to republish results, and then from a full
  run, not a five-episode one.
- No import of PyTorch, TensorFlow, JAX or another learning framework; no new dependency in
  `pyproject.toml` that the plan did not announce.
- A config key added or renamed is updated in every `configs/*.yaml` that uses the agent, and old
  checkpoints still load or the PR says they do not.
- No absolute paths from someone's machine.
- A change under `.github/workflows/` comes with the README list refreshed by
  `scripts/sync_workflow_docs.py`.

**Security.** Run `/security-review` as well when the diff touches any of: loading a checkpoint or config
(a file is untrusted input; `allow_pickle=True`, `pickle`, `yaml.load` without the safe loader and `eval`
are findings), the `subprocess` call in `rl_core/utils/video.py` or any new one, paths built from config or
command-line values, dependencies, or `.github/` (permissions, unpinned third-party actions, secrets in
logs, script injection from PR titles or branch names). Most algorithm changes touch none of these; say
"not applicable" and move on.

## Pass 6: docs and the PR itself

- The roadmap item matches what was built, including what was dropped.
- `TESTING.md` has a row, with a runnable command and an expected number, for anything a person should now
  run.
- README usage, ARCHITECTURE.md (the class diagram, the extension steps) and config comments still
  describe the code.
- The PR description follows `.github/pull_request_template.md` and every ticked box is true.

## Report

**Verdict** first: *Approve*, or *Changes needed* with the count of Critical and Warning findings.

Then findings, most severe first. Each one: severity, `path:line`, then **Symptom** → **Source** →
**Consequence** → **Remedy**. Make the consequence concrete: the input or sequence that goes wrong and what
the person training the agent sees (a crash, a NaN, a curve that plateaus, a number that is quietly wrong).
Keep style opinions out; Ruff owns style.

Finish with **Not checked**: what you could not judge from the code, above all whether the agent still
learns if you did not run it. Never report a pass you did not make.
