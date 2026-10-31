---
name: experiment-runner
description: >
  Runs training experiments on the RL framework and reports what they do and do not show: whether an agent
  learns, whether a change made it better or worse, how two agents or two hyperparameter values compare,
  across several seeds and without touching the maintainer's saved runs or the published benchmark report.
  Use this whenever the question is about learning rather than code: "does it still learn", "does this
  solve CartPole or Pendulum", "is this faster or more stable", "try a different learning rate", "compare
  PPO and DQN", "rerun the benchmark", and as the learning gate of the dev loop after any change to an
  update rule, a loss, a network, the trainer or a committed config.
---

# Experiment runner

A unit test says the code does what was written. Only a training run says the agent learns, and a training
run is noisy: the same code with another seed can solve the task or stall. This skill is how to get an
answer you can stand behind, and how to say plainly how far it goes.

Commands are written `python …`, meaning the virtual environment's interpreter: `.venv/Scripts/python` on
Windows, `.venv/bin/python` elsewhere. `<scratch>` is your session's scratch directory.

## 1. Write the question down first

Before running anything, state:

- **The claim** being tested, as something a number can contradict: "DQN with the committed config reaches
  a rolling average of 475 within 500 episodes on at least 4 of 5 seeds".
- **The metric**: evaluation reward, solve rate, or episodes to solve. Decide now, not after seeing which
  one looks best.
- **The baseline**: the same seeds on unchanged code or the unchanged config. A number with nothing beside
  it says little.
- **The budget**: episodes, seeds, and roughly how long it will take.

Change one thing at a time. Two changes and a better curve do not say which one helped.

## 2. Run it through the benchmark

`src/benchmark.py` already does seeded sweeps, evaluates each run, and applies the challenge's solve rule.
Give it its own output directory and report path, always:

```bash
python src/benchmark.py --challenge cartpole --agents dqn ppo --seeds 1 2 3 4 5 \
    --output-dir <scratch>/exp1 --report <scratch>/exp1/report.md --no-plot
```

- Without `--report`, it rewrites the tracked file `docs/benchmarks/<challenge>_results.md`. Without
  `--output-dir`, it writes under the repository. Neither is acceptable for an experiment.
- Without `--episodes` or `--steps`, each agent trains for its own config's budget (`training.total_steps`
  environment steps when set, else `training.num_episodes`); `--episodes N` or `--steps N` gives every agent
  N (not both). A run counts as solved only on its training average over `--window` episodes (default 100);
  the final evaluation is reported apart, and Held says whether the last window still averages at or above
  the threshold. `--eval-episodes` is how many greedy episodes score each run (default 10). A run with fewer
  episodes than the window can never solve, so short smoke runs report 0%.
- The default is two seeds (42 and 123). That is enough to see that something runs, not enough to compare.
  Use at least three for "it learns" and five for "A is better than B".
- **To try a hyperparameter**, copy `configs/` to `<scratch>/configs`, edit the copy, and pass
  `--config-dir <scratch>/configs`. The committed config changes only with the maintainer's agreement.
- **To compare against unchanged code**, check `develop` out beside the work without disturbing it:
  `git worktree add <scratch>/base develop`, run the same command from there with the working tree's
  virtual environment, then `git worktree remove <scratch>/base`.

Use `train.py` instead when you need what only it does (video, plots, a checkpoint to record from). It
writes to the `checkpoint_dir` and `log_dir` in the config and overwrites an agent's earlier checkpoints,
so copy the config to `<scratch>`, point both paths there, and pass `--video-dir` and `--plot-dir` there
too.

## 3. Budget the time

Time a short run first (one seed, a twentieth of the episodes) and scale up. Episodes get longer as a
CartPole agent improves, so expect the full run to cost more than the linear estimate.

- Up to a couple of minutes: run it in the foreground.
- Longer: run it in the background and carry on with work that does not depend on it. Anything longer than
  an hour writes its driver, per-seed curves and results under `~/rl-experiments/<item>/`, not the session
  scratch directory under `/tmp`: a reboot on 2026-10-30 wiped `/tmp`, taking a night's runs and the
  baseline curves they were to be compared with. Run it from a frozen worktree
  (`git worktree add --detach ~/rl-experiments/<item>/repo HEAD`) when you will edit `src/` meanwhile: each
  seed imports the code when it starts, so later seeds would otherwise train on half-written changes.
- More than about half an hour in total, or anything that would occupy the machine while the maintainer is
  using it: say what you want to run and why, and ask.

## 4. Read the result

`<output-dir>/benchmark_metrics.json` holds everything the report table is built from:

| Field | Meaning |
| :--- | :--- |
| `summary.<agent>.mean_eval_reward`, `std_eval_reward` | Mean and spread across seeds of the evaluation reward after training |
| `summary.<agent>.budget`, `budget_unit` | The budget each run of this agent got, in `steps` or `episodes` |
| `summary.<agent>.episodes` | Mean episodes each run of this agent actually trained for |
| `summary.<agent>.avg_solved_step`, `held_rate` | Mean training steps to the solve, over the seeds that solved; fraction of seeds whose last `window` episodes held the threshold |
| `summary.<agent>.solve_rate` | Fraction of seeds whose rolling training average reached the challenge's `solve_threshold` |
| `summary.<agent>.avg_solved_episode` | Mean first episode at which that happened, over the seeds that solved |
| `summary.<agent>.eval_threshold_rate` | Fraction of seeds whose final evaluation reached the threshold (not a solve) |
| `detailed_runs.<agent>[i]` | One seed: `eval_mean_reward`, `final_rolling_avg`, `solved_episode`, `solved_step`, `is_solved`, `held`, `eval_reached_threshold`, `total_episodes`, `total_steps`, `window`, `duration` |
| `metadata.episodes`, `steps`, `budgets_per_agent`, `episodes_per_agent` | The `--episodes` / `--steps` overrides (`null` without one), each agent's budget and unit, and the mean episodes each trained |

Read the per-seed rows, not only the mean. Things to look for:

- **The floor.** The `random` agent scores roughly 20 on CartPole. An agent near the floor has learned
  nothing, whatever its loss did. Include `random` in the sweep when the floor is not obvious.
- **Training against evaluation.** `final_rolling_avg` comes from training episodes, with exploration on.
  `eval_mean_reward` is the greedy policy. A large gap either way is worth a sentence.
- **Spread.** A difference between two agents that is smaller than the spread across seeds is not a
  difference. One seed that fails where four succeed is a finding about stability, not an outlier to drop.
- **Solved, then lost.** `solved_episode` records the first time the threshold was reached. Compare it
  with the final average to see whether the agent held it.
- **A suspiciously identical number.** Two seeds with exactly the same reward usually means the seed is
  not reaching something.

If the result is bad, do not start adjusting hyperparameters. Use the `training-doctor` skill to find out
why first.

## 5. Report

Give the numbers in a form the maintainer can check:

- the exact command, the branch or commit, and which config (committed, or what was changed)
- per agent: seeds run, solve rate, mean and spread of evaluation reward, episodes to solve
- the baseline beside it
- what this shows, and **what it does not**: seeds not run, challenges not run, agents on the same shared
  code that were not rerun

Say "did not solve in N episodes on k of n seeds", not "struggles". Say "not run" for anything not run.
Never choose which seeds to report after seeing them.

## Republishing benchmark results

`docs/benchmarks/*.md` and the plots under `plots/benchmarks/` are the project's published numbers.
Regenerate them only when asked or when the roadmap item says so, with each agent's config budget and the seeds
the maintainer names, by running the benchmark without `--report` and `--output-dir`. Show the diff of the
report before staging it.
