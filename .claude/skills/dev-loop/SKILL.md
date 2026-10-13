---
name: dev-loop
description: >
  Runs a feature or bug fix on the RL framework (rl_core, rl_cartpole, rl_pendulum) from proposal to hand-off
  as the project's dev team: propose the work, plan it, get the plan reviewed, write tests first, implement,
  pass every gate, sync the docs, and hand over for the maintainer's commit. Use this whenever the user asks
  to build, add, change, tune or fix anything in the agents, environments, trainer, benchmark or scripts,
  asks "what's next" or "what should we work on", picks a roadmap item, or reports a bug, a crash, a NaN or
  an agent that stopped learning, even when they do not mention a workflow. It is the executable form of
  CONTRIBUTING.md §4.
---

# Dev loop

The maintainer wants to make the big decisions and nothing else. This loop is how routine work gets done
without them: it has exactly three points where a person is asked, and everything between those points is
yours to decide, do and report.

## The three touchpoints

1. **Pick the work.** They choose what to build, usually from options you propose.
2. **Approve the plan.** They see the roadmap item and the plan together, after the plan has been reviewed
   and refined, not before.
3. **Commit.** They commit and push once the work has passed every gate. Commits are SSH-signed by them, so
   never run `git commit` or `git push`. They also merge; never run `gh pr merge`.

Between touchpoints, do not stop to ask about implementation, test or refactoring choices. Make the call,
and say what you chose in the hand-off. Stop early only for something that is really theirs:

- a hyperparameter in a committed `configs/*.yaml`, a solve threshold, or any number in
  `docs/benchmarks/` that a reader would quote
- a change of scope, or the order of roadmap work
- a new dependency (above all one that weakens the pure-NumPy rule), or anything under `.github/`
- anything destructive or hard to undo, which here includes the contents of `checkpoints/`, `logs/`,
  `videos/` and `plots/`: they are git-ignored, so a deleted or overwritten training run is gone

If one of those comes up mid-task, carry on with everything that does not depend on the answer.

## Running Python

Every command below is written `python -m …`. `python` means the virtual environment's interpreter:
`.venv/Scripts/python` on Windows, `.venv/bin/python` on Linux and macOS. Never the bare command, and never
`pip install` outside `.venv`. If `.venv` is missing, ask the maintainer to run `.\setup.ps1`.

## The loop

### 1. Propose (skip if they already named the work)

Read `docs/roadmap.md` (the unchecked items under **Upcoming Milestones**) and any follow-up notes on
recently completed items. Offer two or three options with a recommendation and the reason, including what
each one touches. Wait for the pick.

### 2. Roadmap item

Branch from an up-to-date `develop` as `feature/<kebab-case-name>` or `fix/<kebab-case-name>`. Find or add
the item in `docs/roadmap.md`: description, acceptance criteria, known edge cases, out of scope. Mark it
`[~]`.

### 3. Plan

Explore before designing. Search for what already does part of the job: the base classes in
`src/rl_core/agents/` (`BaseAgent`, `ActorCriticBase`, `ContinuousActorCriticBase`), `ReplayBuffer`, the
`make_env` factories, `rl_core/utils/`, and the setup helpers in `train.py`. Reusing them is the default.
Check Gymnasium and NumPy signatures in Context7, not from memory.

For anything that changes a hyperparameter, a schedule or an update rule, **walk the numbers** before
proposing them: play the real values through a typical run. `configs/cartpole_dqn.yaml` decays epsilon by
0.995 per update from 1.0 to a floor of 0.02. That is ln(0.02) / ln(0.995) ≈ 780 updates, and at one update
per environment step with early episodes lasting about 20 steps, exploration is over within the first 40 of
500 episodes. Whether that is right is a judgement; not knowing it is a mistake. Do the same for learning
rate against the number of updates, target-sync period against episode length, buffer warm-up against
batch size, and a Polyak rate against how many updates the target lags by (ln 2 / τ).

For a new algorithm, network piece or update rule, use the `algorithm-crafter` skill for this step: it
pins down the variant, the array shapes and the hand-derived gradients before any code exists.

For a bug report of the form "it does not learn", "it diverges" or "it gives NaN", use the
`training-doctor` skill first. The plan starts from its diagnosis, not from a guess.

The plan covers: why, the approach, files to change, what is reused, the tests that will lock it in, and how
it will be verified. Call out a config schema change, a checkpoint format change, and anything that touches
numerical stability. Large work is split into small PRs that each leave `develop` working, as the DQN work
was (`ReplayBuffer`, then the agent, then the config and CLI integration).

### 4. Plan review, then touchpoint 2

Have the plan reviewed by a fresh pair of eyes before the maintainer sees it: spawn a general-purpose agent,
give it the plan text and the roadmap item (it cannot see this conversation), and tell it to read and follow
`.claude/skills/architect/SKILL.md`. That skill runs brooks-lint's architecture audit and then applies this
project's own lens.

Revise the plan, then present it with the roadmap item and a short note of what the review changed. Wait
for approval.

### 5. Tests first

Use the `test-crafter` skill. Run the new tests and confirm they fail for the right reason: an assertion or
a missing feature, not an import or syntax error. For a bug, the first test reproduces the bug.

### 6. Implement

Write the minimum code that makes the tests pass. Follow CONTRIBUTING.md §1 to §3.

### 7. Gates

Run them in this order, fix what they find, and rerun whatever a fix could have affected.

| Gate | How | Passes when |
| :--- | :--- | :--- |
| Lint | `python -m ruff check .` | no findings |
| Format | `python -m ruff format --check .` | exit code 0 |
| Types | `python -m mypy src` | "Success: no issues found" |
| Tests | `python -m pytest` | 0 failed, and coverage of the files you touched did not fall |
| Smoke run | each entry point the change can reach, for a few episodes, writing to a scratch directory | exit code 0 and every logged metric is finite |
| Security | `python -m bandit -r src/ -ll` | no medium or high findings |
| Duplication | `python -m pylint src --disable=all --enable=duplicate-code --min-similarity-lines=20 --ignore-imports=yes --ignore-signatures=yes` | nothing new |
| Mutation checks | `test-crafter` skill (`scripts/mutation_check.py`) | every new rule has a test that fails when the rule is broken |
| Cleanup | `cleanup` skill, then rerun the gates above | no duplicated rule, no leftovers |
| Review | fresh agent following `.claude/skills/pr-reviewer/SKILL.md` on the diff against `develop` | verdict is Approve; no Critical findings left |
| Test quality | same agent, `brooks-lint:brooks-test` on the changed test files | no Critical findings left |
| Learning run | `experiment-runner` skill, when the change touches an update rule, a loss, a network, the trainer, a wrapper or a committed config | the claim written down beforehand holds across the seeds run |
| Markdown | `python .claude/skills/doc-sync/scripts/check_md_width.py` (line length only), and `npx markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules` where Node is installed | CI's markdownlint job would pass |

**Smoke runs must not touch the maintainer's results.** `train.py` writes to the `checkpoint_dir` and
`log_dir` named in the config and overwrites an agent's earlier checkpoints, and `src/benchmark.py` rewrites
the tracked report `docs/benchmarks/<challenge>_results.md` by default. Copy the config to your scratch
directory with those two paths changed and run that with `--num-episodes`; give the benchmark
`--output-dir` and `--report` in the scratch directory. After the gates, `git status` must show no change
under `docs/benchmarks/`.

**A smoke run proves the code runs, not that the agent learns.** A change to an update rule, a loss, a
network, the trainer or a committed config is only verified by full runs of the committed config that
reach the challenge's `solve_threshold` (475 for CartPole, -200 for Pendulum, as a rolling average). That
is the learning-run gate, and the `experiment-runner` skill says how to run it without touching saved
results, how many seeds to use, and when a run is long enough to ask first. If it was not run, the
hand-off says so and names the `TESTING.md` row for the maintainer to run. If it fails, go to
`training-doctor`, not to the config.

The review runs in a separate agent because a reviewer who did not write the code notices more. Give it a
self-contained prompt: the branch, the base (`develop`), and the roadmap item. The reviewer only reports;
you make the fixes, so each one goes back through the tests. Never use `brooks-sweep` for the same reason.
Run `/security-review` as well when the diff touches checkpoint loading, config parsing, the `subprocess`
call in `rl_core/utils/video.py`, dependencies in `pyproject.toml`, or `.github/`.

Findings: fix every Critical. Fix Warnings unless there is a reason not to, and write that reason in the PR
description. Suggestions are optional. After three rounds, stop looping and report what is left.

`architect`, `cleanup` and `pr-reviewer` build on the brooks-lint plugin. If it is not installed they still
run with the project's own checks; say in the hand-off that the general rubric was skipped, and point to
the setup steps in the README. A plugin installed mid-session is not loaded until Claude Code restarts, for
this session and for the agents it spawns.

### 8. Doc sync

Use the `doc-sync` skill. If `.github/workflows/` changed, it runs `python scripts/sync_workflow_docs.py`.

### 9. Hand-off, then touchpoint 3

Stage the changed files by name with `git add`. Then report:

- what changed, in terms of what someone training an agent or extending the framework will notice
- the decisions you made on their behalf
- what was verified and how, and what was not (see below)
- which `TESTING.md` rows are worth running by hand, with the exact command and the number to look for
- the exact commit command (conventional commit subject, plus the `Co-Authored-By` trailer) and the push
  command

Mark the roadmap item `[x]` only when they confirm the work is right. If they confirm before committing,
make that edit first so it lands in the same commit.

After they have committed and pushed: open the PR against `develop` with `gh pr create`, following
`.github/pull_request_template.md`, with every ticked box true. Watch `gh pr checks` until it finishes, and
fix a red check without being asked.

## Honesty in the hand-off

The maintainer acts on your report without re-checking it, so it has to be exact. Say "not verified" for
anything you did not run. Never describe a five-episode smoke run as evidence of learning, or one seed as a
result. When you quote a reward, say which config, how many episodes, how many seeds, and whether it came
from training or evaluation episodes. If a gate was skipped or a finding was left open, lead with that.
