---
name: dev-loop
description: >
  Runs a feature or bug fix on the RL framework (rl_core and the rl_<challenge> packages) from proposal to hand-off
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
3. **Commit.** They commit once the work has passed every gate. Commits are SSH-signed by them, so never run
   `git commit`; you push once they say it is committed. They also merge; never run `gh pr merge`.

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

Read `docs/roadmap.md` (it holds only open work) and the newest entries in `CHANGELOG.md` for what just
landed. Offer two or three options with a recommendation and the reason, including what
each one touches, and say which of them can run together (see "Working in parallel"): by default propose a
batch of two or three independent items rather than one. Wait for the pick. When the maintainer leaves the
choice to you ("as you see fit"), pick the batch yourself and say why.

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
proposing them: play the real values through a typical run. Until 2026-10-16 `configs/cartpole_dqn.yaml`
decayed epsilon by 0.995 "per update" from 1.0 to a floor of 0.02, and `DQNAgent` applied that once per
`update()` call, which the trainer makes once per episode. Reaching the floor takes ln(0.02) / ln(0.995) ≈ 780
episodes, but the config trained for 500, so epsilon ended near 0.08 and the floor was never reached (the
config now decays it linearly per environment step with `exploration_steps`). Read as "per gradient
step" the same line would end exploration within the first 40 episodes. Whether either is right is a
judgement; not knowing which one the code does is a mistake, and the logged `epsilon` settles it. Do the
same for learning
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
| Markdown | `python .claude/skills/doc-sync/scripts/check_md_width.py` (line length only), and `npx markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules` where Node is installed; without Node, `pymarkdownlnt` from a scratch virtual environment (`pymarkdown --disable-rules md013,md024,md033,md041,md040,md029 scan <files>`, matching `.markdownlint.json`; skip `.claude/` files, whose YAML front matter it misreads and which CI's glob does not reach). Deleting a roadmap item must keep the blank line before the next heading: on 2026-11-09 a removal failed CI's MD022 | CI's markdownlint job would pass |

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
the setup steps in `docs/ai-assistants.md`. A plugin installed mid-session is not loaded until Claude Code restarts, for
this session and for the agents it spawns.

### 8. Doc sync

Use the `doc-sync` skill. If `.github/workflows/` changed, it runs `python scripts/sync_workflow_docs.py`.

### 9. Hand-off, then touchpoint 3

Stage the changed files by name with `git add`. Then report, starting with the PR's **quick summary** (its
bullets are in the `pr-reviewer` skill's Report section), updated for the fixes made after the review:

- what changed, in terms of what someone training an agent or extending the framework will notice
- the decisions you made on their behalf
- what was verified and how, and what was not (see below)
- which `TESTING.md` rows are worth running by hand, with the exact command and the number to look for
- the exact commit command (conventional commit subject; no `Co-Authored-By` trailer, which
  `.claude/settings.json` turns off); you push after they commit

Complete the roadmap item in the hand-off commit: remove it from `docs/roadmap.md` and add a condensed entry to
`CHANGELOG.md` under today's date (`## <date>`, created if it is not already the newest section; `Added`,
`Changed` or `Fixed`): what changed, the branch, and the numbers a reader would quote (config, seeds, result).
The PR then merges complete, and nothing has to be dated after the merge. Stage it with the rest and say so in the hand-off,
so the maintainer can unstage it if they do not agree. For work that spans several PRs, the item stays `[~]`
with a progress note until the last one. Anything still live in the item (a follow-up, a deferred idea) moves to
the open item it belongs to, or a new one, before the item is removed.

After they have committed: push the branch, then open the PR against `develop` with `gh pr create`, following
`.github/pull_request_template.md`, with the quick summary as the first lines of its Description and every
ticked box true. Watch `gh pr checks` until it finishes, and fix a red check without being asked.

A PR stacked on another branch gets no unit tests: CI runs on pull requests into `develop` only. Before the
maintainer merges a stacked PR, run CI on its branch (`gh workflow run CI --ref <branch>`) and wait for it; on
2026-11-10 #205 merged untested and its x86 golden failure surfaced only in #206.

When several PRs are open at once they share `CHANGELOG.md` and `docs/roadmap.md`. After each one merges,
merge `develop` into every other open branch yourself, resolve the conflicts keeping both sides' entries, and
check that each CHANGELOG entry and roadmap edit of both is still there before the maintainer merges it. On
2026-11-04 a resolution kept one side: #172's CHANGELOG entry was lost and its finished item stayed in the
roadmap until a follow-up PR restored them.

Then clean up what the work left behind (the maintainer's decision, 2026-11-04): the git worktrees made for
it (`git worktree remove <path>` once their branch is committed or abandoned; save an abandoned diff as a
patch first), local branches that were abandoned or have merged (`git branch -d`), and scratch or
`~/rl-experiments/<item>/` folders that nothing in `CHANGELOG.md` or the roadmap cites. Keep an experiment
folder a document points to. Never touch `checkpoints/`, `logs/`, `videos/` or `plots/` in the maintainer's
checkout, or anything this work did not create. What you cannot remove (a denied deletion), list in the
hand-off with the command for the maintainer to run.

## Working in parallel

The maintainer wants the loop to go faster where the work allows it (2026-11-04), and parallel is the
default whenever it is possible and a good idea (2026-11-06). The touchpoints do not change: one pick, one
plan approval and one commit per item, batched when several items run together.

**Run items together when they do not collide.** Before starting work, ask of the open items:

- **Do they touch different code?** Two items that edit the same module, config or test file wait for each
  other; `CHANGELOG.md` and `docs/roadmap.md` do not count (you write them at hand-off).
- **Do they need different resources?** Pair a compute-bound item (a multi-hour experiment) with a code-bound
  one (a new library module, a fix, docs), so the cores stay busy without the experiment slowing down. Two
  experiments together must still fit in 3 runs at once.
- **Does one need the other's result?** Then it waits, or is stacked on the other's branch (as the Box2D pin
  fix sat on #184, 2026-11-06), and its PR opens after the first merges.

Run in parallel when all three allow it; say in the proposal what runs together and why, and what waits and
why. Keep doing so inside each item: while an experiment runs, plan or implement the next item, run doc-sync,
or review; a small fix found on the way (a version mismatch, a stale comment) gets its own branch and PR
instead of waiting or riding along. Do not parallelise for its own sake: one small item, steps that need the
previous step's output, or work the maintainer must decide first stay sequential.

**Keep the planner on its item.** The agent that planned an item knows its probes and trade-offs; send the
architect's findings back to it for revision, and after approval have it implement the plan (SendMessage),
rather than briefing a fresh agent from scratch.

**Subagents are your call.** Spawn them when the work splits into parts that do not depend on each other:
the plan review and the PR review (always, as above), two or more roadmap items picked together (one agent
per item), or a search across many files. Not for a few tool calls you could make yourself. Brief each one
as a peer, since it sees only your prompt:

- the skill it follows (`algorithm-crafter`, `training-doctor`, `test-crafter` ...) and the roadmap item;
- its own worktree, branched from `develop` (`git worktree add -b <branch> ~/rl-experiments/<item>-repo
  origin/develop`), with `PYTHONPATH=$PWD/src` for every Python tool, since the editable install points at the
  main checkout;
- what it must not touch: `CHANGELOG.md` and `docs/roadmap.md` (you write them for every item at hand-off,
  so parallel branches do not conflict), the maintainer's `checkpoints/`, `logs/`, `videos/` and `plots/`, and
  `git commit`;
- no training run longer than a few minutes unless you scheduled it. Submit long runs to the run queue
  (`.claude/skills/experiment-runner/scripts/run_queue.py submit ...`, see `experiment-runner`) instead of
  asking for a go: the machine has 6 cores, and the queue enforces at most 3 runs at once across all agents,
  each with `OPENBLAS_NUM_THREADS=1`. Never stop a run with `pkill -f`; use `run_queue.py cancel`.

**The PR stage splits the same way.** Once the code passes the tests, run in parallel:

- `doc-sync` in a subagent, in the branch's worktree: give it the diff (`git diff origin/develop`) and the facts
  that changed (a key, a default, a result and its seeds). It edits the docs and reports what it changed and
  what it could not settle. Meanwhile you run the mutation, duplication and cleanup gates.
- Then the `pr-reviewer` agent on code and docs together. Beside it, for a PR that quotes results, a second
  agent that checks every number in `CHANGELOG.md`, `README.md`, `docs/results.md`, `TESTING.md` and config comments against
  the results files (`benchmark_metrics.json`, the metrics logs), and for a large test change a third that runs
  `brooks-lint:brooks-test` on the changed test files.

Review the edits yourself (`git diff`), fix what the reports find, and rerun the gates the fixes could affect.

Check what comes back before relying on it. A subagent's result is a claim, and its mistakes read as
confidently as its findings.

**Workflows only when the maintainer asks.** A workflow fans out many agents and costs far more tokens, and
Claude Code runs one only on an explicit request ("use a workflow"). When a batch would gain from one (three
or more independent items, or a review or experiment fan-out with a verify step for each finding or arm),
say so in the plan with the rough number of agents and why, and run it only once the maintainer says to.

## Status file for resuming

After a `/clear` or in a new session, "continue" must not mean digging through old transcripts. Keep
`~/rl-experiments/STATUS.md` (outside the repo, so nothing is committed; create it if it is missing)
current at every touchpoint and whenever an agent or a long run starts or ends. It is a pointer list, not
a log: replace lines, do not append.
**"Continue", or a new session, starts by reading `STATUS.md` and `run_queue.py status`.**

```markdown
# Status (updated 2026-11-07 14:05)

## Items
- <item>: stage <proposed|plan|review|approved|implementing|gates|hand-off|PR #n>;
  worktree <path>, branch <branch>
## Agents
- <agent name>: <what it is doing>, on <item>
## Jobs (from `run_queue.py status`)
- <job id> <name> <state>: results <file>
## Waiting on the maintainer
- <decision, with the item it blocks>
```

## Honesty in the hand-off

The maintainer acts on your report without re-checking it, so it has to be exact. Say "not verified" for
anything you did not run. Never describe a five-episode smoke run as evidence of learning, or one seed as a
result. When you quote a reward, say which config, how many episodes, how many seeds, and whether it came
from training or evaluation episodes. If a gate was skipped or a finding was left open, lead with that.
