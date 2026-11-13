---
name: doc-sync
description: >
  Brings the RL framework's documentation back in line with the code after a change: roadmap, TESTING.md,
  README, ARCHITECTURE.md, the assistant instruction files, config comments and docstrings. Use this at the
  end of every feature, fix or refactor before handing over for commit, whenever a hyperparameter, command,
  flag, file path, class, workflow or behaviour that a document mentions has changed, and when asked to
  update, sync or check the docs. It also finds documents that the change made false without touching them.
---

# Doc sync

Documentation here is read by the maintainer and by other AI assistants, and both act on it without
checking. A stale command or number in a doc becomes a wrong decision later. The work is less about writing
new text than about finding every sentence the change made untrue.

This skill can run in a subagent while the dev loop's other gates run (`dev-loop`, "Working in parallel"):
it then works in the branch's worktree, never commits, and reports every file it edited.

## 1. Work out what changed in fact

From `git diff develop...HEAD` (and unstaged work), write down the facts that moved:

- values: hyperparameters, thresholds, defaults, episode counts, coverage figures
- names: files, packages, classes, config keys, commands, script flags, output paths, workflow jobs
- behaviour: what someone training an agent or extending the framework now experiences
- process: steps, gates, who decides what

## 2. Find every place that states the old fact

Search the docs for the old value and the old name, not just the files you expect:

```
Grep  pattern: <old value or name>   glob: *.md
Grep  pattern: <old value or name>   glob: *.yaml
```

Also search for the feature's name, since prose often describes a thing without quoting it ("saves a
checkpoint every hundred episodes"). A moved module is the usual offender: paths are quoted in the roadmap,
CONTRIBUTING.md, GEMINI.md and ARCHITECTURE.md.

## 3. Update what each document owns

| Document | Owns | Update when |
| :--- | :--- | :--- |
| `docs/roadmap.md` | Open and in-progress items: description, acceptance criteria, status, follow-ups | Always. Record what was built and what was dropped while the item is open |
| `CHANGELOG.md` | One condensed entry per completed item, newest first, under the date it was handed off | When an item completes, in its hand-off commit; the item leaves the roadmap at the same time |
| `packages/<dist>/CHANGELOG.md` | The distribution's release notes: the open section (`## [Unreleased]`, or the undated `## [<version>]` of a bumped but unreleased version), then one `## [<version>] - <date>` section per version (`scripts/release.py check` and `tests/test_packages.py` need the current version's) | The distribution's code, requirements or behaviour changed (a line under the open section: `## [Unreleased]`, or the undated `## [<version>]` of a bumped but unreleased version), or its version was bumped (the section) |
| `packages/<dist>/README.md` | What the distribution is, how to install it, its extras (the wheel's long description) | An extra, a requirement, an install command or what the distribution holds changed |
| `TESTING.md` | Manual checks a person runs, as rows with an ID (`ENV-`, `TR-`, `VID-`, `OUT-`, `BM-`, `CKPT-`, `LOG-`, `CI-`), a command and an expected result | A command, flag, output path or expected behaviour was added or changed |
| `README.md` | The front page: what the repo is, what works and what does not, the challenge table (one row per registered challenge, named by its environment id, with its report), the robotics ladder, the quick start, the documentation map | A challenge was added, its result changed, a headline feature or the quick-start command changed, or the ladder moved |
| `docs/external-challenges.md` | How another project (a game) registers a challenge through the entry points and trains here, with code run as written | `BenchmarkChallenge`, `BaseEnvWrapper`, `env_from_config`, `register_challenge`, the entry-point group or how `train.py` and `benchmark.py` select a challenge changed |
| `docs/releasing.md` | Installing a release, making one (the order of tags), a breaking `rl-core` change, the repository settings for releases | `scripts/release.py`, `.github/workflows/release.yml`, a distribution's requirements or the release order changed |
| `requirements-lock.txt` | The exact version of every third-party package the development environments, CI and releases install | A package was added to or dropped from an install, or a pin was bumped; a pin must satisfy its distribution's floor (`tests/test_versions.py`) |
| `docs/usage.md` | Install (including Box2D), every train, record, export, benchmark and robustness command, configuration, curriculum, environment settings and checkpoints, the development commands | A command, flag, prerequisite, config key or output path changed |
| `docs/results.md` | The full table of which agent solves which challenge, with the caveats behind each number | A committed config, a measured result or a threshold changed |
| `docs/navigation.md` | The navigation environments, the simulator and the state-estimation library with its measurements | `rl_navigation` changed, or its numbers did |
| `docs/ai-assistants.md` | MCP server setup, skill setup and skill tooling | A skill, a skill tool, `.mcp.json` or `.claude/settings.json` changed |
| `docs/agents.md` | How each agent works: its idea, its update as the code computes it, this repo's options, what was measured, lessons | An agent was added, or its update rule, options, committed config or measured result changed |
| `ARCHITECTURE.md` | Package boundaries, the agent class diagram, the training sequence, the artifact tree, extension steps | A class, a base-class method in the diagram, a boundary or an output path changed |
| `CONTRIBUTING.md` | Standards (§1 to §3), the workflow (§4), the artifact policy (§5) | A rule or a step changed |
| `CLAUDE.md` | Claude-specific instructions, the skills table, known discrepancies | A skill, tool, git rule or discrepancy changed |
| `GEMINI.md` | Directory layout (§2), build and test commands (§4) | A folder, file or command changed |
| `configs/*.yaml` comments | What each key means and its unit ("per update", "episodes") | A key was added, renamed, or changed meaning |
| Docstrings | `Args`, `Returns` and `Raises` of public classes and functions | A signature or behaviour changed |
| `.brooks-lint.yaml` | Project review rules C1 onwards | A review found a class of mistake worth a standing rule |
| `.claude/skills/*/SKILL.md` | How the dev team works | A lesson was learned the hard way, or a helper named there moved |

Two things are generated, so run the generator and never edit by hand:

- **The workflow list in `README.md`**, between `<!-- WORKFLOWS:START -->` and `<!-- WORKFLOWS:END -->`:
  run `python scripts/sync_workflow_docs.py` (the virtual environment's Python) when `.github/workflows/`
  changed.
- **`docs/benchmarks/<challenge>_results.md`**: written by `benchmark.py`. Regenerate it in the same PR
  whenever the change alters what one of its rows measures: a committed config of an agent the report lists, or
  that agent's training (the maintainer's decision, 2026-11-04: every document a PR makes stale is updated in
  it). Rerun the report's own agents and seeds (its header names them), on the branch's code, as a full run
  with the default `--report`, never a smoke run; say in the hand-off how long it took, and ask first if it
  would take more than half an hour. If a smoke run rewrote a report, restore it.

**Diagrams.** Update every diagram the change makes false (ARCHITECTURE.md's package flowchart, agent class
diagram and training sequence, and any other Mermaid block: `Grep` for `` ```mermaid ``): a renamed or added
class, a base-class method, a new package, a changed call order. Add a diagram where one would explain the
change better than prose: a new module whose parts call each other (a pipeline, a filter's predict and update
loop, a SLAM front end and back end), a new class hierarchy, a state machine, a data flow across packages. A
one-function fix or a docs-only change needs none; say in the report which you added or why none was needed.
Write diagrams as Mermaid in the Markdown (GitHub renders it; it diffs as text and cannot go stale silently),
next to the prose they explain, not as images: a PNG is used only for a real output, such as a frame of a
video. Keep each to what a reader needs (about a dozen nodes), with the same names the code uses.

**Work items no longer needed** go in the same PR. A roadmap entry the change answers (a parked idea now
measured, a follow-up now done, a note that a later item supersedes) is removed or rewritten, with what
was learned condensed into `CHANGELOG.md`, so nothing in the roadmap still asks a settled question. Then
list what the work left on the machine (step 9 of the `dev-loop` skill clears it).

Prefer pointing at the source over restating it. "See `configs/cartpole_dqn.yaml`" cannot go stale; a
copied table of hyperparameters will.

When an item completes, it leaves `docs/roadmap.md` and a condensed entry goes into `CHANGELOG.md` under the
hand-off date (`## <date>`, created if it is not already the newest section): what changed, the branch (the PR
number is not known yet), and the measured numbers other documents quote. The pull request carries its own entry,
so it merges complete and nothing has to be dated afterwards. Before removing the item, move anything still live
in it (follow-ups, deferred ideas) to the open item it belongs to, or to a new one. An item delivered over several
PRs stays `[~]` with a progress note until the last one.
Other documents cite completed work as `(CHANGELOG.md, <date>)`, never by roadmap item name.

When one document contradicts another, the repository is the source of truth: fix the document, and if it
cannot be fixed in this change, add it to **Known Discrepancies** in `CLAUDE.md`.

## 4. Check the edits landed

After editing, `Grep` each file for a phrase you just wrote, and check again immediately before staging: an
editor with the file open can write an old buffer back over the change.

Then confirm the old facts are gone: the searches from step 2 should now return nothing but intended
history (changelog entries keep their original wording).

CI lints every Markdown file with `markdownlint-cli` 0.39.0 and `.markdownlint.json`: prose lines are at
most 120 characters; tables and code blocks are exempt. Run the real linter where Node is installed. Where
it is not, the script beside this skill checks the line-length rule only, which is the one that usually
fails:

```bash
npx markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules
python .claude/skills/doc-sync/scripts/check_md_width.py          # the virtual environment's Python
```

Also run `python scripts/check_doc_links.py`: it fails on any relative link, cross-file anchor or file path in the
Markdown that no longer resolves (a renamed heading, a moved file). CI runs it on every pull request, and
`tests/test_doc_links.py` runs it too.

## 5. Report

List the documents changed and the one-line reason for each, the documents checked and left alone, and any
contradiction you found but did not resolve.

## Style

Write for someone arriving cold: full sentences, the reason alongside the rule, numbers with units. Every
command in a doc is one a reader can paste and run. Match the document's existing structure rather than
adding a new section for each change. Dates are absolute (`2026-10-12`), never "yesterday".
