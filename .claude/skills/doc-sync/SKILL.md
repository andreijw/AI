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
| `docs/roadmap.md` | Each item's description, acceptance criteria, status, follow-ups | Always. Record what was built and what was dropped. Mark `[x]` only after the maintainer confirms |
| `TESTING.md` | Manual checks a person runs, as rows with an ID (`ENV-`, `TR-`, `VID-`, `OUT-`, `BM-`, `CKPT-`, `LOG-`, `CI-`), a command and an expected result | A command, flag, output path or expected behaviour was added or changed |
| `README.md` | Features, project structure, install, usage commands, configuration, architecture summary | A headline feature, a command, a prerequisite or the folder layout changed |
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
- **`docs/benchmarks/<challenge>_results.md`**: written by `src/benchmark.py`. Regenerate it only when the
  roadmap item calls for republished results, from a full run the maintainer has agreed to. If a smoke run
  rewrote it, restore it.

Prefer pointing at the source over restating it. "See `configs/cartpole_dqn.yaml`" cannot go stale; a
copied table of hyperparameters will.

When an item is confirmed complete, the roadmap keeps one line for it under **Completed Releases &
Milestones** and the full description comes out of the upcoming section: an item should not appear both
folded and in full. Before removing the full text, move anything still live in it (follow-ups, deferred
ideas) to the open item it belongs to, or to a new one.

When one document contradicts another, the repository is the source of truth: fix the document, and if it
cannot be fixed in this change, add it to **Known Discrepancies** in `CLAUDE.md`.

## 4. Check the edits landed

After editing, `Grep` each file for a phrase you just wrote, and check again immediately before staging: an
editor with the file open can write an old buffer back over the change.

Then confirm the old facts are gone: the searches from step 2 should now return nothing but intended
history (completed roadmap items keep their original wording).

CI lints every Markdown file with `markdownlint-cli` 0.39.0 and `.markdownlint.json`: prose lines are at
most 120 characters; tables and code blocks are exempt. Run the real linter where Node is installed. Where
it is not, the script beside this skill checks the line-length rule only, which is the one that usually
fails:

```bash
npx markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules
python .claude/skills/doc-sync/scripts/check_md_width.py          # the virtual environment's Python
```

## 5. Report

List the documents changed and the one-line reason for each, the documents checked and left alone, and any
contradiction you found but did not resolve.

## Style

Write for someone arriving cold: full sentences, the reason alongside the rule, numbers with units. Every
command in a doc is one a reader can paste and run. Match the document's existing structure rather than
adding a new section for each change. Dates are absolute (`2026-10-12`), never "yesterday".
