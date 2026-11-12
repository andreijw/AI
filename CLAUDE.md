# RL and Robot Navigation — Claude Code Instructions (CLAUDE.md)

## Tool Efficiency

- **Partial reads**: Use `Read` with `offset` and `limit`. Never load a full file when a section suffices.
- **Search first**: Use `Grep` to locate symbols, definitions, or patterns before opening any file. Use `Glob` for file
  discovery.
- **Precise edits**: Use `Edit` for targeted string replacements. Only use `Write` for brand-new files.
- **Parallel calls**: Issue independent tool calls in the same turn to cut round-trips.
- **No speculative reads**: Do not open a file "for context" unless you have a specific question to answer from it.

## Project Standards

Stack: **Python** distributions `rl-core`, `rl-navigation`, `rl-challenges` (`packages/<dist>/src/`) · CI on
**Python 3.12** · **Gymnasium ≥1.0**
· **NumPy** agents, with an optional **PyTorch** backend for DDPG, TD3 and SAC only (`backend: torch`; see
CONTRIBUTING.md "Framework Independence") · optional **Box2D** for LunarLander and BipedalWalker (`box2d` extra) ·
optional **onnx** and **onnxruntime** for the ONNX export (`onnx` extra) · **pytest** · **Ruff** (line length 100)
· **mypy**.

- **Coding standards** → **[CONTRIBUTING.md](CONTRIBUTING.md)** §1–3.
- **Feature/bug workflow** → **[CONTRIBUTING.md §4](CONTRIBUTING.md#4-feature--bug-workflow)**. Follow it in order for
  every feature or bug fix, starting with Step 0 (create a branch).
- **Logs & checkpoints policy** → [CONTRIBUTING.md §5](CONTRIBUTING.md#5-generated-artifacts-logs--checkpoints).
- **Directory layout and build/test/lint commands** → **[GEMINI.md](GEMINI.md)** §2 and §4. GEMINI.md is written for
  another assistant; translate its tool names with the mapping below.
- **Manual test procedures** → [TESTING.md](TESTING.md). **Architecture** → [ARCHITECTURE.md](ARCHITECTURE.md).

## Dev Team Workflow

You are the dev team; the maintainer makes the big decisions. The workflow in CONTRIBUTING.md §4 is run through the
skills in [.claude/skills/](.claude/skills/). Start every feature, tuning change or bug fix with `dev-loop`, which
calls the others in order.

**The maintainer decides three things:** which work to do (from your proposal), whether the reviewed plan is right,
and when to commit. Everything between those points is yours to decide, do and report. Also theirs: hyperparameters
in committed `configs/`, solve thresholds and published benchmark numbers, scope and roadmap order, new dependencies,
changes under `.github/`, and anything destructive, which includes the git-ignored training outputs in
`checkpoints/`, `logs/`, `videos/` and `plots/`.

| Skill | Use it |
| :--- | :--- |
| `dev-loop` | For any request to build, change, tune or fix, and for "what's next" |
| `architect` | On every plan before the maintainer sees it; for "where should this live" questions |
| `algorithm-crafter` | When the work is a new algorithm, network piece, loss or update rule |
| `test-crafter` | Before production code; whenever tests are added or coverage is questioned |
| `experiment-runner` | When the question is whether an agent learns, or which of two things learns better |
| `training-doctor` | When an agent does not learn, diverges or gives NaN, before touching a hyperparameter |
| `cleanup` | After implementing, before review |
| `pr-reviewer` | As the last gate before the hand-off; for any review request |
| `doc-sync` | Before the hand-off |

- `architect`, `cleanup` and `pr-reviewer` build on the **brooks-lint** plugin (pinned in
  [.claude/settings.json](.claude/settings.json)) and add this project's knowledge. Project review rules C1–C6 live
  in [.brooks-lint.yaml](.brooks-lint.yaml). Use its read-only modes only (`brooks-review`, `brooks-audit`,
  `brooks-test`, `brooks-debt`); never `brooks-sweep`.
- **Parallel work** (`dev-loop`, "Working in parallel"): spawn subagents, each in its own worktree, when work
  splits into independent parts; propose a workflow in the plan, and run one only when the maintainer says
  "use a workflow".
- Each worktree gets its own venv layered on `.venv`: `scripts/worktree_venv.sh <worktree>`, then run every tool as
  `<worktree>/.venv/bin/python`; never `pip install -e` a worktree into `.venv`.
- Run `architect` and `pr-reviewer` in a **fresh agent**: tell it to read the skill's `SKILL.md` and give it the plan
  or branch, since it cannot see the conversation.
- If the plugin is missing, the skills still run with the project checks; say so and point to
  [docs/ai-assistants.md → AI Assistant Skills](docs/ai-assistants.md#ai-assistant-skills).
- **Training runs never write into the maintainer's results.** Send experiment output to the scratch directory:
  `benchmark.py` with `--output-dir` and `--report`, `train.py` with a copied config whose `checkpoint_dir` and
  `log_dir` point there. The benchmark rewrites the tracked `docs/benchmarks/*.md` by default; it does so on
  purpose only as a PR's full rerun of a report the change made stale (`doc-sync`).
- **A smoke run is not evidence of learning.** Report which config, how many episodes and how many seeds; say
  "not verified" for anything not run.
- Shared tooling: gradient checks in `tests/helpers.py` (`assert_gradients_match`), agent internals for maths
  tests in `tests/agent_internals.py` and `tests/trainer_internals.py` (tests touch no private attribute
  elsewhere), the mutation runner in `.claude/skills/test-crafter/scripts/`, a Markdown line-length check in
  `.claude/skills/doc-sync/scripts/`, and the run queue in `.claude/skills/experiment-runner/scripts/run_queue.py`
  (submit, status, wait, cancel and prune long runs; it holds the 3-run cap across agents).
- When a review or a bug teaches a lesson worth keeping, put it where it will be applied: a rule in
  `.brooks-lint.yaml`, or a line in the relevant skill.

## Branch-First TDD

- **Before any edit for a feature or fix, be on a dedicated branch.** If on `develop` (or an unrelated branch), branch
  first:

  ```bash
  git checkout develop && git pull && git checkout -b feature/<kebab-case-name>   # or fix/<kebab-case-name>
  ```

- **Red → green → refactor.** Show a failing test (run it and confirm it fails for the right reason) before writing
  production code, then write the minimum code to pass, then refactor with the suite green. See CONTRIBUTING.md §4
  Steps 4–6.

## Environment

- First-time setup on Windows: `.\setup.ps1` (installs the Python of `.python-version` via winget, creates `.venv`,
  installs `requirements-dev.txt` with the video extras at the versions in `requirements-lock.txt`, runs the CI checks).
  Flags: `-Update`, `-NonInteractive`, `-SkipChecks`, `-InstallHooks`.
- **Run every Python tool from `.venv`**: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m ruff check .`,
  `.venv/Scripts/python -m mypy`. Never use bare `python` — on this machine it is the Microsoft Store stub.
  On Linux and macOS the interpreter is `.venv/bin/python`; the skills write `python` and mean whichever applies.
- If `.venv` is missing, ask the user to run `.\setup.ps1`; do not `pip install` into a global interpreter.
- **Every version is defined once** (`tests/test_versions.py`): third-party packages pinned exactly in
  `requirements-lock.txt`, which every install uses as constraints (`-c requirements-lock.txt`); floors in
  `packages/*/pyproject.toml`, one per package; Python in `.python-version`; markdownlint-cli in `ci.yml`'s
  `MARKDOWNLINT_CLI_VERSION`. PyTorch alone comes from its CUDA index, outside the lock.
- Linux and macOS: `./setup.sh --profile cpu|gpu`. **`cpu`** (the default) builds `.venv` with `requirements-dev.txt`
  plus the `video` and `box2d` extras: no PyTorch, and the interpreter for every tool. **`gpu`** builds `.venv-gpu` with
  the same and `rl-core[torch]` from PyTorch's CUDA index: use `.venv-gpu/bin/python` for the PyTorch backend's tests
  (`tests/test_torch_ddpg_agent.py`, skipped elsewhere) and for `train.py` runs with `backend: torch`. Both build Box2D
  (`box2d-py` from source with `swig`) before the extras. CI installs neither PyTorch nor Box2D, and their tests skip
  there (`needs_torch`, `needs_box2d`/`env_marks` in `tests/helpers.py`; maintainer's decisions, 2026-10-25 and
  2026-10-27). The C export tests (`needs_cc`) skip on Windows and where no `CC`, `cc`, `gcc` or `clang` is found; CI's
  Linux runner has gcc, and `test_ci_has_a_c_compiler` fails there if it ever does not. CI's test job installs the
  `onnx` extra (maintainer's decision, 2026-11-02), so the ONNX tests (`needs_onnx`) run there. `tests/conftest.py` runs
  them after every other test, and the tests that render through pygame (fixture `clear_sdl`) just before them:
  importing onnxruntime or starting SDL starts a native thread, and a fork after it (an async vector environment) may
  deadlock.

## Tool Mapping (GEMINI.md → Claude Code)

| GEMINI.md says | Use in Claude Code |
| :--- | :--- |
| `view_file` with `StartLine`/`EndLine` | `Read` with `offset`/`limit` |
| `replace_file_content` | `Edit` |
| `multi_replace_file_content` | One `Edit` per hunk (or `replace_all` for repeated strings) |
| `grep_search` | `Grep` for content, `Glob` for file names |
| Gemini model names (`gemini-*`) | See Model Selection below |

## Model Selection

| Model | ID | Use for |
| :--- | :--- | :--- |
| Haiku | `claude-haiku-4-5-20251001` | Writing/running tests, lint and type-check fixes, simple refactors |
| Sonnet | `claude-sonnet-5-5` | Features, code reviews, documentation |
| Opus | `claude-opus-5-5` | New algorithms (e.g. DQN), numerical-stability debugging, cross-module architecture |

## MCP Servers

Shared servers are defined in [.mcp.json](.mcp.json) (same `context7` definition and `CONTEXT7_API_KEY` as the
eternal-descent repo). Setup and troubleshooting:
[docs/ai-assistants.md → AI Assistant MCP Servers](docs/ai-assistants.md#ai-assistant-mcp-servers).

| Server | Use it when |
| :--- | :--- |
| `context7` | Version-correct library docs. Library IDs: Gymnasium `/websites/gymnasium_farama`, NumPy `/websites/numpy_doc_stable`, pytest `/pytest-dev/pytest`, Ruff `/websites/astral_sh_ruff`, mypy `/websites/mypy_readthedocs_io_en_stable`, Matplotlib `/websites/matplotlib_stable`, Stable-Baselines3 `/dlr-rm/stable-baselines3` (the reference for rl-zoo defaults and the SB3 loss/target conventions the configs cite), PyTorch `/websites/pytorch` (the torch backend), ONNX Runtime Python `/websites/onnxruntime_ai_api_python` (the ONNX export), ROS 2 `/ros2/ros2_documentation` (the hardware step). For anything else, call `resolve-library-id` first |
| `memory` (user scope) | Cross-session notes; not a source of truth for API details |

### MCP Safety Rules

- **Check Gymnasium APIs in Context7 rather than from memory.** Older Gym tutorials return `done` from `step()`;
  this repo uses the v1.0+ `terminated`/`truncated` split and `reset(seed=...)` returning `(obs, info)`.
- **If a server is disconnected or a tool returns an auth error, point the user to the
  [MCP setup steps](docs/ai-assistants.md#ai-assistant-mcp-servers) rather than working around it.** Context7 reports
  "Connected" even with a missing key; an "Invalid API key" tool result means `CONTEXT7_API_KEY` is not set.

## Git

- Branch off `develop` as `feature/<kebab-case-name>` (or `fix/<kebab-case-name>`); PRs target `develop`.
- Commit messages use conventional commits: `type(scope): summary` (e.g. `feat(agents): …`, `fix(env): …`, `chore: …`,
  `docs: …`).
- **Commits are SSH-signed by the maintainer. Never run `git commit`.** Stage the changed files with `git add` and
  print the exact commit command for the user to run. Once they say it is committed, push the branch yourself (SSH
  is refused from this machine's shell, so push over HTTPS with `gh`'s credentials) and open or update the PR.
- PR descriptions follow [.github/pull_request_template.md](.github/pull_request_template.md).

## Known Discrepancies

Treat the repository as the source of truth when docs disagree:

- CONTRIBUTING.md asks for ≥80% **branch** coverage; CI enforces 50% **line** coverage (`--cov-fail-under=50`, no
  `--cov-branch`).
- GEMINI.md §1 names Gemini models; use the Model Selection table above instead.
- Older PR descriptions and commits say `configs/pendulum_ppo.yaml` solves Pendulum-v1. It did under the 10-episode
  rule, then solved 0 of 5 under the 100-episode rule (2026-10-16), and since 2026-10-18 solves 5 of 5 with
  2048-step rollouts and 200,000 steps, holding on 4 of 5 (`docs/benchmarks/pendulum_results.md`; CHANGELOG.md,
  2026-10-18). Continuous Actor-Critic beats the random agent in evaluation since 2026-10-18 but does not solve
  under the 100-episode rule (CHANGELOG.md, 2026-10-18). CartPole PPO does solve CartPole-v1: 5 of 5 seeds within
  200,000 steps, holding on all 5 (TESTING.md `TR-3` and `docs/benchmarks/cartpole_results.md`).
- The `box2d` extra names `box2d-py` (pinned in `requirements-lock.txt`), not the `box2d==2.3.10` Gymnasium 1.4's own
  `box2d` extra pins: that
  has no aarch64 build, and `box2d-py` builds on the Jetson with `swig` and `--no-build-isolation` (docs/usage.md,
  LunarLander (Box2D)). Rendering uses `pygame-ce`, as Gymnasium does; never install plain `pygame` beside it.
- CONTRIBUTING.md §5 and TESTING.md `CKPT-*` describe `.npz` checkpoints; `Trainer` names them
  `agent_episode_<n>.pt` (NumPy archive content, resolved by `BaseAgent._resolve_save_path`).
