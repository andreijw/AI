# RL CartPole — Claude Code Instructions (CLAUDE.md)

## Tool Efficiency

- **Partial reads**: Use `Read` with `offset` and `limit`. Never load a full file when a section suffices.
- **Search first**: Use `Grep` to locate symbols, definitions, or patterns before opening any file. Use `Glob` for file
  discovery.
- **Precise edits**: Use `Edit` for targeted string replacements. Only use `Write` for brand-new files.
- **Parallel calls**: Issue independent tool calls in the same turn to cut round-trips.
- **No speculative reads**: Do not open a file "for context" unless you have a specific question to answer from it.

## Project Standards

Stack: **Python** package `rl_cartpole` (src layout, `src/rl_cartpole/`) · CI on **Python 3.12** · **Gymnasium ≥0.29**
· **pure NumPy** agents (no PyTorch/TensorFlow) · **pytest** · **Ruff** (line length 100) · **mypy**.

- **Coding standards** → **[CONTRIBUTING.md](CONTRIBUTING.md)** §1–3.
- **Feature/bug workflow** → **[CONTRIBUTING.md §4](CONTRIBUTING.md#4-feature--bug-workflow)**. Follow it in order for
  every feature or bug fix, starting with Step 0 (create a branch).
- **Logs & checkpoints policy** → [CONTRIBUTING.md §5](CONTRIBUTING.md#5-generated-artifacts-logs--checkpoints).
- **Directory layout and build/test/lint commands** → **[GEMINI.md](GEMINI.md)** §2 and §4. GEMINI.md is written for
  another assistant; translate its tool names with the mapping below.
- **Manual test procedures** → [TESTING.md](TESTING.md). **Architecture** → [ARCHITECTURE.md](ARCHITECTURE.md).

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

- First-time setup on Windows: `.\setup.ps1` (installs Python 3.12 via winget, creates `.venv`, installs
  `.[dev,video]`, runs the CI checks). Flags: `-Update`, `-NonInteractive`, `-SkipChecks`, `-InstallHooks`.
- **Run every Python tool from `.venv`**: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m ruff check .`,
  `.venv/Scripts/python -m mypy src`. Never use bare `python` — on this machine it is the Microsoft Store stub.
- If `.venv` is missing, ask the user to run `.\setup.ps1`; do not `pip install` into a global interpreter.

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
eternal-descent repo). Setup and troubleshooting: [README → AI Assistant MCP Servers](README.md#ai-assistant-mcp-servers).

| Server | Use it when |
| :--- | :--- |
| `context7` | Version-correct library docs. Library IDs: Gymnasium `/websites/gymnasium_farama`, NumPy `/websites/numpy_doc_stable`, pytest `/pytest-dev/pytest`, Ruff `/websites/astral_sh_ruff`, mypy `/websites/mypy_readthedocs_io_en_stable`, Matplotlib `/websites/matplotlib_stable`. For anything else, call `resolve-library-id` first |
| `memory` (user scope) | Cross-session notes; not a source of truth for API details |

### MCP Safety Rules

- **Check Gymnasium APIs in Context7 rather than from memory.** Older Gym tutorials return `done` from `step()`;
  this repo uses the v0.29+ `terminated`/`truncated` split and `reset(seed=...)` returning `(obs, info)`.
- **If a server is disconnected or a tool returns an auth error, point the user to the
  [README setup steps](README.md#ai-assistant-mcp-servers) rather than working around it.** Context7 reports
  "Connected" even with a missing key; an "Invalid API key" tool result means `CONTEXT7_API_KEY` is not set.

## Git

- Branch off `develop` as `feature/<kebab-case-name>` (or `fix/<kebab-case-name>`); PRs target `develop`.
- Commit messages use conventional commits: `type(scope): summary` (e.g. `feat(agents): …`, `fix(env): …`, `chore: …`,
  `docs: …`).
- **Commits are SSH-signed by the maintainer. Never run `git commit` or `git push`.** Stage the changed files with
  `git add` and print the exact commit and push commands for the user to run.
- PR descriptions follow [.github/pull_request_template.md](.github/pull_request_template.md).

## Known Discrepancies

Treat the repository as the source of truth when docs disagree:

- CONTRIBUTING.md asks for ≥80% **branch** coverage; CI enforces 50% **line** coverage (`--cov-fail-under=50`, no
  `--cov-branch`).
- `.pre-commit-config.yaml` pins Ruff `v0.1.15` while CI installs the latest Ruff, so the two formatters can disagree
  (commit `645f4a8` fixed one such drift). Its mypy hook depends on the deprecated `types-all`. Trust CI's
  `ruff format --check .`; `setup.ps1` only installs the hooks with `-InstallHooks`.
- `requirements.txt` lists `matplotlib>=3.5.0` and `pytest-cov>=4.0.0`, lagging `pyproject.toml`
  (`matplotlib>=3.9.4`, `pytest-cov>=7.1.0`). Install from `pyproject.toml` (`pip install -e ".[dev,video]"`).
- GEMINI.md §1 names Gemini models; use the Model Selection table above instead.
