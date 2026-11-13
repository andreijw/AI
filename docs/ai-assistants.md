# AI Assistants

How to set up Claude Code (and other assistants) for this repository: the shared MCP server and the development
workflow shipped as skills. The assistants' own rules are in [CLAUDE.md](../CLAUDE.md) and [GEMINI.md](../GEMINI.md).

## AI Assistant MCP Servers

The repo ships a shared [`.mcp.json`](../.mcp.json) that gives Claude Code (and other MCP clients) up-to-date library
documentation. It contains no secrets or machine paths; the API key comes from an environment variable. Usage rules
for the assistant live in [CLAUDE.md](../CLAUDE.md#mcp-servers).

| Server | Scope | Purpose | Needs |
| :--- | :--- | :--- | :--- |
| `context7` ([Context7](https://context7.com)) | Project (`.mcp.json`) | Version-correct Gymnasium, NumPy, pytest, Ruff, mypy, and Matplotlib docs | Free API key in `CONTEXT7_API_KEY` |

The definition is identical to the one in the sibling `eternal-descent` repo, so if `CONTEXT7_API_KEY` is already set
for that project, skip to step 3.

### Setup

1. **Create a Context7 API key** at <https://context7.com/dashboard> (free; keys start with `ctx7sk`).

2. **Store it as a user environment variable** without echoing it:

   ```powershell
   # Windows (PowerShell): copy the key from the dashboard first, then read it from the clipboard.
   [Environment]::SetEnvironmentVariable('CONTEXT7_API_KEY', (Get-Clipboard -Raw).Trim(), 'User')
   Set-Clipboard -Value ' '   # clear the key from the clipboard
   ```

   ```bash
   # macOS / Linux: append to ~/.zshrc or ~/.bashrc (the key is stored in plain text there)
   read -rs -p "Context7 API key: " CONTEXT7_API_KEY && echo
   printf 'export CONTEXT7_API_KEY=%q\n' "$CONTEXT7_API_KEY" >> ~/.zshrc
   unset CONTEXT7_API_KEY
   ```

3. **Restart.** Fully quit VS Code (every window), or open a new terminal, so the variable is inherited.

4. **Approve the project server.** The first time you run `claude` in the repo, trust the folder and approve
   `context7`. Or pre-approve it in your personal, git-ignored `.claude/settings.local.json`:

   ```json
   {
     "enabledMcpjsonServers": ["context7"]
   }
   ```

   To undo approvals: `claude mcp reset-project-choices`.

5. **Verify.** Run `claude mcp list`; `context7` should show `Connected` with no *Missing environment variables*
   warning. "Connected" alone is not proof, so also run a real prompt: *"Use context7 to look up `Env.step` in the
   Gymnasium docs."* Expect snippets mentioning `terminated` and `truncated`, **not** "Invalid API key".

### Troubleshooting

- **"Connected" but tools fail.** HTTP servers connect and list tools even with a missing or invalid key. Context7
  only reports `Invalid API key … should start with 'ctx7sk'` inside the tool result.
- **A new environment variable isn't picked up.** Reloading the window isn't enough: quit **every** VS Code window.
  Check from VS Code's integrated terminal: `[bool]$env:CONTEXT7_API_KEY`.

## AI Assistant Skills

The repo ships the development workflow as Claude Code **skills** in [`.claude/skills/`](../.claude/skills/). They are
Markdown files committed with the code, so cloning the repo installs them: Claude Code loads them when it starts in
this folder and picks the right one from what you ask. Together they let the assistant act as the dev team while a
person makes three decisions: which work to do, whether the plan is right, and when to commit.

| Skill | What it does | Runs |
| :--- | :--- | :--- |
| `dev-loop` | Takes a feature or fix from proposal to hand-off, calling the others in order | Whenever you ask to build, change or fix something, or ask what is next |
| `architect` | Reviews a plan or structure: package boundaries, Python class design, NumPy and Gymnasium hazards | On every plan before you see it |
| `algorithm-crafter` | Pins down an algorithm's variant, array shapes and hand-derived gradients, then fits it to the agent base classes and registries | When the work is a new agent or update rule |
| `test-crafter` | Writes unit tests, gradient checks and integration tests, then breaks the code on purpose to prove they fail | Before the implementation |
| `experiment-runner` | Runs seeded training sweeps outside your saved results and reports what they do and do not show | When the question is whether an agent learns |
| `training-doctor` | Finds why an agent does not learn, diverges or gives NaN, cheapest check first | Before any hyperparameter is changed to fix a bad run |
| `cleanup` | Finds and merges duplication against the whole codebase; removes leftovers | After the implementation |
| `pr-reviewer` | Final review: acceptance criteria, numerical correctness, soundness of the learning, what gets committed, whether the docs still describe every changed file; opens with a quick summary for you | Before you are asked to commit |
| `doc-sync` | Brings roadmap, TESTING, README, ARCHITECTURE and assistant docs back in line | Before you are asked to commit |
| `release` | Takes a distribution to a draft GitHub Release: version bump, local check, your tag commands in dependency order, watching the run, checking the draft, publishing it on your yes | When you ask to release a distribution; you sign the tag and say yes to each publish |

`architect`, `cleanup` and `pr-reviewer` build on the [brooks-lint](https://github.com/hyhmrright/brooks-lint) plugin
(MIT), which supplies a general code-quality rubric. The rules specific to this project live in
[`.brooks-lint.yaml`](../.brooks-lint.yaml) and in the skills themselves. Without the plugin the skills still run, with
the project checks only.

### Skill Tooling

| Tool | Purpose |
| :--- | :--- |
| `tests/helpers.py` → `numerical_grad`, `assert_gradients_match` | Finite-difference check of a hand-written backward pass, parameter by parameter |
| `tests/agent_internals.py` | The one module agent tests may use to reach weights, forward passes and loss methods |
| `tests/trainer_internals.py` | The same for `Trainer`: its metrics-record dispatch; all else goes through `train()` and `evaluate()` |
| `.claude/skills/test-crafter/scripts/mutation_check.py` | Applies small breaks from a JSON list, reports which tests notice, restores the files |
| `.claude/skills/doc-sync/scripts/check_md_width.py` | Checks the Markdown line-length rule on machines without Node |
| `.claude/skills/experiment-runner/scripts/run_queue.py` | Machine-wide job queue: `submit`, `run`, `status`, `wait`, `cancel`, `prune`; keeps at most 3 training runs going across all agents |
| `benchmark.py --output-dir … --report …` | Seeded experiment sweeps that leave `docs/benchmarks/` and saved runs alone |

### Skill Setup

1. **Prerequisites:** [Claude Code](https://code.claude.com) 2.1 or newer (CLI or the VS Code extension) and the
   `.venv` from [Installation](usage.md#installation). Node.js is optional: it is only needed to run `markdownlint`
   locally.

2. **Install the brooks-lint plugin.** [`.claude/settings.json`](../.claude/settings.json) declares its marketplace,
   pinned to the tag `v1.7.0`, and enables the plugin for this project. Start `claude` in the repo, trust the folder,
   and accept the prompt to install it. If no prompt appears, install it from a terminal. The marketplace has to be
   added before the plugin can be found in it:

   ```bash
   claude plugin marketplace add hyhmrright/brooks-lint@v1.7.0
   claude plugin install brooks-lint@brooks-lint-marketplace
   claude plugin list                                   # brooks-lint, Version: 1.7.0, enabled
   ```

3. **Restart Claude Code** so the plugin and skills are loaded.

4. **Verify.** Type `/` in Claude Code: the ten skills above should be listed, along with
   `brooks-lint:brooks-review`, `brooks-lint:brooks-audit` and `brooks-lint:brooks-test`.

5. **Use it.** Ask in plain words ("what should we work on next?", "PPO stopped learning on Pendulum, find out
   why") or name a skill (`/dev-loop`, `/training-doctor`). The assistant stages changes and prints the commit and
   push commands; it never commits, pushes or merges.

The same settings file pre-approves what the loop runs constantly, so it does not stop to ask each time: anything run
with the project's own interpreter (`.venv/bin/python` or `.venv/Scripts/python`, which covers the quality gates,
`train.py`, `benchmark.py` and the skill scripts), edits to the skill files, and `gh run`. That means the
assistant can run any Python in this repository's environment without a prompt; narrow it in the file if you prefer
to be asked. Rules are written as patterns, never with one-off file names; approve anything more specific into your
git-ignored `.claude/settings.local.json`. To opt out of the plugin personally, add this to that file:

```json
{
  "enabledPlugins": { "brooks-lint@brooks-lint-marketplace": false }
}
```

Other assistants can use the skills too: they are ordinary Markdown, and [GEMINI.md](../GEMINI.md) §5 points to them.
