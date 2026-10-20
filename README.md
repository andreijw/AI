# RL CartPole

A modular reinforcement learning framework for CartPole simulation. This project implements a
clean, extensible foundation for experimenting with RL algorithms in physics-based environments.

## Overview

This project implements a reinforcement learning (RL) agent that learns to balance a pole on a
moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI
module: simple enough to complete quickly, but structured in a way that mirrors real robotics
control loops and can be extended later.

The architecture is designed to support:

- Multiple RL algorithms (random, REINFORCE, actor-critic, PPO and DQN for CartPole; random, actor-critic and PPO
  with Gaussian policies, and DDPG with a deterministic policy, for Pendulum)
- Easy experimentation with different configurations
- Clean separation between environments, agents, and training pipelines
- Extensibility for future robotics modules (vision, SLAM, ROS2 integration, etc.)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for environments,
  agents, training, and utilities
- **Environment Wrapper**: Custom CartPole environment wrapper with episode tracking and monitoring
- **Configurable Training**: YAML-based configuration system for easy experimentation
- **Logging & Metrics**: Built-in logging system for tracking training progress
- **Extensible Agent Framework**: Base agent interface for implementing new RL algorithms
- **Comprehensive Testing**: Unit tests for core functionality

## Project Structure

```
AI/
├── src/
│   ├── rl_core/                 # Algorithm-agnostic RL framework
│   │   ├── agents/              # Core discrete & continuous agent implementations
│   │   │   ├── base_agent.py
│   │   │   ├── actor_critic_agent.py
│   │   │   ├── actor_critic_base.py
│   │   │   ├── config_values.py # Strict readers for integer agent settings
│   │   │   ├── continuous_actor_critic.py
│   │   │   ├── continuous_ppo.py
│   │   │   ├── continuous_random_agent.py
│   │   │   ├── dqn_agent.py
│   │   │   ├── optim.py         # Adam, shared by the continuous agents and the `optimizer` option
│   │   │   ├── ppo_agent.py
│   │   │   ├── random_agent.py
│   │   │   ├── reinforce_agent.py
│   │   │   ├── replay_buffer.py
│   │   │   ├── returns.py       # Discounted returns shared by the episodic agents
│   │   │   └── softmax.py       # Numerically stable softmax shared by the discrete policies
│   │   ├── training/            # Generic training pipeline (trainer.py)
│   │   ├── utils/               # Config, logging, video recording & plotting utilities
│   │   ├── challenges/          # BenchmarkChallenge, the challenge registry, discover_challenges
│   │   └── seeding.py           # Separate seeded random streams (derive_rng)
│   ├── rl_cartpole/             # CartPole-v1 discrete balance control challenge
│   │   ├── environments/        # CartPole wrappers and vectorization factories
│   │   └── benchmark.py         # CartPole benchmark challenge implementation
│   ├── rl_pendulum/             # Pendulum-v1 continuous torque control challenge
│   │   ├── environments/        # Pendulum wrappers and continuous noise factories
│   │   └── benchmark.py         # Pendulum benchmark challenge implementation
│   └── benchmark.py             # Benchmark runner & CLI (finds challenges through rl_core)
├── tests/                       # Unit & integration test suites
├── configs/                     # YAML hyperparameter configurations
│   ├── cartpole_*.yaml          # CartPole challenge configs (default, PPO, DQN, etc.)
│   └── pendulum_*.yaml          # Pendulum challenge configs (default, PPO, A2C, DDPG)
├── train.py                     # Main training entrypoint
├── record.py                    # Checkpoint video recorder
├── setup.ps1                   # Windows developer setup
├── pyproject.toml              # Project packaging & tool settings
├── CONTRIBUTING.md             # Coding standards & workflow
├── ARCHITECTURE.md             # Architectural design & Mermaid diagrams
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.12+ (the version CI tests)
- pip or conda package manager

### Windows Quick Setup

From PowerShell in the repository root, run:

```powershell
.\setup.ps1
```

The script installs Git and Python 3.12 via `winget` if they are missing, creates a `.venv` virtual environment,
installs the package with `[dev,video]` extras, and runs the same checks as CI (Ruff, mypy, pytest). Flags:
`-NonInteractive` (no prompts), `-SkipChecks`, `-Update` (upgrade tools and packages), and `-InstallHooks`
(`pre-commit install`). Activate the environment afterwards with `.\.venv\Scripts\Activate.ps1`.

If script execution is disabled, run it once with `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.

### Manual Setup (any OS)

1. Clone the repository:

```bash
git clone https://github.com/andreijw/AI.git
cd AI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Usage

### Quick Start: Train Your First Agent

The fastest way to try the framework is to run the training script with the default
random-agent baseline configuration:

```bash
python train.py --config configs/cartpole_default.yaml
```

By default, `configs/cartpole_default.yaml` trains for `total_steps: 50000` environment steps.

For a quick smoke-test pass `--num-episodes 50` (or `--total-steps 2000`), or create a minimal config:

```yaml
# configs/quick_test.yaml
environment:
  name: "CartPole-v1"
  max_episode_steps: 500
  seed: 42            # reproducible sequence of start states (each episode differs)

agent:
  type: "random"
  config:
    seed: 42          # makes action selection reproducible

training:
  num_episodes: 50
  max_steps_per_episode: 500
  eval_frequency: 10
  save_frequency: 50
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

Then run:

```bash
python train.py --config configs/quick_test.yaml
```

To generate and save learning-curve plots during training, include `--plot` on your initial run:

```bash
python train.py --config configs/quick_test.yaml --plot
```

> Note: Running `train.py` again with `--plot` will start a new training run and then save plots,
> it does **not** only load and plot previous results.

### Train specific agent configurations

```bash
# Random baseline
python train.py --config configs/cartpole_default.yaml

# REINFORCE
python train.py --config configs/cartpole_reinforce.yaml

# Actor-Critic
python train.py --config configs/cartpole_actor_critic.yaml

# PPO
python train.py --config configs/cartpole_ppo.yaml

# DQN
python train.py --config configs/cartpole_dqn.yaml

# Pendulum Continuous Actor-Critic
python train.py --config configs/pendulum_actor_critic.yaml

# Pendulum Continuous PPO
python train.py --config configs/pendulum_ppo.yaml

# Pendulum DDPG
python train.py --config configs/pendulum_ddpg.yaml
```

### Which agents solve which challenge

A run solves the challenge when its average training reward over 100 episodes reaches the threshold: 475 for
CartPole-v1, -200 for Pendulum-v1. It **holds** when the average over its last 100 episodes is still at or above the
threshold. Each agent trains for a budget of environment steps set against its rl-baselines3-zoo reference (see the
config comments). Measured with the committed configs on seeds 1 to 5 on 2026-10-18 (CartPole PPO on 2026-10-19,
after time-limit cuts were bootstrapped), with evaluation on its own environment and named random streams
([`docs/benchmarks/cartpole_results.md`](docs/benchmarks/cartpole_results.md),
[`docs/benchmarks/pendulum_results.md`](docs/benchmarks/pendulum_results.md)). `train.py` uses seed 42, which was
not measured, so a single `train.py` run may land on the unlucky side of these numbers:

| Challenge | Agent | Config | Budget (steps) | Seeds solved | Steps to solve | Held |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| CartPole-v1 | PPO | `cartpole_ppo.yaml` (standard PPO, annealed) | 200,000 | 5 of 5 | 72k to 90k | 5 of 5 |
| CartPole-v1 | Actor-Critic | `cartpole_actor_critic.yaml` (Adam) | 500,000 | 5 of 5 | 209k to 456k | 4 of 5 |
| CartPole-v1 | REINFORCE | `cartpole_reinforce.yaml` (Adam) | 500,000 | 4 of 5 | 255k to 376k | 3 of 5 |
| CartPole-v1 | DQN | `cartpole_dqn.yaml` (Adam, per-step exploration) | 300,000 | 5 of 5 | 144k to 183k | 5 of 5 |
| Pendulum-v1 | DDPG | `pendulum_ddpg.yaml` (rl-zoo settings, 64×64 networks) | 40,000 | 5 of 5 | 31k | 5 of 5 |
| Pendulum-v1 | PPO | `pendulum_ppo.yaml` (2048-step rollouts, annealed) | 200,000 | 5 of 5 | 72k to 157k | 4 of 5 |
| Pendulum-v1 | Actor-Critic | `pendulum_actor_critic.yaml` (bounded mean, 8-step updates) | 1,000,000 | 0 of 5 | — | 0 of 5 |

CartPole DQN trains for 300,000 steps rather than rl-zoo's 50,000: holding a 100-episode average of 475 alone takes
47,500 steps, and rl-zoo's 50,000-step result is a greedy evaluation after training (roadmap: "DQN Reaches the rl-zoo
Reference"). Pendulum DDPG uses rl-zoo's settings with 64×64 networks instead of 400×300, which take about 16 times
longer in NumPy; its greedy evaluation averages -158 (± 25 across the 5 seeds' means) against rl-zoo's -152
(± 94 across episodes of one run) (roadmap: "Continuous Off-Policy
Learning & DDPG"). Pendulum PPO trains on rollouts of whole episodes for 200,000 steps, twice rl-zoo's 100,000;
updating on one episode at a time it solved no seed (roadmap: "Pendulum PPO Holds -200 Under the 100-Episode
Rule").
Continuous Actor-Critic does not solve: its training average ends at -286 to -307, though its greedy evaluation
of -172 ± 27 beats the random agent on every seed (roadmap: "Continuous Actor-Critic Learns Pendulum-v1").

### Run any agent from one base config (quick overrides)

You can override both the agent type and number of episodes directly from CLI without
editing YAML files:

```bash
# quick random agent run
python train.py --config configs/cartpole_default.yaml --agent-type random --num-episodes 50

# quick REINFORCE run
python train.py --config configs/cartpole_default.yaml --agent-type reinforce --num-episodes 200

# quick Actor-Critic run
python train.py --config configs/cartpole_default.yaml --agent-type actor_critic --num-episodes 200

# quick PPO run
python train.py --config configs/cartpole_default.yaml --agent-type ppo --num-episodes 200

# quick DQN run
python train.py --config configs/cartpole_default.yaml --agent-type dqn --num-episodes 200
```

The challenge comes from the config's `environment.name`, which must be a registered challenge's environment id
exactly (`CartPole-v1` or `Pendulum-v1`); any other name is an error that lists the options. `--challenge cartpole`
or `--challenge pendulum` must agree with that name, and is required for a config that has no `environment.name`.
An agent the challenge does not support (for example `--agent-type dqn` on Pendulum) is rejected before training.

### Visualization and video recording

Outputs are partitioned into per-challenge and per-agent folders so runs never overwrite one another:
`checkpoints/<challenge>/<agent_type>/`, `videos/<challenge>/<agent_type>/`, `plots/<challenge>/<agent_type>/`,
and `logs/<challenge>/<agent_type>/` (for example `checkpoints/cartpole/ppo/agent_episode_1400.pt` or
`checkpoints/pendulum/ppo/agent_episode_200.pt`). Re-training the same agent replaces that agent's previous
outputs; copy its folders first to keep a run. Logs in `logs/` are timestamped per run.

For a full run with saved learning-curve plots (saved to `plots/ppo/`):

```bash
python train.py --config configs/cartpole_ppo.yaml --plot
```

To visualize live training locally (requires display):

```bash
python train.py --config configs/cartpole_ppo.yaml --render
```

For headless visualization (saved MP4 videos):

```bash
python train.py --config configs/cartpole_ppo.yaml --record-video
```

### Basic/default run

Run training with defaults (`configs/cartpole_default.yaml`):

```bash
python train.py
```

### Record Training Videos (Headless / SSH)

If you are running on a headless machine such as an **NVIDIA Orin Nano accessed via SSH**,
there is no display available so `--render` will not work. Use `--record-video` instead to
save MP4 videos of training episodes to disk. No display or X server is required.

First, install the optional video dependencies:

```bash
pip install -e ".[video]"
# or: pip install moviepy "gymnasium[classic-control]"
```

Then run training with video recording:

```bash
python train.py --record-video
```

Videos are saved to `./videos/<agent_type>/` by default. Use `--video-dir` to choose a different
location (used exactly as given):

```bash
python train.py --record-video --video-dir /path/to/videos
```

By default a clip is recorded every `eval_frequency` training episodes (configured in the YAML
file, default 100). Clips are named after the training episode they show
(`cartpole-training-episode-<N>.mp4`); evaluation episodes are never recorded. Use
`--video-every` to change the interval:

```bash
# Record every 25th training episode as its own clip
python train.py --record-video --video-every 25
```

Every recorded episode is rendered and encoded frame by frame, so small intervals slow training
down (recording all 1000 random-agent episodes takes ~6 minutes instead of ~8 seconds).

### Record a Trained Agent

To get one longer video of a trained agent, record it from a checkpoint after training. This
does not slow training down:

```bash
python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_1400.pt --episodes 5
```

`record.py` loads the agent type from the config (or `--agent-type`), plays `--episodes`
evaluation episodes (default 5) and saves them as one
`videos/<challenge>/<agent_type>/<challenge>-playback-full.mp4` (`--video-dir` to change).
A well-trained agent balances for the full 500 steps, i.e. 10 seconds per episode.

After training, copy the videos to your local machine and open them in any video player:

```bash
# From your local machine
scp -r user@orin-nano:/path/to/AI/videos .
```

### Multi-Agent Benchmarking

Run an automated comparative benchmark across all algorithms over reproducible seeds:

```bash
# Benchmark all agents on CartPole, each for its own config's budget, across seeds 42 and 123
python src/benchmark.py --challenge cartpole \
    --agents random reinforce actor_critic ppo dqn --seeds 42 123

# Quick comparison between PPO and DQN, both for 150 episodes
python src/benchmark.py --challenge cartpole --agents ppo dqn --episodes 150 --seeds 42
```

Without `--episodes` or `--steps`, each agent trains for its own config's budget (`configs/<challenge>_<agent>.yaml`,
or `<challenge>_default.yaml` for the random agent): `training.total_steps` environment steps when set, else
`training.num_episodes` episodes. A step budget finishes the episode that crosses it. A run counts as solved when its
average training reward over `--window` consecutive episodes (default 100) reaches the challenge's threshold (475 for
CartPole-v1, -200 for Pendulum-v1); the report gives the episode and the environment steps it took, and **Held**: the
share of runs whose last `--window` episodes still average at or above the threshold. The final greedy evaluation is
reported in its own column and does not count as a solve. `train.py --total-steps N` trains for a step budget too.

Outputs:

- Comparative learning curve plot: `plots/benchmarks/cartpole/benchmark_comparison.png`
- Markdown summary table: [`docs/benchmarks/cartpole_results.md`](docs/benchmarks/cartpole_results.md)
- Raw metrics JSON: `plots/benchmarks/cartpole/benchmark_metrics.json`

## Configuration

Configuration files are in YAML format. Here's an example:

```yaml
# Environment settings
environment:
  name: "CartPole-v1"
  render_mode: null
  max_episode_steps: 500
  seed: 42

# Agent settings
agent:
  type: "random"  # Options: "random", "reinforce", "actor_critic", "ppo", "dqn"
  config:
    seed: 42

# Training settings
training:
  num_episodes: 1000
  # total_steps: 100000  # optional: train for this many environment steps instead (num_episodes is then ignored)
  max_steps_per_episode: 500
  eval_frequency: 100
  save_frequency: 100
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

## Development

### Running Tests

Run the test suite:

```bash
pytest -q
```

Run tests with coverage (CI-aligned):

```bash
pytest --cov-report=xml --cov-fail-under=50
```

### Code Formatting

Format code with Ruff:

```bash
ruff format .
```

### Linting

Check code quality with Ruff:

```bash
ruff check .
```

Fix auto-fixable issues:

```bash
ruff check --fix .
```

### Type Checking

Run type checking with mypy:

```bash
mypy src/
```

### Security Scanning

Run security checks with Bandit:

```bash
bandit -r src/ -ll
```

### Markdown Linting

Lint markdown documentation:

```bash
npx --yes markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules
```

## CI/CD

This repository includes comprehensive GitHub Actions workflows for maintaining code quality and security:

### Automated Quality Checks

- **Linting**: Automated code quality checks with Ruff (format & linting)
- **Type Checking**: Static type checking with MyPy for type safety
- **Testing**: Automated unit tests with pytest and coverage reporting (50% minimum)
- **Security Scanning**:
  - CodeQL for comprehensive security analysis
  - Bandit for Python-specific security vulnerabilities
- **Dependency Management**: Dependabot for automated dependency updates

### Local Development Tools

Pre-commit hooks are available for local development. Install them with:

```bash
pip install -e ".[dev]"
pre-commit install
```

The pre-commit hooks will automatically run:

- Code formatting (Ruff)
- Linting checks (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit)
- Markdown linting (markdownlint)
- File validation (trailing whitespace, YAML/JSON syntax, etc.)

### Workflows

Workflows are automatically synchronized from `.github/workflows` by
`scripts/sync_workflow_docs.py` (executed by the nightly maintenance workflow).

<!-- WORKFLOWS:START -->

- **codeql.yml** - CodeQL Advanced
- **copilot-deploy.yml** - Copilot Deployment
- **lint.yml** - Lint
- **nightly-maintenance.yml** - Nightly Maintenance
- **security.yml** - Security Check
- **test.yml** - Tests
- **type-check.yml** - Type Check

<!-- WORKFLOWS:END -->

For more information about the Copilot deployment workflow and GPG signing setup,
see [.github/COPILOT_DEPLOYMENT.md](.github/COPILOT_DEPLOYMENT.md).

## Architecture

### Environment Wrapper

The `CartPoleEnv` class wraps Gymnasium's CartPole-v1 environment and provides:

- Episode tracking and statistics
- Standardized interface for training pipeline
- Easy monitoring and logging integration

### Agent Framework

The `BaseAgent` abstract class defines the interface for all RL agents:

- `select_action()`: Choose actions based on observations
- `update()`: Update policy based on collected experience
- `save()` / `load()`: Checkpoint management

Implemented agents (in `src/rl_core/agents/`):

- **Discrete Control Agents**:
  - `RandomAgent`: Random action baseline policy
  - `ReinforceAgent`: Monte Carlo policy-gradient agent
  - `ActorCriticAgent`: Advantage Actor-Critic (A2C) policy + value network agent
  - `PPOAgent`: Clipped-policy Proximal Policy Optimization agent
  - `DQNAgent`: Deep Q-Network with replay buffer and target network
- **Continuous Control Agents**:
  - `ContinuousRandomAgent`: Uniform random continuous torque baseline policy
  - `ContinuousActorCriticAgent`: Continuous Gaussian policy A2C with state-value critic
  - `ContinuousPPOAgent`: Continuous Gaussian policy PPO with clipped surrogate objective
  - `DDPGAgent`: Deep Deterministic Policy Gradient with replay, a Q critic and Polyak target networks

### Training Pipeline

The `Trainer` class orchestrates the training process:

- Episode collection and management
- Agent policy updates
- Evaluation and checkpointing
- Metrics logging

## Next AI Progression

With discrete control (CartPole: Random, REINFORCE, Actor-Critic, PPO, DQN) and continuous
control (Pendulum: Random, Actor-Critic, PPO, DDPG) fully operational, the framework provides
a decoupled foundation for future AI challenges.

### Next Steps

1. **More Continuous Off-Policy Learning**: TD3 (twin critics, delayed actor updates) or SAC (Soft Actor-Critic) on
   the replay buffer, networks and target updates DDPG already uses.
2. **Advanced Environments**: BipedalWalker, MountainCar, or Acrobot challenges.
3. **Hyperparameter Sweeps & Automated Tuning**: Population-based training or grid-search runners.

## Future Enhancements

After DQN, the current structure also supports:

- **Additional Algorithms**: A3C, SAC, etc.
- **Advanced Environments**: More complex physics simulations
- **Vision Integration**: Camera-based observations
- **SLAM Integration**: Simultaneous localization and mapping
- **ROS2 Integration**: Robot Operating System 2 support
- **Embedded Deployment**: Deploy trained agents to edge devices

## AI Assistant MCP Servers

The repo ships a shared [`.mcp.json`](.mcp.json) that gives Claude Code (and other MCP clients) up-to-date library
documentation. It contains no secrets or machine paths; the API key comes from an environment variable. Usage rules
for the assistant live in [CLAUDE.md](CLAUDE.md#mcp-servers).

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

The repo ships the development workflow as Claude Code **skills** in [`.claude/skills/`](.claude/skills/). They are
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
| `pr-reviewer` | Final review: acceptance criteria, numerical correctness, soundness of the learning, what gets committed | Before you are asked to commit |
| `doc-sync` | Brings roadmap, TESTING, README, ARCHITECTURE and assistant docs back in line | Before you are asked to commit |

`architect`, `cleanup` and `pr-reviewer` build on the [brooks-lint](https://github.com/hyhmrright/brooks-lint) plugin
(MIT), which supplies a general code-quality rubric. The rules specific to this project live in
[`.brooks-lint.yaml`](.brooks-lint.yaml) and in the skills themselves. Without the plugin the skills still run, with
the project checks only.

### Skill Tooling

| Tool | Purpose |
| :--- | :--- |
| `tests/helpers.py` → `numerical_grad`, `assert_gradients_match` | Finite-difference check of a hand-written backward pass, parameter by parameter |
| `tests/agent_internals.py` | The one module agent tests may use to reach weights, forward passes and loss methods |
| `.claude/skills/test-crafter/scripts/mutation_check.py` | Applies small breaks from a JSON list, reports which tests notice, restores the files |
| `.claude/skills/doc-sync/scripts/check_md_width.py` | Checks the Markdown line-length rule on machines without Node |
| `src/benchmark.py --output-dir … --report …` | Seeded experiment sweeps that leave `docs/benchmarks/` and saved runs alone |

### Skill Setup

1. **Prerequisites:** [Claude Code](https://code.claude.com) 2.1 or newer (CLI or the VS Code extension) and the
   `.venv` from [Installation](#installation). Node.js is optional: it is only needed to run `markdownlint` locally.

2. **Install the brooks-lint plugin.** [`.claude/settings.json`](.claude/settings.json) declares its marketplace,
   pinned to the tag `v1.7.0`, and enables the plugin for this project. Start `claude` in the repo, trust the folder,
   and accept the prompt to install it. If no prompt appears, install it from a terminal. The marketplace has to be
   added before the plugin can be found in it:

   ```bash
   claude plugin marketplace add hyhmrright/brooks-lint@v1.7.0
   claude plugin install brooks-lint@brooks-lint-marketplace
   claude plugin list                                   # brooks-lint, Version: 1.7.0, enabled
   ```

3. **Restart Claude Code** so the plugin and skills are loaded.

4. **Verify.** Type `/` in Claude Code: the nine skills above should be listed, along with
   `brooks-lint:brooks-review`, `brooks-lint:brooks-audit` and `brooks-lint:brooks-test`.

5. **Use it.** Ask in plain words ("what should we work on next?", "PPO stopped learning on Pendulum, find out
   why") or name a skill (`/dev-loop`, `/training-doctor`). The assistant stages changes and prints the commit and
   push commands; it never commits, pushes or merges.

The same settings file pre-approves what the loop runs constantly, so it does not stop to ask each time: anything run
with the project's own interpreter (`.venv/bin/python` or `.venv/Scripts/python`, which covers the quality gates,
`train.py`, `src/benchmark.py` and the skill scripts), edits to the skill files, and `gh run`. That means the
assistant can run any Python in this repository's environment without a prompt; narrow it in the file if you prefer
to be asked. Rules are written as patterns, never with one-off file names; approve anything more specific into your
git-ignored `.claude/settings.local.json`. To opt out of the plugin personally, add this to that file:

```json
{
  "enabledPlugins": { "brooks-lint@brooks-lint-marketplace": false }
}
```

Other assistants can use the skills too: they are ordinary Markdown, and [GEMINI.md](GEMINI.md) §5 points to them.

## Contributing

Contributions are welcome! Before opening a Pull Request, read [CONTRIBUTING.md](CONTRIBUTING.md). It holds the coding
standards and the branch-first, test-driven feature & bug workflow (§4) that every change follows. AI coding
assistants additionally follow [CLAUDE.md](CLAUDE.md) (Claude Code) or [GEMINI.md](GEMINI.md) (Gemini).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
