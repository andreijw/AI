# RL CartPole

A modular reinforcement learning framework for CartPole simulation. This project implements a
clean, extensible foundation for experimenting with RL algorithms in physics-based environments.

## Overview

This project implements a reinforcement learning (RL) agent that learns to balance a pole on a
moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI
module: simple enough to complete quickly, but structured in a way that mirrors real robotics
control loops and can be extended later.

The architecture is designed to support:

- Multiple RL algorithms (currently includes random, REINFORCE, actor-critic, and PPO agents)
- Easy experimentation with different configurations
- Clean separation between environments, agents, and training pipelines
- Extensibility for future robotics modules (vision, SLAM, ROS2 integration, etc.)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for environments, agents, training, and utilities
- **Environment Wrapper**: Custom CartPole environment wrapper with episode tracking and monitoring
- **Configurable Training**: YAML-based configuration system for easy experimentation
- **Logging & Metrics**: Built-in logging system for tracking training progress
- **Extensible Agent Framework**: Base agent interface for implementing new RL algorithms
- **Comprehensive Testing**: Unit tests for core functionality

## Project Structure

```
AI/
├── src/
│   └── rl_cartpole/
│       ├── __init__.py
│       ├── environments/        # Environment wrappers and factories
│       │   ├── __init__.py
│       │   ├── cartpole_env.py
│       │   └── make_env.py
│       ├── agents/              # Agent implementations
│       │   ├── __init__.py
│       │   ├── base_agent.py
│       │   ├── random_agent.py
│       │   ├── reinforce_agent.py
│       │   ├── actor_critic_agent.py
│       │   └── ppo_agent.py
│       ├── training/            # Training pipeline
│       │   ├── __init__.py
│       │   └── trainer.py
│       └── utils/               # Utilities
│           ├── __init__.py
│           ├── config.py        # Configuration loading
│           ├── logger.py        # Logging utilities
│           └── visualization.py # Plotting utilities
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_reinforce_agent.py
│   ├── test_actor_critic_agent.py
│   ├── test_trainer.py
│   └── ...
├── configs/                     # Configuration files
│   ├── cartpole_default.yaml
│   ├── cartpole_reinforce.yaml
│   ├── cartpole_actor_critic.yaml
│   └── cartpole_ppo.yaml
├── train.py                     # Main training script
├── record.py                    # Record a trained agent from a checkpoint
├── setup.ps1                   # Windows developer setup (Python 3.12 + .venv + CI checks)
├── .mcp.json                   # Shared MCP servers for AI assistants (context7)
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── CONTRIBUTING.md             # Coding standards and feature/bug workflow
├── CLAUDE.md / GEMINI.md       # AI assistant instructions
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

By default, `configs/cartpole_default.yaml` uses `num_episodes: 1000`, so this run may take a while to complete.

For a quick smoke-test you can override `num_episodes` by editing
`configs/cartpole_default.yaml` (change `num_episodes: 1000` to e.g. `50`) or by
creating a minimal config:

```yaml
# configs/quick_test.yaml
environment:
  name: "CartPole-v1"
  max_episode_steps: 500
  seed: 42

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
```

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
```

### Visualization and video recording

Outputs are kept in per-agent folders so training one agent never overwrites another's:
`checkpoints/<agent_type>/`, `videos/<agent_type>/` and `plots/<agent_type>/` (for example
`checkpoints/ppo/agent_episode_1400.pt`). Re-training the same agent replaces that agent's previous
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
python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/ppo/agent_episode_1400.pt --episodes 5
```

`record.py` loads the agent type from the config (or `--agent-type`), plays `--episodes`
evaluation episodes (default 5) and saves them as one
`videos/<agent_type>/cartpole-playback-full.mp4` (`--video-dir` to change). A well-trained agent balances for the full 500 steps, i.e. 10 seconds
per episode.

After training, copy the videos to your local machine and open them in any video player:

```bash
# From your local machine
scp -r user@orin-nano:/path/to/AI/videos .
```

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
  type: "random"  # Options: "random", "reinforce", "actor_critic", "ppo"
  config:
    seed: 42

# Training settings
training:
  num_episodes: 1000
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

For more information about the Copilot deployment workflow and GPG signing setup, see [.github/COPILOT_DEPLOYMENT.md](.github/COPILOT_DEPLOYMENT.md).

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

Implemented agents:

- `RandomAgent`: Random baseline policy
- `ReinforceAgent`: Monte Carlo policy-gradient agent
- `ActorCriticAgent`: Policy + value network agent with entropy regularization
- `PPOAgent`: Clipped-policy actor-critic agent for more stable updates

### Training Pipeline

The `Trainer` class orchestrates the training process:

- Episode collection and management
- Agent policy updates
- Evaluation and checkpointing
- Metrics logging

## Next AI Progression

With random, REINFORCE, actor-critic, and PPO in place, the next natural step is to add a
value-based baseline so policy-gradient and value-learning approaches can be compared directly.

### Immediate target: `DQNAgent`

1. Add a `DQNAgent` implementation that follows the same `BaseAgent` lifecycle (`select_action`,
   `update`, `save`, `load`).
2. Extend `train.py` dispatch and config options so DQN can run through the same CLI flow used by
   existing agents.
3. Add focused tests for replay handling, target updates, and epsilon-greedy behavior.
4. Benchmark DQN against PPO on CartPole with the existing metrics and plotting pipeline.

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

## Contributing

Contributions are welcome! Before opening a Pull Request, read [CONTRIBUTING.md](CONTRIBUTING.md). It holds the coding
standards and the branch-first, test-driven feature & bug workflow (§4) that every change follows. AI coding
assistants additionally follow [CLAUDE.md](CLAUDE.md) (Claude Code) or [GEMINI.md](GEMINI.md) (Gemini).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
