# RL CartPole

A modular reinforcement learning framework for CartPole simulation. This project implements a
clean, extensible foundation for experimenting with RL algorithms in physics-based environments.

## Overview

This project implements a reinforcement learning (RL) agent that learns to balance a pole on a
moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI
module: simple enough to complete quickly, but structured in a way that mirrors real robotics
control loops and can be extended later.

The architecture is designed to support:

- Multiple RL algorithms (currently includes random, REINFORCE, and actor-critic agents)
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
│       │   └── actor_critic_agent.py
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
│   └── cartpole_actor_critic.yaml
├── train.py                     # Main training script
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

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
python train.py --config configs/quick_test.yaml --plot --plot-dir ./plots
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
```

### Basic Training

Run training with the default configuration:

```bash
python train.py
```

### Custom Configuration

Use a custom configuration file:

```bash
python train.py --config configs/cartpole_default.yaml
```

### Render Environment

Visualize the training process (requires a local display):

```bash
python train.py --render
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

Videos are saved to `./videos/` by default. Use `--video-dir` to choose a different location:

```bash
python train.py --record-video --video-dir /path/to/videos
```

A new video is recorded every `eval_frequency` episodes (configured in the YAML file, default
100). After training, copy the videos to your local machine and open them in any video player:

```bash
# From your local machine
scp user@orin-nano:/path/to/AI/videos/*.mp4 .
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
  type: "random"  # Options: "random", "reinforce", "actor_critic"
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
pytest --cov=src --cov-report=xml --cov-report=term-missing --cov-fail-under=50
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

All workflows run on push/PR to `main` and `develop` branches:

1. **lint.yml** - Ruff linting and formatting checks
2. **type-check.yml** - MyPy type checking (informational)
3. **test.yml** - Pytest with coverage enforcement and Codecov integration
4. **security.yml** - Bandit security scanning (also runs weekly)
5. **codeql.yml** - CodeQL security analysis (also runs weekly)
6. **copilot-deploy.yml** - Automated deployment with signed commits (manual trigger)

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

### Training Pipeline

The `Trainer` class orchestrates the training process:

- Episode collection and management
- Agent policy updates
- Evaluation and checkpointing
- Metrics logging

## Future Enhancements

This initial structure sets the foundation for:

- **PPO Implementation**: Full Proximal Policy Optimization algorithm
- **Additional Algorithms**: DQN, A3C, SAC, etc.
- **Advanced Environments**: More complex physics simulations
- **Vision Integration**: Camera-based observations
- **SLAM Integration**: Simultaneous localization and mapping
- **ROS2 Integration**: Robot Operating System 2 support
- **Embedded Deployment**: Deploy trained agents to edge devices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
