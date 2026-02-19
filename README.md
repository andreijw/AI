# RL CartPole

A modular reinforcement learning framework for CartPole simulation. This project implements a clean, extensible foundation for experimenting with RL algorithms in physics-based environments.

## Overview

This project implements a reinforcement learning (RL) agent that learns to balance a pole on a moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI module: simple enough to complete quickly, but structured in a way that mirrors real robotics control loops and can be extended later.

The architecture is designed to support:

- Multiple RL algorithms (currently includes baseline random agent, PPO coming soon)
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
│       ├── environments/        # Environment wrappers
│       │   ├── __init__.py
│       │   └── cartpole_env.py
│       ├── agents/              # Agent implementations
│       │   ├── __init__.py
│       │   ├── base_agent.py    # Abstract base class
│       │   └── random_agent.py  # Random baseline agent
│       ├── training/            # Training pipeline
│       │   ├── __init__.py
│       │   └── trainer.py
│       └── utils/               # Utilities
│           ├── __init__.py
│           ├── config.py        # Configuration loading
│           └── logger.py        # Logging utilities
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   └── test_config.py
├── configs/                     # Configuration files
│   └── cartpole_default.yaml
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

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Usage

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

Visualize the training process:

```bash
python train.py --render
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
  type: "random"
  config:
    learning_rate: 0.0003
    gamma: 0.99

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
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=src/rl_cartpole --cov-report=term-missing
```

### Code Formatting

Format code with Ruff (recommended):

```bash
ruff format .
```

Or use Black (also supported):

```bash
black src/ tests/
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
