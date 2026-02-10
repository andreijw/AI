# RL CartPole

A modular reinforcement learning framework for CartPole simulation. This project implements a clean, extensible foundation for experimenting with RL algorithms in physics-based environments.

## Overview

This project implements a reinforcement learning (RL) agent that learns to balance a pole on a moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI module: simple enough to complete quickly, but structured in a way that mirrors real robotics control loops and can be extended later.

The architecture is designed to support:
- Multiple RL algorithms (currently includes baseline random agent, PPO coming soon)
- Easy experimentation with different configurations
- Clean separation between environments, agents, and training pipelines
- Environment augmentation with noise and domain randomization for robust training
- Extensibility for future robotics modules (vision, SLAM, ROS2 integration, etc.)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for environments, agents, training, and utilities
- **Enhanced Environment Wrapper**: Custom CartPole environment wrapper with:
  - Episode tracking and monitoring
  - Observation noise injection (Gaussian)
  - Action noise (probabilistic action flipping)
  - Domain randomization (gravity, pole length, cart mass)
  - Vectorized environments for parallel training
- **Configurable Training**: YAML-based configuration system for easy experimentation
- **Logging & Metrics**: Built-in logging system for tracking training progress
- **Extensible Agent Framework**: Base agent interface for implementing new RL algorithms
- **Comprehensive Testing**: Unit tests for core functionality

## Project Structure

```
AI/
├── src/
│   ├── env/                     # Environment factory (for creating environments)
│   │   ├── __init__.py
│   │   ├── cartpole_env.py      # Noise & domain randomization wrapper
│   │   └── make_env.py          # Factory functions for env creation
│   └── rl_cartpole/
│       ├── __init__.py
│       ├── environments/        # Environment wrappers
│       │   ├── __init__.py
│       │   └── cartpole_env.py  # Base CartPole wrapper
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
│   ├── test_env.py              # Tests for environment factory
│   ├── test_environment.py      # Tests for base environment
│   ├── test_agents.py
│   └── test_config.py
├── configs/                     # Configuration files
│   ├── env.yaml                 # Environment augmentation config
│   └── cartpole_default.yaml    # Training config
├── examples/
│   └── demo_environment.py      # Demo of environment features
├── train.py                     # Main training script
├── example.py                   # Example usage
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

### Environment Augmentation

The project supports advanced environment features for robust training:

#### Create Environment with Noise and Domain Randomization

```python
from src.env import make_single_env, make_vec_env, make_env_from_config

# Create a basic environment
env = make_single_env()

# Create environment with observation noise, action noise, and domain randomization
env = make_single_env(
    observation_noise_std=0.01,   # Add Gaussian noise to observations
    action_noise_std=0.1,          # Probabilistic action flipping
    domain_randomization={
        'gravity': (9.0, 11.0),
        'pole_length': (0.4, 0.6),
        'cart_mass': (0.8, 1.2),
    }
)

# Create vectorized environments for parallel training
vec_env = make_vec_env(num_envs=4)

# Load from configuration file
env = make_env_from_config('configs/env.yaml')
```

#### Environment Configuration

Edit `configs/env.yaml` to customize environment augmentation:

```yaml
env_name: "CartPole-v1"
num_envs: 1

observation_noise:
  enabled: false
  std: 0.01

action_noise:
  enabled: false
  std: 0.1

domain_randomization:
  enabled: false
  gravity:
    min: 8.0
    max: 12.0
  pole_length:
    min: 0.4
    max: 0.6
  cart_mass:
    min: 0.8
    max: 1.2
```

### Render Environment

Train with visualization:
```bash
python train.py --render
```

### Run Example

See a quick demonstration:
```bash
python example.py
```

Or run the environment demo:
```bash
python examples/demo_environment.py
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test module:
```bash
pytest tests/test_environment.py -v
pytest tests/test_agents.py -v
pytest tests/test_env.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Style

This project follows PEP 8 style guidelines. Format code with:
```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

Run type checks with:
```bash
mypy src/
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Environment Features

### CartPoleEnv Wrapper (src/env/cartpole_env.py)

The enhanced environment wrapper adds robustness features:

1. **Observation Noise**: Adds Gaussian noise to observations to make agents more robust to sensor noise
2. **Action Noise**: For discrete actions, randomly flips actions with a given probability to encourage exploration
3. **Domain Randomization**: Randomizes physics parameters on each episode reset:
   - Gravity
   - Pole length  
   - Cart mass

These features help train more robust policies that generalize better to variations in the environment.

### Factory Functions (src/env/make_env.py)

- `make_single_env()`: Create a single environment with custom augmentation settings
- `make_vec_env()`: Create vectorized environments for parallel execution
- `make_env_from_config()`: Load environment from YAML configuration file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## Future Work

- Implement PPO agent
- Add more environment wrappers
- Integration with ROS2
- Vision-based control
- SLAM integration
- Transfer to real robotics hardware
