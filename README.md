# AI
This project implements a reinforcement learning (RL) agent that learns to balance a pole on a moving cart using pure simulation. It is intentionally designed as a foundational robotics/AI module: simple enough to complete quickly, but structured in a way that mirrors real robotics control loops and can be extended later.

## Project Structure

```
cartpole-rl/
│
├── configs/
│   └── env.yaml                    # Environment configuration
│
├── src/
│   └── env/
│       ├── cartpole_env.py         # CartPole wrapper with noise/randomization
│       ├── make_env.py             # Factory functions for environments
│       └── __init__.py
│
├── tests/
│   └── test_env.py                 # Environment tests
│
├── examples/
│   └── demo_environment.py         # Demo script
│
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

The environment setup includes:

- **Basic Gymnasium CartPole wrapper**: Clean interface for CartPole-v1 environment
- **Observation noise**: Add Gaussian noise to observations for robustness
- **Action noise**: Add stochastic noise to actions for exploration
- **Domain randomization**: Randomize physics parameters (gravity, pole length, cart mass)
- **Vectorized environments**: Support for parallel environment execution

### Quick Start

```python
from src.env import make_single_env, make_vec_env, make_env_from_config

# Create a basic environment
env = make_single_env()

# Create environment with noise and domain randomization
env = make_single_env(
    observation_noise_std=0.01,
    action_noise_std=0.1,
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

### Configuration

Edit `configs/env.yaml` to customize environment parameters:

```yaml
env_name: "CartPole-v1"
num_envs: 1

observation_noise:
  enabled: false
  std: 0.01

action_noise:
  enabled: false
  std: 0.01

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

## Examples

Run the demo script to see all features in action:

```bash
python examples/demo_environment.py
```

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

## Features

### CartPoleEnv Wrapper

The `CartPoleEnv` class wraps the standard Gymnasium CartPole environment and adds:

1. **Observation Noise**: Adds Gaussian noise to observations to make the agent more robust
2. **Action Noise**: For discrete actions, randomly flips actions with a given probability
3. **Domain Randomization**: Randomizes physics parameters on each episode reset:
   - Gravity
   - Pole length
   - Cart mass

### Factory Functions

- `make_single_env()`: Create a single environment with custom settings
- `make_vec_env()`: Create vectorized environments for parallel execution
- `make_env_from_config()`: Load environment from YAML configuration file

## Next Steps

Future additions will include:
- PPO agent implementation
- Training pipeline
- Policy inference and export
- Logging and monitoring

