# Environment Setup Implementation Summary

> **Historical.** This summary describes the environment setup as it was when the repository held only CartPole;
> its paths are updated to the `packages/` layout, but the files it lists may no longer match. For the current one see
> [GEMINI.md](../GEMINI.md#2-codebase-directory-layout) and
> [ARCHITECTURE.md](../ARCHITECTURE.md).

## Overview

This document summarizes the implementation of the basic environment setup for the CartPole RL
project, including observation noise, action noise, domain randomization, and vectorized environment support.

## Files Added/Modified

### New Files

1. **`packages/rl-challenges/src/rl_cartpole/environments/make_env.py`**
   - Factory functions for creating single and vectorized environments
   - Support for configuration-based environment creation

2. **`configs/env.yaml`**
   - Environment configuration template with all new parameters
   - Documentation and examples for each setting

3. **`examples/demo_env_features.py`**
   - Comprehensive demonstration of all environment features
   - Multiple demo functions showcasing different capabilities

### Modified Files

1. **`packages/rl-challenges/src/rl_cartpole/environments/cartpole_env.py`**
   - Added observation noise injection
   - Added action noise with configurable probability
   - Implemented domain randomization
   - Enhanced compatibility with VectorEnv

2. **`packages/rl-challenges/src/rl_cartpole/environments/__init__.py`**
   - Exported new factory functions

3. **`tests/test_environment.py`**
   - Added new test cases for new features
   - All environment tests passing

## Features Implemented

### 1. Observation Noise

**Purpose**: Add Gaussian noise to observations for robustness training

**Implementation**:

- Parameter: `obs_noise_std` (float, default=0.0)
- Applied after environment step and reset
- Uses numpy's normal distribution

**Example**:

```python
env = CartPoleEnv(obs_noise_std=0.1)
```

### Velocity Masking, Frame Stacking and Frame Differences

**Purpose**: Make the task partially observable, then give the agent a short history to recover what was hidden.

**Implementation** (`rl_core.environments.observation`, applied by `BaseEnvWrapper` for every challenge):

- Parameters: `mask_velocity` (bool, default false), `frame_stack` (positive integer, default 1, which is off) and
  `frame_deltas` (bool, default false; needs `frame_stack` of 2 or more)
- Order: observation noise, then the mask, then the stack, then the differences, so a frame keeps the noise it was
  observed with and a masked entry is exactly zero
- A reset fills every slot of the stack with the first observation, so its differences are zero
- `mask_velocity: true` is refused for an environment id whose wrapper lists no velocity entries: Navigation-v0 and
  NavigationContinuous-v0, whose observation holds no velocity of the robot's state
- The stack is oldest frame first; with `frame_deltas` it becomes the newest frame, then the differences between
  consecutive frames, newest first (an invertible re-expression of the same entries)
- Measured on CartPole PPO: masking alone drops greedy evaluation from 500 to 42; `frame_stack: 4` brings it back
  to 337 but solves on 0 of 5 seeds (CHANGELOG.md, 2026-11-01); with `frame_deltas: true` and the agent's
  `normalize_observations: true` it solves and holds on 5 of 5 (CHANGELOG.md, 2026-11-02)

**Example**:

```python
env = CartPoleEnv(mask_velocity=True, frame_stack=4)  # CartPole: 16 entries, velocities zero
env = CartPoleEnv(
    mask_velocity=True, frame_stack=4, frame_deltas=True
)  # newest frame, 3 differences
```

### 2. Action Noise

**Purpose**: Randomly flip actions to test robustness

**Implementation**:

- Parameter: `action_noise_prob` (float, 0.0-1.0, default=0.0)
- Flips action (0→1, 1→0) with specified probability
- Applied before environment step

**Example**:

```python
env = CartPoleEnv(action_noise_prob=0.05)
```

### 3. Domain Randomization

**Purpose**: Randomize physical parameters for sim-to-real transfer

**Parameters Supported**:

- `gravity`: Default 9.8 m/s²
- `masscart`: Cart mass, default 1.0 kg
- `masspole`: Pole mass, default 0.1 kg
- `length`: Half-length of pole, default 0.5 m

**Implementation**:

- Applied at episode reset
- Uniform sampling within specified ranges
- Automatically updates derived parameters (total_mass, polemass_length)

**Example**:

```python
domain_rand = {
    "gravity": (9.0, 10.0),
    "masscart": (0.9, 1.1),
    "masspole": (0.09, 0.11),
    "length": (0.45, 0.55),
}
env = CartPoleEnv(domain_randomization=domain_rand)
```

### 4. Vectorized Environments

**Purpose**: Parallel environment execution for faster training

There are no vector factories (`make_vec_env` and `make_vec_env_from_config` were removed in rl-challenges 0.3.0, and
so was the keyword factory `make_env`: build `CartPoleEnv(...)` directly). To train from several environments, set
`training.num_envs` (docs/usage.md, "Several training environments"); its environments are seeded with
`rl_core.seeding.env_seed`. For a Gymnasium vector environment of your own, build one over the wrapper as below.

**Example**:

```python
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from rl_cartpole.environments import CartPoleEnv

# 8 environments, each in its own process
vec_env = AsyncVectorEnv([lambda i=i: CartPoleEnv(seed=42 + i) for i in range(8)])

# 4 environments stepped in turn in this process
vec_env = SyncVectorEnv([lambda i=i: CartPoleEnv(seed=42 + i) for i in range(4)])
```

### 5. Configuration-Based Creation

**Purpose**: Easy environment creation from config files

**Function**: `make_env_from_config(config)`: a single environment

**Example**:

```python
config = {
    "max_episode_steps": 500,
    "seed": 42,
    "obs_noise_std": 0.01,
    "action_noise_prob": 0.05,
    "domain_randomization": {
        "gravity": [9.0, 10.0],
        "length": [0.4, 0.6],
    },
}
env = make_env_from_config(config)
```

## API Reference

### Factory Function

#### `make_env_from_config(config)`

Creates environment from configuration dictionary. `CartPoleEnv(...)` takes the same settings as keywords
(`env_name` for the config's `name`).

**Parameters**:

- `config`: Dictionary with environment parameters

**Returns**: `CartPoleEnv` instance

## Testing

The environment implementation includes comprehensive tests covering:

- **Basic Environment Functionality**: Environment creation, reset, step, episode tracking, and completion
- **Noise Injection**: Observation noise and action noise application
- **Domain Randomization**: Parameter randomization and per-episode variation
- **Factory Function**: Environment creation from a config, with and without features
- **Configuration-based Creation**: Creating environments from configuration dictionaries
- **Environment Name Support**: Using different Gymnasium environments with the wrapper

All tests are located in `tests/test_environment.py` and can be run with pytest.

## Example Usage

### Basic Training Loop

```python
from rl_cartpole.environments import CartPoleEnv

env = CartPoleEnv(seed=42)
obs, info = env.reset()

for episode in range(100):
    done = False
    while not done:
        action = policy(obs)  # Your policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    obs, info = env.reset()

env.close()
```

### Robust Training with All Features

```python
from gymnasium.vector import AsyncVectorEnv
from rl_cartpole.environments import CartPoleEnv

# 8 parallel environments with noise and randomization
settings = {
    "obs_noise_std": 0.02,
    "action_noise_prob": 0.05,
    "domain_randomization": {
        "gravity": (9.0, 10.0),
        "masscart": (0.9, 1.1),
        "masspole": (0.09, 0.11),
        "length": (0.45, 0.55),
    },
}
vec_env = AsyncVectorEnv([lambda i=i: CartPoleEnv(seed=42 + i, **settings) for i in range(8)])

obs, info = vec_env.reset()
for step in range(1000):
    actions = policy(obs)  # Your policy for all envs
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

vec_env.close()
```

### Configuration-Based Setup

```python
import yaml
from rl_cartpole.environments import make_env_from_config

# Load from YAML file
with open("configs/env.yaml", "r") as f:
    config = yaml.safe_load(f)

env = make_env_from_config(config)
```

## Future Enhancements

Potential future additions:

1. Additional noise types (multiplicative, adversarial)
2. More domain randomization parameters (friction, force magnitude)
3. Curriculum learning support
4. Episode recording and replay
5. Automatic hyperparameter tuning for noise/randomization

## References

- Gymnasium Documentation: <https://gymnasium.farama.org/>
- Domain Randomization: OpenAI's work on sim-to-real transfer
- VectorEnv: <https://gymnasium.farama.org/api/vector/>
