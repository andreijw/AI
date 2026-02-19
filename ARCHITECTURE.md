# Architecture Documentation

## Overview

This document describes the architecture of the RL CartPole framework, explaining the design
decisions and how the components interact.

## Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new agents, environments, and training strategies
3. **Configurability**: All parameters controllable via configuration files
4. **Testability**: Components are independently testable
5. **Simplicity**: Clear, readable code that mirrors real robotics control loops

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
│                      (trainer.py)                           │
│  - Episode collection                                        │
│  - Agent updates                                             │
│  - Evaluation & checkpointing                                │
└───────────────┬──────────────────────────┬─────────────────┘
                │                          │
                ↓                          ↓
    ┌───────────────────┐      ┌───────────────────┐
    │   Environment     │      │      Agent        │
    │ (cartpole_env.py) │      │  (base_agent.py)  │
    │                   │      │                   │
    │ - State tracking  │      │ - Action selection│
    │ - Episode stats   │      │ - Policy updates  │
    │ - Gym wrapper     │      │ - Checkpointing   │
    └───────────────────┘      └───────────────────┘
                │                          │
                ↓                          ↓
    ┌───────────────────┐      ┌───────────────────┐
    │   Gymnasium       │      │  Agent Impls      │
    │   CartPole-v1     │      │ - RandomAgent     │
    │                   │      │ - (PPO - future)  │
    └───────────────────┘      └───────────────────┘

    ┌────────────────────────────────────────────────┐
    │            Utilities & Config                  │
    │  - Configuration loading (config.py)           │
    │  - Logging & metrics (logger.py)               │
    └────────────────────────────────────────────────┘
```

## Module Details

### 1. Environment Module (`environments/`)

**Purpose**: Wraps the Gymnasium CartPole environment to provide a standardized interface.

**Key Components**:

- `CartPoleEnv`: Main environment wrapper class
  - Manages episode lifecycle
  - Tracks episode statistics (steps, rewards)
  - Provides consistent API for training pipeline

**Design Decisions**:

- Wrapper pattern allows easy extension to other environments
- Episode tracking built-in for simplified metrics collection
- Configurable via constructor parameters (render mode, seed, etc.)

### 2. Agent Module (`agents/`)

**Purpose**: Defines the agent interface and implementations.

**Key Components**:

- `BaseAgent`: Abstract base class defining the agent contract
  - `select_action()`: Policy function
  - `update()`: Learning/optimization step
  - `save()`/`load()`: Persistence
- `RandomAgent`: Baseline implementation for testing

**Design Decisions**:

- Abstract base class enforces consistent interface across algorithms
- Separate training/inference modes via `training` flag
- Agent owns its own hyperparameters and state

**Future Extensions**:

- `PPOAgent`: Proximal Policy Optimization
- `DQNAgent`: Deep Q-Network
- `A3CAgent`: Asynchronous Advantage Actor-Critic

### 3. Training Module (`training/`)

**Purpose**: Orchestrates the training loop and manages the interaction between environment and agent.

**Key Components**:

- `Trainer`: Main training pipeline class
  - Episode collection loop
  - Calls agent updates
  - Periodic evaluation
  - Checkpoint saving
  - Metrics logging

**Design Decisions**:

- Trainer is algorithm-agnostic (delegates learning to agent)
- Configurable evaluation and checkpoint frequencies
- Trajectory collection for batch updates (ready for PPO)
- Clean separation of training and evaluation

### 4. Utilities Module (`utils/`)

**Purpose**: Shared utilities for configuration and logging.

**Key Components**:

- `config.py`: Configuration management
  - YAML/JSON loading
  - Config merging
  - Config persistence
- `logger.py`: Logging infrastructure
  - Console and file logging
  - Structured metrics logging (JSONL format)
  - Multiple logging levels

**Design Decisions**:

- YAML as primary config format (human-readable, hierarchical)
- Structured logging for easy metrics analysis
- Logger integration with training pipeline

## Data Flow

### Training Loop

1. Trainer initializes environment and agent
2. For each episode:
   - Environment reset
   - Agent selects actions based on observations
   - Environment steps and returns next state, reward
   - Trajectory data collected
   - Agent updates policy based on trajectory
3. Periodic evaluation on separate episodes
4. Periodic checkpoint saving

### Configuration Flow

1. Load YAML config file
2. Parse into nested dictionary
3. Pass relevant sections to components
4. Components use config for initialization

### Metrics Flow

1. Environment tracks episode statistics
2. Trainer aggregates metrics across episodes
3. Logger writes to console and structured files
4. Metrics can be analyzed post-training

## Extension Points

### Adding a New Agent

1. Inherit from `BaseAgent`
2. Implement required methods:
   - `select_action()`
   - `update()`
   - `save()` and `load()`
3. Add agent type to config
4. Update `train.py` to instantiate new agent

### Adding a New Environment

1. Create wrapper class (similar to `CartPoleEnv`)
2. Ensure it provides standard Gym interface
3. Add episode tracking if desired
4. Update config to specify new environment
5. Ensure observation/action spaces are compatible with agent

### Adding New Training Strategies

1. Extend `Trainer` class or create new trainer
2. Modify training loop as needed
3. Preserve agent/environment abstractions
4. Update config schema if needed

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
  - Environment wrapper functionality
  - Agent action selection and updates
  - Config loading and merging
- **Integration Tests**: Test component interactions
  - Environment + Agent
  - Trainer + Environment + Agent
- **End-to-End Tests**: Full training runs with small episode counts

## Future Architecture Considerations

As the project evolves toward more advanced robotics/AI capabilities:

1. **Vision Integration**: Add observation preprocessing pipeline
2. **SLAM**: Integrate localization and mapping modules
3. **ROS2**: Add ROS2 nodes for real robot deployment
4. **Embedded Inference**: Add model optimization for edge devices
5. **Multi-Agent**: Extend to support multiple agents
6. **Curriculum Learning**: Progressive environment difficulty

## Code Style Guidelines

- Type hints for function signatures
- Docstrings for all public methods
- Configuration over hard-coding
- Minimal dependencies
- Clear variable names
- Consistent formatting (Black)
