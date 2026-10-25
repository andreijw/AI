# Architecture Documentation

## Overview

This document describes the decoupled architecture of the reinforcement learning framework, explaining its design
principles, module boundaries, continuous and discrete agent abstractions, challenge packages, and artifact scoping.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           src/benchmark.py & CLI                            │
│                         (Multi-Agent Benchmarking)                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│      Challenge packages: src/rl_<name>/  (found by discover_challenges)     │
│                                                                             │
│  rl_cartpole     CartPoleEnv     CartPoleChallenge     discrete; noise, DR  │
│  rl_pendulum     PendulumEnv     PendulumChallenge     continuous; noise, DR│
│  rl_acrobot      AcrobotEnv      AcrobotChallenge      discrete swing-up    │
│  rl_mountaincar  MountainCarEnv  MountainCarChallenge  discrete, exploration│
│                                  MountainCarContinuousChallenge  continuous │
│                                                                             │
│  Each holds a BaseEnvWrapper subclass, ENV_DEFAULTS + make_env_from_config, │
│  and a BenchmarkChallenge subclass with AGENT_CLASSES and CONFIG_MAP        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 src/rl_core/                                │
│                     (Algorithm-Agnostic Core Framework)                     │
│                                                                             │
│  agents/               training/                 utils/                     │
│  - BaseAgent           - Trainer (loop)          - config.py (YAML/JSON)    │
│  - Discrete Agents     - Checkpoint manager      - logger.py (JSONL)        │
│  - Continuous Agents                             - video.py (MP4 recording) │
│  - ReplayBuffer                                  - visualization.py (plots) │
│                                                                             │
│  challenges/                                                                │
│  - BenchmarkChallenge, CHALLENGES registry, discover_challenges()           │
│                                                                             │
│  environments/                                                              │
│  - BaseEnvWrapper (seeding, observation noise, episode statistics)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Core Decoupling**: Algorithms and trainers in `rl_core` have zero coupling to specific Gymnasium environments.
2. **Challenge Isolation**: Challenge-specific environments, physics domain randomizations, and configs reside in
   independent packages (`rl_cartpole`, `rl_pendulum`, `rl_acrobot`, `rl_mountaincar`).
3. **Discrete & Continuous Symmetry**: Shared base interfaces support discrete action selection and continuous
   bounded control (Gaussian policies).
4. **Structured Artifact Scoping**: Generated logs, checkpoints, plots, and videos are partitioned deterministically
   by challenge and algorithm: `<artifact_root>/<challenge>/<agent>/`.
5. **Pure NumPy Neural Foundations**: Neural network architectures and gradient optimizations (Adam, He-init,
   clipped PPO, GAE) are implemented natively in NumPy without heavyweight external frameworks. One `Adam` in
   `rl_core/agents/optim.py` serves the continuous agents and, with `optimizer: adam` in the agent config,
   REINFORCE, Actor-Critic, PPO and DQN, which otherwise use plain SGD (DQN's SGD clips each gradient element at
   `grad_clip_value`). DDPG's two-hidden-layer actor and critic use the shared `rl_core/agents/mlp.py`
   (stateless forward and backward passes over named weights), and every continuous agent reads its bounds through
   `rl_core/agents/bounds.py`. Discrete PPO takes the standard form when its
   config sets `rollout_steps`, `minibatch_size` and `gae_lambda`: it buffers whole episodes until it holds
   `rollout_steps` transitions, computes GAE over them (`rl_core/agents/returns.py`, shared with continuous PPO)
   and takes `ppo_epochs` passes of shuffled `minibatch_size` steps. Without those keys it trains on each episode
   alone, full batch, with return-minus-value advantages. The Trainer still calls `update` once per episode; PPO
   returns no metrics on the episodes it only stores. Both PPO agents keep those episodes in one `EpisodeRollout`
   (`rl_core/agents/rollout.py`), which saves them in checkpoints and restores them on load; a rollout still partly
   filled when training ends is not trained on. Each episode ends its GAE trace, and an episode cut by the time limit
   (or by `max_steps_per_episode`) adds γ·V(next state) to its last reward, so CartPole's 500-step limit is not a
   terminal state. Continuous PPO buffers whole episodes the same way with its own `rollout_steps` and
   applies the same rule; both read it from `episode_end` in `rl_core/agents/returns.py`. DQN and DDPG
   store the trainer's `terminated` flag, so they bootstrap through time limits too. The continuous Actor-Critic
   replays each episode in `n_steps` chunks, each with GAE targets bootstrapped from the chunk's next state unless
   the episode truly terminated. Both continuous Gaussian agents can bound their policy mean inside the action
   bounds (`bounded_mean`: bias + scale·tanh(z)); without it the linear mean head can drift past a bound, where every
   executed action is the same. DDPG explores with Gaussian noise or, with `noise_type: ornstein_uhlenbeck`, with
   the temporally correlated noise of `rl_core/agents/exploration_noise.py`, and resets that noise in `end_episode`,
   which the Trainer calls after every training episode's `update` and before any checkpoint save
   (`BaseAgent.end_episode`, a no-op by default). `update` itself no longer resets it, so an agent updated
   mid-episode keeps its noise process running. TD3 (`TD3Agent`, a `DDPGAgent` subclass) adds a second critic and
   target critic (taking the smaller target), target-policy smoothing noise drawn from its own seed stream, and
   actor and target updates on every `policy_delay`-th gradient step.

---

## Component Architecture

The high-level interaction between the training orchestration, environment wrappers, and agent policies is
visualized below:

```mermaid
flowchart TD
    subgraph CLI ["Entrypoints"]
        Train["train.py"]
        Record["record.py"]
        Bench["src/benchmark.py"]
    end

    subgraph Challenges ["Challenge Layer"]
        CartPole["rl_cartpole: CartPoleEnv (Discrete)"]
        Pendulum["rl_pendulum: PendulumEnv (Continuous)"]
        Acrobot["rl_acrobot: AcrobotEnv (Discrete)"]
        MountainCar["rl_mountaincar: MountainCarEnv (Discrete and Continuous)"]
    end

    subgraph Core ["rl_core Framework"]
        Trainer["Trainer (training loop & eval)"]
        
        subgraph Agents ["Agent Implementations"]
            BaseAgent["BaseAgent"]
            DiscreteAgents["Discrete: Random, Reinforce, A2C, PPO, DQN"]
            ContinuousAgents["Continuous: Random, A2C, PPO, DDPG"]
        end

        EnvBase["environments: BaseEnvWrapper"]

        subgraph Utils ["Utility Infrastructure"]
            Config["config.py"]
            Logger["logger.py"]
            Video["video.py"]
            Viz["visualization.py"]
        end
    end

    Train --> Core
    Record --> Core
    Bench --> Core
    Train --> Challenges
    Record --> Challenges
    Bench --> Challenges
    Trainer --> Challenges
    Trainer --> Agents
    Trainer --> Utils
    CartPole --> EnvBase
    Pendulum --> EnvBase
    Acrobot --> EnvBase
    MountainCar --> EnvBase
```

---

## Agent Class Hierarchy

The agent hierarchy provides clean inheritance across discrete and continuous control paradigms:

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +int observation_dim
        +int action_dim
        +dict config
        +select_action(obs, training)
        +update(batch)* dict
        +save(path)
        +load(path)
        #_checkpoint_arrays()* dict
        #_load_checkpoint_arrays(data)*
    }

    class RandomAgent {
        +select_action(obs, training) int
        +update(batch) dict
    }

    class ContinuousRandomAgent {
        +ndarray action_low
        +ndarray action_high
        +select_action(obs, training) ndarray
        +update(batch) dict
    }

    class ReinforceAgent {
        +select_action(obs, training) int
        +update(batch) dict
    }

    class ActorCriticBase {
        <<abstract>>
        #ndarray _W1, _b1, _W_pi, _b_pi, _W_v, _b_v
        +select_action(obs, training) int
    }

    class ActorCriticAgent {
        +update(batch) dict
    }

    class PPOAgent {
        +float clip_epsilon
        +int ppo_epochs
        +int rollout_steps
        +int minibatch_size
        +float gae_lambda
        +update(batch) dict
    }

    class DQNAgent {
        +ReplayBuffer replay_buffer
        +int anneal_steps
        +select_action(obs, training) int
        +update(batch) dict
    }

    class ContinuousActorCriticBase {
        <<abstract>>
        +ndarray action_low
        +ndarray action_high
        +bool bounded_mean
        #ndarray _W1, _b1, _W_mu, _b_mu, _log_std, _W_v, _b_v
        +forward_actor(obs) tuple
        +forward_critic(obs) float
        +select_action(obs, training) ndarray
    }

    class ContinuousActorCriticAgent {
        +update(trajectory) dict
    }

    class ContinuousPPOAgent {
        +float clip_ratio
        +int anneal_steps
        +int ppo_epochs
        +update(trajectory) dict
    }

    class TorchDDPGAgent {
        +BACKEND = "torch"
        +device, dtype, cuda_graph
        #_train_steps(num_updates) dict
    }

    class DDPGAgent {
        +BACKENDS = torch: TorchDDPGAgent
        +ReplayBuffer replay_buffer
        +ndarray action_low
        +ndarray action_high
        #dict _actor, _critic, _target_actor, _target_critic
        #ExplorationNoise _noise
        +select_action(obs, training) ndarray
        +update(batch) dict
        #_train_steps(num_updates) dict
    }

    BaseAgent <|-- RandomAgent
    BaseAgent <|-- ContinuousRandomAgent
    BaseAgent <|-- ReinforceAgent
    BaseAgent <|-- ActorCriticBase
    ActorCriticBase <|-- ActorCriticAgent
    ActorCriticBase <|-- PPOAgent
    BaseAgent <|-- DQNAgent
    BaseAgent <|-- ContinuousActorCriticBase
    ContinuousActorCriticBase <|-- ContinuousActorCriticAgent
    ContinuousActorCriticBase <|-- ContinuousPPOAgent
    BaseAgent <|-- DDPGAgent
    DDPGAgent <|-- TorchDDPGAgent
    DDPGAgent <|-- TD3Agent
```

Every agent is NumPy by default. `BaseAgent.class_for_config(config)` returns the class to build for the config's
`backend` key (`numpy` when absent): the class itself, or the version its `BACKENDS` names, imported only then.
`BenchmarkChallenge.make_agent` builds through it, so `backend: torch` in a DDPG config builds `TorchDDPGAgent`, which
runs DDPG's gradient steps in PyTorch (`_train_steps`, on the GPU as a CUDA graph) and keeps everything else, acting
included, in NumPy. Its checkpoints are DDPGAgent's format, so either backend loads the other's. Only
`rl_core/agents/torch_*.py` import torch; nothing imports them directly. Inheriting DDPGAgent brings its NumPy
training maths too: TorchDDPGAgent overrides those methods to raise, since running them would train the NumPy
mirror of its weights rather than the device. If TD3 or SAC need the same backend, splitting DDPG's maths from its
bookkeeping is the cleaner shape.

---

## Training and Rollout Execution Flow

The sequence diagram illustrates how `Trainer` coordinates episode rollouts, trajectory tracking, gradient updates,
evaluation checkpoints, and video captures. `train.py` and the benchmark give the trainer a separate evaluation
environment (built by the challenge's `make_env`, or `setup_eval_env` in the benchmark) and an evaluation seed from
`rl_core.seeding.eval_seed_for`: evaluation episode i resets that environment with `eval_seed + i`, so every
evaluation in a run plays the same start states and how often a run evaluates does not change its training episodes.
The random agents draw their evaluation actions from a separate generator for the same reason. An agent checkpoint
holds every generator the agent draws from and, for DQN and DDPG, the replay buffer, so a loaded agent continues
exactly as the saved one would have. Beside it the trainer writes `trainer_episode_<n>.npz` (episode and step
counts, reward and length histories, the update-metric window, the evaluation seed and, for an environment that
implements `rl_core.environments.ResumableEnv`, its generators) as a `TrainerState` that also records the agent
file's SHA-256, both files through
`rl_core.training.checkpoints.write_atomically`; `Trainer.resume` (`train.py --resume`) reads both, so a resumed
run ends exactly as an uninterrupted one. Without an evaluation
environment (direct `Trainer` callers and tests) evaluation runs on the training environment. `record.py` passes its
playback environment as the evaluation environment with the same seed, so a playback shows the training run's
evaluation start states.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / CLI
    participant Trainer as Trainer
    participant Env as Challenge Env (CartPole, Pendulum, Acrobot, MountainCar)
    participant Agent as Agent (Discrete / Continuous)
    participant Logger as Structured Logger
    participant Disk as Artifact Storage

    User->>Trainer: train()
    loop Episode Loop (1 to num_episodes)
        Trainer->>Env: reset()
        Env-->>Trainer: obs, info
        loop Step Loop (until done or max_steps)
            Trainer->>Agent: select_action(obs, training=True)
            Agent-->>Trainer: action
            Trainer->>Env: step(action)
            Env-->>Trainer: next_obs, reward, terminated, truncated, info
            Trainer->>Trainer: record transition (obs, action, reward, next_obs, done)
        end
        Trainer->>Agent: update(trajectory / batch)
        Agent-->>Trainer: loss_metrics
        opt Periodic Evaluation
            Trainer->>Trainer: evaluate(eval_episodes) on the evaluation env, reset(seed=eval_seed + i)
            Trainer->>Logger: log eval stats & rewards
        end
        opt Periodic Checkpoint
            Trainer->>Agent: save(checkpoint_path)
            Agent->>Disk: write agent_episode_<n>.pt
            Trainer->>Disk: write trainer_episode_<n>.npz (counters, histories, env generators)
        end
    end
    Trainer-->>User: training_summary_dict
```

---

## Artifact Directory Scoping

To avoid collisions when multiple agents or challenges are evaluated, artifacts are partitioned into structured
challenge and algorithm subdirectories:

```
checkpoints/
├── cartpole/
│   ├── ppo/
│   │   ├── agent_episode_500.pt
│   │   └── trainer_episode_500.npz
│   └── dqn/
└── pendulum/
    ├── ppo/
    │   └── agent_episode_200.pt
    └── actor_critic/

logs/
├── cartpole/
│   └── ppo/
│       ├── rl_cartpole_<timestamp>.log
│       └── metrics_<timestamp>.jsonl
└── pendulum/
    └── ppo/

videos/
├── cartpole/
│   └── ppo/
│       └── cartpole-training-episode-100.mp4
└── pendulum/
    └── ppo/

plots/
├── cartpole/
│   └── ppo/
│       └── training_metrics.png
└── pendulum/
    └── ppo/
```

---

## Extension Guidelines

### Adding a New Challenge

1. Create a new package under `src/rl_<challenge>/`:
   - `environments/<challenge>_env.py`: the challenge's wrapper, a subclass of
     `rl_core.environments.BaseEnvWrapper`. The base already handles seeding (the constructor seed on the first
     reset only), observation noise, episode statistics and `wrap_env`. The subclass validates its own settings,
     calls `_make_env()`, and overrides `_prepare_action` (action noise) and `_apply_domain_randomization`
     (per-episode physics); both default to doing nothing.
   - `environments/make_env.py`: `make_env`, an `ENV_DEFAULTS` dict naming each environment setting the challenge
     supports with its default, and `make_env_from_config`, which reads them with
     `rl_core.utils.config.read_env_settings`: null takes the default, and any other setting given a value is
     rejected rather than ignored. A vectorized factory only if something uses it (`rl_acrobot` has none).
   - `benchmark.py`: a `BenchmarkChallenge` subclass from `rl_core.challenges`, decorated with `@register_challenge`. It
     implements `make_env(env_config, render_mode)` (usually the package's `make_env_from_config`) and lists its agents
     in the class variable `AGENT_CLASSES` (agent type to class). The base class's `supported_agents` and `make_agent`
     read that table and size each agent from the environment's spaces; a challenge overrides only
     `agent_config(agent_config, env)` when its environment adds to the agent's config: a continuous challenge returns
     `with_box_action_bounds(agent_config, env, self.name)` from `rl_core.challenges`, as Pendulum and
     MountainCarContinuous do, to pass on its Box action bounds. The
     base class builds benchmark runs from these. `train.py` and `record.py` pick the challenge whose `name` equals the
     config's `environment.name` exactly (`select_challenge`), or the one `--challenge` names, and build through it, so
     no script needs an edit. Two challenges may share an environment id; the config then needs `--challenge`.
     One package may hold several challenges sharing a wrapper, each with its own `ENV_DEFAULTS`-style table and
     factory (`rl_mountaincar`: `make_env_from_config` and `make_continuous_env_from_config`).
     `discover_challenges()` imports every `src/rl_*/benchmark.py`, so the runner needs no edit; a package that fails to
     import is reported with a warning, in `run_benchmark`'s unknown-challenge error and at the end of `src/benchmark.py
     --help`. A challenge's `__init__` must work with no arguments or with `config_dir` alone (the base class provides
     one), and its `key` must not already be registered.
2. Add reference YAML configurations under `configs/<challenge>_<agent>.yaml`.
3. Add corresponding test suites in `tests/test_<challenge>.py`, and the challenge's rows to the per-challenge
   tables in `tests/test_challenge_selection.py` (`EXPECTED_AGENTS`, `ENV_CLASSES`, `CONFIGURED`) and
   `tests/test_committed_configs.py` (`STEP_BUDGETS`).

### Adding a New Continuous Algorithm

1. Inherit from `ContinuousActorCriticBase` (or `BaseAgent`) in `src/rl_core/agents/`.
2. Implement `update(trajectory: list[dict[str, Any]]) -> dict[str, float]` with closed-form or autodiff gradients.
3. Expose the agent in `src/rl_core/agents/__init__.py`.
4. Register the agent in the challenge's `AGENT_CLASSES` map.
