# Changelog

What changed in `rl-core`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [0.2.0]

### Fixed

- `ActorCriticAgent` and `ContinuousActorCritic` (with `n_steps: 0`) take `bootstrap_truncation` (default false):
  a time-limit cut is bootstrapped from V(next state) instead of treated as an end. PPO and continuous PPO share
  the same bootstrap; their behaviour is unchanged.
- `TorchTD3Agent` raises on the NumPy-only `_td3_targets` and `_target_actions` instead of silently running them,
  like the other NumPy training maths the torch backends replace.

### Changed

- **Breaking:** the public API is the `__all__` of `rl_core.agents`, `rl_core.challenges`, `rl_core.environments`,
  `rl_core.export` (its modules `c_header`, `onnx_model`, `rollout` and `verify`, each with its own `__all__`),
  `rl_core.seeding`, `rl_core.training` and `rl_core.utils`; `rl_core.__all__` is `["__version__"]`. Every other
  module path (`rl_core.utils.config_values`, `rl_core.agents.base_agent`, ...) is private and may change in any
  release. No name the 0.1.0 packages exported has left. `read_env_settings`, `check_positive_int` and
  `non_negative_int` now import from `rl_core.utils`. `rl_core.seeding` and the four export modules now have an
  `__all__`, so a star import gives only those names, and the rest at those public paths (`WRAPPER_STREAM`, `ATOL`,
  ...) are private. `import rl_core.export` now loads its four modules, and through `rollout` also
  `rl_core.challenges` and Gymnasium (never onnx or onnxruntime).
- Checkpoint save and load live in `rl_core.agents.checkpointing.CheckpointMixin`, which `BaseAgent` inherits; every
  name it defined in 0.1.0 still imports from `rl_core.agents.base_agent` (a private path from 0.2.0: import
  from `rl_core.agents`). The checkpoint format is unchanged.

## [0.1.0] - 2026-11-11

### Added

- First release as its own distribution, split out of `rl-cartpole` 0.1.0 with unchanged import names.
