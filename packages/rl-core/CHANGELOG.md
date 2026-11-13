# Changelog

What changed in `rl-core`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [Unreleased]

### Fixed

- `ActorCriticAgent` and `ContinuousActorCritic` (with `n_steps: 0`) take `bootstrap_truncation` (default false):
  a time-limit cut is bootstrapped from V(next state) instead of treated as an end. PPO and continuous PPO share
  the same bootstrap; their behaviour is unchanged.
- `TorchTD3Agent` raises on the NumPy-only `_td3_targets` and `_target_actions` instead of silently running them,
  like the other NumPy training maths the torch backends replace.

### Changed

- Checkpoint save and load live in `rl_core.agents.checkpointing.CheckpointMixin`, which `BaseAgent` inherits; every
  name `rl_core.agents.base_agent` exported in 0.1.0 still imports from there. The checkpoint format is unchanged.

## [0.1.0] - 2026-11-11

### Added

- First release as its own distribution, split out of `rl-cartpole` 0.1.0 with unchanged import names.
