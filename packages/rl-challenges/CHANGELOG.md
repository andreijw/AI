# Changelog

What changed in `rl-challenges`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [0.2.0]

### Changed

- **Breaking:** requires `rl-core~=0.2.0`. The CartPole and Pendulum factories import `read_env_settings` from
  `rl_core.utils`, rl-core's public API.

### Deprecated

- `make_env`, `make_vec_env` and `make_vec_env_from_config` of `rl_cartpole` and `rl_pendulum` warn on every call
  (`DeprecationWarning`) and are removed in 0.3.0. Build `CartPoleEnv(...)` or `PendulumEnv(...)` directly, or use
  `make_env_from_config`, which the challenges build with and which does not warn; for several training
  environments set `training.num_envs` (seeded with `rl_core.seeding.env_seed`, where these seed environment i
  with seed + i).

## [0.1.0] - 2026-11-11

### Added

- First release as its own distribution, split out of `rl-cartpole` 0.1.0 with unchanged import names.
