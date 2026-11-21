# Changelog

What changed in `rl-challenges`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [0.3.0]

### Changed

- **Breaking:** requires `rl-core~=0.3.0`.
- **Breaking:** an environment section of CartPole or Pendulum that sets `async_envs` to a value now raises
  "does not support environment setting(s) async_envs", as for any unknown setting; `false` and `null` are still
  accepted, being "left off". The key had no effect in `make_env_from_config`.

### Removed

- `make_env`, `make_vec_env` and `make_vec_env_from_config` of `rl_cartpole` and `rl_pendulum`, deprecated since
  0.2.0. Build `CartPoleEnv(...)` or `PendulumEnv(...)` directly, or use `make_env_from_config`; for several training
  environments set `training.num_envs` (environment i is seeded with `rl_core.seeding.env_seed`; the removed
  factories seeded it with seed + i). Stay on `~=0.2.0` to keep the old names.

## [0.2.0] - 2026-11-20

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
