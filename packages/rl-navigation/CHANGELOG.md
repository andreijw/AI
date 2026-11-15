# Changelog

What changed in `rl-navigation`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [0.2.0]

### Added

- `FeatureSettings.max_bridged_angle` (default 3°): `segment_scan` joins usable beams across dropped ones, so circle
  extraction keeps working under dropout. This changes `extract_circles` and `EkfSlam` results on the 360-beam lidar;
  the 64-beam lidar is unchanged. `summarize_features` reports `reachable_rate`.
- `rl_navigation.planning`: `PlanningGrid` (`from_arena`, `from_occupied`), `astar`, `reachable`, `segments_free` and
  `shortcut`. `rl_navigation.sim.layouts.connected` is the flood fill `path_exists` uses.
- `rl_navigation.planning`: `PurePursuit`, `FollowerParams`, `Command`, `continuous_action`, `safe_discrete_action`,
  `Navigator`, `NavigatorParams`, `TrueMap`, `OnlineMap`, `polyline_free` and `PlanningGrid.cells_of`.

### Changed

- **Breaking:** the `challenge` extra requires `rl-core~=0.2.0`; the navigation environment imports
  `check_positive_int` and `non_negative_int` from `rl_core.utils`, rl-core's public API. The robot side (NumPy only)
  is unchanged.
- `rl_navigation.estimation.comparison` is a package of six modules with the same public names and the same `python -m`
  entry point. New `rl_navigation.estimation.averages` (`mean_or_nan`, `median_or_nan`).

### Fixed

- `EkfSlam` adds the second-order heading term to its measurement noise (`SlamSettings.second_order`, default on): a
  behaviour change; `second_order=False` restores the 0.1.0 filter. `summarize_slam` reports `collapses`,
  `low_nees_runs` and `run_nees_mean_median`.

## [0.1.0] - 2026-11-11

### Added

- First release as its own distribution, split out of `rl-cartpole` 0.1.0 with unchanged import names.
- Requires NumPy 1.26 or newer (ROS 2 Jazzy's), the oldest the robot-side tests are run with.
