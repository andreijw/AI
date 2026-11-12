# Changelog

What changed in `rl-navigation`, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with [semantic versions](https://semver.org/) (0.x: a minor version may break the API). A release's notes are its
section here. The repository's day-by-day log is the root CHANGELOG.md.

## [Unreleased]

### Added

- `FeatureSettings.max_bridged_angle` (default 3°): `segment_scan` joins usable beams across dropped ones, so circle
  extraction keeps working under dropout. This changes `extract_circles` and `EkfSlam` results on the 360-beam lidar;
  the 64-beam lidar is unchanged. `summarize_features` reports `reachable_rate`.

## [0.1.0] - 2026-11-11

### Added

- First release as its own distribution, split out of `rl-cartpole` 0.1.0 with unchanged import names.
- Requires NumPy 1.26 or newer (ROS 2 Jazzy's), the oldest the robot-side tests are run with.
