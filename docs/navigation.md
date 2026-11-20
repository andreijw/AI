# Robot Navigation

The navigation half of the project: a NumPy robot simulator with Gymnasium environments to train agents in, and a
state-estimation, mapping, SLAM and planning library meant to run on a Jetson robot. Phase 8 of [roadmap.md](roadmap.md)
holds what is still open (the state-estimation and planning follow-ups, the real robot); the planner's judged comparison
with the trained policies is [benchmarks/planning_results.md](benchmarks/planning_results.md). How to train and
benchmark the navigation agents is in [usage.md](usage.md); their numbers are in [results.md](results.md).

The explanations, the measurements and the history of each step moved to the navigation track of the teaching pages,
[docs/learn/navigation/](learn/navigation/sim-and-sensors.md). This page keeps what a developer of `rl_navigation` or a
robot's installer needs: the install, the module map and the environment's standalone use.

## The track

| Step | Page | Module | Imports |
| :--- | :--- | :--- | :--- |
| The simulator, sensors and encoders | [sim-and-sensors](learn/navigation/sim-and-sensors.md) | `rl_navigation.sim` | NumPy |
| The Gymnasium environments and their agents | [the challenge page](learn/challenges/navigation.md) | `rl_navigation.gym_env`, `rl_navigation.environments` | Gymnasium; the second also `rl_core` |
| State estimation: EKF and Monte Carlo localization | [estimation](learn/navigation/estimation.md) | `rl_navigation.estimation` | NumPy, `sim` |
| Occupancy-grid mapping | [mapping](learn/navigation/mapping.md) | `rl_navigation.mapping` | NumPy, `sim` |
| Landmark EKF-SLAM | [slam](learn/navigation/slam.md) | `rl_navigation.slam` | NumPy, `sim`, the `estimation` core |
| A*, pure pursuit and the navigator | [planning](learn/navigation/planning.md) | `rl_navigation.planning` | NumPy, `sim`, `mapping` |
| Policy export to C and ONNX | [export-to-robot](learn/navigation/export-to-robot.md) | `rl_core.export` | the `onnx` extra for ONNX |

The desk tools are not imported by the libraries: `rl_navigation.estimation.comparison`,
`rl_navigation.estimation.runs`, `rl_navigation.mapping.evaluation`, `rl_navigation.slam.evaluation`,
`rl_navigation.slam.feature_evaluation`, `rl_navigation.localization_video` and `rl_navigation.planning_comparison`
(which imports the Gymnasium layer but never `rl_core`).

## Installing on the robot

All of this is the `rl-navigation` distribution (`packages/rl-navigation/`). On its own it needs NumPy only: the
simulator, `estimation`, `mapping`, `slam` and `planning` import nothing else. Its extras add the rest: `gym`
(Gymnasium, for `rl_navigation.gym_env`), `challenge` (`rl-core`, for the navigation challenge on the framework) and
`video` (`rl_navigation.localization_video`). Ubuntu 24.04's system Python is externally managed (PEP 668), so a ROS 2
node uses a virtual environment created with `--system-site-packages`, which keeps `rclpy` and the apt NumPy visible,
and installs `rl-navigation` from its release tag into it (the newest release's tag; `@develop` for unreleased
work):

```bash
python3 -m venv --system-site-packages ~/robot-venv
~/robot-venv/bin/pip install \
    "rl-navigation @ git+https://github.com/andreijw/AI@rl-navigation-v0.2.0#subdirectory=packages/rl-navigation"
```

ROS 2 Jazzy runs Python 3.12, which the distribution requires, and its apt NumPy is 1.26, the oldest it allows:
the `Packages` CI job runs the robot-side tests from the rl-navigation wheel alone, without rl-core or Gymnasium, on
NumPy 1.26.

## Using the environment on its own

The environment does not need the training framework. `rl_navigation.gym_env` imports only Gymnasium and the NumPy
simulator `rl_navigation.sim`, never `rl_core`, so another project can use it directly:

```python
import gymnasium as gym

import rl_navigation.gym_env  # registers Navigation-v0 and NavigationContinuous-v0

env = gym.make("NavigationContinuous-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`info` holds `is_success`, `collision` and `goal_distance`. The framework's noise and domain randomization live in
`rl_navigation.environments`, which does import `rl_core`.
