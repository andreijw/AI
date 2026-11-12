# rl-navigation

Range-sensor robot navigation for a real robot's program: a NumPy simulator (`rl_navigation.sim`), pose
estimation with an EKF and a particle filter (`rl_navigation.estimation`), occupancy-grid mapping
(`rl_navigation.mapping`) and landmark EKF-SLAM on lidar circle features (`rl_navigation.slam`). These need NumPy
only. Part of [andreijw/AI](https://github.com/andreijw/AI).

```bash
pip install "rl-navigation @ git+https://github.com/andreijw/AI@rl-navigation-v0.1.0#subdirectory=packages/rl-navigation"
```

Extras: `gym` (the Gymnasium environments `Navigation-v0` and `NavigationContinuous-v0`), `challenge` (the
navigation challenge on `rl-core`), `video` (a localization trial as an MP4). Changes: [CHANGELOG.md](CHANGELOG.md).
