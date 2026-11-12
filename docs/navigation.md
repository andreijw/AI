# Robot Navigation

The navigation half of the project: a NumPy robot simulator with Gymnasium environments to train agents in, and a
state-estimation library meant to run on a Jetson robot. Phase 8 of [roadmap.md](roadmap.md) holds what is still
open (SLAM, planning, the real robot). How to train and benchmark the navigation agents is in
[usage.md](usage.md); their numbers are in [results.md](results.md).

## Range-Sensor Navigation

`Navigation-v0` and `NavigationContinuous-v0` are this project's own Gymnasium environments (Phase 8 of
[`docs/roadmap.md`](roadmap.md)): a disc robot drives to a goal through randomly placed obstacles, seeing its
range sensors (see Observation) and told the goal's distance and bearing, as an oracle: a real robot would need
localisation to supply them. Each reset draws a new layout; `reset(options={"layout": layout})` plays a given
`rl_navigation.sim.Layout` instead.

- **Actions**: `Navigation-v0` has five discrete actions (straight ahead, forward arcs left and right at half speed,
  turns on the spot left and right). `NavigationContinuous-v0` takes two actions in [-1, 1]: forward speed from 0 to
  0.5 m/s (no reverse, as the sonars face forward) and turn rate within ±2 rad/s.
- **Observation**: goal distance, cos and sin of the goal's bearing, the last executed forward speed and turn rate,
  then the range readings: five HC-SR04 sonars by default (`sensor: "hc_sr04"`, 10 values in all), or a full-circle
  lidar (`sensor: "lidar"`, `lidar_beams`, default 64).
- **Reward**: 10 × the progress toward the goal over the starting distance, -0.01 per step and -10 on a collision.
  Reaching the goal or colliding ends the episode; otherwise it is truncated after 400 steps.
- **Framework settings** (`rl_navigation.environments.NavigationEnv`): `action_noise_prob` (discrete) or
  `action_noise_std` (continuous), `obs_noise_std`, `frame_stack`, `frame_deltas`, `num_obstacles`, and
  `domain_randomization` ranges of `wheel_slip`, `sensor_noise_std`, `dropout_prob` and `num_obstacles`.
  `mask_velocity: true` is refused.
- **Rendering**: `render_mode: "rgb_array"` draws the arena in NumPy (`rl_navigation/sim/render.py`), so
  `--record-video` and `record.py` work as for any challenge; `"human"` opens a window through Gymnasium's
  `HumanRendering`, which needs pygame-ce (installed by the `box2d` extra).

The benchmark runs `random`, `ppo` and `dqn` on `Navigation-v0` (`--challenge navigation`) and `random`, `ppo` and
`td3` on `NavigationContinuous-v0` (`--challenge navigation_continuous`), each from its committed config
(`configs/navigation_*.yaml`). Both ids are solved when the training average over 100 episodes reaches 7.0, which
is about 90% of training episodes at the goal (the maintainer's choice, 2026-11-07); the reward has no goal bonus, so
the threshold also asks for short trips. It also counts successes (`info["is_success"]` of an episode's last step):
the training log, the evaluation, the Markdown report's "Eval Success" column and the JSON carry the rate. Measured on
seeds 1 to 5 with greedy evaluation over 100 held-out layouts (2026-11-07,
[`docs/benchmarks/navigation_results.md`](benchmarks/navigation_results.md) and
[`docs/benchmarks/navigation_continuous_results.md`](benchmarks/navigation_continuous_results.md)): TD3 reaches
the goal in 96% of them after 100,000 steps and holds the threshold on 5 of 5 seeds, DQN 94% after 300,000 (5 of 5;
with squared error and a fixed rate it reached 11%; `td_loss: huber` and a decaying rate are the fix), PPO 88% after
200,000 with either action space (it reaches 7.0 on every seed but holds it on 1 of 5 discrete and 0 of 5
continuous), and the random agent 1%. A sweep over sensor noise and dropout (TESTING.md `ROB-2`, three TD3 seeds) kept
success between 87 and 98% on every row, from no noise to 0.4 m and from no dropout to 40% (one DQN seed fell to
82% at dropout 0.4), and a domain-randomization curriculum did not make TD3 more robust. A step with a sonar read
takes about 0.65 ms and a reset about 20 ms (measured on the Jetson in the design review).

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

## State estimation

`rl_navigation.estimation` estimates the robot's pose `(x, y, θ)` (metres and radians, in the map's frame) from its
wheel encoders and range readings on a known map. It is a library for the robot's own program: it imports only NumPy
and `rl_navigation.sim`, never Gymnasium or `rl_core` (`tests/test_import_footprint.py`), and nothing in training or
the benchmark calls it. It holds two filters with the same calls: an extended Kalman filter, `EkfLocalizer`, which
tracks a pose it is given at the start, and a particle filter, `MclLocalizer` (Monte Carlo localization), which can
also start from anywhere and recover from a kidnapping (the robot carried elsewhere without being told), as the EKF
cannot, though the EKF recovers from a small bump while standing still (below; [`CHANGELOG.md`](../CHANGELOG.md),
2026-11-06 to 2026-11-09, and the open item "State estimation follow-ups" in [`docs/roadmap.md`](roadmap.md)).

- **Encoders**: `rl_navigation.sim.WheelOdometry` models them. A reading is a `(2,)` array, the left and the right
  wheel's travel in metres since the previous reading, with noise variance `k × |travel|` per wheel. Its wheel base
  (0.2 m) and `k` (2.5e-4 m) are placeholders until they are calibrated on the robot.
- **Map**: the EKF reads the map only through the `RangeMap` protocol (`min_range`, `max_range`, `noise_std`,
  `expected_ranges(poses)` and `free(points, margin)`), and the particle filter only through `ParticleMap`, which
  extends it with `num_readings`, `bounds` and `expected_beam_ranges`. `SensorMap(arena, sensor)` provides both for
  a known `Arena` and a `RangeSensor`.
- **Loop**: call `predict(wheel_travel)` with each encoder reading, then `update(readings)` with the scan taken at
  that pose; `pose` and `covariance` hold the estimate. A reading at or beyond `max_range`, at or below `min_range`,
  or NaN counts as a dropout. `update` returns an `UpdateResult` saying which beams it used, gated as outliers, or
  left out as nonlinear (across a corner). A still robot's covariance shrinks to almost nothing, so after a bump
  of a few centimetres every beam fails the gate; after `reset_after=3` such scans (`None` disables) the EKF raises
  its variances to at least `reset_std²` (0.2 m, 0.2 m, 0.1 rad) and recovers (`UpdateResult.reset`). It handles
  bumps up to about 0.3 m and 0.1 to 0.3 rad with the lidar; the HC-SR04 sonars recover from position bumps up to
  0.3 m and 0.1 rad. Two or more resets in `EkfLocalizer.recent_resets` (the last 20 scans) mean the filter is lost.
  An update whose iteration does not converge, or whose scan is inconsistent with the prediction (its NIS above the
  chi-square 0.999 quantile), is *held*: the estimate moves but the covariance keeps its prior instead of shrinking, and
  the scan counts toward the reset (`UpdateResult.held`; `nis_quantile=None` turns it off). A scan with no usable beam
  counts too when the estimate is not in free space (`RangeMap.free` at margin 0: inside an obstacle or beyond a wall,
  where no robot can be): a lost estimate that has left the arena sees no usable beam, and before this off-map count
  (`count_off_map=False` turns it off) it reported itself fine while metres off. A reset raises the covariance and never
  moves the estimate, so this changes what the filter reports, not where it is. Without the hold a kidnapped or wrongly
  matched EKF reset and, on the very next scan, collapsed onto a wrong pose; with it the lost HC-SR04 run of `EST-3` is
  gone and kidnap recovery roughly doubles with the lidar. The HC-SR04 does not recover from a 0.3 rad heading bump:
  three of its four candidate sonars happen to agree with the wrong heading, so no reset fires (a documented limit). A
  large heading bump can settle it on a symmetric room's alias (the lidar, after 1.5 rad in a square room), and once its
  resets stop nothing signals it, as with MCL.

In the simulator, `World` stands in for the robot (run from the repository with the package installed):

```python
import numpy as np

from rl_navigation.estimation import EkfLocalizer, SensorMap
from rl_navigation.sim import LayoutParams, WheelOdometry, World, generate_layout, hc_sr04

rng = np.random.default_rng(0)
layout = generate_layout(rng, LayoutParams())  # on the robot: the known map of its room
sensor, encoders = hc_sr04(), WheelOdometry()  # placeholder noise until calibrated
world = World(layout.arena, sensor, layout.start, layout.goal)  # stands in for the robot
ekf = EkfLocalizer(
    SensorMap(layout.arena, sensor),
    encoders,
    pose=layout.start,
    covariance=np.diag([0.05, 0.05, 0.05]) ** 2,
)
readings, _ = world.read_sensors(rng)
for _ in range(100):
    v, omega = (0.2, 0.0) if readings.min() > 0.4 else (0.0, 1.0)  # drive, or turn away
    motion = world.step(v, omega, rng)
    ekf.predict(encoders.read(motion.v, motion.omega, world.dt, rng))  # (left, right) in metres
    readings, _ = world.read_sensors(rng)  # metres; max_range or NaN when there is no echo
    result = ekf.update(readings)
print(np.round(ekf.pose - world.pose, 3), int(result.used.sum()), "beams used")
```

It prints an error of at most 2 mm and 6 mrad, with 3 of the 5 sonars used in the last update. On the robot, the
encoder and sonar drivers replace `encoders.read` and `world.read_sensors`, and the `Arena` is the room's. The filter's
model breaks at a collision, where the wheels turn while the robot stays put, so its estimate drifts from then on.

Measured over 50 seeded layouts × 600 steps of a wandering robot with the filter's defaults (2026-11-06;
`TESTING.md` `EST-1`): position RMSE 5.7 mm and heading 10.8 mrad with the HC-SR04 sonars, 1.8 mm and 1.7 mrad with
a 64-beam lidar; with 30 % of readings dropped, 6.3 mm (sonars) and 2.0 mm (lidar). Dead reckoning on the same
encoders has a position RMSE of 0.93 m, and 45 of its 50 runs exceed 0.5 m of error at some step. The filter's mean
NEES is 3.2 to 3.3 and its median 2.4 to 2.5 (3 degrees of freedom); 96 to 97 % of steps lie under 9.35, the
chi-square 97.5 % point, where a consistent filter puts 97.5 %. Occasional spikes remain: the worst step of any run
reaches a NEES of about 340 with the sonars (seed 2) and 2,700 with the lidar (seed 30, a start-up transient from a
first update made 0.04 rad off). In an empty arena the lidar filter's ANEES over 30 runs, 3.05, lies inside the
95 % band [2.19, 3.94], a loose check that catches only a large miscalibration (CHANGELOG.md, 2026-11-06).

**Monte Carlo localization.** `MclLocalizer(range_map, odometry, pose=None, covariance=None, *, rng, settings)` takes
the same `predict(wheel_travel)` and `update(readings)` and exposes `pose`, `covariance`, `particles`, `weights` and
`cluster_weight`. Give it `pose` and `covariance` to start near a known pose, or leave them out to start with the
particles spread over the free space. It is augmented MCL (Thrun, Burgard and Fox, *Probabilistic Robotics*, Tables
8.2 and 8.3, with the low-variance resampler of Table 4.4) with a cut-down beam model: when the likelihood falls
against its long-run average, a share of the particles is redrawn over the free space, which is how it recovers from a
kidnapping. One deviation from Table 8.3: the particles injected on the previous scan are left out of that average,
since counted in they drag it down and keep injecting more for about 1000 scans after the robot is found; what remains
after a disturbance is about 1 % of the particles injected per scan, and a resampling on every scan, until the
long-run average catches up (about 700 scans, 70 s at 10 Hz, after a 10 cm bump with 300 particles).
`MclSettings` (frozen) holds the knobs; the defaults are 2000 particles, 16 beams per scan and a `model_std` of
0.15 m. At the defaults on one Jetson core a lidar step costs about 45 ms for the filter alone and 50 ms with the
simulator, and an HC-SR04 step about 65 ms with the simulator (about 120 ms with `max_beams=64`, every lidar beam),
so size `num_particles` and `max_beams` to the robot's control rate. `update`
returns an `MclUpdate`; its `used` marks the beams in the likelihood, so it is not comparable with the EKF's
`UpdateResult.used`. The reported `covariance` is conservative: `model_std` widens the cloud;
`MclSettings.covariance_scale` corrects it per axis (default `(1, 1, 1)`, unscaled; fitted values per sensor in
its docstring and `TESTING.md` `EST-2`). Known limitation: in a
symmetric arena, resampling collapses one of the mirror-image hypotheses within 1 to 3 scans, so `cluster_weight` ends
near 1 and cannot flag the ambiguity. Its evaluation (global localization and recovery from a kidnapping at step 300,
over fresh seeds 101 to 150) is `TESTING.md` `EST-2`; `evaluation.ekf()` and `evaluation.mcl()` build either filter
for `run_trial(make_localizer=...)`.

**EKF against MCL.** `python -m rl_navigation.estimation.comparison` runs both filters on the same seeded trials (each
sensor, with its own dropout and with 30 %) and prints a Markdown report: tracking accuracy and consistency (position
and heading RMSE, NEES, per-axis error against the reported variance), then a global start and a kidnapping at step 300
(shares localized and recovered, steps to get there, how often a filter is confidently wrong, and how often it is so
with a position σ under 5 cm). It saves one record per trial with `--records`, so an interrupted run resumes and jobs
run apart merge with `--from-records`; it writes no file unless `--report` is given. Run with the `TESTING.md` `EST-3`
settings (seeds 101 to 150, 600 steps; the report is
[`docs/benchmarks/localization_results.md`](benchmarks/localization_results.md)), the EKF tracks more accurately from
the noisy start (lidar position RMSE about 2 mm against MCL's 7 mm) but recovers from the kidnapping in only 12 to 28 %
of runs against MCL's 86 to 98 %, at about 2.5 to 4.4 ms a step against MCL's 37 to 51 ms; no EKF tracking run is lost
since the hold (2026-11-09; before it the HC-SR04 lost one of 50 at 30 % dropout). Since the off-map count (2026-11-10)
a lost EKF whose estimate has left the arena reports itself lost, and its HC-SR04 share of confidently wrong steps after
a kidnapping fell from 0.35 and 0.34 to 0.16 and 0.19. The module is a desk tool and `rl_navigation.estimation` does not
import it.

**Watching one trial.** `python -m rl_navigation.localization_video --seed 101 --mcl-scale 0.130 0.124 0.308 --output
<scratch>/seed101.mp4` writes one seeded lidar trial as a video: the true robot, its scan and path, MCL's particles
and estimate from a global start, and the EKF's estimate from the noisy start, each with its 2σ position ellipse,
with a kidnapping at step 300 (`--kidnap-at none` for none). Both filters run on the same trajectory
(`rl_navigation.estimation.recording`, painted with NumPy at 160 pixels per metre). `--stills` saves chosen frames as
PNGs; `--output` is required and never overwritten. The MP4 needs the `video` extra; `TESTING.md` `EST-4` says what
to look for.

![One frame of the seed-101 lidar localization video: the robot among obstacles with its scan and path, and a
filter's 2σ position ellipse](images/localization_seed101_step381.png)

*Frame 381 of the seed-101 lidar trial (`TESTING.md` `EST-4`): the true robot with its lidar scan and path, and a
position estimate drawn with its 2σ ellipse.*

## Mapping

`rl_navigation.mapping` builds a log-odds occupancy grid from range scans at known poses (Thrun, Burgard and Fox,
*Probabilistic Robotics*, Table 9.1), the first step of Phase 8's "Occupancy-grid mapping and SLAM"; SLAM, which
needs no known poses, comes in later pull requests. Like `estimation` it imports only NumPy and `rl_navigation.sim`,
so it runs on the robot.

```python
from rl_navigation.mapping import LIDAR_LOG_ODDS, OccupancyGrid
from rl_navigation.sim import lidar

grid = OccupancyGrid(lidar(), shape=(100, 100), resolution=0.05, model=LIDAR_LOG_ODDS)  # 5 x 5 m
grid.update(pose, readings)  # one scan at a known pose (x, y, theta)
occupied = grid.occupied()  # (rows, cols) booleans: more likely occupied than not
```

Each beam's cells are traced exactly (half-open intervals, so a hit on a cell border goes to the far cell): the
endpoint's cell is marked occupied and the cells before it free, with log-odds clamped to [0.12, 0.97]. Readings at
maximum range, dropouts and NaN change nothing. The lidar preset makes a cell occupied after one hit and frees it
again after 18 misses; the HC-SR04 preset traces each of its five cone rays with the mount's single reading.

On fresh seeds 101 to 150 (`TESTING.md` `MAP-1`, 600 steps a run) the lidar map is 99.96 % accurate with occupied
precision 1.0000, recall of the surfaces its beams reached 0.94 to 0.95 and free-space coverage 0.98; the HC-SR04's
is 99.1 % accurate with precision 0.87 to 0.88. Built from dead reckoning instead of the true poses, the lidar map
falls to precision 0.34, with hundreds of cells mapped free inside obstacles. An update takes 1.3 ms on the Jetson
(lidar, 100 × 100 cells; `MAP-2`). `rl_navigation.mapping.evaluation` drives and scores such runs; the package does
not import it.

## SLAM

`rl_navigation.slam` estimates the pose and a map of landmarks together, with no known map: landmark EKF-SLAM
(Phase 8, step 3, PR 2). Its landmarks are the arena's cylinders, fitted as circles to a lidar's scan; the robot
will carry a 360° lidar at about 1° per beam (the maintainer's decision, 2026-11-09), so the evaluation uses
`lidar(num_beams=360)`. Like `estimation` it runs on the robot: it imports NumPy, `rl_navigation.sim` and the
`estimation` core only.

```python
from rl_navigation.sim import lidar
from rl_navigation.slam import extract_circles

sensor = lidar(num_beams=360)
for feature in extract_circles(sensor, readings):  # one scan, metres
    x, y, radius = feature.z  # the centre in the robot's frame (x ahead) and the radius
    feature.covariance  # (3, 3), from the fit
```

A scan's usable readings are split where neighbouring points jump apart, and each segment of at least 8 points is
fitted with a circle: an algebraic start, then a geometric Gauss-Newton fit in which each residual is weighted by
the range noise projected on the surface (`σ·max(cos α, 0.3)`, α the beam's incidence), which keeps the fit's
covariance honest (mean NEES 2.8 to 2.9 on tuning seeds and 3.22 on seeds 101 to 150, against 3). A fit is kept when
its residuals pass a χ² test, its radius lies in 0.1 to 0.8 m, its centre lies beyond the points (a convex face),
two straight lines do not explain the points better (a box corner), and its result is sound (finite, a
positive-definite covariance, no σ above 0.1 m); nothing raises on a scan's content. The HC-SR04's five sonars, 30°
apart, never give a feature: landmark SLAM needs the lidar.

On fresh seeds 101 to 150 (`TESTING.md` `CIRC-1`, judged once on 2026-11-10) the 360-beam lidar finds 76 % of the
circles at least 8 beams reach, with a median centre error of 5.7 mm (p95 2.4 cm), a mean fit NEES of 3.22 and
0.05 % false fits, in 6.4 ms a scan (p99 11 ms) on one Jetson core; every gate passes. Most missed circles touch a
box, a wall or another circle, so their points join one segment that no circle fits. A circle's fits from successive
scans are not independent: they share its viewing geometry, and the radius and the centre trade off on a short arc,
so on tuning seeds a circle's mean radius error sat up to a millimetre or two from zero (a per-circle probe on
tuning seeds 1 to 5 while the SLAM was planned; `radius_bias_t2` above 1 reflects this and the scans' serial
correlation alike). Fusing hundreds of them therefore needs a model-error floor in the filter (1.5σ, 15 mm on this
lidar; below). `rl_navigation.slam.feature_evaluation` scores such runs; the package does not import it.

### Landmark EKF-SLAM

```python
from rl_navigation.sim import WheelOdometry, lidar
from rl_navigation.slam import EkfSlam

slam = EkfSlam(lidar(num_beams=360), WheelOdometry(), start_pose, start_covariance)
slam.predict(wheel_travel)  # (left, right) metres, as EkfLocalizer.predict
result = slam.update(readings)  # extract circles, associate, one batch update, extend the map
slam.pose, slam.covariance  # the robot, as a Localizer
# (L, 3) circles (x, y, radius) and their (L, 3, 3) covariance blocks:
slam.landmarks, slam.landmark_covariances
```

`EkfSlam` is Thrun's EKF-SLAM (*Probabilistic Robotics*, Tables 10.1 and 10.2): one Gaussian over the pose and every
landmark `(x, y, radius)`, the wheel travel as the control, and the body-frame circle as the measurement, with the fit's
covariance plus a **model-error floor** of `floor_factor` × the range noise (1.5: 15 mm at 1 cm), which absorbs the
fit's per-circle bias; without it the radius grows overconfident as hundreds of fits are fused (the floor brings its
NEES to about 1 on tuning seeds) and a landmark's own features fall outside its gate and come back as duplicates. A new
feature is associated by maximum likelihood (a χ²₃ gate at 0.99, one to one per scan); one beyond `new_landmark_nis`
(30) from every landmark becomes a candidate, and a landmark after 3 sightings within 10 cm, in 3 scans (the provisional
list); one in between is discarded. Its Jacobians are **first-estimate Jacobians** (FEJ, Huang, Mourikis and Roumeliotis
2008): a plain EKF-SLAM linearizes at estimates that move between steps and so learns about the absolute heading, which
nothing measures (in a test, a robot starting at σθ = 0.3 rad and seeing two circles for 30 scans: standard Jacobians
shrink σθ to 0.02 rad, FEJ keeps 0.30); FEJ keeps the system's unobservable directions (global translation and rotation)
unobservable, which the tests check to 1e-12. There is no iteration and no hold, both of which would break that. The
filter never raises on a scan's content: unsound features are skipped, labelled ones gated, and an update that leaves
the state non-finite, its covariance indefinite or the pose outside its bounds is undone (`SlamUpdate` counts each).

`rl_navigation.slam.evaluation` drives `run_trial`'s wanderer with the filter (`slam_trial`), optionally with an
oracle that labels each feature from the true pose (known correspondence), and scores it after aligning the
estimated trajectory to the true one (`TESTING.md` `SLAM-1`): SLAM cannot recover the start's error or the heading
drift before its first sighting, so its **aligned trajectory error** and the map's figures are the SLAM figures, and
the absolute error against dead reckoning is a low bar. Judged once on fresh seeds 101 to 150 (`SLAM-1`, 2026-11-10),
the 360-beam row passes 7 of its 9 gates: aligned error 6.4 cm mean and 1.1 cm median, 80 % of seen circles mapped, 0.24
spurious and 0.06 duplicate landmarks a run, 96 % of steps under the NEES bound, 13 of 50 runs lost in absolute terms.
It fails two: its absolute error is 0.44 of dead reckoning's (gate 0.40), and 9 runs have an aligned error above 10 cm
(gate 8), 4 of them association failures that the oracle avoids. An update takes 7.5 ms (8.4 ms with 30 landmarks) on
one Jetson core (`SLAM-2`). Known limits: FEJ stays linearized at a landmark's first
estimate, so a landmark first seen while the heading was poorly known leaves the run wrong (the lost runs; OC-EKF is
the follow-up); layouts with at most one circle barely beat dead reckoning (box corners are the follow-up); and with
30 % of beams dropped the extraction splits each arc at every dropped beam, so few circles are found.
