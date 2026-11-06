# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

Order (the maintainer's decision, 2026-11-04): the repo is the foundation for robotics work, so Phase 8 (Robot
Navigation) leads. Its first item, the range-sensor navigation environment, ran in parallel with Phase 7's SAC and
BipedalWalker item (done; CHANGELOG.md, 2026-11-05), and Phase 8 continues in order. The
Training and Code Health items are picked as they become useful (parallel environments before a PPO comparison on
BipedalWalker or navigation).

### Workflow Tooling

No open items.

### Training

- [ ] **Parallel environments for the on-policy agents** (added by the maintainer, 2026-11-04)
  - **Description**: The `Trainer` steps one environment, one episode at a time. rl-zoo's A2C collects from 8
    environments in lockstep (64 transitions per update on Pendulum), and its PPO configs usually from several too;
    ours cannot follow either recipe. Add `training.num_envs` for the on-policy agents (REINFORCE, and Actor-Critic
    and PPO in both action kinds): Gymnasium's `SyncVectorEnv` (or `AsyncVectorEnv` with `async_envs`) with autoreset,
    each update built from every environment's steps, episode records and the solve rule per finished episode.
  - **Acceptance criteria**: `num_envs: 1`, or no key, is bit-for-bit today's training (the golden tests); with N
    environments each one's episodes are recorded, cut and bootstrapped as now (a time-limit cut bootstraps, a true
    termination does not); evaluation, checkpoints and `--resume`, the curriculum and observation normalisation work
    as with one; the replay-buffer agents refuse `num_envs` above 1 until they are designed for it. Then the
    measurement the Code Health idea asks for: the continuous Actor-Critic with 8 environments and `n_steps: 8`
    against the committed config, seeds 1 to 5, under the win rule.
  - **Known edge cases**: environments finishing episodes at different steps; Gymnasium's autoreset mode (the
    final observation of a cut episode is needed to bootstrap); seeding N environments reproducibly from one seed; a
    step budget crossed partway through a vector step; `update_every` and the actor-critic's chunking across
    environments.
  - **Out of scope**: vectorised DQN, DDPG and TD3; changing any committed config before it is measured.

### Code Health

The last health dashboard (brooks-health, 2026-10-22) scored 94/100 with 0 Critical findings; its four Warnings are
fixed (CHANGELOG.md, 2026-10-22 and 2026-10-25). The five ideas the maintainer picked on 2026-10-28 are done
(CHANGELOG.md, 2026-10-28 and 2026-10-29), and the ideas picked on 2026-11-03 are measured (CHANGELOG.md,
2026-11-04). A learning change is committed only if, on seeds 1 to 5 against the
committed config, more seeds hold the threshold or the mean final average is better on at least 4 of 5 paired seeds
with no seed clearly worse; otherwise it stays opt-in and is recorded as tried.

Parked ideas, not scheduled. Each needs the maintainer's pick, and some a decision on the win rule, before it becomes
an item.

- [ ] **The continuous Actor-Critic's training average**: its greedy policy matches rl-zoo's A2C, but its 100-episode
  training average stays near -300 on Pendulum (CHANGELOG.md, 2026-10-18), -283 with observation normalisation
  (CHANGELOG.md, 2026-11-04). Tried without closing it: true A2C (`update_every: 8`; CHANGELOG.md, 2026-10-30) and
  50-step chunks (50 transitions per update from one environment, against rl-zoo's 64 from eight; CHANGELOG.md,
  2026-11-04). Untried: rl-zoo's 8 parallel environments, which wait on the Training item "Parallel environments
  for the on-policy agents"; and gSDE exploration (its PPO version is in `~/rl-experiments/item5/` on the
  development machine).
- [ ] **SAC's BipedalWalker training average does not hold**: `configs/bipedalwalker_sac.yaml` solved 300 on 4
  of seeds 1 to 5 within 500,000 steps but held on 1 (final 100-episode averages 251 to 307), while every seed's
  greedy evaluation was 321 to 327 (CHANGELOG.md, 2026-11-05). The training average includes exploration noise.
  Untried: rl-zoo's gSDE exploration (`use_sde`, `log_std_init: -3`), which its recipe uses on this environment and
  this framework lacks (dropped from the item by the maintainer, 2026-11-05); TD3 at rl-zoo's 1,000,000 steps.
- [ ] **Why Pendulum PPO collapses with observation normalisation**: with `normalize_observations: true`
  `configs/pendulum_ppo.yaml` held -200 on 2 of seeds 1 to 5 against 4; seed 5 reached -187 by episode 310 and slid
  to about -1000 by episode 900 with the entropy steady and a small value loss, while the learning rate annealed
  (CHANGELOG.md, 2026-11-04). A scratch variant with the statistics frozen per rollout collapsed on 3 of 5
  (2026-10-29). The same option helps the continuous Actor-Critic and CartPole PPO (`TR-24`), so a defect in how PPO
  uses normalised inputs is possible, not only a poor fit. A `training-doctor` diagnosis on seed 5 would settle
  which: the statistics' drift late in the run, the policy's mean action and log standard deviation, and the update
  metrics around the slide.

### Phase 6 — Advanced RL & Robotics Extensions

No open items.

### Phase 7 — Challenge Ladder

New environments in order of difficulty, each tier adding one new hardship so that a failure points at one cause.
Every challenge is its own `rl_<name>` package that `rl_core.challenges` discovers; solve thresholds are
Gymnasium's `reward_threshold` unless the maintainer sets another, judged by the 100-episode training average.
Tier 1 began with Acrobot-v1 (CHANGELOG.md, 2026-10-22). The items below are sketches to turn into full items
when picked.

- [ ] **MountainCar-v0: Hold -110**
  - **Description**: MountainCar-v0's DQN does not reliably hold -110: with a 100,000-transition buffer, 2 of 10 seeds
    end at or above it (both crossing it only near the end of the run), and the final averages are -107 to -132
    (CHANGELOG.md, 2026-10-25). With observations normalised when sampled, committed 2026-11-04, all of seeds 1 to 5
    and 11 to 15 reach -110 and 3 of 10 hold it, final averages -106 to -119 (CHANGELOG.md, 2026-11-04).
  - **Diagnosis (2026-10-25)**: the agent reaches the flag reliably, so exploration is unlikely to be the limit. On seed
    4 the greedy policy evaluated -100 to -107 from episode 550 to 700, then settled near -125 with a spread of about
    20 and did not stay good: consistent with a good policy found and then forgotten. A larger buffer helps the
    training average (better on 10 of 10 seeds, by +6 on the held-out seeds 6 to 10) but not the held-out greedy
    evaluation. Note that -110 is judged on the training average, exploration included, while rl-zoo's -100.8 is a
    greedy evaluation.
  - **Diagnosis (2026-10-27)**: no bug. Through `train.py` on seeds 1 to 5 (final averages -131, -122, -108, -118,
    -107; mean -117.2, bit-for-bit repeatable single-threaded; the -117.8 of 2026-10-25 came through the benchmark on
    an older commit and was not reconciled): no divergence; mean Q (about -31 late) matches the discounted return of
    the policy's episode lengths, so no gross overestimation; exploration at 0.02 costs 1 to 6 points against greedy
    on the same 100 starts. The policy does not collapse: it wanders (greedy evaluations -100 to -180 while the rate is
    above about 30% of its start) and freezes as the rate decays, at greedy -105 to -138 depending on the seed.
  - **Screened 2026-10-27, no better** (seeds 1 to 5, one change each, paired against the baseline run alongside, all
    `train.py` single-threaded): Polyak `tau` 0.0067 (the hard sync's mean lag; mean final -121.1, 1 of 5 hold, paired
    -3.9), `learning_rate` 0.002 with the same decay (-125.3, 1 of 5, paired -8.1) and `batch_size` 256 (-118.0, 2 of
    5, paired -0.9, one seed -20 worse); the baseline held 2 of 5. A lower rate cut greedy-action churn on a fixed grid
    of probe states from 0.12 to 0.07 between checkpoints yet ended worse, and churn did not track the swings in
    greedy return (correlation +0.10 over 20 runs), so "noisy Q estimates against small action gaps" is not supported
    as the cause.
  - **Prioritised replay, not shown to help (2026-10-27)**: proportional prioritised replay (Schaul et al. 2016) for
    DQN, opt-in, on the parked branch `feature/dqn-prioritized-replay` (tested, not merged). Same protocol, seeds 1 to
    5: alpha 0.6 held 1 of 5 (mean final -120.0, paired -2.9, one seed -21 worse), alpha 0.4 held 0 of 5 (-121.1,
    paired -4.0); the baseline again held 2 of 5 with the same finals. The prediction written beforehand, fewer
    mid-run losses of a greedy policy at -110 or better and smaller late swings, did not hold (losses 1 and 3 against
    1; swing 13.3 and 12.2 against 11.1). Untried: keeping the best greedy policy (which does not change the training
    average the rule judges).
  - **Double DQN, re-checked 2026-10-29** (first probed 2026-10-23 on an older config, mean -141 against -133): through
    the benchmark, seeds 1 to 5, it held 1 of 5 against the committed config's 2 (mean final -120.9 against -117.8;
    +17 on two seeds, -19 and -21 on two others). There was little overestimation to remove: on seed 1 the logged mean
    Q sat within about 1 of the discounted return in both arms (CHANGELOG.md, 2026-10-29).
  - **Dueling DQN, 2026-10-29**: held 0 of 5 against the committed config's 2 (mean final -125.0 against -117.8; 18
    and 19 worse on seeds 3 and 5) (CHANGELOG.md, 2026-10-29).
  - **PPO**: never reached the flag without reward normalisation (CHANGELOG.md, 2026-10-23); a separate problem, not
    examined by the diagnosis above.

- [ ] **MuJoCo Tasks (Tier 3)**
  - **Description**: Hopper and HalfCheetah, with the SAC and TD3 agents (BipedalWalker: CHANGELOG.md,
    2026-11-05); needs the `mujoco` package (a new dependency, the maintainer's decision).

- [ ] **Learning from Pixels (Tier 4)**
  - **Description**: CarRacing or Atari. Needs convolutional networks and a GPU; not practical in pure NumPy.

### Phase 8 — Robot Navigation

This repo prepares for robotics work (the maintainer, 2026-11-04). Phase 8 builds a navigation stack on the RL
foundation, one robotics skill per step: sensing, state estimation, mapping and SLAM, planning and control, then
real hardware. The maintainer has an HC-SR04 ultrasonic sensor, a wheeled robot base and a lidar or depth camera.
The core stays plain Python with no ROS dependency; ROS 2 comes as thin nodes at the hardware step. Each simulated
piece is tested against the simulator's ground truth, so the real robot later meets code that is known to be
right in simulation. The goal is a navigation library other projects import (the maintainer, 2026-11-04: a
Jetson robot built on it): `rl_navigation.sim` and the later estimation, mapping and planning modules import
neither Gymnasium nor `rl_core`, and a trained policy reaches the robot through the ONNX or C export, so the
robot needs neither the trainer nor the agents. Step 1 is a full item; steps 2 to 5 are sketches to turn into
full items when picked.

- [~] **Range-sensor navigation environment (`rl_navigation`)** (picked 2026-11-04; PR 4 of 5, the PPO, DQN and
  TD3 configs; PR 5 the threshold, success-rate metric, curriculum and robustness sweep)
  - **Progress (PR 4, 2026-11-06)**: TD3 (continuous) and PPO (both ids) reach at least 80 % greedy success on 100
    held-out layouts on 5 of 5 seeds; DQN does not (0 to 50 %): its greedy policy mostly runs out the clock (a
    timeout costs less than a collision), cause not diagnosed (CHANGELOG.md, 2026-11-06). Success was counted by a
    scratch harness; the metric is PR 5. Final 100-episode training averages against the floated
    threshold 7.0: TD3 5 of 5, discrete PPO 1 of 5, continuous PPO 0 of 5.
  - [ ] **Before PR 5**: the `training-doctor` skill on navigation DQN, or the maintainer drops DQN from the
    discrete id.
  - **Description**: A 2D Gymnasium environment and challenge package: a differential-drive robot (unicycle
    kinematics, linear and angular velocity commands, integrated at a fixed time step) in a walled arena with
    static obstacles (circles and axis-aligned boxes), reaching a goal. Its sensor is a configurable set of range
    beams ray-cast in NumPy. One preset models the HC-SR04 (about 2 cm to 4 m, a cone of about 15°, approximated
    by several rays whose minimum is the reading, Gaussian range noise and dropouts that read as maximum range),
    with the figures checked against the datasheet when planned; another models a 2D lidar (many narrow beams,
    the count and range set by the config). The observation is the readings, the goal's distance and bearing in
    the robot's frame and the last command; actions are continuous (linear and angular velocity) with a discrete
    variant for DQN. The reward is progress towards the goal, a collision penalty and a small cost per step; a
    collision or reaching the goal terminates, the time limit truncates. Obstacle layouts are drawn per episode
    from the seed. The existing wrappers and tools apply: observation and action noise, frame stacking (the
    sensor sees only a slice of the world), domain randomization (sensor noise, wheel slip, obstacle count) as
    schedulable settings for the curriculum, the robustness benchmark, `record.py` and the benchmark.
  - **Acceptance criteria**: `gymnasium.utils.env_checker.check_env` passes; the same seed gives the same layout,
    readings and trajectory; unit tests with hand-computed values for a beam meeting a wall, a circle and a box
    corner at a known distance, the cone minimum, the range limits, the dropout rate over many seeded draws, one
    kinematics step, and collision against each obstacle kind; an `rgb_array` render for `record.py`; the random
    agent's score and committed PPO and TD3 configs measured on seeds 1 to 5 against a solve threshold the
    maintainer sets; TESTING rows for training, recording and the robustness sweep over sensor noise.
  - **Known edge cases**: a spawn or a goal inside an obstacle, or a goal with no free path (layouts are
    rejected and redrawn, a bounded number of times); a beam parallel to a wall or grazing a circle; a reading at
    exactly the minimum or maximum range; a fast robot passing through a thin wall in one step (collision is
    checked along the swept segment); angles wrapping at ±π.
  - **Out of scope**: maps given to the agent, moving obstacles, 3D, odometry drift (step 2), real hardware
    (step 5), and a physics engine (kinematics only).

- [~] **State estimation (`rl_navigation.estimation`)** (picked 2026-11-06; PR 1 of 3, wheel odometry and EKF
  localization; PR 2 Monte Carlo localization; PR 3 the comparison report, its CLI, and any hooks the maintainer
  picks)
  - **Description**: The robot's pose `(x, y, θ)` is estimated from wheel odometry and range readings on a known
    map, in the step 1 simulator with odometry noise added.
    - **Odometry.** The simulator gets a wheel-encoder model, `WheelOdometry`. Each step it gives the left and
      right wheel travel, with noise whose variance grows with the distance each wheel rolls (Siegwart,
      Nourbakhsh and Scaramuzza, *Introduction to Autonomous Mobile Robots*, 2nd ed., §5.2.4). That is what the
      real base's encoders measure.
    - **EKF.** An extended Kalman filter predicts with this model on the exact unicycle arc. It corrects with
      HC-SR04 or lidar ranges, expected from a map through a small `RangeMap` protocol: `Arena` with a
      `RangeSensor` now, an occupancy grid at step 3. Its update is iterated (Gauss–Newton, guarded) and in
      Joseph form. The Kalman update it calls is the linear Kalman filter, tested on its own.
    - **MCL.** A particle filter (Monte Carlo localization) on the same map adds global localization and recovery
      from a kidnapping, which the EKF cannot do.
    - **Library.** Both are a library a robot program calls (`predict(wheel_travel)`, `update(readings)`, `pose`,
      `covariance`). Both import only NumPy and `rl_navigation.sim`.
    - **Judged by** the simulator's true pose over many seeded runs, with and without sensor dropouts, against
      dead reckoning on the same odometry.
  - **Acceptance criteria**:
    - Predict values and their Jacobians match hand computation and central differences.
    - The measurement Jacobian matches analytic values for a beam on a wall and on a circle.
    - The Joseph form keeps `P` positive-definite where the standard form gives 0.
    - A stationary robot converges.
    - An all-dropout scan or an all-NaN scan changes nothing.
    - An outlier is gated out.
    - In an empty 4 × 4 m arena with the lidar, the ANEES averaged over 30 seeded runs × 100 steps lies in the
      χ²(90)/30 95 % band [2.19, 3.94]. With the filter's process noise a quarter of the true one,
      it lies above 3.94 (halved, it measured 3.84, inside the band).
    - Over seeded layouts, a function reports for each preset, with and without 30 % dropout: position and
      heading RMSE, mean NEES, beams used, and the lost count. The EKF beats dead reckoning on every row.
    - MCL localizes globally and recovers from a kidnapping on a set share of seeded layouts.
    - A seeded navigation-environment trajectory is unchanged (golden test).
    - `rl_navigation.estimation` imports neither Gymnasium nor `rl_core` (subprocess test).
    - There are TESTING rows for the evaluation.
  - **Known edge cases**:
    - A dropout, a NaN from a driver and "nothing within range" are indistinguishable for the EKF, and all count
      as dropouts.
    - Readings clipped at `min_range`.
    - Beams near a corner or at a circle's edge, where the expected range jumps.
    - A probe pose past a wall, which casts `inf`.
    - Beams with no wall in range.
    - A stationary robot, which has zero process noise: its covariance keeps shrinking, so after a bump of more
      than about 3 cm every beam is gated and the EKF never recovers (PR 1 review, 2026-11-06). PR 3 decides
      between a process-noise floor, re-initialisation after several all-gated scans, and leaving it to MCL.
    - The heading wrapping at ±π.
    - The `sinc` derivative near Δθ = 0.
    - A whole scan gated out, which loses the track.
    - An iterate leaving the linear region.
    - A collision, where the wheels turn while the robot stays put. Runs end at a collision.
    - Symmetric arenas, which give MCL several clusters.
  - **Out of scope**: SLAM and unknown maps (step 3), systematic odometry error and its calibration (UMBmark, step
    5), an IMU, a UKF, moving obstacles, ROS 2, and hardware.

- [ ] **Occupancy-grid mapping and SLAM**
  - **Description**: Occupancy-grid mapping from range readings with known poses (log-odds, an inverse sensor
    model for both presets), then SLAM without them: landmark EKF-SLAM, then a particle-filter grid SLAM
    (FastSLAM-style) with scan matching for the lidar preset. Judged by map and pose error against ground truth,
    and loop closure on layouts that revisit a place.

- [ ] **Planning and control**
  - **Description**: A* on the occupancy grid (from step 3 or the true map), path smoothing, and a pure-pursuit or
    PID follower that outputs the same velocity commands the RL agents do, as a classical baseline. Compared with
    the RL policy on the same seeded layouts: success rate, path length, time to goal and collisions, and how each
    degrades as sensor noise rises (the robustness benchmark).

- [ ] **Hardware and ROS 2**
  - **Description**: The HC-SR04 on the Jetson's GPIO (a 5 V echo needs a level shifter or divider for the
    3.3 V pins): a driver, timing and calibration that fit the step 1 noise model to real readings. Then the
    wheeled base (motor driver, wheel odometry) and the lidar, and ROS 2 nodes (sensor publisher, estimator,
    mapper, planner, policy) over the plain-Python core. Ends with the simulated policy and planner run on the
    robot, and a report of what transferred and what did not. New dependencies (`Jetson.GPIO`, ROS 2, a lidar
    driver) are the maintainer's decision when the item is planned.
