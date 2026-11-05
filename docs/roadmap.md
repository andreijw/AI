# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

Order (the maintainer's decision, 2026-11-04): the repo is the foundation for robotics work, so Phase 8 (Robot
Navigation) leads. Its first item, the range-sensor navigation environment, and Phase 7's SAC and BipedalWalker
item run in parallel (separate worktrees; one is mostly code, the other mostly GPU time), then Phase 8 continues in
order. The Training and Code Health items are picked as they become useful (parallel environments before a PPO
comparison on BipedalWalker or navigation).

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

- [~] **SAC and BipedalWalker-v3 (Tier 3)** (picked 2026-11-04; PR 3 of 4, the PyTorch SAC and LunarLanderContinuous)
  - **Description**: Soft Actor-Critic (Haarnoja et al. 2018, with the automatic entropy temperature of the
    follow-up paper, as Stable-Baselines3 implements it): a squashed Gaussian actor, twin critics with Polyak
    targets, and a learned temperature against a target entropy of minus the action size. In NumPy and on the
    PyTorch backend, as TD3 is (CHANGELOG.md, 2026-10-27). SAC is the second agent with twin critics and a replay
    buffer, so it is the trigger for a shared off-policy base class with DDPG and TD3. Then a `bipedalwalker`
    challenge (BipedalWalker-v3, `box2d` extra, 1,600-step episodes, threshold 300) with a SAC config translated
    from rl-baselines3-zoo's, checked in Context7 and the zoo's `hyperparams/sac.yml` when planned.
  - **Acceptance criteria**: the SAC loss, the entropy temperature's update and the tanh log-probability
    correction pass finite-difference gradient checks and hand-computed values; the float64 PyTorch agent trains
    exactly as the NumPy one (as TD3's parity tests do); on Pendulum-v1 and LunarLanderContinuous-v3 a SAC config
    reaches the threshold on seeds 1 to 5 before BipedalWalker is tried; BipedalWalker is measured on seeds 1 to 5
    against TD3 on the same budget, with the time a seed takes reported; `record.py` and the ONNX and C export
    work for the SAC actor (its deterministic action, tanh of the mean).
  - **Known edge cases**: the log-probability of an action squashed to ±1 (the correction's log of 1 - tanh²
    needs the numerically stable form); the temperature's log parameter running away when the target entropy is
    unreachable; BipedalWalker's -100 on a fall, which is a true termination, against the 1,600-step time limit,
    which bootstraps; a seed taking hours, so the runs go in the background with the per-seed curves under
    `~/rl-experiments/`.
  - **Out of scope**: BipedalWalkerHardcore; MuJoCo (needs the `mujoco` dependency, the maintainer's decision);
    n-step returns; PPO on BipedalWalker until parallel environments exist (Training).

- [ ] **MuJoCo Tasks (Tier 3)**
  - **Description**: Hopper and HalfCheetah after BipedalWalker, with the SAC and TD3 agents above; needs the
    `mujoco` package (a new dependency, the maintainer's decision).

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

- [~] **Range-sensor navigation environment (`rl_navigation`)** (picked 2026-11-04; PR 1 of
  5, the `rl_core` environment hooks)
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

- [ ] **State estimation**
  - **Description**: A Kalman filter and an extended Kalman filter for the robot's pose from wheel odometry and range
    readings, then a particle filter (Monte Carlo localization) on a known map, in the step 1 simulator with
    odometry noise added. Judged by the pose error against ground truth over many seeded runs, with and without
    sensor dropouts.

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
