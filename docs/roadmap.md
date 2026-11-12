# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

## Robotics ladder (the direction)

The repo is the foundation for robotics work. Each rung below is a view over the items in this file; where it stands
is in the item itself and on the [README](../README.md). A rung marked "proposed" is an item sketch below, to be
turned into a full item when the maintainer picks it. The ladder uses no numbers, so it does not collide with Phase
7's tiers.

| Rung | Item |
| :--- | :--- |
| Classic control and Box2D | Phase 7 (MountainCar-v0 "Hold -110" is open) |
| Inverted pendulums, locomotion, robot arms in MuJoCo | Phase 7 "MuJoCo Tasks": InvertedPendulum and InvertedDoublePendulum first, then Hopper and HalfCheetah, then Reacher and Pusher; one `mujoco` dependency decision for all |
| Further arms | Phase 7 "Gymnasium-Robotics arms" (proposed): Fetch and Franka Kitchen; a new dependency |
| GPU-parallel simulation | Phase 8 "GPU-parallel navigation simulation" (proposed): the simulator as PyTorch tensors on the Orin's GPU; builds on `training.num_envs` (CHANGELOG.md, 2026-11-10) |
| Isaac Lab, deployed to the Jetson | Phase 8 "Isaac Lab on cloud or x86" (proposed, blocked on hardware): training needs an x86-64 RTX machine or cloud, deployment goes through the ONNX export |
| Isaac ROS on the robot | Phase 8 "Hardware and ROS 2" (Isaac ROS 4.6 supports Jetson Orin on JetPack 7.2; Isaac ROS documentation, read 2026-11-09) |
| Autonomous navigation | Phase 8 "Occupancy-grid mapping and SLAM" and "Planning and control" |
| Sim-to-real | Phase 8 "Hardware and ROS 2" (fitting the sensor noise model to real readings) |
| Multi-agent coordination | Phase 8 "Multi-agent navigation" (proposed): several robots in the navigation simulator, or PettingZoo (a new dependency) |
| Learning from pixels | Phase 7 "Learning from Pixels (Tier 4)" |

---

## Upcoming Milestones

Order (the maintainer's decision, 2026-11-04): the repo is the foundation for robotics work, so Phase 8 (Robot
Navigation) leads. Its first item, the range-sensor navigation environment (done; CHANGELOG.md, 2026-11-07), ran in
parallel with Phase 7's SAC and BipedalWalker item (done; CHANGELOG.md, 2026-11-05), and Phase 8 continues in order. The
Training and Code Health items are picked as they become useful (parallel environments before a PPO comparison on
BipedalWalker or navigation).

### Workflow Tooling

- [~] **Importable libraries with GitHub Releases** (the maintainer's request and approved plan, 2026-11-11).
  The code becomes three distributions with unchanged import names: `rl-core`
  (the framework a game project imports), `rl-navigation` (the robot's libraries, NumPy only, with the Gymnasium
  environment and the navigation challenge as extras) and `rl-challenges` (the six classic and Box2D challenges).
  Acceptance criteria:
  - Challenges register through the `rl_core.challenges` entry-point group, in a fixed order with this
    repository's distributions first; another project registers one from its own `pyproject.toml`.
  - Each distribution has its own `pyproject.toml`, version (`__version__`, SemVer 0.x from 0.1.0) and
    `CHANGELOG.md` under `packages/<dist>/`; one editable install of all three is the development environment.
  - The installed `rl-navigation` wheel needs only NumPy; a git worktree tests its own code, never the main
    checkout's.
  - A pushed `<dist>-v<version>` tag builds, tests and smoke-tests that distribution and opens a draft GitHub
    Release with the wheel, the sdist, `SHA256SUMS` and the notes from its CHANGELOG section.
  - A guide shows a game project registering its own challenge and training here.

  Progress: PR 1 (`feature/challenge-entry-points`): entry-point discovery and per-worktree environments.

### Training

No open items.

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
  2026-11-04). rl-zoo's 8 parallel environments, measured 2026-11-10 (`num_envs: 8`, `update_every: 8`, one A2C step
  over 64 transitions; seeds 1 to 5 paired): final training averages -244.1 against -282.7, better on all 5 seeds,
  but none holds -200 (CHANGELOG.md, 2026-11-10); adopting it in `configs/pendulum_actor_critic.yaml` is the
  maintainer's decision. Untried: gSDE exploration (its PPO version is in `~/rl-experiments/item5/` on the
  development machine). Asynchronous training environments (deferred from the parallel-environments item until a
  measured speedup on navigation or BipedalWalker) also remain open.
- [ ] **SAC's BipedalWalker training average does not hold**: `configs/bipedalwalker_sac.yaml` solved 300 on 4
  of seeds 1 to 5 within 500,000 steps but held on 1 (final 100-episode averages 251 to 307), while every seed's
  greedy evaluation was 321 to 327 (CHANGELOG.md, 2026-11-05). The training average includes exploration noise.
  Untried: rl-zoo's gSDE exploration (`use_sde`, `log_std_init: -3`), which its recipe uses on this environment and
  this framework lacks (dropped from the item by the maintainer, 2026-11-05); TD3 at rl-zoo's 1,000,000 steps.

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
  - **Description**: Inverted pendulums, locomotion and robot arms, with the SAC and TD3 agents (BipedalWalker:
    CHANGELOG.md, 2026-11-05), in this order: InvertedPendulum and InvertedDoublePendulum first, then Hopper and
    HalfCheetah, then Reacher and Pusher. Needs the `mujoco` package: one new dependency, one decision of the
    maintainer's for all three steps.

- [ ] **Gymnasium-Robotics arms (proposed)**
  - **Description**: Fetch (reach, push, pick and place) and Franka Kitchen from the `gymnasium-robotics` package, as
    the next arms after Reacher and Pusher. (The Panda arm is not in Gymnasium-Robotics.)
  - **Feasibility**: plausible after the MuJoCo item, which brings the same `mujoco` dependency. The Fetch
    observations are dictionaries and their rewards sparse by default, which the agents here have not been tried on;
    expect a flattening wrapper and a look at the reward before any learning claim. Not examined further.
  - **Needs**: the maintainer's decision on the `gymnasium-robotics` dependency, taken with the `mujoco` one.

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
robot needs neither the trainer nor the agents. Step 1, the navigation environment, is done (CHANGELOG.md,
2026-11-04 to 2026-11-07), and so is step 2, state estimation (2026-11-06 to 2026-11-09);
steps 3 to 5 are sketches to turn into full items when picked.

- [ ] **State estimation follow-ups** (left open when the State estimation item closed, 2026-11-09; CHANGELOG.md,
  2026-11-06 to 2026-11-09)
  - **Symmetric arenas** (deferred until occupancy-grid mapping has landed, the maintainer's decision, 2026-11-09).
    Both filters can settle on a mirror pose with nothing to flag it: MCL's resampling keeps one mirror hypothesis
    within 1 to 3 scans, so `cluster_weight` ends near 1, and after a 1.5 rad bump the lidar EKF settled on the
    square room's 90° alias once its resets stopped. Options: keep several MCL clusters alive and report the
    ambiguity, prefer symmetry-breaking moves, or rely on distinctive features from the map.
  - **Accepted limit: the 0.3 rad HC-SR04 heading bump** (the maintainer's decision, 2026-11-09). No reset fires:
    three of the four candidate sonars happen to agree with the wrong heading (one turns nonlinear, one is gated),
    so the scan looks consistent. Documented in `EkfLocalizer` and pinned by
    `test_a_heading_bump_the_sonars_barely_see_fires_no_reset`; the earlier explanation, "the beams turn nonlinear
    rather than gated", was wrong.
  - **Accepted limit: MCL's covariance scale for the HC-SR04 with its own dropout** fails the ANEES-band gate on
    seeds 101 to 150 (82.5 % against 85 %) while passing the per-axis gate (0.92 to 1.06); the default scale stays
    (1, 1, 1). Kept as a documented limit (the maintainer's decision, 2026-11-09); refit when MCL's settings change.
  - **Out of scope**: SLAM and unknown maps (step 3), odometry calibration (step 5).

- [~] **Occupancy-grid mapping and SLAM** (picked 2026-11-09; PR 1 mapping with known poses; PR 2 landmark EKF-SLAM;
  PR 3 lidar scan matching and the grid as a `RangeMap` (`GridMap`); PR 4 grid FastSLAM; possibly PR 5 a report and
  CLI)
  - **Progress (PR 1, 2026-11-09)**: `rl_navigation.mapping` builds a log-odds occupancy grid from range scans at
    known poses (5 cm cells, an exact vectorised beam traversal, presets for both sensors), as a peer of
    `estimation`. On fresh seeds 101 to 150 the lidar maps pass every gate (accuracy 0.9996, precision 1.0000, recall
    of seen surfaces 0.94 to 0.95, free-space coverage 0.98, at most 2 false-free cells per run, at the gate); the
    HC-SR04 rows are reported; maps built from dead reckoning instead fall to precision 0.34 (CHANGELOG.md,
    2026-11-09). Left for later PRs: SLAM, and a `deep` false-free metric whose neighbours lie in one obstacle (one
    seed's 2 cells straddle a real 2 cm gap between two boxes).
  - **Progress (PR 2a, 2026-11-10)**: PR 2 is split into 2a, circle features, and 2b, the landmark EKF-SLAM filter,
    for the robot's 360° lidar at about 1° per beam (the maintainer's choice, 2026-11-09; the HC-SR04 cannot see
    landmarks). `rl_navigation.slam.features` fits circles to scan segments with a covariance; on fresh seeds 101 to
    150 it passes every `CIRC-1` gate (fit NEES 3.22, false fits 0.0005, detection 0.76, centre error 5.7 mm median,
    6.4 ms a scan; CHANGELOG.md, 2026-11-10).
  - **Progress (PR 2b, 2026-11-10)**: `rl_navigation.slam.EkfSlam`, landmark EKF-SLAM with first-estimate Jacobians,
    judged once on seeds 101 to 150 against gates registered beforehand: 7 of 9 pass (error after alignment 0.064 m
    mean, 11 mm median); it fails the ratio to dead reckoning (0.443 against ≤ 0.40) and the count of runs above 0.10 m
    after alignment (9 against ≤ 8). Shipped with the failure recorded (the maintainer's decision; CHANGELOG.md,
    2026-11-10).
  - **Follow-ups**:
    - **Association drift**: 4 of the 9 aligned failures are association errors (the oracle row has 5), with first
      estimates drifting 0.05 to 0.60 m; a fix is judged on new seeds 151 to 200, as 101 to 150 are spent.
    - **Dropped beams split arcs**: `segment_scan` splits at every unusable beam, so at 30 % dropout detection
      falls from 75 to 10 % and false fits rise to 28 % (4 seeds); bridging isolated dropped beams should let
      the 30 % dropout row be gated (lost 25 of 50 today).
    - Left for later PRs: lidar scan matching and the grid as a `RangeMap` (PR 3), grid FastSLAM (PR 4).
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

- [ ] **GPU-parallel navigation simulation (proposed)**
  - **Description**: The navigation simulator's ray casting and motion as PyTorch tensors on the Orin's GPU, stepping
    thousands of robots at once, so an agent trains on far more experience than the NumPy simulator allows.
  - **Feasibility**: plausible, since the backend, `.venv-gpu` and CUDA graphs already run on the Jetson
    (`docs/pytorch_jetson_timing.md`), and the sim is plain array code. The agents would need batched collection
    first, which `training.num_envs` now provides on the CPU (CHANGELOG.md, 2026-11-10). Not prototyped.
  - **Needs**: the maintainer's decision to allow PyTorch outside `rl_core/agents/torch_*.py` (a new module that
    leaves `rl_navigation.sim` NumPy only, so the robot keeps importing the NumPy simulator).

- [ ] **Multi-agent navigation (proposed)**
  - **Description**: Several robots in one navigation arena that must reach their goals without colliding with each
    other, trained as independent learners first and with shared information later.
  - **Feasibility**: plausible as a simulator extension (`World` models one robot today); or through PettingZoo's
    interface, which is a new dependency. Not prototyped.
  - **Needs**: the maintainer's decision between an own multi-robot API and PettingZoo (a new dependency).

- [ ] **Isaac Lab on cloud or x86 (proposed, blocked on hardware)**
  - **Description**: Train a robot policy in NVIDIA Isaac Lab and deploy it to the Jetson through the ONNX export
    (`export.py`).
  - **Feasibility**: not on the current hardware. Isaac Sim's only supported aarch64 host is DGX Spark, and Isaac Lab
    asks for 32 GB of RAM (Isaac Sim and Isaac Lab documentation, read 2026-11-09), more than an Orin Nano has. Training
    needs an x86-64 machine with an RTX GPU, or a cloud instance. Isaac ROS 4.6, on the robot, does support Jetson
    Orin on JetPack 7.2 (Isaac ROS documentation, read 2026-11-09).
  - **Needs**: the maintainer's hardware decision (an x86 RTX machine or cloud) and the new dependencies, which do
    not belong in this repository's NumPy-only core.
