# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

### Workflow Tooling

No open items.

### Training

No open items.

### Code Health

The last health dashboard (brooks-health, 2026-10-22) scored 94/100: 0 Critical, 4 Warnings. The first Warning,
each challenge copying its agent construction and config reading, is fixed (CHANGELOG.md, 2026-10-22); so are the
second and third, the benchmark's solve numbers and report, and the fourth, trainer tests calling private methods
(CHANGELOG.md, 2026-10-25). What follows are unscheduled ideas.

- [ ] **Ideas: Close the Actor-Critic Gap; Larger Networks on a GPU**
  - **Description**: Lower-priority ideas recorded 2026-10-18, to be turned into items when picked.
    - The continuous Actor-Critic's greedy policy matches rl-zoo's A2C, but its 100-episode training average stays
      at about -300 (CHANGELOG.md, 2026-10-18). True A2C updates every n steps, which needs the
      trainer to call `update` mid-episode for every agent; gSDE exploration is the other untried difference.
    - Larger networks and pixel-based tasks on the Jetson's GPU: the optional PyTorch backend item above holds the
      measured case (CHANGELOG.md, 2026-10-25).
    - Not tried, though the bars were met without them: Pendulum PPO with gSDE exploration, separate tanh actor and
      critic networks, or observation normalisation; CartPole DQN with more than `max_updates_per_call: 64` (about
      0.13 gradient steps per transition against rl-zoo's 0.5).
    - Candidates once wanted: double, dueling or prioritised-replay DQN.
    - `src/benchmark.py` still imports matplotlib at module level, so `--no-plot` loads it too; importing
      `plot_benchmark_comparison` inside `_save_plot` would end that (review suggestion, 2026-10-25).
    - Found 2026-10-18 and not yet investigated: on the one-state overfit batch of
      `tests/test_policy_gradient_maths.py`, the discrete Actor-Critic (SGD, learning rate 0.05) reaches probability
      1.0 and can then collapse to the other action (seeds 6 and 7; 9 of 30 seeds under the old seeding) once the
      rejected action's log-probability runs away. The test only requires the batch to be fitted within 1,000
      updates.

### Phase 6 — Advanced RL & Robotics Extensions

- [ ] **Curriculum Learning & Domain Randomization Scheduling**
  - **Description**: Add dynamic curriculum pacing that scales domain randomization and noise injection difficulty
    as the agent reaches performance milestones during training.
  - **Acceptance criteria**:
    - `CurriculumScheduler` with linear, step, and adaptive reward-threshold difficulty scheduling.
    - Dynamic updates to `DomainRandomizationWrapper` and noise wrapper parameters during rollouts.
    - Robustness evaluation benchmark assessing zero-shot transfer under out-of-distribution physical perturbations.
    - Configurable curriculum specification in environment YAML files.
    - Comprehensive unit tests in `tests/test_curriculum.py`.
  - **Known edge cases**: Catastrophic forgetting across difficulty stages; oscillation near threshold boundaries.
  - **Out of scope**: Adversarial environment generation; automatic domain randomization (ADR) meta-policies.

- [ ] **Observation Preprocessing & Frame Stacking**
  - **Description**: Support temporal history and partial observability via frame stacking and custom feature
    transformations.
  - **Acceptance criteria**:
    - `FrameStackWrapper` concatenating $K$ sequential observations into a flattened feature vector.
    - Velocity masking option simulating partially observable Markov decision processes (POMDP).
    - Full compatibility with vectorized environment execution (`make_vec_env`).
    - Unit tests in `tests/test_frame_stack.py`.
  - **Known edge cases**: Padding on episode resets; memory footprint for long history buffers.
  - **Out of scope**: Convolutional neural networks for raw RGB pixels; recurrent neural network policies (LSTM/GRU).

- [ ] **Lightweight Embedded Inference Export (C Header & ONNX)**
  - **Description**: Export trained pure NumPy model parameters into zero-dependency standalone C header files
    (`policy_weights.h`) and standard ONNX formats for edge microcontroller and robotics deployment.
  - **Acceptance criteria**:
    - CLI export script `scripts/export_c_header.py` converting checkpoints into self-contained C forward pass code.
    - Standalone C verification harness compiled via GCC/Clang and checked against NumPy numerical output (tolerance
      $10^{-5}$).
    - Export utility for standard ONNX format compatible with visualizers (Netron).
    - Unit tests in `tests/test_export.py`.
  - **Known edge cases**: Floating-point precision discrepancies (FP32 vs FP64); activation clipping overflows.
  - **Out of scope**: Quantization-aware training (INT8); target-specific MCU hardware optimizations (CMSIS-NN).

### Phase 7 — Challenge Ladder

New environments in order of difficulty, each tier adding one new hardship so that a failure points at one cause.
Every challenge is its own `rl_<name>` package that `rl_core.challenges` discovers; solve thresholds are
Gymnasium's `reward_threshold` unless the maintainer sets another, judged by the 100-episode training average.
Tier 1 began with Acrobot-v1 (CHANGELOG.md, 2026-10-22). The items below are sketches to turn into full items
when picked.

- [ ] **MountainCar-v0: Hold -110**
  - **Description**: MountainCar-v0's DQN does not reliably hold -110: with a 100,000-transition buffer, 2 of 10 seeds
    end at or above it (both crossing it only near the end of the run), and the final averages are -107 to -132
    (CHANGELOG.md, 2026-10-25).
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
  - **PPO**: never reached the flag without reward normalisation (CHANGELOG.md, 2026-10-23); a separate problem, not
    examined by the diagnosis above.

- [~] **LunarLander-v3, Discrete and Continuous (Tier 2)**
  - **Description**: Gymnasium's LunarLander-v3 and LunarLanderContinuous-v3: 8-D state, contact dynamics, shaped
    reward, 1,000-step limit, threshold 200. Three pull requests: the challenges (random agents), then discrete DQN,
    then continuous TD3 on the PyTorch backend. The POMDP variant of CartPole is Tier 2 too, through the Phase 6
    frame-stacking item.
  - **Decided (maintainer, 2026-10-27)**: `box2d` is an optional extra naming `box2d-py==2.3.5`, which `./setup.sh`
    builds from source with `swig` (Gymnasium 1.4's own extra pins `box2d==2.3.10`, which has no aarch64 build); CI
    does not install it, and the LunarLander tests skip there.
  - **Acceptance criteria (confirmed by the maintainer, 2026-10-27)**: PR 2, a DQN config from rl-zoo's
    LunarLander entry solving 200 on at least 4 of 5 seeds, judged on held-out seeds 11 to 15; PR 3, a TD3 config
    from rl-zoo's LunarLanderContinuous entry on the PyTorch backend, the same. Each against the random floor and with
    a 5-seed run.
  - **Progress**: the first pull request (`feature/lunarlander-challenge`) adds `rl_lunarlander` with both
    challenges and their random configs, the `box2d` extra and its setup.sh build, and one shared factory for the
    observation-noise-only wrappers (CHANGELOG.md, 2026-10-27). The second (`feature/lunarlander-dqn`) adds
    `configs/lunarlander_dqn.yaml`: rl-zoo's DQN reaches 200 on 5 of 5 held-out seeds and holds it on 4 of 5
    (CHANGELOG.md, 2026-10-27). Open: PR 3, TD3 on the PyTorch backend for LunarLanderContinuous-v3.
  - **Agent**: TD3, in NumPy and on the PyTorch backend (CHANGELOG.md, 2026-10-27); SAC is the next candidate
    and the trigger for a shared off-policy base class. n-step returns stay out of scope.
  - **Out of scope**: Gymnasium's wind, gravity and turbulence options; Box2D on Windows and in CI.

- [ ] **BipedalWalker-v3 and MuJoCo Tasks (Tier 3)**
  - **Description**: High-dimensional continuous control over millions of steps (BipedalWalker threshold 300;
    MuJoCo Hopper, HalfCheetah). Needs `box2d` or `mujoco` and larger networks, so it waits on the optional
    PyTorch/GPU backend in the Code Health ideas.

- [ ] **Learning from Pixels (Tier 4)**
  - **Description**: CarRacing or Atari. Needs convolutional networks and a GPU; not practical in pure NumPy.
