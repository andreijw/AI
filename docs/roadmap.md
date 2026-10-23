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

The last health dashboard (brooks-health, 2026-10-21) scored 92/100. Its one Critical, the curve alignment inside
`plot_benchmark_comparison`, and the two complexity hotspots it left, `DQNAgent.__init__` and
`Trainer._log_metrics`, are fixed (CHANGELOG.md, 2026-10-21). The loose ends gathered from completed items are
closed too (CHANGELOG.md, 2026-10-21). What remains here are unscheduled ideas.

- [ ] **Ideas: Close the Actor-Critic Gap; Larger Networks on a GPU**
  - **Description**: Lower-priority ideas recorded 2026-10-18, to be turned into items when picked.
    - The continuous Actor-Critic's greedy policy matches rl-zoo's A2C, but its 100-episode training average stays
      at about -300 (CHANGELOG.md, 2026-10-18). True A2C updates every n steps, which needs the
      trainer to call `update` mid-episode for every agent; gSDE exploration is the other untried difference.
    - The maintainer expects to use the Jetson's GPU later, so PyTorch should be allowed then. That would make
      rl-zoo's 400×300 DDPG networks practical (about 16 times slower than 64×64 in NumPy here) and pixel-based or
      larger tasks possible. It would be an optional backend beside the NumPy agents, with the NumPy versions kept
      as the gradient reference, and needs the pure-NumPy rule in CLAUDE.md and CONTRIBUTING.md updated.
    - Not tried, though the bars were met without them: Pendulum PPO with gSDE exploration, separate tanh actor and
      critic networks, or observation normalisation; CartPole DQN with more than `max_updates_per_call: 64` (about
      0.13 gradient steps per transition against rl-zoo's 0.5).
    - Candidates once wanted: double, dueling or prioritised-replay DQN.
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

- [ ] **Learning-Rate Decay for DQN (MountainCar-v0 follow-up)**
  - **Description**: MountainCar DQN learns to reach the flag but does not solve: its 100-episode training average
    peaks at -112 to -130 and then drifts back (-130 to -146 at the end of 120,000 steps; CHANGELOG.md, 2026-10-23).
    Input scale, a lower fixed learning rate, update timing, a Huber loss, depth and a longer budget were ruled out.
    The untested suspect is DQN's fixed learning rate of 0.004 with no decay; PPO already decays its rate with
    `anneal_fraction`. Add an optional linear decay to `DQNAgent` (off by default), test it on MountainCar, and
    re-check the CartPole and Acrobot DQN configs, which it would not change unless turned on.
  - **Also open**: PPO on MountainCar-v0 never reached the flag without reward normalisation; it returns once the
    framework has an exploration aid or normalisation.

- [ ] **MountainCarContinuous-v0 Challenge with Ornstein-Uhlenbeck Exploration (Tier 1, exploration)**
  - **Description**: The continuous version (`Box(-1, 1)` force, threshold 90, 999-step limit) rewards reaching
    the flag with +100 but charges for force, so "do not move" is a strong local optimum. rl-baselines3-zoo's DDPG
    and TD3 use Ornstein-Uhlenbeck noise (temporally correlated, std 0.5); our DDPG has Gaussian noise only, so this
    adds OU noise as an exploration option (`algorithm-crafter`) and the challenge to the `rl_mountaincar` package.

- [ ] **LunarLander-v3, Discrete and Continuous (Tier 2), with TD3 or SAC**
  - **Description**: 8-D state, contact dynamics, longer horizons; threshold 200. Needs the `box2d` dependency
    (a swig build, not checked on the Jetson's aarch64) and a stronger off-policy continuous agent. The POMDP
    variant of CartPole is Tier 2 too, through the Phase 6 frame-stacking item.

- [ ] **BipedalWalker-v3 and MuJoCo Tasks (Tier 3)**
  - **Description**: High-dimensional continuous control over millions of steps (BipedalWalker threshold 300;
    MuJoCo Hopper, HalfCheetah). Needs `box2d` or `mujoco` and larger networks, so it waits on the optional
    PyTorch/GPU backend in the Code Health ideas.

- [ ] **Learning from Pixels (Tier 4)**
  - **Description**: CarRacing or Atari. Needs convolutional networks and a GPU; not practical in pure NumPy.
