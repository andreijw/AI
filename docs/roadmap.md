# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

### Workflow Tooling

No open items.

### Code Health

The last health dashboard (brooks-health, 2026-10-21) scored 92/100. Its one Critical, the curve alignment inside
`plot_benchmark_comparison`, and the two complexity hotspots it left, `DQNAgent.__init__` and
`Trainer._log_metrics`, are fixed (CHANGELOG.md, 2026-10-21).

- [~] **Loose Ends from Completed Items**
  - **Description**: Small gaps recorded in completed items and never resolved, gathered when those items moved to
    CHANGELOG.md (2026-10-21). Each can become its own item when picked.
    - Discrete PPO never trains on steps still buffered when training ends, and its checkpoints do not hold the
      rollout buffer (continuous PPO likewise).
  - **Progress**: the DQN, `log_std` and video-warning loose ends are done (`feature/code-health-loose-ends`,
    CHANGELOG.md, 2026-10-21). Left for `feature/ppo-rollout-checkpoints`, decided by the maintainer on 2026-10-21:
    both PPO agents save the buffered steps in their checkpoints, through one shared rollout class, and do not
    train on a partly filled rollout when training ends.

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
    - Candidates once wanted: a resume option for `train.py` (checkpoints already hold everything needed), and
      double, dueling or prioritised-replay DQN.
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
