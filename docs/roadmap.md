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
each challenge copying its agent construction and config reading, is fixed (CHANGELOG.md, 2026-10-22), and so are
the second and third, the benchmark's solve numbers and report (CHANGELOG.md, 2026-10-25). The trainer-test item
below is the last; then come unscheduled ideas.

- [ ] **PyTorch Backend: Timing Test on the Jetson**
  - **Description**: Agreed 2026-10-23 as the next step towards GPU training. Measure, in a scratch virtual
    environment and not `.venv`, one DDPG gradient step (batch 256) at 64×64 and 400×300 with PyTorch on the CPU and
    on the Jetson's GPU, against NumPy's 5.6 ms and 40 ms here. PyPI's PyTorch wheels do not drive the Jetson's
    integrated GPU: it needs NVIDIA's Jetson builds matching the installed L4T R39 and Python 3.12, and the CUDA
    toolkit is not under `/usr/local`, so whether the GPU is usable at all is part of the answer.
  - **Acceptance criteria**: a table of milliseconds per gradient step for the four cases, the install steps that
    worked (or why they did not), and a recommendation the maintainer can decide on: an optional `[torch]` backend
    beside the NumPy agents (which stay the gradient reference, with a test that both compute the same gradients),
    or not yet.
  - **Out of scope**: adding PyTorch to `pyproject.toml`, porting any agent, changing CI, and the pure-NumPy rule in
    CLAUDE.md and CONTRIBUTING.md; those follow the decision.

- [ ] **Test Hygiene: Trainer Tests Through a Seam**
  - **Description**: The 2026-10-22 dashboard's fourth Warning: about 13 call sites in `tests/test_trainer.py` call
    private `Trainer` methods (`_run_episode`, `_save_checkpoint`, `_log_metrics`, `_log_info`), against the
    rule every agent test already follows. Its suggestions belong here too: the per-challenge tables in
    `tests/test_challenge_selection.py`, and the fake-package prefix shared implicitly between `tests/conftest.py`
    and `tests/helpers.py`.
  - **Acceptance criteria**:
    - Trainer tests go through `train()`, `evaluate()` and the files and log lines they produce, or through a
      `tests/trainer_internals.py` like `tests/agent_internals.py`; `test_source_conventions.py` enforces it.
    - The `rl_fake` prefix is one named constant used by both files.
    - No loss of coverage on `rl_core/training/trainer.py` (99% today).
  - **Out of scope**: Changing `Trainer` behaviour.

- [ ] **Ideas: Close the Actor-Critic Gap; Larger Networks on a GPU**
  - **Description**: Lower-priority ideas recorded 2026-10-18, to be turned into items when picked.
    - The continuous Actor-Critic's greedy policy matches rl-zoo's A2C, but its 100-episode training average stays
      at about -300 (CHANGELOG.md, 2026-10-18). True A2C updates every n steps, which needs the
      trainer to call `update` mid-episode for every agent; gSDE exploration is the other untried difference.
    - The maintainer expects to use the Jetson's GPU later, so PyTorch should be allowed then (the timing test
      above comes first). That would make rl-zoo's 400×300 DDPG networks practical (about 16 times slower than
      64×64 in NumPy here) and pixel-based or larger tasks possible. It would be an optional backend beside the
      NumPy agents, with the NumPy versions kept as the gradient reference, and needs the pure-NumPy rule in
      CLAUDE.md and CONTRIBUTING.md updated.
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
    evaluation. Next
    levers, untried: a smaller step once the policy is good (the current rate is still half its start at episode 700),
    prioritised replay, or keeping the best greedy policy. Note that -110 is judged on the training average, exploration
    included, while rl-zoo's -100.8 is a greedy evaluation.
  - **PPO**: never reached the flag without reward normalisation (CHANGELOG.md, 2026-10-23); a separate problem, not
    examined by the diagnosis above.

- [ ] **LunarLander-v3, Discrete and Continuous (Tier 2), with TD3 or SAC**
  - **Description**: 8-D state, contact dynamics, longer horizons; threshold 200. Needs the `box2d` dependency
    (a swig build, not checked on the Jetson's aarch64) and a stronger off-policy continuous agent. The POMDP
    variant of CartPole is Tier 2 too, through the Phase 6 frame-stacking item.
  - **Episode-end hook**: DDPG resets its Ornstein-Uhlenbeck noise inside `update`, relying on the Trainer calling
    it once per episode (ARCHITECTURE.md). A TD3 that reuses `rl_core/agents/exploration_noise.py` is the moment
    to replace that with an explicit `BaseAgent` episode-end hook, before a per-step update loop arrives.

- [ ] **BipedalWalker-v3 and MuJoCo Tasks (Tier 3)**
  - **Description**: High-dimensional continuous control over millions of steps (BipedalWalker threshold 300;
    MuJoCo Hopper, HalfCheetah). Needs `box2d` or `mujoco` and larger networks, so it waits on the optional
    PyTorch/GPU backend in the Code Health ideas.

- [ ] **Learning from Pixels (Tier 4)**
  - **Description**: CarRacing or Atari. Needs convolutional networks and a GPU; not practical in pure NumPy.
