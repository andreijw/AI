# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

Order (the maintainer's decision, 2026-10-31): finish Training and Code Health first, the TD3 decision and then
gradient clipping with SGD, then Phase 6 starting with frame stacking. The parked ideas under Code Health do not block
Phase 6.

### Workflow Tooling

No open items.

### Training

- [ ] **Adopt Updates During the Episode for LunarLanderContinuous TD3** (evidence complete; needs the maintainer's
  decision)
  - **Description**: `training.update_every: 1` against the committed config, which updates once per episode, from the
    per-episode curves of the 400,000-step runs (CHANGELOG.md, 2026-10-30). The 300,000-step column cuts the same
    curves at 300,000 steps. That is exact, because the budget only decides when training stops: no agent or schedule
    reads it. The greedy evaluation at 300,000 steps was not recorded.

    | | Committed, 400,000 steps | Every step, 300,000 steps | Every step, 400,000 steps |
    | :--- | :--- | :--- | :--- |
    | Seeds 1 to 5: held 200 (mean final) | 3 of 5 (202.8) | 3 of 5 (210.8) | 5 of 5 (225.9) |
    | Seeds 11 to 15: held 200 (mean final) | 4 of 5 (214.1) | 4 of 5 (188.2; seed 14 at 52) | 5 of 5 (242.8) |
    | Time per seed | 36 to 47 minutes | about 40 to 58 minutes (scaled by gradient steps; not measured) | 54 to 78 minutes |

  - **Acceptance criteria**: the maintainer chooses between keeping the committed config, `update_every: 1` at
    400,000 steps (5 of 5 held on both seed sets, one seed more than 20 worse in each, 1.5 to 2 times the time), and
    `update_every: 1` at 300,000 steps (as many seeds held as today, in more time). A change goes into
    `configs/lunarlander_continuous_td3.yaml` with a republished `docs/benchmarks/` for that config, in one PR.
  - **Out of scope**: the other off-policy configs (DQN, DDPG, MountainCarContinuous TD3), which were not run with it.

### Code Health

The last health dashboard (brooks-health, 2026-10-22) scored 94/100 with 0 Critical findings; its four Warnings are
fixed (CHANGELOG.md, 2026-10-22 and 2026-10-25). The five ideas the maintainer picked on 2026-10-28 are done
(CHANGELOG.md, 2026-10-28 and 2026-10-29). A learning change is committed only if, on seeds 1 to 5 against the
committed config, more seeds hold the threshold or the mean final average is better on at least 4 of 5 paired seeds
with no seed clearly worse; otherwise it stays opt-in and is recorded as tried.

- [ ] **Clip Gradients with SGD** (next, picked 2026-10-31)
  - **Description**: `max_grad_norm` is refused unless `optimizer: adam` (`rl_core/agents/optim.py`). With plain SGD
    the discrete actor-critics' critic step grows with the squared norm of the hidden activations (‖h‖²), which a
    growing trunk inflates, and nothing bounds it. Never seen in a real run: the committed Actor-Critic configs use
    Adam.
  - **Acceptance criteria**: `max_grad_norm` clips the global gradient norm with `optimizer: sgd` as it does with
    Adam, in every agent that accepts both optimizers; a non-positive value is still refused. Every committed config
    and golden test is unchanged, since none combines SGD with a clip.
  - **Out of scope**: changing which optimizer or clip any committed config uses.

Parked ideas, not scheduled. Each needs the maintainer's pick, and some a decision on the win rule, before it becomes
an item.

- [ ] **The continuous Actor-Critic's training average**: its greedy policy matches rl-zoo's A2C, but its 100-episode
  training average stays at about -300 on Pendulum (CHANGELOG.md, 2026-10-18). True A2C, each chunk trained on as it
  is collected (`update_every: 8`), did not close it: mean final -295.0 against -298.5, better on 3 of 5 seeds
  (CHANGELOG.md, 2026-10-30; PR #153). Untried: rl-zoo's 8 parallel environments (64 transitions per update, against
  8 here), and gSDE exploration (a PPO version was tried on Pendulum, CHANGELOG.md, 2026-10-29; its code is in
  `~/rl-experiments/item5/` on the development machine).
- [ ] **Faster to the threshold, same final average**: these two reached the threshold sooner on most seeds but did not
  end higher, and the win rule judges the final average only. Adopting either needs a shorter-budget run showing it
  holds as often, or a win rule that also counts steps to the threshold, which is the maintainer's to set.
  - Pendulum PPO with gSDE and separate 64-64 tanh actor and critic, rl-zoo's recipe: it reached -200 sooner on 9 of
    seeds 1 to 5 and 11 to 15 (46,400 to 127,200 steps against 72,400 to 193,000) but held on 8 of 10, as the
    committed config does, and failed the win rule on 11 to 15 (CHANGELOG.md, 2026-10-29).
  - LunarLander DQN with `dueling: true`: it reached 200 sooner on all of seeds 1 to 5 (100,000 to 148,000 steps
    against 152,000 to 228,000) with the same final average (234.8 against 235.3; one seed 34 lower). Held-out seeds
    were not run.
- [ ] **Prioritised-replay DQN**: parked, tested but not merged, on `feature/dqn-prioritized-replay`; it did not help
  MountainCar (MountainCar item, Phase 7). Double and dueling DQN stay opt-in (`double_q`, `dueling`; CHANGELOG.md,
  2026-10-29).

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

- [ ] **Observation Preprocessing & Frame Stacking** (first in Phase 6)
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
  - **Double DQN, re-checked 2026-10-29** (first probed 2026-10-23 on an older config, mean -141 against -133): through
    the benchmark, seeds 1 to 5, it held 1 of 5 against the committed config's 2 (mean final -120.9 against -117.8;
    +17 on two seeds, -19 and -21 on two others). There was little overestimation to remove: on seed 1 the logged mean
    Q sat within about 1 of the discounted return in both arms (CHANGELOG.md, 2026-10-29).
  - **Dueling DQN, 2026-10-29**: held 0 of 5 against the committed config's 2 (mean final -125.0 against -117.8; 18
    and 19 worse on seeds 3 and 5) (CHANGELOG.md, 2026-10-29).
  - **PPO**: never reached the flag without reward normalisation (CHANGELOG.md, 2026-10-23); a separate problem, not
    examined by the diagnosis above.

- [ ] **BipedalWalker-v3 and MuJoCo Tasks (Tier 3)**
  - **Description**: High-dimensional continuous control over millions of steps (BipedalWalker threshold 300;
    MuJoCo Hopper, HalfCheetah). BipedalWalker can use the `box2d` extra and the PyTorch backend that LunarLander
    used (CHANGELOG.md, 2026-10-27 and 2026-10-28); MuJoCo needs `mujoco`.
  - **Agent**: TD3, in NumPy and on the PyTorch backend (CHANGELOG.md, 2026-10-27); SAC is the next candidate
    and the trigger for a shared off-policy base class. n-step returns stay out of scope.

- [ ] **Learning from Pixels (Tier 4)**
  - **Description**: CarRacing or Atari. Needs convolutional networks and a GPU; not practical in pure NumPy.
