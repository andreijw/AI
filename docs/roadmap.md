# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress

Open and in-progress work only. Completed work, with its measured results, is in [CHANGELOG.md](../CHANGELOG.md):
an item that is done moves there, condensed, in the hand-off commit that completes it.

---

## Upcoming Milestones

Order (the maintainer's decision, 2026-10-31): finish Training and Code Health first, then Phase 6 starting with frame
stacking. The TD3 decision and gradient clipping with SGD are done (CHANGELOG.md, 2026-11-01), so Phase 6 is next. The
parked ideas under Code Health do not block Phase 6.

### Workflow Tooling

No open items.

### Training

No open items.

### Code Health

The last health dashboard (brooks-health, 2026-10-22) scored 94/100 with 0 Critical findings; its four Warnings are
fixed (CHANGELOG.md, 2026-10-22 and 2026-10-25). The five ideas the maintainer picked on 2026-10-28 are done
(CHANGELOG.md, 2026-10-28 and 2026-10-29). A learning change is committed only if, on seeds 1 to 5 against the
committed config, more seeds hold the threshold or the mean final average is better on at least 4 of 5 paired seeds
with no seed clearly worse; otherwise it stays opt-in and is recorded as tried.

Parked ideas, not scheduled. Each needs the maintainer's pick, and some a decision on the win rule, before it becomes
an item.

- [ ] **Observation normalisation for the replay-buffer agents**: `normalize_observations` is refused for DQN, DDPG and
  TD3, whose buffers would keep inputs normalised by statistics that later change; SB3 stores raw observations and
  normalises at sample time. Not measured: whether Pendulum PPO or the continuous Actor-Critic learn better with it
  (only smoke runs, CHANGELOG.md, 2026-11-02).
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

- [~] **Curriculum Learning & Domain Randomization Scheduling**
  - **Description**: Add dynamic curriculum pacing that scales domain randomization and noise injection difficulty
    as the agent reaches performance milestones during training. The config's `environment` section stays the
    target (the hardest settings, which evaluation always uses); `environment.curriculum` gives the easiest
    settings and the pacing, and one difficulty from 0 to 1 moves every scheduled setting between the two.
  - **Progress**: three PRs (plan approved 2026-11-03). Done: the wrappers' schedulable settings
    (`feature/curriculum-wrapper-settings`, CHANGELOG.md, 2026-11-03). Next: the scheduler, the trainer wiring,
    resume and the scripts; then the robustness benchmark. For the next PR: CartPole accepts a randomization range
    with a lower bound of 0 (`length`, `masspole`, `masscart`, `gravity`), which Pendulum refuses; a curriculum
    reaching it would divide by zero at reset, so the curriculum's config check, or CartPole's rule, should refuse
    it (review of PR 1a, 2026-11-03).
  - **Acceptance criteria** (rewritten 2026-11-03: the `DomainRandomizationWrapper` the first version named does
    not exist; noise and randomization live in each challenge's wrapper):
    - `CurriculumScheduler` in `rl_core/training/curriculum.py`: linear, step and reward-threshold pacing of one
      difficulty from 0 to 1, an integer stage, exact at both ends, no randomness.
    - Each wrapper lists `SCHEDULABLE_SETTINGS` (observation noise everywhere; action noise and domain randomization
      on CartPole and Pendulum) and changes them between episodes through `apply_settings`: all or nothing, the
      constructor's checks, only settings on at construction, the same randomization parameters. (Done.)
    - The curriculum is set in `environment.curriculum`; `read_env_settings` passes the key through, so `record.py`,
      `export.py` and the benchmark load the same config. Evaluation runs at the target on its own environment; a
      curriculum without one is refused.
    - Resumable mid-curriculum by replaying the saved training rewards; the curriculum is saved, and a resume with
      a different one is refused (the maintainer's decision, 2026-11-03).
    - No curriculum means unchanged training, pinned by the wrappers' recorded trajectories
      (`tests/test_env_golden.py`, recorded before the change).
    - Robustness benchmark: a trained checkpoint evaluated greedily over perturbations inside and beyond its training
      range, one evaluation environment per perturbation built from the config.
    - Unit tests in `tests/test_curriculum.py`, plus wrapper, trainer and script tests.
  - **Solve rule under a curriculum** (the maintainer's decision, 2026-11-03): unchanged; a curriculum result also
    quotes greedy evaluation at the target.
  - **Known edge cases**: Catastrophic forgetting across difficulty stages; oscillation near threshold boundaries
    (difficulty never falls, and a stage lasts at least `min_episodes_per_stage`, no fewer than `window`).
  - **Out of scope**: Adversarial environment generation; automatic domain randomization (ADR) meta-policies;
    curricula over anything but the wrappers' noise and randomization; vector environments; mixing easier episodes
    back in against forgetting (a follow-up if a learning run shows it).

- [~] **Lightweight Embedded Inference Export (C Header & ONNX)**
  - **Progress (2026-11-02)**: the C header PR is done (`export.py`, `rl_core.export`; CHANGELOG.md, 2026-11-02).
    Open: the ONNX export, which waits for the maintainer's decision on the `onnx` dependency.
  - **Description**: Export a trained agent's greedy policy into a zero-dependency standalone C header
    (`policy_weights.h`) for edge microcontroller and robotics deployment, and later into ONNX. Two PRs: the C header
    first (`feature/c-header-export`), then ONNX, which needs the `onnx` package as a new dependency (the
    maintainer's decision, not yet taken).
  - **Acceptance criteria (C header PR)**:
    - A CLI `export.py` beside `train.py` and `record.py`, taking a run config and a checkpoint as `record.py` does
      (`--challenge` included), that writes one header holding the weights
      as `float` arrays and a C forward pass: the network's raw outputs and the greedy action, the same action
      `select_action(obs, training=False)` takes.
    - Every NumPy agent with a network: REINFORCE, Actor-Critic and PPO (discrete and continuous), DQN (plain and
      dueling), DDPG and TD3; a checkpoint from the PyTorch backend exports through the NumPy agent, without PyTorch.
      Observation normalisation, when the agent uses it, is folded into the header. The random agents are refused.
    - A C verification harness compiled with GCC or Clang that runs the header on observations from the challenge's
      environment and compares outputs with NumPy's float64 forward pass (tolerance $10^{-5}$, absolute and relative)
      and greedy actions with the agent's (a discrete action may differ only where the top two outputs are within the
      tolerance; such near-ties are counted and reported).
    - Unit tests in `tests/test_export.py`, compiling C where a compiler is found and skipping where not.
  - **Known edge cases**: Floating-point precision discrepancies (FP32 vs FP64); ties between greedy actions (the C
    code takes the first maximum, as `np.argmax` does); activation clipping overflows (`tanh` of large inputs).
  - **Out of scope**: Quantization-aware training (INT8); target-specific MCU hardware optimizations (CMSIS-NN);
    velocity masking, frame stacking and frame differences in C (the header takes the agent's input vector, after
    them, and names the settings it was trained with); the ONNX export (second PR).

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
