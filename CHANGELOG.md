# Changelog

What changed in the framework, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
There are no version tags: work merges to `develop`. Each pull request adds its own entry under the day it was
handed off for merge, so it merges complete; sections before 2026-10-21 use the day the pull requests merged (a
month, before October 2026). Each entry is the condensed form of its roadmap item; the full discussion is in the
pull requests and the git history. Open work is in [docs/roadmap.md](docs/roadmap.md).

Results quote the committed config and seeds 1 to 5 unless they say otherwise. They use the 100-episode solve rule
from 2026-10-16; earlier results use the 10-episode rule.
"Solves" is the rolling training average reaching the threshold (475 on CartPole-v1, -200 on Pendulum-v1, -100 on
Acrobot-v1) and
"holds" is the final 100-episode average still at or above it.

## 2026-10-29

### Added

- **Double DQN, opt-in** (`feature/double-dqn`, the third of the five Code Health pull requests): `double_q: true` in
  a DQN config lets the online network choose the next action and the target network value it,
  `y = r + (1 - terminated)·γ·Q_target(s', argmax_a' Q_online(s', a'))` (van Hasselt, Guez & Silver 2016), against
  the upward bias of a max over noisy estimates. Off by default, and a run without it is bit-for-bit unchanged
  (`DQN_GOLDEN_EXACT=1 pytest tests/test_dqn_golden.py`). `rl_core.agents.config_values.flag` reads it and rejects
  anything but true or false, so a quoted `"false"` cannot switch it on. SB3's DQN, and so rl-zoo, has no double
  option. Through the benchmark on seeds 1 to 5, against each committed config run alongside, it won on none of the
  four and no config changes:

  | Config | Held (committed → double) | Mean final 100-episode average | Seeds clearly worse |
  | :--- | :--- | :--- | :--- |
  | CartPole | 5 → 3 | 497.5 → 479.7 | 2 (by 25 and 46) |
  | Acrobot | 5 → 5 | -92.6 → -92.0 (better on 3 of 5) | 0 |
  | MountainCar | 2 → 1 | -117.8 → -120.9 | 2 (by 19 and 21) |
  | LunarLander | 5 → 4 | 235.3 → 221.9 | 1 (by 68) |

  "Clearly worse" was set beforehand at 25, 10, 10 and 20 points. On seed 1 of CartPole, Acrobot and MountainCar,
  double DQN's logged mean Q ended 0.5 to 1.6 lower, but the committed configs barely overestimate: mean Q sat within
  about 4 of the discounted return of the policy's own episodes in every case. Double DQN also ran on MountainCar on
  2026-10-23, on an older config, with the same verdict (mean -141 against -133).

### Fixed

- **Discrete Actor-Critic "overfit collapse" was the test's** (`fix/actor-critic-overfit-collapse`, the second of
  the five Code Health pull requests): `test_policy_gradient_maths.py` replayed one fixed batch of alternating
  actions 1,000 times, and the discrete Actor-Critic could fit it and then flip to another action. The batch keeps
  an action the policy has stopped choosing, with an advantage of about -0.5, so its -A·log π term has no lower
  bound. Through the shared ReLU trunk the policy head and hidden layer grew each other until the SGD critic
  diverged and the next step killed every hidden unit: on all 30 seeds tried, including the 25 that looked fitted.
  Dropping actions below probability 1e-3 from the batch, as sampling would, removed every collapse. No agent code
  changed. The test is now `test_on_policy_updates_in_one_state_learn_the_rewarded_action`: each update samples its
  batch from the agent's own policy, and REINFORCE, Actor-Critic and PPO must all end above 0.9 for the rewarded
  action, with the critics within 0.05 of the true return. Over seeds 0 to 29 every agent ends at 0.98 or above and
  both critics within 0.013. The `test-crafter`, `training-doctor` and `algorithm-crafter` skills now keep fixed
  batches for value and Q losses. Allowing `max_grad_norm` with SGD is recorded as a Code Health idea.

## 2026-10-28

### Added

- **LunarLanderContinuous TD3 on the GPU** (`feature/lunarlander-td3`, the last pull request of the LunarLander roadmap
  item, which is now complete): `configs/lunarlander_continuous_td3.yaml` takes rl-zoo's LunarLanderContinuous-v3 TD3
  entry (400×300 networks, learning rate 1e-3, gamma 0.98, Gaussian noise 0.1, 200,000-transition buffer, 10,000
  warm-up actions; SB3's TD3 defaults for the rest) with `backend: torch`, and `td3` is registered for the
  `lunarlander_continuous` challenge. It needs PyTorch and Box2D (`.venv-gpu`); CI checks its settings from the YAML.
  The budget is 400,000 steps (rl-zoo: 300,000), the maintainer's choice from a screen of seeds 1 to 5 at 500,000
  steps, which reached 200 on 3 of 5 by 300,000 and on 4 of 5 by 400,000, held on 3 of 5 at each (one seed peaked at
  199.6). On held-out seeds 11 to 15 at 400,000 it reached 200 on 5 of 5 at 162,000 to 330,000 steps and held it on 4 of
  5 (final 100-episode training averages 177 to 241); greedy evaluation varied from 87 to 269 across seeds. A seed took
  46 to 47 minutes with two sharing the Jetson's GPU, 28 alone. Seed 42 through `train.py` (TR-20) reaches 200 at
  about episode 510 and ends at about 272, greedy evaluation 262 ± 16; its recording (VID-10) scores 249 ± 17. The
  random agent ends at -214 to -228. The trainer updates once per episode where SB3 updates every step; whether that
  is why rl-zoo's 300,000 steps were not enough is a new roadmap item (Training).

### Changed

- **matplotlib only when plotting** (`fix/benchmark-lazy-matplotlib`, the first of the five Code Health pull requests
  picked 2026-10-28): `train.py` and `src/benchmark.py` import `rl_core.utils.visualization` inside the code that
  plots instead of at module level, so a run without `--plot` (or with `--no-plot`), and every `record.py` run, no
  longer loads matplotlib and pyplot. Importing `record` takes 0.33 s instead of 1.10 s on the Jetson.
  `tests/test_import_footprint.py` now fails if `train`, `record` or `benchmark` loads matplotlib on import. The Code
  Health idea "larger networks on a GPU" is dropped from the roadmap: it pointed at the PyTorch backend, which has
  shipped (CHANGELOG.md, 2026-10-25).

## 2026-10-27

### Added

- **LunarLander DQN** (`feature/lunarlander-dqn`, the second pull request of the LunarLander roadmap item):
  `configs/lunarlander_dqn.yaml` translates rl-zoo's LunarLander-v3 DQN (the same entry as Acrobot's) with one hidden
  layer of 256 and a squared-error loss (SB3 uses Huber): one gradient step per transition, target sync every 250
  gradient steps, epsilon to rl-zoo's 0.1 over 12,000 actions, 300,000 steps (rl-zoo: 100,000, judged by a greedy
  evaluation), and the trainer's episode cut set to the 1,000-step limit (its default is 500). Measured through the
  benchmark's runner, single-threaded: on seeds 1 to 5 the 0.1 floor reached 200 on 5 of 5 at 152,000 to 228,000
  steps and held it on 5 of 5 (final 100-episode training averages 214 to 251, greedy evaluation 224 on average),
  against 0.02 holding on 3 of 5 (181 to 260); on held-out seeds 11 to 15 it reached 200 on 5 of 5 at 166,000 to
  297,000 steps and held it on 4 of 5 (198 to 231), greedy evaluation 238 ± 25. Seed 42 through `train.py` reaches 200
  at about episode 450 and ends at 236.9. The random agent ends at -178 to -194.

- **LunarLander-v3 challenges, discrete and continuous** (`feature/lunarlander-challenge`, the first pull request of
  the LunarLander roadmap item): `rl_lunarlander` with `LunarLanderEnv` (observation noise only) and two challenges,
  `lunarlander` (`LunarLander-v3`, four engine actions) and `lunarlander_continuous` (`LunarLanderContinuous-v3`, two
  throttles in [-1, 1]), threshold 200 (Gymnasium's), random agents only, with `configs/lunarlander_default.yaml` and
  `configs/lunarlander_continuous_default.yaml` (50,000 steps). Box2D is a new optional `box2d` extra naming
  `box2d-py==2.3.5` and `pygame-ce`: Gymnasium 1.4's own extra pins `box2d==2.3.10`, which has no build for the
  Jetson, so `./setup.sh` builds `box2d-py` from source with `swig` first, in both profiles (about 90 seconds). CI
  does not install it; the tests that build a Box2D environment skip without it (`env_marks` in `tests/helpers.py`,
  from Gymnasium's entry point), and without it building a LunarLander environment fails at once naming `./setup.sh`.
  Acrobot, MountainCar and LunarLander now build their wrappers through one `rl_core.environments.env_from_config`.
  Measured on seeds 1 to 5 through the benchmark: the random agent's final 100-episode training averages are -178 to
  -194 (discrete) and -214 to -228 (continuous), greedy evaluation -233 ± 31 and -230 ± 26.

- **MountainCarContinuous TD3 at rl-zoo's 400×300 on the GPU** (`feature/mountaincar-continuous-td3`, the last pull
  request of the TD3 roadmap item, which is now complete): `configs/mountaincar_continuous_td3.yaml` takes rl-zoo's
  TD3 entry (400×300 networks, OU noise sigma 0.5, learning rate 1e-3, batch 256, 300,000 steps; SB3's TD3 defaults
  for the rest) with `backend: torch`, and `td3` is registered for the `mountaincar_continuous` challenge. The config
  needs PyTorch (`.venv-gpu`): CI and `.venv` never build it, and the tests that would are skipped without PyTorch
  (`needs_torch` in `tests/helpers.py`, chosen from each config's `backend`), while its settings are still checked
  from the YAML. The benchmark now checks that every selected agent's config can be built before any agent trains
  (`BenchmarkChallenge.check_agent_buildable`): without it, `src/benchmark.py --challenge mountaincar_continuous`
  outside `.venv-gpu` would have trained DDPG for hours, then failed on TD3 and saved nothing. Now it fails at once
  with the PyTorch install hint; `--agents random ddpg` runs the others. Measured 2026-10-26
  through the benchmark on seeds 1 to 5 on the Jetson's GPU: 5 of 5 reach 90 at 36,800 to 70,500 steps (episodes 125
  to 163) and hold it, final 100-episode training averages 92.3 to 93.9, greedy evaluation 93.8 ± 0.2 across seed
  means, against DDPG's 93.4 ± 0.2 at 64×64 (CHANGELOG.md, 2026-10-23). A seed took about 20
  minutes alone and 35 with two benchmark processes running at once. Seed 42 through `train.py` (TR-18): 90 at about
  episode 150, held with a last-100 average of 92.3, greedy evaluation 93.9; its recording (VID-9) scores 93.9 ± 0.1.

## 2026-10-26

### Added

- **TD3 on the PyTorch backend** (`feature/torch-td3-agent`, the third pull request of the TD3 roadmap item):
  `backend: torch` in a TD3 config builds `TorchTD3Agent(TorchDDPGAgent, TD3Agent)`, which runs TD3's gradient steps
  in PyTorch, on the GPU as two CUDA graphs: a critic-only step and a full step that also trains the actor and moves
  the three targets, chosen by TD3Agent's `_is_actor_step`. The target-policy noise is drawn from TD3Agent's own
  generator with each batch, so float64 on the CPU or GPU matches the NumPy `TD3Agent` to rounding (metrics in the
  same order, checkpoints, greedy actions; delays of 2, 3 and 100, with and without clipping;
  `tests/test_torch_ddpg_agent.py`). Checkpoints are TD3Agent's format. `TorchDDPGAgent`'s step is now built from
  `_critic_update`, `_actor_update` and `_move_targets`, which TD3's step reuses.
  - **Measured on the Jetson** (2026-10-26, MountainCarContinuous shapes, 400×300, batch 256, float32 with CUDA
    graphs, `scripts/pytorch_timing/gpu_memory.py`): both TD3 graphs hold 56 MiB on the device (104 MiB reserved)
    against DDPG's 38 MiB, of 7.3 GiB; an update of 999 steps, acting included, takes 3.3 s against DDPG's 2.8 s
    (`docs/pytorch_jetson_timing.md`). No learning run on MountainCarContinuous
    yet: that config is the next pull request.

### Changed

- **The PyTorch backend reads its layout from the NumPy agent** (`feature/torch-backend-hooks`, the second pull
  request of the TD3 roadmap item; no change to any number): `TorchDDPGAgent` no longer names DDPG's two trained
  networks, its target pairs, its step inputs or its metrics. It takes them from `_networks`, the new
  `_optimizers_by_net` and `_target_names`, and from overridable `_input_shapes`, `_sample_inputs`, `STEP_METRICS`
  and `_reduce_metrics`; CUDA graphs are captured per step kind (`_step_kind`), lazily. So a `TorchTD3Agent` only
  supplies its device step. TD3's delay rule and metric means each have one home (`_is_actor_step`, `_step_means`),
  and its checkpoints derive their `<net>_adam_*` keys from `_optimizers_by_net`: a TD3 file lists
  `critic_2_adam_*` earlier, with the same keys and values. The alternative, moving the NumPy maths out of the agents,
  was rejected in review, since it left all of those names hard-coded.
  - **Unchanged numbers:** `tests/test_ddpg_td3_golden.py` pins seeded DDPG (Gaussian, OU, clipped) and TD3 training
    to arrays recorded from `develop` (bit for bit on the recording machine, 1e-12 relative elsewhere). Full runs of
    `pendulum_ddpg.yaml` and `pendulum_td3.yaml` (seed 42, 40,000 steps) on the branch and on `develop` wrote
    identical checkpoints at episodes 50, 100, 150 and 200.

## 2026-10-25

### Changed

- **Benchmark solve numbers apart from plotting** (`feature/benchmark-hygiene`, the second and third Warnings of the
  2026-10-22 health dashboard): the rolling average behind the solve episode, `final_rolling_avg` and the plotted
  curves is now the public `rl_core.utils.smoothing.rolling_average` (the same per-episode mean as before, so no
  number moves; it raises for a window below 1 instead of returning NaN). `rl_core.utils` no longer re-exports
  `plot_training_metrics` and `plot_benchmark_comparison` (import them from `rl_core.utils.visualization`), so
  importing the trainer or the challenges no longer loads matplotlib; `tests/test_import_footprint.py` checks it in
  a fresh process. `run_benchmark(challenge_name, *, settings, config_dir, report_path, save_plot)` and
  `BenchmarkRunner.run(settings, *, report_path, save_plot)` take a `BenchmarkSettings` (agents, seeds, budget,
  window, evaluation episodes, output folder; its defaults are the CLI's, which now reads them from it) instead of
  up to 11 keyword arguments, so an older keyword call fails with a TypeError. The benchmark now rejects
  `--window` or `--eval-episodes` below 1 before training (`--window 0` used to train every run, then write NaN to
  the metrics JSON). The Markdown report is built from five named parts. Console output, metrics JSON and reports
  are unchanged: the recorded characterization files pass untouched, and a review compared the old and new report
  code on 288 generated cases byte for byte. No learning run: no agent, trainer or config changed.

- **MountainCar DQN keeps a 100,000-transition replay buffer** (`feature/mountaincar-hold`), up from rl-zoo's 10,000.
  - **Diagnosis:** the update showed no sign of a fault: the loss stayed finite and small, and the mean Q settled near
    -30, which fits episodes of about 110 steps. A failed episode is stored as cut by the time limit, not as ended.
    The agent reaches the flag reliably. On seed 4, the greedy policy evaluated -100 to -107 between episodes 550 and
    700, then sat near -122 to -131 with a spread of about 20 from episode 850. That is consistent with a good policy
    being forgotten once 10,000 transitions hold only the last ~85 episodes. The learning rate, still 45% of its start
    at episode 700, is a competing explanation that was not tested.
  - **Seeds 1 to 10** (100,000 against 10,000), the size chosen on seeds 1 to 5 and checked on 6 to 10:
    - Final 100-episode training averages: -107 to -132 (mean -120.5) against -121 to -146 (mean -131.2), better on
      all 10 seeds. The gain was +15 on seeds 1 to 5 and +6 on seeds 6 to 10.
    - Greedy evaluation: -120 ± 13 against -127 ± 12. It was better on seeds 1 to 5, but on seeds 6 to 10 it was no
      better on average (-122.6 against -121.9).
    - Seeds 3 and 5 end at or above -110, both crossing it only in their last 5,000 to 14,000 steps, while the
      learning rate is near zero. Seed 6 reaches -110 at 146k steps and drifts back. With 10,000, only seed 4
      reached it and none ended above it.
  - **Other sizes, seeds 1 to 5 (final training averages):** 30,000, 50,000 and 200,000 each beat 10,000, with means of
    -125.6, -125.6 and -127.3. 100,000 scored highest on these same seeds (-117.8), but the differences among the
    larger sizes are within the seed spread. 100,000 with a 300,000-step budget and decay, which changes two things at
    once, scored -124.6.
  - **Still open:** MountainCar still does not reliably hold -110; the roadmap item stays open with the diagnosis.

- **Trainer tests through a seam** (`feature/trainer-test-seam`, the fourth Warning of the 2026-10-22 health
  dashboard; tests only, no behaviour change): `tests/test_trainer.py` no longer calls private `Trainer` methods. Its
  episode, checkpoint and progress-line tests go through `train()` and `evaluate()` and the files and log lines they
  produce, with recording agents in place of patched methods; the five metrics-dispatch tests go through the new
  `tests/trainer_internals.py`. The file left `PRIVATE_ACCESS_EXEMPT`, so the convention test now enforces it.
  `trainer.py` keeps the same coverage (255 statements, the same 3 uncovered). The fake-package prefix is one constant,
  `tests.helpers.FAKE_PACKAGE_PREFIX`, used by `conftest.py`. A new agent is listed in one table fewer: the script
  test reads the supported agents from `EXPECTED_AGENTS` in `test_challenge_selection.py`, and the Pendulum test no
  longer repeats that check.

### Added

- **TD3 agent** (`feature/td3-agent`, the first pull request of the TD3 roadmap item): `TD3Agent`, a `DDPGAgent`
  subclass, as Stable-Baselines3 implements TD3: twin critics with the smaller target, target-policy smoothing
  (noise 0.2 clipped to 0.5 in the scaled action space, from its own seed stream, `AGENT_TARGET_NOISE_STREAM` 14) and
  actor and target updates on every second gradient step, counted as SB3 counts them and checkpointed. For a seed,
  its actor and first critic start as DDPG's and exploration and replay draw as DDPG's. `configs/pendulum_td3.yaml`
  is rl-zoo's TD3 Pendulum settings with `pendulum_ddpg.yaml`'s two departures (64×64, 40,000 steps).
  - **Learning:** on seeds 1 to 5, every run solves -200 at 32,600 to 33,400 steps and holds it (final averages -143
    to -165, greedy evaluation -157 ± 24), against DDPG's 31,200 steps and -158 ± 25 on the same seeds: TD3's actor
    updates half as often and solves about 1,700 steps later (1,400 to 2,200 by seed). 163 s per run on average
    against DDPG's 144 s. A DDPG agent loads a TD3 checkpoint as actor and first critic only; the reverse is
    refused.
  - **Episode-end hook:** `BaseAgent.end_episode()`, called by the Trainer after every training episode's `update`
    and before any checkpoint save. DDPG's noise reset moved there from `update`; seeded DDPG training is
    byte-identical before and after (Gaussian and OU noise, 62 checkpoint arrays each), so its published results
    stand.
  - Behaviour-preserving DDPG refactors for TD3: `_build_networks(weights_rng)`, `_target_pairs()` and a `params`
    argument to `_critic_loss_and_grads`.

- **Optional PyTorch backend for DDPG** (`feature/torch-ddpg-backend`), completing the roadmap item:
  `backend: torch` in a DDPG config's `agent.config` builds `TorchDDPGAgent`, which runs DDPG's gradient steps in
  PyTorch, on the GPU as a CUDA graph, and keeps acting, exploration and the replay buffer in NumPy. NumPy stays the
  default; no committed config changes.
  - **How a backend is chosen:** `BaseAgent.class_for_config(config)`, which `BenchmarkChallenge.make_agent` now
    builds through; each agent class names its other backends in `BACKENDS`, imported only when chosen, so nothing
    else needs PyTorch. `backend: torch` on an agent without a PyTorch version is an error. `DDPGAgent.update` now
    hands its gradient steps to `_train_steps`, the one method the PyTorch backend overrides.
  - **Same training, same checkpoints:** in float64 the backend trains as DDPGAgent does to rounding, on the CPU and
    on CUDA, and its checkpoints are DDPGAgent's format in both directions (`tests/test_torch_ddpg_agent.py`, run
    on the board: CI never installs PyTorch).
  - **Speed through the agent** (25W mode, 200 gradient steps per update as in a Pendulum episode): 0.86 ms per
    step at 64×64 and 2.80 ms at 400×300 on the GPU, against NumPy's 5.43 and 36.6 ms (`docs/pytorch_jetson_timing.md`,
    "Through the Agent"). End to end, `configs/pendulum_ddpg.yaml` runs took 34 s on the GPU against 144 s in
    NumPy (seeds 1 to 5) and, with 400×300 networks, 98 s against 1,074 s (seeds 1 to 3). Every run on both
    backends solved -200 and held it, with the same solve steps to within one episode.
  - **No committed config switches backend:** committed configs must run in `.venv` and CI, which have no PyTorch.
    A GPU run adds `backend: torch` to a copy. `./setup.sh --profile gpu` builds the PyTorch environment
    (`.venv-gpu`); the default `cpu` profile builds `.venv` as before.
  - **Rules and setup:** the `[torch]` extra; the pure-NumPy rule in CLAUDE.md, CONTRIBUTING.md and
    `.brooks-lint.yaml` now allows PyTorch only in `rl_core/agents/torch_*.py`, which only `class_for_config` loads
    (`tests/test_source_conventions.py`). The timing test's `torch_ddpg.py`, `time_torch.py` and
    `test_ddpg_torch_reference.py`, described in the entry below, are replaced by the agent, `time_ddpg.py` and
    `test_torch_ddpg_agent.py`.

- **PyTorch timing test on the Jetson** (`feature/pytorch-timing-test`), a measurement only: nothing in the
  framework uses PyTorch. Results, install steps and the recommendation are in `docs/pytorch_jetson_timing.md`; the
  scripts are in `scripts/pytorch_timing/`, the PyTorch step in its `torch_ddpg.py`.
  - **Setup:** PyTorch's own `torch 2.14.1+cu132` aarch64 wheel drives the Orin's GPU on JetPack 7.2.1 (its `sm_80`
    kernels run on compute capability 8.7). No NVIDIA-specific build is needed, and it installs into a separate
    5.2 GiB virtual environment.
  - **Measurement:** one DDPG gradient step (Pendulum shapes, batch 256), median of 300 steps, 25W power mode.
  - **Results:**
    - NumPy float64: 5.46 ms at 64×64 and 44.7 ms at 400×300.
    - PyTorch CPU float32: 11.3 and 22.2 ms.
    - PyTorch GPU float32 without CUDA graphs: about 22 ms at both sizes, apparently limited by kernel launches.
    - The same step replayed as a CUDA graph: 0.84 and 2.50 ms, 6.5 and 18 times faster than NumPy (float32
      against NumPy's float64; NumPy float32 was not timed).
    - One greedy action costs about 1.1 ms on the GPU against 0.04 ms in NumPy.
  - **Same computation:** `tests/test_ddpg_torch_reference.py` (skipped without PyTorch) runs one NumPy `update`
    and one PyTorch step on the same batch in float64 on the CPU; all four networks agree to a relative 1e-10,
    Adam and Polyak included. Float32, the GPU and CUDA-graph replay are not covered.
  - **Next:** the recommendation is an optional backend for the off-policy agents, DDPG first. It awaits the
    maintainer's decision in `docs/roadmap.md`.

## 2026-10-23

### Changed

- **Learning-rate decay for DQN** (`feature/dqn-lr-decay`): `DQNAgent` takes an optional `anneal_steps`, as PPO
  does: each update's step size is `learning_rate` times `max(0, 1 - steps_trained / anneal_steps)`, counting the
  transitions given to `update` before it (warm-up included). Only the step is scaled, so Adam's moments see the full
  gradient; SGD scales after its element clip. `anneal_fraction` is logged when it is set, and the counter is
  checkpointed (`steps_trained`; an older checkpoint resumes the decay from the full rate). Unset, nothing changes: 30
  episodes of the committed CartPole and Acrobot DQN configs (seed 42) give identical weights and logs.
  `configs/mountaincar_dqn.yaml` now decays its rate over a 200,000-step budget (was a fixed rate over 120,000). Seeds
  1 to 5: final 100-episode training averages -121 to -146 (mean -133) against -135 to -146 (mean -143) for a fixed
  rate over the same 200,000 steps, greedy evaluation -132 ± 11 against -148, and seed 4 reaches -110 at about 103k
  steps before drifting back; over 120,000 steps the decay was mixed (3 of 5 seeds better). MountainCar still does not
  solve under the 100-episode rule. Also probed on the decaying config, seeds 1 to 5, and no better: n-step returns
  (n 3: final averages mean -141; n 5: mean -153, two seeds collapsing late) and Double DQN targets (mean -141). For
  comparison, rl-baselines3-zoo publishes -100.8 for its DQN, an evaluation of the trained agent over about 1,400
  episodes rather than a training average; our greedy evaluation is -132 ± 11.

### Added

- **MountainCarContinuous-v0 challenge with Ornstein-Uhlenbeck exploration** (`feature/mountaincar-continuous-ou`),
  the third rung of the Phase 7 challenge ladder: `MountainCarContinuousChallenge` in `src/rl_mountaincar/` (key
  `mountaincar_continuous`, threshold 90, Gymnasium's `reward_threshold`; 999-step limit) shares `MountainCarEnv`,
  with `random` and `ddpg`. DDPG takes `noise_type: ornstein_uhlenbeck` (`rl_core/agents/exploration_noise.py`,
  SB3's discretisation with `ou_theta` 0.15 and `ou_dt` 0.01 by default, reset at the end of every episode); the
  Gaussian default draws exactly as before and checkpoints gain no keys. `with_box_action_bounds` in
  `rl_core.challenges` now passes the Box action bounds for Pendulum and MountainCarContinuous. Measured through the
  benchmark on seeds 1 to 5 with `configs/mountaincar_continuous_ddpg.yaml` (rl-zoo's DDPG settings with OU sigma
  0.5, 64×64 networks instead of 400×300, 300,000 steps): 5 of 5 reach 90 at 25,000 to 69,000 steps (episodes 112 to
  252) and hold it, final 100-episode training averages 91.0 to 93.2, greedy evaluation 93.4 ± 0.2 across seed means
  against rl-zoo's 93.5. The same config with Gaussian noise of std 0.5 solved 0 of 5: four seeds learned to stand
  still (final averages about -23, evaluation about 0) and one reached the flag in greedy evaluation (92.3) with a
  final training average of 48.8. The random agent scores about -33.

- **MountainCar-v0 challenge** (`feature/mountaincar-challenge`), the second rung of the Phase 7 challenge ladder and
  the first built on the shared challenge base class: `src/rl_mountaincar/` (`MountainCarEnv`, observation noise only;
  `MountainCarChallenge`, key `mountaincar`, threshold -110, Gymnasium's `reward_threshold`) with `random` and `dqn`.
  It does not solve. Measured through the benchmark on seeds 1 to 5 with `configs/mountaincar_dqn.yaml`: DQN learns
  to reach the flag but no seed reaches -110 (final 100-episode training averages -130 to -146; greedy evaluation
  -157 ± 15); the random agent scores -200. About 3 minutes per seed. The config starts from rl-baselines3-zoo's
  MountainCar-v0 DQN with two changes: rl-zoo counts its target interval in environment steps, so its 600 is 300
  gradient steps here (left at 600 gradient steps, seeds 1 to 5 ended at -157 to -165, at 300 at -141 to -148), and
  the exploration floor is 0.02 instead of 0.07 (best 100-episode windows -112 to -130 against -127 to -135 over
  300,000 steps). Ruled out on seeds 1 to 3 with scratch probes: scaled inputs, a learning rate of 0.001, updates
  every 16 steps mid-episode, a Huber loss, two hidden layers of 256 (worse: four of six runs never found the flag)
  and 300,000 steps (no seed solved; runs peak and then drift). PPO is not offered: with rl-zoo's settings and no
  reward normalisation it never reached the flag in 1,000,000 steps on any of seeds 1 to 5. The per-file seeding and
  noise tests of Acrobot and MountainCar moved into the shared `tests/test_env_seeding.py`, which now covers all four
  wrappers; the "unknown environment" tests use `Taxi-v3`.

## 2026-10-22

### Changed

- **One agent table and one settings check per challenge** (`feature/challenge-base-agents`, the first of three
  pull requests from the 2026-10-22 health dashboard, 94/100): `BenchmarkChallenge` now owns `AGENT_CLASSES` (a
  class variable; the base's is empty and read-only) with default `supported_agents` and `make_agent`, and a
  challenge overrides only `agent_config(agent_config, env)` to add what its environment implies (Pendulum's
  torque bounds). Each package's `make_env_from_config` reads an `ENV_DEFAULTS` dict through the new
  `rl_core.utils.config.read_env_settings`, replacing three copies of the config-reading code and Acrobot's list
  of the other challenges' settings. CartPole and Pendulum now reject a setting they do not support, or a
  misspelt one, when it is given a value (`action_noise_std: 0.1` on CartPole used to be ignored); left at null,
  0 or empty it is still accepted. No committed config carries such a setting. 30 episodes of the committed
  CartPole, Pendulum and Acrobot DQN and PPO configs and Pendulum DDPG (seed 42) give bit-identical checkpoints
  and logged metrics before and after.

### Added

- **Acrobot-v1 challenge** (`feature/acrobot-challenge`), the first rung of the Phase 7 challenge ladder:
  `src/rl_acrobot/` (`AcrobotEnv` on `BaseEnvWrapper` with observation noise only; a config that sets action noise
  or domain randomization is rejected) with `AcrobotChallenge` (key `acrobot`, solve threshold -100, Gymnasium's
  `reward_threshold`), found by `discover_challenges` with no edit to `rl_core` or the scripts. Configs
  `acrobot_default.yaml`, `acrobot_dqn.yaml` and `acrobot_ppo.yaml` start from rl-baselines3-zoo's Acrobot-v1
  settings (DQN with one hidden layer of 256 instead of two and an exploration floor of 0.02 instead of 0.1; PPO
  on one environment, without normalisation, for 500,000 steps instead of 1M). Measured on seeds 1 to 5 through the
  benchmark: PPO solves 5 of 5 (72k to 278k steps) and holds 5 of 5 (final averages -83 to -99), evaluation
  -86 ± 9; DQN solves 5 of 5 (25k to 32k steps) and holds 5 of 5 (final averages -90 to -96), evaluation -85 ± 6;
  random scores -500. With rl-zoo's floor of 0.1 DQN held on 1 of 5 (final averages -95 to -103, evaluation
  -91 ± 5), and with 0.05 on 5 of 5 by under 1.2 points on three seeds. About 3 minutes per seed for each. Not
  published under `docs/benchmarks/`.

### Fixed

- **Test registry fixture dropped real challenges** (`feature/acrobot-challenge`): `isolated_registry` restored
  its snapshot of `CHALLENGES` after each test, dropping a real challenge first imported during the test; the
  module stayed imported, so it never registered again, and `pytest tests/test_script_challenge_selection.py` run
  alone failed 5 tests. It now keeps every real `rl_*` challenge and drops only the `rl_fake*` ones.

## 2026-10-21

### Added

- **Resume training from a checkpoint** (`feature/checkpoint-generators` and `feature/train-resume`):
  `python train.py --config <the run's config> --resume <checkpoint_dir>/<challenge>/<agent>/agent_episode_<n>.pt`
  continues a run so that it ends exactly as one that never stopped. Beside each agent checkpoint the trainer now
  writes `trainer_episode_<n>.npz` (episode and step counts, reward and length histories, the update-metric window,
  the evaluation seed and the training environment's generators, through `rl_core.environments.ResumableEnv`); both
  files are written to a temporary name and moved into place. `Trainer.resume` reads the whole trainer file
  (`rl_core.training.checkpoints.TrainerState`) and checks it before loading anything: the SHA-256 of the agent
  file it was saved with, the agent class, the environment, the evaluation environment and that its histories agree
  with each other and with the step count. Budgets count the whole run (resuming episode 600 of 1,000 trains 400 more),
  and checkpoint and video numbering continue. Through `train.py` with seed 42, 200 episodes of
  `configs/cartpole_ppo.yaml` and 100 resumed + 100 give identical checkpoint arrays and a byte-identical plot, as do
  100 episodes of `configs/pendulum_ddpg.yaml` against 50 + 50; with `--record-video`, 20 episodes of the CartPole
  PPO config against 10 + 10 give byte-identical clips under the same episode numbers. `tests/test_resume.py` checks
  every agent on both challenges. Uninterrupted training is unchanged: 100 episodes of the CartPole PPO and DQN
  configs give the same agent checkpoints as before. Maintainer's decisions: an exact resume, budgets as totals. Not
  covered: mid-episode stops, changed hyperparameters, `src/benchmark.py`, old log files (a resumed run writes new
  ones), and checkpoints saved before this change (no trainer state).

### Changed

- **Code health loose ends** (`feature/code-health-loose-ends`): `DQNAgent.__init__` drops from 94 lines and
  complexity 12 to 44 lines, through `_validate_rates`, `_validate_buffer_and_optimizer`, `_init_q_networks` and a
  `_sync_target` shared with the hard target sync; seeded weights are bit-identical before and after.
  `Trainer._log_metrics` drops from complexity 11 to below 10: the metric normaliser is the public
  `rl_core.utils.logger.json_safe`, which `Logger.log` now applies too, so a metrics line written without the
  trainer is also strict JSON; every logger dispatch branch is now tested. The Gaussian agents keep `log_std` inside
  [-20, 2] by projecting the parameter after every Adam step and on `load`, instead of only clipping it in the
  forward pass, where the entropy bonus could push the hidden raw value past the bound without limit;
  `init_log_std` outside the bounds is rejected. No saved Pendulum checkpoint came near a bound (120 checkpoints,
  -2.01 to 0.57), and 100 episodes of the committed Pendulum PPO and Actor-Critic configs (seed 42) give
  bit-identical weights to `develop`.
- **Agent checkpoints hold every generator and the replay buffer** (`feature/checkpoint-generators`, the first of
  two pull requests for resuming training): `BaseAgent.save` and `load` are now one shared template, writing a
  compressed `.npz` of each agent's `_checkpoint_arrays`, its extras and the state of every generator attribute
  (`rng_*`, through `rl_core.seeding.generator_state` and `restore_generator`). DQN and DDPG also save the filled part
  of their replay buffer (`replay_*`, `ReplayBuffer.checkpoint_state` and `restore`, which refuses another capacity or
  an inconsistent write index). A loaded agent now continues exactly as the saved one would have, for every agent;
  the resume tests no longer hand the loaded agent the saved one's generators or buffer. Older checkpoints load as
  before (the random agents' `rng_state` keys too), and seeded training is unchanged: 60 episodes of the committed
  CartPole PPO, DQN, REINFORCE and Actor-Critic configs and Pendulum DDPG (seed 42) give bit-identical weights. A
  stored transition costs about 36 to 41 bytes compressed, so at the committed configs a CartPole DQN checkpoint with
  its full 20,000-transition buffer is about 0.8 MB (34 KB before) and a Pendulum DDPG one at the end of its
  40,000-step run about 2 MB (302 KB before); `save` and `load` reject a key saved twice and check every generator
  state before changing anything.
- **PPO checkpoints hold the partly filled rollout** (`feature/ppo-rollout-checkpoints`): both PPO agents keep
  buffered episodes in one shared `rl_core.agents.rollout.EpisodeRollout`, replacing two copies of the
  bookkeeping. Checkpoints save the buffered steps as `ppo_rollout_*` arrays (none while the buffer is empty), and
  a load replaces the buffer with the checkpoint's; older checkpoints load with an empty buffer. Generator states
  are not saved, so a resume is not bit-identical. By the maintainer's decision, a rollout still partly filled
  when training ends is not trained on: the reported policy stays the one trained on full rollouts. 100 episodes
  of the committed CartPole and Pendulum PPO configs (seed 42) give bit-identical weights before and after.
- **Tighter, faster, safer CI** (`feature/ci-pipeline-hardening`): every pull request runs one `ci.yml` with two
  jobs, `Checks` (Ruff, format, MyPy, Bandit, pylint duplicate-code, markdownlint, README workflow list) and
  `Unit Tests` (pytest, coverage, Codecov), down from about 7 jobs that each installed Python, plus CodeQL. Merges
  no longer re-run the checks, a new push cancels a pull request's older run, and docs-only pull requests skip the
  Python steps but still report both checks. Every action is pinned to a commit SHA; every token is read-only
  except CodeQL's results upload; `copilot-deploy.yml`, which held `contents: write` and a GPG key on pull requests
  while doing nothing, is gone. Nightly maintenance became a weekly re-run of `ci.yml` on `develop` with no write
  access, and its docs-sync pull request is gone (the `Checks` job now fails on a stale list). Dependabot groups
  updates into one pull request per ecosystem. `tests/test_ci_policy.py` fails when a workflow breaks any of these
  rules. Measured before, October 1–21: 1,001 runs, 1,177 jobs, 1,108 of them under a minute. The required checks
  become `Checks` and `Unit Tests` in the ruleset.
- **Benchmark plot curves are a pure, tested function** (#117): `rl_core.utils.visualization.aligned_curves` smooths one
  agent's runs and aligns them by episode or on a shared environment-step grid; `plot_benchmark_comparison` only
  draws (complexity 14 to below 10). New tests work the alignment out by hand; every line, band and the PNG are
  byte-identical before and after. It rejects a window below 1, empty runs and non-positive episode lengths, which
  gave NaN curves or a silently wrong step grid. Clears the last Critical of the 2026-10-21 health dashboard
  (92/100).
- **`train.py` `main` split** (#114): 167 lines down to 48, through `parse_args(argv)`, `output_dirs`,
  `resolve_render_mode`, `attach_video_recorder` and `train_and_plot`. `--help`, every CLI error and seeded outputs
  are unchanged; new tests cover Ctrl-C, `--plot` and closing both environments when training fails.
- **One forward and backward pass for the Gaussian agents** (#115): `ContinuousActorCriticBase` owns
  `_batch_forward` and `_policy_value_grads` (the policy loss enters as d(loss)/d(log π)), and `_PARAM_NAMES` drives
  Adam, `save` and `load`. Gradients are bit-identical on the recording machine, locked in by
  `tests/test_gaussian_gradients_golden.py`. The duplicate-code audit at 8 lines now reports no pair in `src/`.
  Completes the tech-debt paydown from the 2026-10-20 health dashboard (86/100, Tech Debt 69).
- **Roadmap and changelog** (#116): completed roadmap items now live here in condensed form and the roadmap holds
  only open work; finishing an item moves it here in its hand-off commit, and loose ends from completed items became
  open items. The same PR turned off the `Co-Authored-By` commit trailer and lets Claude push once the maintainer
  has committed.
- **Benchmark runner split** (#113): `BenchmarkRunner.run` drops from 210 lines and complexity 15 to 31 lines, via a
  frozen `BenchmarkSettings`, per-seed runs, the public `summarise_runs` and separate writers; the report generator
  takes 5 parameters instead of 10. Console output, metrics JSON, report and plot are unchanged, locked in by
  `tests/test_benchmark_characterization.py`.

### Fixed

- **DQN trained on a batch it did not store** (`feature/code-health-loose-ends`): a batch without `terminated` or
  `dones`, without `next_observations`, or empty, stores nothing and now runs no gradient step and no epsilon decay,
  as in warm-up. The trainer always sends the flags, so committed runs are unaffected.
- **"Overwriting existing videos" on every `--record-video` run** (`feature/code-health-loose-ends`): `train.py`
  created the video folder before Gymnasium's `RecordVideo`, which warns whenever its folder exists. The tests that
  triggered it now record into fresh folders, and pytest turns the warning into an error.
- **Private-access guard allows inherited hooks** (#112): a zero-argument `super()._name` is allowed like
  `self._name`; `super(T, obj)._name` is still flagged. `develop` had been red since #111 met #107.

## 2026-10-20

### Changed

- **Agent tests go through the public interface** (#109, #110, #111): about 560 private-attribute reads in 22 test
  files now go through `select_action`, `update`'s metrics, save/load round trips, or `tests/agent_internals.py`
  for maths checks. An AST guard holds every `tests/test_*.py` to this, apart from a listed set of non-agent files.
- **Shared environment wrapper base** (#107): `rl_core.environments.BaseEnvWrapper` owns seeding, observation noise,
  reset, step statistics and wrapping for `CartPoleEnv` and `PendulumEnv`, which keep only their action noise and
  domain-randomization keys. Seeded rollouts with every noise source on are bit-identical before and after.
- **One softmax for the discrete policies** (#103): `rl_core.agents.softmax.stable_softmax` replaces three inline
  copies in REINFORCE, Actor-Critic and PPO; checkpoints and logs are bit-identical.
- **Complete optimizer state in checkpoints** (#108, roadmap only): every learning agent already saved its Adam
  moments, counters and target networks (#74, #88, #92) and has an exact-resume test; checkpoints holding only
  `adam_t` still load.

### Fixed

- **Discrete PPO bootstraps time-limit cuts** (#104): `rl_core.agents.returns.episode_end` is the rule both PPOs
  share, so CartPole's 500-step cut is no longer terminal. CartPole PPO now solves in 79,556 steps on average
  (was 108,325), every seed faster, and holds 5 of 5. PPO's `rollout_steps`, `ppo_epochs`, mini-batch sizes and
  `anneal_steps` are strict positive integers.
- **Strict integer settings for every agent** (#105, #106): `hidden_dim` and DDPG's `batch_size`, `buffer_capacity`,
  `learning_starts`, `max_updates_per_call` and `hidden_sizes`, and continuous Actor-Critic's `n_steps`, reject 2.5,
  4.0, `True` and `"4"` through the shared readers in `rl_core.agents.config_values`.

## 2026-10-19

### Added

- **DQN config validation** (#100): invalid `target_update_frequency`, `tau`, buffer sizes and integer settings
  raise a `ValueError` at construction; warm-up updates report only `epsilon`, so the log averages real updates.

### Changed

- **Separate random streams for agents and buffers** (#98, #99): each agent and `ReplayBuffer` draws weights,
  exploration, shuffling and replay sampling from named streams of its seed (`rl_core.seeding`). Gymnasium's
  `reset(seed=s)` had drawn the same numbers as `default_rng(s)`. Both benchmarks republished, the figures the
  README and TESTING.md quote: CartPole REINFORCE 4 of 5, Actor-Critic and PPO 5 of 5, DQN 5 of 5 at 143,952 to
  183,439 steps, holding, evaluation 500 ± 0. Pendulum DDPG 5 of 5, evaluation -158 (± 25 across seed means); PPO
  5 of 5 at 72,400 to 156,800 steps, 4 holding; Actor-Critic solves none (final averages -286 to -307) but its
  evaluation of -172 ± 27 beats the random agent's -1293 on 5 of 5.
- **Benchmarks republished with evaluation on its own environment** (#97).

### Fixed

- **Quiet video recorder cleanup** (#102): a rejected `TrainingVideoRecorder` no longer raises in `__del__`, and any
  unraisable exception now fails the suite. Gymnasium is required at 1.0 or later.

## 2026-10-18

### Added

- **DDPG** (#90, #91, #92): pure NumPy `DDPGAgent` with a shared MLP, shared action bounds, prefixed Adam state and
  a `ReplayBuffer` for continuous actions. `configs/pendulum_ddpg.yaml` uses rl-zoo's settings with 64×64 networks
  (400×300 takes about 16 times longer in NumPy) for 40,000 steps: 5 of 5 solve at 31,000 to 31,600 steps and hold;
  greedy evaluation -150 (± 27 across seed means) against rl-zoo's -152 (± 94 across episodes).
- **Separate evaluation environment** (#96): evaluation runs on its own environment with a fixed set of start
  states per run, so `eval_frequency` no longer changes training.

### Changed

- **Pendulum PPO holds -200 under the 100-episode rule** (#93): continuous PPO trains on multi-episode rollouts
  (`rollout_steps`), bootstrapping time-limit cuts. Updating on one episode at a time solved 0 of 5; rollouts of
  2048 steps over 200,000 steps (twice rl-zoo's 100,000) solve 5 of 5 (78k to 155k steps) and hold 4 of 5; on seeds
  1 to 10, 9 solve and 8 hold.
- **Continuous Actor-Critic beats random on Pendulum** (#94, #95): a bounded (tanh) policy mean and 8-step chunked
  updates with GAE. Greedy evaluation -152 ± 45 (rl-zoo A2C -163) beats the random agent on 5 of 5, but the training
  average ends at -286 to -341 and solves no seed. True n-step A2C or gSDE might close the gap (not attempted).
- **DQN reaches the rl-zoo reference** (#89): Adam, a target sync every 128 gradient steps and epsilon decayed to
  0.04 over 8,000 steps, for 300,000 steps: 5 of 5 solve and hold, evaluation 500. rl-zoo's 50,000-step budget
  cannot satisfy the rule, because holding a 100-episode average of 475 alone takes 47,500 steps; rl-zoo's 50,000-step
  result is a greedy evaluation after training.

## 2026-10-16

### Added

- **Budgets and solve rule in environment steps** (#85, #86): training can be budgeted in steps; the benchmark
  reports steps to solve and whether a run held, over a 100-episode window. Budgets are set against rl-zoo. Under
  the new rule: CartPole PPO (200k steps) solves 5 of 5 and holds 4, Actor-Critic (500k) 5 of 5 holding 4,
  REINFORCE (500k) 4 of 5 holding 2, DQN (50k) 0 of 5; Pendulum PPO (100k) 0 of 5, Actor-Critic (1M) 0 of 5.
- **DQN options** (#88): per-step exploration (`exploration_steps`), bootstrapping through time-limit cuts, the
  shared `optimizer` option (the element-wise clip renamed `grad_clip_value`) and complete DQN checkpoints.

### Changed

- **Challenge selection through the registry** (#87): `train.py` and `record.py` build the environment and agent
  through the challenge whose environment id matches `environment.name`, or the one `--challenge` names; an unknown
  id is an error naming the options. Adding a challenge needs no edit to either script.

## 2026-10-15

### Added

- **Shared Adam optimizer** (#74): `optimizer: adam` (with optional global-norm clipping) for the policy-gradient
  agents. CartPole REINFORCE with SGD solved 0 of 5 in 2000 episodes, with Adam 5 of 5 (#75).
- **Standard PPO for discrete actions** (#78, #79): rollouts across episodes, GAE, shuffled mini-batch epochs and
  annealing of the learning rate and clip range. CartPole PPO with rl-zoo's settings plus annealing solves 5 of 5
  (episodes 323 to 427) with a greedy evaluation of 500 on every seed.
- **Pendulum PPO learns** (#82, #83, #84): five causes fixed together: the log-probability of the clipped action
  (it caused the NaN), an initial mean far outside the bounds (`mean_init_scale`), unscaled rewards
  (`reward_scale`), the time limit treated as terminal, and γ 0.99 (now 0.9, learning rate 0.001). Continuous PPO
  gains `anneal_steps`, `clip_epsilon` as its clip key (`clip_ratio` kept as an alias) and a clamp on the log-ratio.
  Under the 10-episode rule 5 of 5 seeds reached -200 but did not hold it.
- **CODEOWNERS** and how pull requests are approved (#80).

### Changed

- **CartPole agents learn within their budgets** (#73, #75, #76, #77): each agent trains for its own config's
  budget, and only the rolling training average counts as a solve. Actor-Critic with Adam first reaches 475 at
  episodes 490 to 839 (SGD: 906 to 1269). Results published in `docs/benchmarks/cartpole_results.md`.
- **`update` split into loss-and-gradients and apply** (#69, #71, #72): every learning agent has a side-effect-free
  loss-and-gradients method, checked against finite differences and hand-worked losses.
- **One discounted-returns function** (#68): `rl_core.agents.returns.discounted_returns` replaces three copies, with
  hand-computed tests that fail without the discount, and softmax stability is locked in at logits of ±1000.
- **Benchmark import cycle broken** (#81): `BenchmarkChallenge` and the registry live in `rl_core`, and the runner
  discovers challenges without naming them.

## 2026-10-13

### Added

- **Dev team skills for Claude Code** (#64): `dev-loop`, `architect`, `test-crafter`, `cleanup`, `pr-reviewer`,
  `doc-sync`, `algorithm-crafter`, `experiment-runner` and `training-doctor`, with the brooks-lint plugin, project
  review rules C1 to C6 and gradient-check helpers.
- **Training diagnostics in logs** (#65): what `update` returns is averaged into the periodic metrics record, and the
  first non-finite value of each metric is logged as a warning.

### Fixed

- **Seeded environments replay one start state** (#67): a seeded environment now gives a reproducible sequence of
  different start states instead of the same one every episode.
- **Continuous PPO mini-batch shuffle seeded** (#66): it drew from NumPy's global generator; a source test now bans
  bare `np.random` calls.

## 2026-10-12

### Added

- **Continuous control and core decoupling** (#63): the algorithm-agnostic framework moved to `src/rl_core/`;
  Gaussian-policy `ContinuousActorCriticAgent`, `ContinuousPPOAgent` and `ContinuousRandomAgent`; the
  `src/rl_pendulum/` challenge for Pendulum-v1.

## 2026-10-10

### Added

- **Cross-agent benchmark suite** (#62): `src/benchmark.py` trains, evaluates and compares agents across seeds, with
  comparison plots and Markdown reports in `docs/benchmarks/`.

## 2026-10-09

### Added

- **DQN** (#58, #59, #60): `ReplayBuffer`, a pure NumPy `DQNAgent` with online and target networks and
  epsilon-greedy exploration, `configs/cartpole_dqn.yaml`, and a contract test harness for every agent.

## 2026-10-08

### Added

- **Per-agent output directories** (#57): checkpoints, videos and plots go to a folder per agent, so training one
  agent no longer overwrites another's.

## 2026-10-04

### Added

- **Training video interval and trained-agent recording** (#56): `--video-every N` and `record.py`, which plays a
  checkpoint back into one video.

### Changed

- **Python 3.12** across CI, config and docs (#55).

## 2026-10-02

### Added

- **CLAUDE.md, `setup.ps1` and the Context7 MCP server** (#54); the workflow moved into CONTRIBUTING.md.

## 2026-09

### Added

- **GEMINI.md workflow guides** (#53), with Ruff SIM108 and formatting fixes.

## 2026-08

### Changed

- GitHub Actions dependency bumps (#51, #52).

## 2026-07

### Changed

- GitHub Actions dependency bumps (#49, #50).

## 2026-05

### Added

- **PPO agent** (#45, #46) and README run and visualization guides; the roadmap set DQN as the next milestone (#48).

### Changed

- **Runnable training paths** streamlined for every CartPole agent (#47).

## 2026-04

### Added

- **Nightly maintenance workflow** (#41, #42): scheduled scans, duplication checks and workflow-doc sync.
- **Extensive agent test suite** (#44).

### Changed

- **Documentation aligned** with the tooling (#33), the training config wiring fixed (#34), exception handling and
  environment-variable lifecycle hardened (#40), and dependency floors raised (#35 to #39, #43).

## 2026-03

### Added

- **REINFORCE** (#26) and **Actor-Critic (A2C)** (#29) agents for CartPole, and training-metrics plots (#23).

### Changed

- **Seeded `RandomAgent`, deep config merging and consistent trainer logging** (#25); logging discipline and a
  leak-proof logger (#28); stricter environment and trainer input validation (#30); CI hardened (#27); docs
  updated (#31); dependency bumps (#22, #24, #32).

## 2026-02

### Added

- **Initial framework** (#1): the CartPole package structure, `CartPoleEnv` on Gymnasium's `terminated`/`truncated`
  API, `BaseAgent`, `RandomAgent`, the `Trainer`, YAML configs and JSONL logging.
- **Noise injection, domain randomization, configurable environments and vectorized environments** (#7).
- **CI workflows** (#4, #5, #6, #8, #9, #14, #15, #16): Ruff, unit tests with coverage, mypy, Bandit, CodeQL,
  markdownlint and the Copilot deployment workflow.
- **Headless video recording** for training over SSH on the NVIDIA Orin Nano (#21).

### Changed

- **Test coverage from 64% to 99%** (#17), a refactor of `src/rl_cartpole` (#18), workflow fixes (#20) and
  GitHub Actions dependency bumps (#10 to #13, #19).
