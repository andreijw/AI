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

## 2026-11-12

### Added

- **A `release` skill** (`feature/release-skill`; the maintainer's request, 2026-11-12). `.claude/skills/release/`
  runs `docs/releasing.md` in order: the version bump and CHANGELOG section in a pull request, the local
  `scripts/release.py check`, the maintainer's signed tag commands in dependency order, watching the Release
  workflow, and checking the draft (assets, `SHA256SUMS`, notes against the CHANGELOG, prerelease flag). Signing the
  tag and publishing the draft stay the maintainer's. First used on `rl-core-v0.1.0` (run 43286857713, green; draft
  checked). `docs/releasing.md` no longer says the two release settings are off.

- **Circle extraction bridges dropped beams** (`feature/segment-scan-bridging`; roadmap, Phase 8 SLAM follow-up).
  `segment_scan` joins two usable beams across up to `FeatureSettings.max_bridged_angle` of unusable ones (default
  3°: 3 beams of the robot's 1° lidar, none of the 5.6° 64-beam lidar), with the breakpoint scaled by the angle
  spanned; a dropped beam never enters a fit. `feature_evaluation` reports `reachable_rate`, the share of visible
  circles that keep 8 usable beams. Judged once on seeds 151 to 200 (TESTING.md `CIRC-2`) against gates registered
  beforehand; the detection gate was lowered from 0.8 to 0.75 of `reachable_rate` after tuning on seeds 1 to 50 and
  before 151 to 200 were read (the maintainer's decision; under 0.8 the row fails, 0.622 against 0.661). At 30 %
  dropout on the 360-beam lidar: detection 0.622 against a bar of 0.620 (0.104 without bridging on seeds 1 to 50),
  false fits 0.004 (0.285); without dropout every `CIRC-1` gate still passes (NEES 2.93, detection 0.773). SLAM-1 at
  30 % dropout: 4 aligned failures against 5 without dropout, circles mapped 0.907 against 0.914 (17 failures without
  bridging on seeds 1 to 50). It was judged as a count of 3 beams and capped by angle after the review, because on the
  64-beam lidar a 3-beam bridge spans 22° and gave overconfident fits (NEES 24); the cap gives the judged segments
  exactly on 360 beams (0 of 4,800 scans differ) and no bridging on 64. Not verified: SLAM-1 on the 64-beam lidar, and
  the extraction timing alone on the machine (it was timed under shared load: 6.1 ms median, 10.8 ms p99).

### Fixed

- **Actor-Critic bootstraps a time-limit cut** (`fix/ac-time-limit-bootstrap`; review of 2026-11-11, W2 and W3).
  A new agent key `bootstrap_truncation` (default false) makes the discrete Actor-Critic, and the continuous one
  with `n_steps: 0`, add γ^(T−t)·V(s_T) to the Monte Carlo returns of an episode cut by a time limit or
  `max_steps_per_episode`, with V at the true next state. A termination is unchanged bit for bit. The bootstrap is
  written once per base class (`ActorCriticBase._bootstrap_cut`, `ContinuousActorCriticBase._add_bootstrap`), and
  PPO and continuous PPO now use it. REINFORCE has no critic and still treats a cut as an end, which a test now
  pins. On `configs/cartpole_actor_critic.yaml`, seeds 1 to 5 paired, 500,000 steps: it held 475 on 5 of 5 seeds
  against 4 of 5, and the final average was better on all 5 (mean 495.9 against 481.3; +6.8 to +23.5). Steps to
  solve fell from 374,332 to 231,271 on average. The committed config turns it on (the maintainer's decision), and
  `docs/benchmarks/cartpole_results.md` was rerun in full: the Actor-Critic row is now 945.6 episodes to solve and
  100% held, and the other agents' learning figures are unchanged. Not verified: Pendulum with `n_steps: 0` (the
  committed Pendulum config uses `n_steps: 8`, which already bootstrapped), and seeds other than 1 to 5.

### Changed

- **The `release` skill publishes a draft once the maintainer says yes** (`docs/release-skill-publish`; the
  maintainer's request, 2026-11-12). After its draft check it asks for each tag by name and, on a yes, runs
  `gh release edit <tag> --draft=false`; tags stay the maintainer's. rl-core and rl-navigation 0.1.0 were published
  this way on 2026-11-12.

- **Tests that can fail, and CI on branch coverage** (`fix/tests-that-can-fail`; code-health passes of 2026-11-12,
  the maintainer's approved cleanup plan, PR 1 of 4). Tests that held no assertion or checked only a type now pin
  behaviour (logger levels and file output, the Random agent's exploration stream, hand-worked REINFORCE losses and
  Actor-Critic values), each shown to fail under a mutation of the code it covers; the REINFORCE and Actor-Critic
  copies in `test_agents_extensive.py` are one parametrized class; 34 unseeded `np.random` calls now use seeded
  generators; the run-queue overlap test no longer depends on sleeps overlapping. New nets: every challenge's
  `CONFIG_MAP` equals its `AGENT_CLASSES` (all 10 do), and every torch agent overrides or stubs the NumPy training
  maths it inherits (runs only where PyTorch is installed, not in CI); `TorchTD3Agent` now stubs `_td3_targets` and
  `_target_actions`. `scripts/write_reference_checkpoints.py` writes 15 reference checkpoints (380 KB in
  `tests/data/checkpoints/`), pinned by keys, dtypes, shapes and value hashes, for the checkpoint refactor that
  follows. CI and the Release workflow now require 90 % branch coverage (`--cov-branch --cov-fail-under=90`; the
  maintainer's decision), against 50 % line coverage before; the suite measures 94.8 % in `.venv`. Not verified: the
  branch coverage on CI's runner, which has no Box2D.

## 2026-11-11

### Added

- **Importable libraries with GitHub Releases** (roadmap, Workflow Tooling; the maintainer's request and approved
  plan of 2026-11-11; PRs 1 to 5 below). The code is three distributions with unchanged import names: `rl-core` (the
  framework a game project imports), `rl-navigation` (the robot's libraries, NumPy only) and `rl-challenges` (the six
  classic and Box2D challenges), each with its own version from 0.1.0 and CHANGELOG under `packages/<dist>/`.
  Challenges register through the `rl_core.challenges` entry-point group, a pushed `<dist>-v<version>` tag drafts a
  GitHub Release, and a worktree tests its own code in its own layered environment. The deferred follow-ups (CLIs as
  console scripts, configs as package data, an optional `rl-core[plots]`, native ROS 2 packaging) are the roadmap's
  "Library follow-ups". Not verified: a real tag run of `release.yml` (it first runs on the first tag), the `Packages`
  job on GitHub's runners, `setup.ps1` on Windows, and the repository settings `docs/releasing.md` lists for the
  maintainer.
- **Training on another project's environment** (`feature/external-challenge-guide`). `docs/external-challenges.md`
  builds a toy game project end to end: a Gymnasium environment, a `BaseEnvWrapper`, a `@register_challenge`
  challenge declared as one entry point, a config, and training from the game's own virtual environment with this
  checkout's `train.py` and `benchmark.py` (run as written: 20 PPO episodes, a 50-episode benchmark report). README
  gains a section on using the distributions elsewhere. Not verified: installing from a release URL (no release
  yet), a continuous-action game, Windows.

- **Releases from a pushed tag** (`feature/release-workflow`). `.github/workflows/release.yml` runs on a tag
  `<dist>-v<version>`: `scripts/release.py check` refuses a tag off `develop`, one that differs from the distribution's
  version (spelled out: `v0.1` is not `0.1.0`), a version without a CHANGELOG section, and a distribution whose in-repo
  requirements, its extras' included, have no tag at their in-tree versions; then the tests run, `python -m build`
  makes the sdist and wheel, `release.py smoke` installs the wheel (with the published wheels of what a plain install
  requires) in a fresh environment, imports every module a plain install can and checks each package's `py.typed`,
  `SHA256SUMS` is written, and a **draft** GitHub Release is opened, a prerelease (`rc`, `.dev`) marked as one. A run on
  a tag whose draft exists refreshes its notes, title, prerelease flag and files; a published release is never changed.
  A dependency whose release is still a draft stops the run with "has no published release", and any other `gh` error
  is printed as it is. Only the publish job can write, to releases; `gh` is told the repository with `GH_REPO`, which a
  new `tests/test_ci_policy.py` rule checks. The non-required `Packages` CI job builds the three distributions on every
  pull request and smoke-tests each (rl-core imported 54 modules, rl-navigation 28 with only NumPy loaded,
  rl-challenges 30, measured locally), then runs the 19 robot-side test files from the rl-navigation wheel alone on
  NumPy 1.26 (783 passed locally, `pytest --noconftest -o addopts=""` with only that wheel, NumPy 1.26.4 and pytest
  installed). `rl-navigation` now requires NumPy 1.26 or newer, and its `challenge` extra Gymnasium too. `build` joins
  the dev tools. `docs/releasing.md` has installing by release URL (every `rl-*` wheel in one command, since there is
  no index to fall back on), the steps, the order of the first tags, a breaking `rl-core` change and the two
  repository settings for the maintainer (immutable releases, a tag ruleset).

- **Per-worktree environments** (`feature/challenge-entry-points`). `scripts/worktree_venv.sh <worktree>` makes
  `<worktree>/.venv`, layered on the main checkout's `.venv` (one `.pth` line) and holding an editable install of
  the worktree, offline in about 10 s. `tests/conftest.py` stops a run whose `rl_core` comes from outside the
  repository it tests, so a worktree can no longer pass or fail on the main checkout's code. The `dev-loop` and
  `experiment-runner` skills and CLAUDE.md use the worktree's interpreter instead of `PYTHONPATH=<worktree>/src`.

### Changed

- **Every version is defined once** (`feature/pin-dependencies`; the maintainer's request, 2026-11-11). A new
  `requirements-lock.txt` pins the 70 third-party packages the development environments, CI and the Release workflow
  install, at the versions of CI's last green run (run 43092816464) plus the local environment's for what CI does not
  install (the video and box2d extras, the box2d build tools; pillow 11.3.0, which moviepy 2.2.1 needs, so Dependabot
  ignores pillow 12 until moviepy allows it), and colorama for Windows. `requirements-dev.txt` starts with
  `-c requirements-lock.txt`, so every `pip install -r requirements-dev.txt` is pinned, extras included; `setup.sh`,
  `setup.ps1`, every `ci.yml` job, `release.yml` and `release.py smoke` (which now refuses to run without the lock)
  install through it. The Packages job's NumPy leg overrides only NumPy, on purpose, with rl-navigation's floor read
  from its `pyproject.toml`. PyTorch stays outside the lock, from PyTorch's CUDA index. The distributions keep floors,
  one per package (NumPy 1.26 everywhere; box2d-py a floor, its exact version now the lock's), so a robot or a game is
  not tied to the pins. rl-challenges' version comes from `rl_cartpole.__version__`, the same `attr` style as the
  other two. Python's version is `.python-version` (setup-python's `python-version-file`, `setup.sh`, `setup.ps1`),
  and markdownlint-cli's is `ci.yml`'s `MARKDOWNLINT_CLI_VERSION`. Pre-commit's Ruff, mypy and Bandit hooks run the
  activated environment's own tools (`repo: local`), which retires the drift between pre-commit's Ruff 0.1.15 and
  CI's. `-Update` and `--upgrade` now re-sync an environment to the lock. `tests/test_versions.py` holds all of it: one
  exact pin per package, one floor per package, every pin within every floor, every package the dev install pulls in
  pinned, no version written in a script, workflow or module outside these places, every install through the lock,
  one Python and one markdownlint version. Not verified: `setup.ps1` on Windows (and Windows wheels for every pin), the
  gpu profile with the lock, Dependabot's first run on the lock (GitHub's docs say its pip updater reads any `.txt`
  file; they do not say how it treats the `-c` line).

- **Docs, skills and review rules follow the `packages/` layout** (`feature/package-docs-sweep`). Every path to the
  code (in the skills, CLAUDE.md, GEMINI.md, ARCHITECTURE.md, CONTRIBUTING.md, TESTING.md, `docs/` and
  `.brooks-lint.yaml`) and every install and gate command now names `packages/<dist>/src/`,
  `requirements-dev.txt`, `mypy` (its file list is in `pyproject.toml`) and `bandit -r packages/ benchmark.py`.
  ARCHITECTURE.md gains a Distributions diagram (the three distributions, their requirements and the entry points)
  and labels the component diagram's layers by distribution. `docs/navigation.md` says how a ROS 2 node installs
  `rl-navigation`: a virtual environment made with `--system-site-packages` on Ubuntu 24.04. `tests/test_packages.py`
  lists what a distribution must have (its folder's name, README, CHANGELOG section for its version, LICENSE, no
  `[tool.ruff]`, a `requirements-dev.txt` line, installed metadata at its version, requirements on another
  distribution that admit its version, no tracked file left in `src/`, a `py.typed` marker in every import package,
  every third-party import declared as a requirement or in an extra, and NumPy only on rl-navigation's robot side), so
  a new library starts as a copied folder. `rl-challenges` now declares Gymnasium and NumPy, which its modules import
  directly. The doc-sync, dev-loop and pr-reviewer skills know the per-distribution CHANGELOGs and that dependencies
  live in `packages/*/pyproject.toml`.

- **Three distributions under `packages/`** (`feature/split-distributions`). The single `rl-cartpole` distribution is
  split, with unchanged import names, into `rl-core` (`packages/rl-core/src/rl_core`), `rl-navigation`
  (`rl_navigation`: NumPy only, extras `gym`, `challenge` and `video`) and `rl-challenges` (the six challenge
  packages, extra `box2d`), each with its own `pyproject.toml`, README, CHANGELOG, LICENSE and `py.typed` marker, at
  version 0.1.0 (`rl_core.__version__`, `rl_navigation.__version__`). `src/benchmark.py` is now `benchmark.py` at
  the root. The root `pyproject.toml` holds only tool configuration, and `requirements-dev.txt` (which replaces the
  stale `requirements.txt`) installs all three editable with the dev tools; `setup.sh`, `setup.ps1`, CI, Dependabot
  (which now ignores the three `rl-*` names, unregistered on PyPI) and pre-commit follow. The scripts and tests no
  longer put `src/` on `sys.path`. Coverage names each `src/` root, since `--cov=packages` would leave out every
  module no test imports. A library's missing-extra message names the extra (rl-core's `onnx` or `torch`,
  rl-navigation's `video`) rather than a checkout path. An environment made before this change runs `./setup.sh` (or
  `setup.ps1`) again, which removes the old `rl-cartpole`; the untracked `src/` left behind is safe to delete. **This
  PR merges together with the next one** (`feature/package-docs-sweep`): until then the docs, skills and review rules
  still name `src/` paths and `pip install -e ".[...]"`, which no longer installs the code.

- **Challenges are found through entry points** (`feature/challenge-entry-points`). `discover_challenges()` imports the
  modules declared in the `rl_core.challenges` entry-point group (`pyproject.toml`, one line per package) instead of
  scanning the directory beside `rl_core`, so a challenge can live in another distribution or project. This repository's
  distributions load first and the rest by distribution and entry name, so the order no longer depends on `sys.path` and
  a plugin cannot take an in-repo key; a module declared twice loads once, an entry name reused for another module is
  recorded as a clash, a module that raises or calls `sys.exit` on import is recorded rather than ending the run, and
  failures are keyed by the module's package, so the messages are unchanged. With no challenge registered the
  unknown-challenge error says how to install them. The tests see only this repository's entry points (they replace the
  public `registry.challenge_entry_points`), and the private-access scan now reads `tests/conftest.py` and
  `tests/helpers.py` too. An existing environment must re-run its editable install once (`./setup.sh`), since entry
  points are installed metadata.

- **Docs and diagrams are a step of every PR** (`fix/review-doc-fixes`). The `doc-sync` skill gains a "Diagrams"
  step: update every Mermaid diagram a change makes false, and add one where it explains the change better than
  prose (a module whose parts call each other, a class hierarchy, a state machine, a cross-package data flow).
  The `pr-reviewer` skill checks that the docs were updated in the PR and that diagrams match the code, defines its
  three severities, and requires `TESTING.md` `TR-15` from `.venv-gpu` when a change can reach the PyTorch agents
  (CI has no PyTorch, so a broken `backend: torch` config passes it); `dev-loop` step 8 points at both.

### Fixed

- **Stale numbers and wrong claims found by the agent and challenge review** (`fix/review-doc-fixes`). Comments
  and docs only; no config value changed. Pendulum Continuous Actor-Critic and DDPG numbers in `docs/results.md`
  and the config headers brought to the current report (-170 ± 26; -158.3 ± 24.8); the CartPole REINFORCE and
  Actor-Critic optimizer comments matched to the report (4 of 5, holding 3; solve at episode 1272 on average);
  the navigation robot is described as also told the goal's distance and bearing, not seeing "only its range
  sensors"; `value_coef` comments state that this repo's critic loss carries a ½, so 0.5 here is SB3's `vf_coef`
  0.25; the discrete `ActorCriticAgent` is described as REINFORCE with a learned baseline, not A2C, and REINFORCE's
  per-episode return standardisation as a (slightly biased) heuristic; README notes which reports lack an agent
  and the held-out seeds of LunarLander DQN and LunarLanderContinuous TD3, and Acrobot DQN's tuning on its
  reporting seeds. Review reports: `~/rl-experiments/review-2026-11-11/`.

- **A checkpoint that does not fit the agent is refused before anything loads**
  (`fix/checkpoint-shape-check`). A review of every agent (2026-11-11) found that `load()` assigned a
  checkpoint's arrays before checking them: one saved with another `hidden_dim` loaded silently under SGD (weights
  of the other width, `hidden_dim` unchanged) and half-loaded under Adam (weights replaced, then the optimizer state
  raised; DQN had already restored its replay buffer), and one missing a weight raised `KeyError` part-way.
  `BaseAgent.load` now checks, before any assignment, that every array the agent saves is present and has the
  agent's shape, and raises `ValueError` naming the entry (and both shapes); the agent is then unchanged. Entries an
  older file legitimately lacks (Adam moments of an SGD run, DQN's counters, a replay buffer) are listed per class
  and still load. `ContinuousRandomAgent` refuses a checkpoint of other dimensions or bounds instead of adopting
  them. `record.py` builds its agent on the NumPy backend, as `export.py` and `robustness.py` do, so a
  `backend: torch` run records without PyTorch; a test pins SAC's PyTorch gradient at the `log_std` bounds. Tests:
  `tests/test_checkpoint_shapes.py` (64), full suite 4108 passed; `TESTING.md` `TR-15` from `.venv-gpu` 441 passed.
  No update rule changed, so no learning run.

## 2026-11-10

### Added

- **Landmark EKF-SLAM on the 360-beam lidar** (`feature/landmark-ekf-slam`; PR 2b of the Phase 8 item "Occupancy-grid
  mapping and SLAM"; judged against registered gates and failing 2 of 9, shipped with the failure recorded, the
  maintainer's decision of 2026-11-10): `rl_navigation.slam.EkfSlam` maps PR 2a's circle features without known poses.
  It follows Thrun's Tables 10.1 and 10.2 with a provisional landmark list, uses first-estimate Jacobians (Huang et al.
  2008) so the unobservable heading is not made falsely certain, keeps each circle's radius in the state, makes one
  batch Joseph update per scan with maximum-likelihood association gated at χ²₃ 0.99 (labelled features at 0.999), adds
  a model-error floor of 1.5 × the range noise per feature, and never raises on a scan's content (unsound features
  skipped; an update leaving the state non-finite, P indefinite or the pose off the arena is undone).
  `rl_navigation.slam.evaluation` is the desk tool behind TESTING.md `SLAM-1` and `SLAM-2`. Gates were set by the
  maintainer after tuning on seeds 1 to 20 and a check on 21 to 50, registered, then judged once on seeds 101 to 150,
  600 steps. The 360-beam row passes 7 of 9: error after SE(2) alignment 0.064 m mean and 11 mm median (≤ 0.10, ≤ 0.03),
  NEES under the 95 % bound on 0.960 of steps (≥ 0.88), circles mapped 0.803 (≥ 0.80), spurious 0.24 and duplicates 0.06
  per run (≤ 0.6, ≤ 0.2), lost 13 of 50 (≤ 15). It fails 2: position RMSE 0.443 of dead reckoning's (≤ 0.40; 0.354 m
  against 0.799 m, dead reckoning better than on tuning seeds' 0.93 m) and 9 runs above 0.10 m after alignment (≤ 8; 4
  of them association failures, as the oracle-labelled row has 5, and seed 113 has no circle). Reported without gates:
  30 % dropout (lost 25 of 50; PR 2a's segmentation splits at every dropped beam), oracle association, σ 2 cm, the
  64-beam lidar (lost 49 of 50) and the ceiling, `EkfLocalizer` on the true map (0.9 mm). `SLAM-2` on one Jetson core:
  an update 7.5 ms median with 10 landmarks and 8.4 ms with 30 (≤ 10 ms; p99 ≤ 30 ms), `predict` 0.19 ms (≤ 0.2 ms). -
  **Circle features from a 360-beam lidar** (`feature/lidar-circle-features`; PR 2a of the Phase 8 item "Occupancy-grid
  mapping and SLAM", whose PR 2, landmark EKF-SLAM, is split into 2a features and 2b the filter; stacked on the EKF
  fix): `rl_navigation.slam.features` (NumPy and `sim` only) splits a scan into segments at range jumps, fits a circle
  to each of 8 points or more by weighted Gauss-Newton from a Kasa start, with each residual weighted by the range noise
  projected on the surface normal, and reports its covariance from (JᵀJ)⁻¹. A fit is kept only if it passes a chi-square
  test, its radius is 0.1 to 0.8 m, it is convex towards the robot, two lines (a box corner or a wall) do not explain
  the points better, and its standard deviations are under 0.1 m. The robot's lidar is the maintainer's choice of
  2026-11-09: 360° at about 1° per beam. `rl_navigation.slam.feature_evaluation` is the desk tool behind TESTING.md
  `CIRC-1`. Pre-registered and judged once on seeds 101 to 150 (360 beams, σ 1 cm), all gates pass: fit NEES mean 3.22
  (gate 2.5 to 3.5; 2.8 to 2.9 on tuning seeds), false fits 0.0005 (≤ 0.005), detection 0.764 (≥ 0.70), centre error
  median 5.7 mm and p95 2.4 cm (≤ 10 mm, ≤ 4 cm), extraction 6.4 ms median and 11.1 ms p99 on one Jetson core (≤ 8 ms, ≤
  15 ms). Reported without gates: σ 2 cm (false fits 0.022, centre median 12 mm) and the 64-beam default (detection
  0.80, 0.19 fits per scan). The SLAM filter is PR 2b. - **Several training environments on one update clock, and PPO
  with several environments** (`feature/parallel-envs-one-clock`; PR 2, the last of the roadmap item "Parallel
  environments for the on-policy agents", which is now complete; stacked on PR 1): - **`update_every` with `num_envs`**:
  every `update_every` vector steps the trainer hands the agent one window holding each environment's steps
  (`update_segments`): segments of episodes that ended first, then the open one cut at the boundary, which bootstraps.
  The continuous Actor-Critic takes one A2C step over the window (targets from the same weights, each segment
  bootstrapped from its own next state, advantages normalised jointly, gradient clipping over the window); one segment
  trains exactly as `update` does. PPO, both kinds, accepts several environments only with this layout (`rollout_steps`
  a multiple of `update_every × num_envs`, trained once per full rollout, so the stored old log-probabilities stay those
  of collection); whole-episode PPO at N > 1 stays refused. With `update_every`, an evaluation or checkpoint can come
  before an episode's last steps are trained on. Asynchronous environments are deferred until a measured speedup. - **N
  = 1 unchanged**: the trainer golden's three arms, now including the Pendulum Actor-Critic with `update_every: 8`
  through the new collector, match bit for bit. - **CartPole PPO, 4 environments × 64 steps** (seeds 1 to 5, 200,000
  steps; claim fixed beforehand): solves and holds 475 on 5 of 5, 87,165 steps to solve on average against 79,556 with
  one environment. - **Pendulum continuous Actor-Critic, 8 environments × 8 steps** (rl-zoo's A2C layout; seeds 1 to 5
  paired against the committed config, 1,000,000 steps): final 100-episode training averages -244.1 against -282.7,
  better on all 5 seeds (by 7.2 to 63.6), so it wins under the project's rule; no seed holds -200; greedy evaluation
  -184.0 ± 42.7 against -170.0 ± 25.9; about 260 s a seed against 329 s. A second arm at `learning_rate: 0.002`
  (measurement only): -264.7, one seed 21.3 below the baseline. The committed config is unchanged; adopting the layout
  is the maintainer's decision (Code Health item). - The experiment-runner skill now says queued runs import a frozen
  copy of `src`: PR 2's first runs were cancelled because they could have imported code under a mutation check.
  TESTING.md `TR-38`.

### Fixed

- **The EKF reports itself lost when its estimate leaves the map** (`fix/ekf-sparse-held-scans`; the State estimation
  follow-up "The EKF's hold leaves sparse HC-SR04 scans quiet"; the maintainer's decisions of 2026-11-10):
  - **The cause was not sparse held scans.** Held scans with one candidate explained 6 and 21 of about 2,100 and
    2,400 confidently-wrong steps. About half of the HC-SR04's came from a lost estimate outside the arena's walls,
    where every sonar is nonlinear or reads nothing, so no scan counted toward the reset; the estimate gets there
    through held updates (56 to 74 %) or dead reckoning, and the silence predates the hold. The rise the 2026-11-09
    entry reported (0.321 to 0.354, 0.308 to 0.341) is within seed noise (paired over 50 seeds, +0.033 with 95 %
    intervals spanning 0); the hold's lidar gains stand.
  - **The fix.** A scan with no candidate beam counts toward the reset when the estimate is off the map
    (`EkfLocalizer(count_off_map=True)`, the default; `RangeMap.free(points, margin)` moved up from `ParticleMap`,
    margin 0 so a robot tracked against a wall is not flagged). Designed on seeds 101 to 120 and 1 to 100, judged
    once on 121 to 150, every pre-registered claim passed: HC-SR04 confidently wrong 0.162 and 0.190 (limit 0.25),
    lidar 0.135 and 0.188 (limits 0.203 and 0.282), recoveries within 2 of 30, collapsed steps within 15 %.
  - **Over all 50 seeds** (`docs/benchmarks/localization_results.md`, regenerated): HC-SR04 confidently wrong 0.354
    to 0.163 and 0.341 to 0.189, lidar 0.123 to 0.124 and 0.201 to 0.165; kidnap recovery 12 to 28 % (was 14 to 30 %;
    one run fewer on two rows). Every tracking run (`EST-1`, both with the default and `count_off_map=False`), every
    MCL figure and `EST-4` are bit-identical.
  - **A new column, "Wrong with σ < 5 cm"**: the confidently-wrong share counting only steps whose position standard
    deviation (larger eigenvalue) is under 5 cm, so a filter waiting at the 0.2 m reset floor is not called
    confident. EKF: HC-SR04 0.095 and 0.126, lidar 0.052 and 0.094; MCL 0.280, 0.282, 0.127 and 0.118.
  - Supersedes the 2026-11-09 figures "14 to 30 %" recovery and "12.3 to 35.4 %" confidently wrong. TESTING.md
    `EST-1`, `EST-3`, `EST-4`.

## 2026-11-09

### Added

- **Several training environments for REINFORCE and the Actor-Critics** (`feature/parallel-envs-n-envs`; PR 1 of the
  roadmap item "Parallel environments for the on-policy agents", stacked on PR 0):
  - **`training.num_envs: N`** (default 1) steps N copies of the training environment in lockstep through an
    in-process batched pool (`rl_core/training/parallel.py`): select every action, then step every environment. Each
    finished episode is handed to the agent as its own batch, so a time-limit or trainer cut still bootstraps and a
    termination does not. Environment 0 keeps the run's seed; the others get independent `SeedSequence` streams
    (`seeding.env_seed`, not `seed + i`). Evaluation and checkpoints fire once per vector step, named by the
    episodes recorded; a step budget lets running episodes finish (at most (N−1) × the longest episode over).
  - **Refused at N > 1**: PPO and the replay-buffer agents (an `enable_parallel_envs` hook that agents opt into),
    `update_every`, rendering and video, and resuming with a different N. A resume at N > 1 is reproducible but drops
    the episodes in flight.
  - **N = 1 is unchanged**: no key and `num_envs: 1` run the old loop, and the new collector forced to one
    environment matches the trainer golden bit for bit (its third arm); both one-environment benchmark reruns
    reproduce `docs/benchmarks/cartpole_results.md` exactly.
  - **CartPole, seeds 1 to 5, 500,000 steps** (pre-registered: no worse than one environment): REINFORCE solves 5 of
    5 with 4 environments against 4 of 5 (steps to solve 320,676 against 318,029); the Actor-Critic 5 of 5 against 5
    of 5, holding 475 on 5 against 4 and solving in 272,136 steps against 374,332. Wall time per seed unchanged.
    TESTING.md `TR-37`. No committed config changes.

- **A golden of seeded training runs** (`feature/parallel-envs`; PR 0 of the roadmap item "Parallel environments for
  the on-policy agents"): `tests/test_trainer_golden.py` runs `Trainer.train()` on 13 tiny seeded cases (REINFORCE,
  Actor-Critic, PPO and DQN in both action kinds where they exist; linear and reward-threshold curricula; observation
  normalisation; a trainer cut and time-limit truncation; evaluation with and without an evaluation environment;
  `save_frequency` and `save_final`; a step budget; `is_success`) and compares every episode, metric record,
  checkpoint name and key, the final agent state and the environment's generators with arrays recorded on develop
  (`tests/data/trainer_training.npz`; floats to a relative 1e-12, the rest exactly; `RECORD_TRAINER_GOLDEN=1`
  re-records, `TRAINER_GOLDEN_EXACT=1` demands bit-for-bit). Each case also runs with `num_envs: 1`; six deliberate
  trainer mutations (a draw order, a lost bootstrap, chunks off by one, an early evaluation, a skipped final save, a
  curriculum fed the wrong reward) are all caught. Tests only.

- **EKF robustness: a held update keeps its covariance** (`feature/ekf-robustness`; the maintainer's decisions of
  2026-11-09):
  - **The cause.** After a reset (a kidnapping, or a wrong wall matched near a corner) the next scan's update could
    not converge and was plainly inconsistent with the prediction (NIS in the thousands), yet it still shrank the
    covariance from 200 mm to about 2 mm round a wrong pose. Seed 131's lost HC-SR04 run (`EST-3`, 30 % dropout)
    began when a sonar matched the wrong wall near a corner while its neighbour was a dropout.
  - **The hold.** An update whose iteration refused a step, or whose NIS at the first linearization exceeds the
    chi-square 0.999 quantile (Wilson-Hilferty, at most 3 % high at one beam), keeps the moved estimate but the prior
    covariance, and counts toward the reset when it has at least `min_candidates` candidates. `UpdateResult.held`;
    `EkfLocalizer(nis_quantile=0.999)`, with `None` turning the hold off (bit-identical to the old filter on all 200
    `EST-1` runs). Chosen over freezing the estimate as well (4 of 14 bump recoveries instead of 11, lidar tracking
    error six times higher), inflating the covariance (a lost `EST-1` run) and refused-step-only (seed 131 stays
    lost).
  - **Results** (full reruns of `EST-1` and the EKF part of `EST-3`; MCL's rows bit-identical): tracking unchanged
    to 4 decimals (mean NEES 3.05 to 3.25); the lost HC-SR04 run is gone (0 of 50, position RMSE 6.8 mm against
    0.33 m, mean NEES 3.38 against 279); kidnap recovery with the lidar 0.16 to 0.30 and 0.10 to 0.24, with the
    HC-SR04 0.16 to 0.20 and 0.18 to 0.14 (9 against 7 of 50, within noise); steps collapsed onto a wrong pose after
    a kidnapping fall by 89 % and 81 % (lidar) and 41 % and 10 % (HC-SR04). `EST-4`'s seed 101 now re-localizes 138
    steps after the kidnapping.
  - **Not better: the HC-SR04's "confidently wrong" share** rose from 0.321 to 0.354 and 0.308 to 0.341, mostly
    filters sitting at the 0.2 m reset floor 1 to 2 m off; shipped with a follow-up item (the maintainer's decision).
  - **Correction.** The 2026-11-08 entry says an HC-SR04 0.3 rad bump fires no reset because its beams turn
    nonlinear; in fact three of its four candidate sonars agree with the wrong heading. It stays a documented limit,
    and the roadmap criterion "a reset fires for the 0.3 rad HC-SR04 bump" was dropped (the maintainer's decision).
  - Superseded figures: the EKF rows of `docs/benchmarks/localization_results.md` (regenerated), the EKF's 10 to 18 %
    recovery and its 23.9 to 32.1 % confidently-wrong share (now 14 to 30 % and 12.3 to 35.4 %), and the 2026-11-08
    "199 of 200 bit-identical" `EST-1` figures (now 183 of 200 runs differ from before the hold). TESTING.md `EST-1`,
    `EST-3`, `EST-4`.

- **Occupancy-grid mapping with known poses** (`feature/occupancy-grid-mapping`; PR 1 of the roadmap item
  "Occupancy-grid mapping and SLAM"):
  - **`rl_navigation.mapping`** builds a log-odds occupancy grid (Thrun, Burgard and Fox, *Probabilistic Robotics*,
    Table 9.1) from range scans at known poses: an exact vectorised traversal of each beam's cells (half-open
    intervals, so a hit on a cell border goes to the far cell), the endpoint cell occupied and the cells before it
    free, clamped to [0.12, 0.97]. Presets: the lidar `p_hit` 0.7 and `p_miss` 0.45 (one hit makes a cell occupied,
    18 misses free it again), the HC-SR04 0.6 and 0.45, each of its five cone rays traced with the mount's reading.
    Readings at maximum range, dropouts and NaN change nothing. 5 cm cells by default.
  - **Layering.** The core imports NumPy and `rl_navigation.sim` only, a peer of `estimation`; only the desk tool
    `mapping.evaluation` imports `estimation`, to drive runs through `run_trial` with a recording stub. The sensors'
    mount geometry (`RangeSensor.mount_origins`, `ray_angles`) and the usable-reading rule (`usable_readings`,
    `RangeLimits`) moved into `sim`; `true_ranges` is bit-identical to before (an oracle test against the old code),
    and MCL's update time is unchanged.
  - **`MAP-1`** (judged once on fresh seeds 101 to 150, 600 steps, own dropout and 30 %; criteria fixed beforehand):
    the lidar passes every gate: accuracy 0.9996, occupied precision 1.0000, recall of seen surfaces 0.937 and 0.953,
    free-space coverage 0.984 and 0.983, at most 2 false-free cells in any run (at the gate: one seed's 2 cells
    straddle a real 2.1 cm gap between two boxes, which the metric counts; another's 1 cell is a 3.4σ reading). The
    HC-SR04, reported without a gate (the maintainer's decision): accuracy 0.991, precision 0.87 to 0.88, recall
    0.84 to 0.85, coverage 0.91 to 0.92. Built from dead reckoning instead of the true poses, the lidar map falls to
    precision 0.34 with 439 false-free cells per run on average: the case for SLAM. About 5 minutes on the Jetson.
  - **`MAP-2`**: an update takes 1.3 ms (lidar, 64 beams, 100 × 100 cells), 0.7 ms (HC-SR04) and 3.4 ms (lidar,
    400 × 400) on the Jetson Orin Nano. TESTING.md `MAP-1` and `MAP-2`.

- **Why Pendulum PPO collapses with observation normalisation: a fit problem, not a defect**
  (`fix/pendulum-ppo-normalisation`; closes the Code Health item; the maintainer's decision, 2026-11-09):
  - **No code defect.** With `normalize_observations: true`, the old log-probs PPO stores match those at collection
    to 1e-14 and the first mini-batch's ratio is exactly 1, on all 90 updates of seed 5 and every update of 11 runs.
    Seed 5 reproduces exactly (final 100-episode average -1005.6; seeds 1 to 5 hold -200 on 2 of 5, mean -360.7).
  - **Cause.** The normaliser's statistics keep moving all run (seed 5's mean of θ̇ from -0.04 at episode 506 to
    +0.91 at the end) while the learning rate and clip range anneal to 0, so late in the run the policy cannot follow
    its shifting inputs; its actions saturate and its failures shift the statistics further. Decisive: seed 5's
    weights from episode 990 score -1028 greedy with their own statistics and -201 with those of episode 781 (10
    evaluation episodes each); every snapshot from episode 451 on scores -183 to -315 under mid-run statistics.
  - **Freezing is not a better config.** Statistics frozen after episode 310 remove seed 5's slide (final -145.8,
    held); a rule fixed beforehand, frozen after episode 100 on seeds 1 to 5, holds on 3 of 5 (mean -191.8) against
    the committed config's 4 of 5 (-186.5, without normalisation). `configs/pendulum_ppo.yaml` is unchanged and no
    freeze key is added. Runs and plots: `~/rl-experiments/ppo-norm/` on the development machine.
  - **Regression test** in `tests/test_observation_normalizer.py`,
    `test_ppo_old_log_probs_are_those_of_collection_under_moving_statistics`, for continuous PPO on Pendulum and
    discrete PPO on CartPole, pins that the stored old log-probs are those of collection under statistics that move
    between rollouts (4 of 4 mutations caught).

- **The README as the project's front page, and the robotics ladder** (`feature/readme-restructure`; the
  maintainer's choices of 2026-11-09):
  - The project presents itself as **RL and Robot Navigation**, not "RL CartPole": README, CLAUDE.md, CONTRIBUTING.md,
    the `setup.ps1` header and `pyproject.toml`'s `description` (the package name `rl-cartpole` is unchanged).
  - The README is a 140-line front page (from 1,253): what works and what does not yet, the ten challenges with their
    solve counts, the navigation library with one frame of the localization video, the ladder, a quick start and a
    documentation map. The rest moved, checked by a scripted inventory (145 command lines before and after), to
    `docs/usage.md`, `docs/navigation.md`, `docs/results.md` and `docs/ai-assistants.md`, and the architecture notes
    into `docs/agents.md`; `docs/environment_setup_summary.md` is marked historical.
  - `scripts/check_doc_links.py` checks every relative link, cross-file anchor and file path in the Markdown (0 broken
    of 126); it runs in a test and, approved by the maintainer, in CI's `Checks` job, which also runs on docs-only
    pull requests. `tests/test_readme_front_page.py` fails when a benchmark challenge or report is missing from the
    front page. TESTING.md `CI-8`.
  - `docs/roadmap.md` opens with the **robotics ladder**: the direction from simulated control to the real robot, as a
    view over the roadmap's items. Phase 7's MuJoCo item widens to inverted pendulums, locomotion and arms; four
    proposed items are added as sketches, each naming its feasibility and the decision it needs: Gymnasium-Robotics
    arms, GPU-parallel simulation on the Jetson, Isaac Lab on cloud or x86 hardware (Isaac Sim does not run on the
    Jetson), and multi-agent navigation.

- **Localization on video** (`feature/localization-recording`; stacked on `feature/localization-comparison`):
  - `python -m rl_navigation.localization_video` writes one seeded trial as an MP4 (and PNG stills with
    `--stills`): the true robot, its scan and path, MCL's particles and estimate from a global start, and the EKF's
    estimate from the noisy start, each with its 2σ position ellipse, and a kidnapping at step 300 by default.
    `--output` is required and never overwritten; the MP4 needs the `video` extra (TESTING.md `EST-4`).
  - Frames come from the same `run_trial` trials the comparison measures (`rl_navigation.estimation.recording`,
    NumPy only): the EKF and MCL trials of one seed share their trajectory and scans, which the recorder checks.
    New NumPy painters in `rl_navigation.sim.render` (`paint_points`, `paint_path`, `paint_pose`), and the beam
    geometry moved from the navigation environment to `RangeSensor.beams` (its frames unchanged).
  - Seed 101, lidar: MCL re-localizes 16 steps (1.6 s) after the kidnapping; the EKF does not. Its reset fires
    every three or four scans and the next scan collapses it back to a few millimetres round a wrong pose, so its
    ellipse flickers between about 0.4 m and a few millimetres while it stays lost. 601 frames in about 55 s on the
    Jetson.

- **EKF against MCL: the comparison report and its command line** (`feature/localization-comparison`; PR 3b, the
  last of the roadmap item "State estimation", which is now complete):
  - **`rl_navigation.estimation.comparison`** runs the EKF, MCL and dead reckoning on the same seeded trials and
    prints a Markdown report (`python -m rl_navigation.estimation.comparison`). `--records` saves each trial as it
    finishes, so a rerun of the same command resumes, and `--from-records` merges jobs run apart. It writes no file
    without `--report`, and the package does not import it.
  - **Report** (`docs/benchmarks/localization_results.md`; TESTING.md `EST-3`): seeds 101 to 150, 600 steps, both
    sensors with their own dropout and with 30 %, MCL with the fitted per-sensor covariance scales. MCL's localized
    and recovered shares equal `EST-2`'s exactly (lidar 96 % and 96 %, 96 % and 98 %; HC-SR04 80 % and 94 %, 88 %
    and 86 %).
  - **Tracking from the noisy start.** The EKF is more accurate: lidar position RMSE 1.8 to 2.1 mm against MCL's
    7.0 mm, HC-SR04 6.8 mm against 11.8 mm with its own dropout. The exception is the HC-SR04 at 30 % dropout,
    where one of 50 EKF runs is lost (row RMSE 0.33 m, mean NEES 279) and MCL holds 13.2 mm. Over all steps, MCL's
    scaled mean NEES is 2.94 to 3.28 and its per-axis ratios 0.95 to 1.15.
  - **Global start and kidnapping.** The EKF recovers from the kidnapping in 10 to 18 % of runs, MCL in 86 to 98 %.
    MCL's median steps to localize from anywhere are 0 for the lidar (the first scan already places it within
    0.2 m and 0.2 rad, since each estimate is taken after that step's update) and 2.5 and 5.5 for the HC-SR04; to
    recover, 21 to 21.5 steps (lidar) and 39 to 55 (HC-SR04). Over all kidnapped runs, the share of post-kidnap
    steps that are confidently wrong (error over 0.5 m and NEES over 9.35; for the EKF, also no reset in the last
    20 scans) is 11.8 to 12.7 % for MCL on the lidar, 28.1 to 28.4 % on the HC-SR04, and 23.9 to 32.1 % for the
    EKF.
  - **Cost.** MCL takes 37 to 51 ms a step against the EKF's 2 to 3 ms (Jetson, one BLAS thread, two jobs at once).
  - The report's prose wraps at 120 characters, as markdownlint requires.

## 2026-11-08

### Added

- **EKF still-robot reset and MCL per-axis covariance scale** (`feature/estimation-filter-fixes`; PR 3a of the
  roadmap item "State estimation"):
  - **EKF reset.** A still robot's covariance used to collapse, so after a bump of more than about 3 cm every beam
    was gated and the EKF never recovered. `EkfLocalizer` now counts a scan with at least `min_candidates=2`
    candidate beams of which at least half are gated. After `reset_after=3` such scans in a row it raises the
    diagonal of `P` to at least `reset_std=(0.2, 0.2, 0.1)` squared; the floor is idempotent, so repeated resets
    do not compound.
  - **Reset signals.** `UpdateResult.reset` flags each reset, and `recent_resets` counts those in the last 20
    scans: 2 or more means lost. `reset_after=None` restores the old behaviour.
  - **Recovery.** In a 5 × 5 m room the lidar recovers to below 1 mm from bumps of 5 to 30 cm, 0.1 to 0.3 rad and
    0.1 + 0.1 m + 0.2 rad. The HC-SR04 recovers from position bumps up to 0.3 m and from 0.1 rad, but not from
    0.3 rad (its beams turn nonlinear and no reset fires) or the combined bump. After a 1.5 rad bump the lidar EKF
    settled on the square room's 90° alias after 2 resets, and once the resets stopped nothing signalled it, the
    same limitation as MCL's.
  - **EST-1 re-run with the reset on** (seeds 1 to 50, 4 rows): 199 of 200 runs are bit-identical to the old filter.
    One false alarm (HC-SR04, 30 % dropout, seed 40) moves that row's `mean_nees` from 3.29 to 3.28 and its
    `nees_below_upper_95` from 0.967 to 0.968.
  - **MCL covariance scale.** `MclSettings.covariance_scale` (default `(1, 1, 1)`, the maintainer's decision)
    scales the reported covariance per axis as D^½ P D^½; poses and EST-2's shares are unchanged. Fitted on
    tuning seeds 1 to 20, the values are lidar (0.130, 0.124, 0.308) and HC-SR04 (0.174, 0.180, 0.525). One
    scalar could not work: EST-2's published mean NEES of 0.57 to 0.94 would need s ≥ 0.94 / 3.72 = 0.253 and
    s ≤ 0.57 / 2.36 = 0.242 to put all four rows in the 50-run band.
  - **MCL judgement on seeds 101 to 150** (criteria fixed before the held-out seeds were read):
    - Per-axis error ratios are 0.92 to 1.06 on every row, inside the [0.5, 2.0] gate.
    - The share of steps whose 50-run ANEES lies in [2.36, 3.72] is 93.1 % and 91.3 % for the lidar and 87.1 %
      for the HC-SR04 with 30 % dropout. For the HC-SR04 with its own dropout it is **82.5 %, failing the 85 %
      gate**. The maintainer lowered that gate from 90 % after the tuning seeds gave 89.6 %, before seeds 101 to
      150 were read.
    - Mean ANEES after scaling is 2.77 to 3.11, against 0.56 to 0.91 unscaled (both from step 50; EST-2's 0.57 to
      0.94 counts every step).
  - TESTING.md `EST-1` and `EST-2`.

### Changed

- **A final checkpoint at the end of training** (`fix/final-checkpoint`; the roadmap item found in navigation PR 5):
  `Trainer.train()` saves the agent and trainer files once more at the last episode, so the newest
  `agent_episode_<n>.pt` holds the final weights instead of weights up to `save_frequency - 1` episodes older. Only
  `train.py` asks for it (`Trainer(..., save_final=True)`, a constructor argument, not a config key), and it saves
  only when this call trained at least one episode that was not just saved, so a resume whose budget is already met
  writes nothing. The benchmark, `record.py`, `robustness.py` and other trainers write nothing new: the first design
  keyed the save on a configured `checkpoint_dir`, which the review found would have made every benchmark run write
  `./checkpoints`, since nearly every committed config names one.
  An interrupted or failed run saves nothing extra (the maintainer's decision: its state can be mid-episode). The
  file format is unchanged and resuming from a final file equals an uninterrupted run (tested). Every `train.py` run
  now leaves one extra checkpoint pair unless it ended on a multiple of `save_frequency`; for the replay-buffer agents
  (DQN, DDPG, TD3, SAC) it holds the buffer and can be hundreds of MB, for example BipedalWalker SAC and TD3 (the
  maintainer accepted the disk cost). TESTING.md `CKPT-5`.

## 2026-11-07

### Added

- **Success-rate metric** (`feature/navigation-solve-metric`; PR 5a of 6 of the roadmap item "Range-sensor
  navigation environment"): an episode's success is `info["is_success"]` at its last step (the Gymnasium and SB3
  Monitor convention), unknown when that step lacks the key, and a rate is reported only when every episode in its
  window is known. `Trainer.evaluate()` reports `success_rate`, the metrics log `success_rate` over its 10
  episodes, `train()` `final_success_rate` over the last 100; the benchmark adds `eval_success_rate` and
  `final_success_rate` (over `--window`) and an "Eval Success" column; `robustness.py` adds a "Success" column and
  labels its threshold column "Reward ≥ <threshold>". `Trainer.evaluate_episodes` returns `EpisodeResult`s. The
  trainer file gains an optional `episode_successes` array: old files resume (earlier outcomes unknown), files with
  it are refused by older code, as with `curriculum`. Challenges that never set `is_success` (every one but
  navigation) give unchanged output, pinned by exact-key tests on a real `Trainer` and the benchmark
  characterization files. Code only: no result was rerun, and the navigation success figures quoted elsewhere still
  come from the scratch harness until PR 5b.

- **Navigation solve threshold, benchmark and robustness results** (`feature/navigation-solve-threshold`; PR 5b
  of 6, the last, of the roadmap item "Range-sensor navigation environment"; stacked on PR 5a): both navigation
  challenges solve at a training reward of 7.0 (`SOLVE_THRESHOLD`, the maintainer's decision; roughly 90 % training
  success, and with no goal bonus it also rewards speed), with success reported beside it and not gating. New
  reports `docs/benchmarks/navigation_results.md` and `navigation_continuous_results.md` (seeds 1 to 5, 100 greedy
  evaluation episodes): TD3 solved and held 5 of 5, eval reward 8.08 ± 0.38, eval success 96.4 %; DQN 5 and 5, 7.76
  ± 0.49, 94.4 %; discrete PPO solved 5, held 1, 6.75 ± 0.85, 88.4 %; continuous PPO solved 5, held 0, 6.66 ±
  0.43, 87.6 %; random 0, about 1 %. Every per-seed eval reward and success matched the earlier scratch harness
  exactly (for DQN, whose per-seed rewards were not kept, success per seed and the reward mean and
  range). Robustness (`ROB-2`, TD3 seeds 42, 1 and 2, 200 episodes a row): success stays at 87 to 98 % on every
  row up to sensor noise 0.4 and dropout 0.4; DQN seed 42 falls to 82 % at dropout 0.4. Two predictions written
  before the runs failed: dropout 0.2 did not cost TD3 10 points (it gained 2.3), and a domain-randomization
  curriculum (noise to [0, 0.2], dropout to [0, 0.3]) did not make TD3 more robust (85.3 % against 91.7 % at
  dropout 0.4, 3 seeds), so no curriculum config is committed. New TESTING rows `TR-36`, `BM-13` and `ROB-2`.

- **Monte Carlo localization** (`feature/state-estimation-mcl`; PR 2 of 3 of the roadmap item "State
  estimation"): `rl_navigation.estimation.MclLocalizer`, augmented MCL (Thrun, Burgard and Fox, Tables 8.2, 8.3
  and 4.4) with a cut-down beam model, has the EKF's `predict` / `update` / `pose` / `covariance` contract and adds
  what the EKF cannot do: start with no known pose and recover after the robot is carried elsewhere. It reports the
  heaviest particle cluster and its weight share (`cluster_weight`), keeps particles that would hit an obstacle in
  place, and imports only NumPy and the simulator. One deviation from Table 8.3: particles injected on the previous
  scan are left out of the recovery average, because injection otherwise sustains itself for about 100 s after a
  disturbance (an averages reset instead cut recovery from 9 to 5 of 10 seeds). The map seam gains a
  `ParticleMap` protocol; the evaluation gains an `ekf()` / `mcl()` localizer factory, a kidnapping at a chosen
  step and `summarize_localization`; EKF results are bit-identical. At the defaults (2000 particles, 16 beams,
  `model_std` 0.15; about 50 ms a step with the simulator on one Jetson core), tuned on seeds 1 to 80 and judged
  once on seeds 101 to 150, 600 steps with a kidnapping at step 300: the lidar localized globally in 96 % of layouts
  and recovered in 96 % (98 % with 30 % dropout), against the maintainer's 70 % gate; the HC-SR04 80 to 88 % and 86
  to 94 % (ungated).
  Tracking from a known start: position RMSE 7 mm (lidar) and 12 to 13 mm (HC-SR04), pooled over every step,
  against 0.80 m for dead reckoning, no lost run, mean NEES 0.57 to 0.94, so the reported covariance is
  conservative (TESTING.md `EST-2`).

- **Machine-wide run queue and a resume file for the dev team** (`chore/run-queue-and-status`): a stdlib job queue,
  `.claude/skills/experiment-runner/scripts/run_queue.py` (`submit`, `run`, `status`, `wait`, `cancel`, `prune`),
  holds every agent to 3 training runs at once on the Jetson (`RL_QUEUE_SLOTS`). Each job runs with one BLAS
  thread in its own process group, with the submitter's environment minus secret-looking names (`KEY`, `TOKEN`,
  `SECRET` and similar; `--env` passes one on purpose), and is cancelled by its stored group, never by a name
  pattern, after `pkill -f` killed its own calling shell twice. A worker holds `worker.lock` for its life and is
  started by `submit`, `wait` or `status`; records are written atomically with mode 0600, a job runs at most once
  even if its worker dies mid-launch, keeps its slot while any process of its group lives, and has its real exit
  code recorded by a wrapper. 72 tests (about 26 s) and 44 of 44 mutations killed, after three review rounds. The
  `dev-loop` skill now keeps `~/rl-experiments/STATUS.md` (items, worktrees, running agents and jobs, pending
  decisions) so "continue" or a new session resumes from one read. CLAUDE.md lists Context7 IDs for
  Stable-Baselines3, PyTorch, ONNX Runtime (Python) and ROS 2.

### Fixed

- **Navigation DQN learns the task** (`fix/navigation-dqn-huber`; the roadmap's "Before PR 5" check on the item
  "Range-sensor navigation environment"): `DQNAgent` gains an opt-in Huber TD loss (`td_loss: huber`, `huber_delta`
  default 1.0; PyTorch's `huber_loss(delta)`, which at 1.0 is SB3's smooth-L1), and `configs/navigation_dqn.yaml`
  uses it with `anneal_steps: 300000`, the learning rate decaying to 0 over the budget. Squared error stays the
  default, bit-identical (golden test). The training-doctor (2026-11-06) found two stacked causes: the squared-error
  loss let rare collisions (-10, about 200 times the step reward's std) dominate the gradient, overestimating Q (12
  to 18 where no return exceeds about 10) so the robot parked and dithered; and a fixed rate on action gaps of
  about 0.07 kept flipping the greedy policy. Double DQN, dueling and an epsilon floor of 0.02 did not help (seed
  1). Seeds 1 to 5, 300,000 steps, final weights, greedy evaluation on 100 held-out layouts: 95, 96, 89, 96 and 96 %
  at the goal (94 %; seeds 3 and 5 were not used to choose the fix), against 0 to 50 % before, PPO's 88 % and TD3's
  96 %; eval reward 7.76 ± 0.49, final 100-episode training averages 7.0 to 8.2, mean Q about 3.3, no non-finite
  update. Each change alone falls short on the same seeds: Huber with a fixed rate 0 to 18 % (mean 4 %), squared
  error with the decaying rate 14 to 86 % (mean 64 %).

## 2026-11-06

### Added

- **Wheel odometry and EKF localization** (`feature/state-estimation`; PR 1 of 3 of the roadmap item "State
  estimation"): `rl_navigation.sim.WheelOdometry` reads each wheel's travel with noise whose variance grows with
  the distance rolled (placeholder wheel base 0.2 m and k = 2.5e-4 m until the robot is calibrated, the
  maintainer's decision), and `rl_navigation.estimation.EkfLocalizer` tracks the pose from it and from HC-SR04 or
  lidar ranges on a known map, through a `RangeMap` protocol (`SensorMap` for the simulator's arena now, a grid
  later). The update is an iterated EKF (up to 3 guarded Gauss–Newton steps) in Joseph form, with a
  finite-difference Jacobian, a linearity check that drops beams across a corner (at every iterate), max-range,
  min-range and NaN readings as dropouts, and a 3σ gate. It imports only NumPy and the simulator, for the robot.
  Over 50 seeded layouts × 600 steps at the filter's defaults: position RMSE 5.7 mm (HC-SR04) and 1.8 mm (64-beam
  lidar), 6.3 and 2.0 mm with 30 % dropout, against 0.93 m for dead reckoning on the same encoders (45 of 50 runs
  whose error exceeded 0.5 m at some step). NEES (3 degrees of freedom): mean 3.2 to 3.3, median 2.4 to 2.5, and 96
  to 97 % of steps under the chi-square 97.5 % point 9.35 (97.5 % expected), with occasional spikes (the worst step
  of any run about 340 on the sonars, seed 2, and 2,700 on the lidar, seed 30, a start-up transient). In an empty
  arena the lidar filter's ANEES is 3.05, inside the 95 % band [2.19, 3.94], a loose check that catches a filter
  told the encoders are four times quieter (6.66) but not one told they are half as noisy (3.84). A navigation
  golden test shows the environment unchanged. Found and fixed before review: iterates skipped the linearity check,
  so a beam across a corner collapsed the heading variance (lidar seed 18, step 96: a corner gave dh/dθ = -17,938
  at an iterate and σθ fell to 5.8e-7 rad against a 1.2e-3 rad error; mean NEES was 238 on the lidar and 23,978 on
  the sonars with dropout). The particle filter is PR 2.

- **Learning configs for the navigation challenges** (`feature/navigation-configs`; PR 4 of 5 of the roadmap item
  "Range-sensor navigation environment"): `configs/navigation_ppo.yaml` and `configs/navigation_dqn.yaml` for
  `Navigation-v0`, `configs/navigation_continuous_ppo.yaml` and `configs/navigation_continuous_td3.yaml` for
  `NavigationContinuous-v0` (PPO at SB3's defaults, with an entropy bonus of 0.01 on the discrete id, 200,000 steps;
  DQN after `lunarlander_dqn.yaml`, 300,000 steps; TD3 after `pendulum_td3.yaml`, NumPy, updating every step, 100,000
  steps; the maintainer's decisions). Seeds 1 to 5, greedy evaluation over 100 held-out layouts, success counted from
  `info["is_success"]` by a scratch harness (the metric itself is PR 5): TD3 96 % success (93 to 100 %), greedy reward
  8.08 ± 0.38, final training average 7.4 to 8.4; continuous PPO 88 % (82 to 91 %), 6.66 ± 0.43, 5.4 to 7.0; discrete
  PPO 88 % (82 to 97 %), 6.75 ± 0.85, 6.0 to 7.4; DQN 11 % (0 to 50 %), -0.58 ± 2.61, -3.8 to -1.3; random 1 %, about
  -11. Predicted beforehand: at least 80 % success and a training average of 5.0 on 4 of 5 seeds for every learning
  agent; held for TD3 and both PPOs (5 of 5), failed for DQN (0 of 5): in greedy evaluation it mostly runs out the
  400-step limit (43 to 88 % of episodes), while 53 to 66 % of its last 100 training episodes end in a collision; the
  cause is not diagnosed. A seed took about 5 minutes for PPO, 15 for TD3 and DQN (three at a time, CPU only). No
  threshold yet (PR 5).

### Fixed

- **One `box2d-py` version everywhere** (`fix/box2d-shared-version`): a Dependabot bump had moved the `box2d` extra in
  `pyproject.toml` to `box2d-py==2.3.8` while `./setup.sh`, the install hint, README, TESTING `ENV-6` and CLAUDE.md
  still built and named 2.3.5, so `./setup.sh` built 2.3.5 and then the extras asked pip for 2.3.8. All now name
  2.3.8, which builds on the Jetson with the same tools (swig 4.5.1, about 95 s) and plays byte-identical episodes
  of LunarLander, LunarLanderContinuous and BipedalWalker to 2.3.5 (same random actions, 3 episodes each), so no
  published result changes. A test holds the extra, `setup.sh` and the hint to one pin.

## 2026-11-05

### Added

- **The BipedalWalker-v3 challenge, measured with SAC and TD3** (`feature/bipedalwalker-sac`; PR 4 of 4, which
  completes the roadmap item "SAC and BipedalWalker-v3"): challenge `bipedalwalker` (package `rl_bipedalwalker`,
  `box2d` extra, 1,600-step episodes, threshold 300) with `configs/bipedalwalker_sac.yaml` and
  `configs/bipedalwalker_td3.yaml` (`backend: torch`, 500,000 steps each, translated from rl-baselines3-zoo at
  `61956d9`; SAC without gSDE, and TD3 at half the zoo's 1,000,000 steps, the maintainer's decisions) and
  `configs/bipedalwalker_default.yaml` (random, 50,000 steps). Seeds 1 to 5 on the Jetson's GPU, three runs at a
  time (`docs/benchmarks/bipedalwalker_results.md`): SAC solved on 4 of 5 (mean 307,090 steps) and held on 1, and
  its greedy evaluation was 324.4 ± 1.9 (321 to 327, all 5 at 300 or more); TD3 solved on 2 and held on none, with
  greedy evaluation 300.1 ± 12.3 (2 of 5 at 300). SAC's greedy evaluation beat TD3's on all 5 paired seeds. A seed
  took 6,029 s for SAC and 7,458 s for TD3 with three sharing the machine. Random: -98.8. Predicted beforehand: SAC
  solving on 0 to 2 of 5 (wrong; it solved on 4) and SAC ahead of TD3 on 4 of 5 seeds (right; 5 of 5). A `train.py`
  SAC run on seed 42 (final 100-episode training average 302.4) recorded at 325.2 over 3 episodes, and its actor
  exports to C and ONNX (1,993 observations, 0 mismatches). Why the training average does not hold is parked in the
  roadmap.

- **The navigation environment and challenges** (`feature/navigation-env`; PR 3 of 5 of the roadmap item
  "Range-sensor navigation environment"): Gymnasium ids `Navigation-v0` (five actions: forward, two arcs, two spins)
  and `NavigationContinuous-v0` (linear speed 0 to 0.5 m/s, no reverse, and turn rate up to 2 rad/s, clipped), 400
  steps, on `rl_navigation.sim`. The raw environment, `rl_navigation/gym_env.py`, imports only Gymnasium and the
  simulator (a subprocess test holds it), so other projects can `gym.make` it without the trainer. The observation is
  the goal's distance, its bearing as cosine and sine, the last executed command and the range readings (HC-SR04 by
  default, or a 64-beam lidar); the reward is 10 times the progress as a fraction of the start distance, -0.01 a step
  and -10 on a collision (the maintainer's decision), so a crash never pays. A collision or the goal ends the episode.
  `reset(options={"layout": ...})` fixes a layout. The `NavigationEnv` wrapper adds action noise, frame stacking and
  domain randomization of wheel slip, sensor noise, dropout and obstacle count (checked against what the arena holds
  before any episode, also in a curriculum or robustness sweep); a NumPy renderer draws `rgb_array` frames for
  `record.py`. Challenges `navigation` and `navigation_continuous` with the random agent only and no solve threshold
  yet; `configs/navigation_default.yaml` and `configs/navigation_continuous_default.yaml` (random, 40,000 steps).
  Tests use hand-computed rewards, bearings and collisions; 19 of 19 mutations caught. Nothing is trained yet: the
  PPO, DQN and TD3 configs are PR 4.

### Changed

- **One Box2D check for every Box2D environment** (`feature/bipedalwalker-sac`): `BaseEnvWrapper._make_env`
  recognises Gymnasium's Box2D environments by their registered entry point (`uses_box2d`) and, without Box2D,
  stops with the install hint naming the environment and `./setup.sh`; LunarLander's own check is gone. The
  seeding, settings and frame-stack tests build one environment per discovered challenge
  (`challenge_env_factories` in `tests/helpers.py`), so a new challenge is covered without editing them. The
  pygame-rendering tests now run after the others (before the ONNX tests), since SDL starts a thread and a later
  fork could deadlock.
- **One domain-randomization check for every wrapper** (`feature/navigation-env`): `checked_randomization` and
  `ParameterBounds` in `rl_core.environments` check a randomization table against per-parameter bounds; Pendulum
  uses it (its tests unchanged) and the navigation wrapper does, while CartPole keeps its own. The
  `action_noise_std` and `action_noise_prob` checks moved into `BaseEnvWrapper`, with the same rules and messages.

## 2026-11-04

### Added

- **`rl_navigation.sim`, a 2D robot simulator library** (`feature/navigation-sim`; PR 2 of 5 of the roadmap item
  "Range-sensor navigation environment"): pure NumPy, importing neither Gymnasium nor `rl_core` (a subprocess test
  holds the boundary), so the later estimation, mapping and planning steps and a real robot can import it. It has
  an `Arena` of walls, circles and boxes; batched ray casting (`cast_rays`, chunked to bound memory); swept-disc
  collision; an occupancy grid; batched unicycle kinematics on the exact arc; range sensors with an HC-SR04 preset
  (2 cm to 4 m and a 15° cone from the datasheet, mounted on the robot's perimeter; noise and dropout are
  placeholders until the real sensor is calibrated) and a lidar preset; a seeded layout generator that checks a path
  exists; and a `World` that steps the robot with wheel slip. No environment, config or challenge yet. Tests use
  hand-computed distances and arcs; 38 of 38 mutations caught. One robot step with a sonar read runs at about 1,700
  a second on the Jetson's CPU.
- **SAC on the PyTorch backend, and LunarLanderContinuous SAC** (`feature/torch-sac`; PR 3 of 4 of the roadmap item
  "SAC and BipedalWalker-v3"): `TorchSACAgent` (`backend: torch`) on a new `TorchOffPolicyAgent` base, which now also
  carries what `TorchDDPGAgent` and `TorchTD3Agent` shared; each group is clipped on the device exactly when its NumPy
  Adam clips it. In float64 on the CPU it replays the NumPy golden cases to a relative 1e-9. The NumPy agents' minimum
  target Q and bootstrapped target moved into `OffPolicyAgent`. `configs/lunarlander_continuous_sac.yaml`: rl-zoo's
  settings (400×300, learning rate 7.3e-4 held constant, as the agents have no schedule) for 500,000 steps on the
  GPU. Through the benchmark, seeds 1 to 5: every seed reaches 200 at 79,587 to 244,525 steps (episodes 272 to 523)
  and holds it, final 100-episode averages 266 to 284, greedy evaluation 282 ± 5, all 5 at 200 or more. TD3's
  published run reaches 200 at 174,000 to 294,000 steps within 400,000, with greedy evaluation 182 ± 69 (2 of 5).
  1.6 to 2.3 hours a seed with two or three runs at once on the Jetson's GPU. The published LunarLanderContinuous
  report has no SAC row yet. Seed 42 was not measured.

- **Soft Actor-Critic** (`feature/sac-agent`; PR 2 of 4 of the roadmap item "SAC and BipedalWalker-v3"): `SACAgent`
  in NumPy, as Stable-Baselines3 implements it: a squashed Gaussian actor, twin critics with Polyak targets and a
  learned entropy temperature against a target entropy of minus the action size, with the numerically stable form of
  the tanh correction. Built on `OffPolicyAgent`; its own random stream for the policy noise
  (`AGENT_POLICY_NOISE_STREAM`); exports to ONNX and C (the greedy action, tanh of the mean). Gradient checks for the
  actor, both critics and the temperature, and a float64 step checked against PyTorch autograd in review (actor
  gradients to 1e-15). `configs/pendulum_sac.yaml`: rl-zoo's Pendulum settings (learning rate 1e-3, SB3's defaults
  otherwise) for 40,000 steps with 64×64 networks, as DDPG and TD3 here (the maintainer's decision; SB3's 256×256
  takes about an hour a seed in NumPy and was not measured). Through the benchmark, seeds 1 to 5: all solve -200 at
  21,800 to 23,200 steps (TD3: 32,600 to 33,400) and hold, final 100-episode averages -135.9 to -163.8, greedy
  evaluation -157 ± 20 (TD3: -157 ± 24); about 9 minutes a seed. The published Pendulum report has no SAC row yet
  (the maintainer's decision).
- **Observation normalisation for DQN, DDPG and TD3** (`feature/replay-obs-normalization`): `normalize_observations`
  is no longer refused for the replay-buffer agents. As Stable-Baselines3 does, the `ReplayBuffer` holds the agent's
  running statistics, stores the raw observations and normalises each sampled batch (observations and next
  observations) with the statistics as they are then, so no stored input goes stale. The `Trainer` adds
  `raw_observations` and `raw_next_observations` to a normalising agent's batch; a normalising buffer refuses a batch
  without them. Acting, export, the PyTorch backend and the checkpoint format are unchanged (the buffer saves raw
  observations, the agent its statistics as before), and without the setting batches and training are bit-for-bit
  as before. `SUPPORTS_OBSERVATION_NORMALIZATION` is gone. Through the benchmark, paired by seed against the
  committed configs: MountainCar DQN reached -110 on 5 of seeds 1 to 5 against 2 and held it on 1 against 2, with
  a better final 100-episode average on 4 of 5 (+13.3, +10.8, +1.5, +7.6, -5.6; mean -112.3 against -117.8); on
  held-out seeds 11 to 15 it reached -110 on 5 against 1, held it on 2 against 0 and was better on 4 of 5 (+7.0,
  +11.8, -0.2, +16.0, +4.5; mean -110.8 against -118.7); greedy evaluation -110 ± 7 against -119 across the ten.
  `configs/mountaincar_dqn.yaml` now sets it (the maintainer's decision); a MountainCar DQN checkpoint saved before
  this no longer loads with the committed config. A run takes about 10% longer. Pendulum TD3 showed no difference
  (5 of 5 hold either way, mean final -151.7 against -152.2) and its config is unchanged.

### Changed

- **Shared off-policy base for DDPG and TD3** (`feature/off-policy-base`; PR 1 of 4 of the roadmap item "SAC and
  BipedalWalker-v3"): `OffPolicyAgent` (`rl_core/agents/off_policy.py`) now holds what DDPG and TD3 share and SAC will
  need: the config keys with per-class defaults, action bounds, the replay buffer, warm-up, update scheduling, the
  critic step (one critic or twin), Polyak averaging and checkpoints. `DDPGAgent` keeps its noise, actor target and
  actor loss; `TD3Agent` still extends it. A pure refactor: DDPG and TD3 train bit-for-bit as before, NumPy and
  PyTorch, shown by new golden cases recorded on the old code (updates during the episode, save, load and continue,
  and DDPG loading a TD3 checkpoint). A DDPG checkpoint missing a network is now refused with "not a DDPG
  checkpoint" (a `ValueError`, before a `KeyError`); the checkpoint format is unchanged.
- **Environment hooks for the navigation challenge** (`feature/navigation-env-hooks`; PR 1 of 5 of the roadmap item
  "Range-sensor navigation environment"): `BaseEnvWrapper._make_env` forwards extra keywords to `gym.make`, for an
  environment with constructor options, and `reset` applies domain randomization before the inner environment's
  reset, so a parameter that reset reads (such as an obstacle count) applies to the episode it was drawn for.
  CartPole and Pendulum, the only environments that randomize, read none at reset, so their training is
  bit-for-bit unchanged (the golden tests, and the review's side-by-side run of both orders with resume).
- **Observation normalisation for the continuous Actor-Critic** (`feature/code-health-trials`):
  `configs/pendulum_actor_critic.yaml` sets `normalize_observations: true` (the maintainer's decision). Pendulum-v1,
  1,000,000 steps, through the benchmark, paired against the config without it: on seeds 1 to 5 the final
  100-episode training average was better on 5 of 5 (+18.7, +9.1, +9.0, +18.4, +23.5; mean -282.7 against -298.5),
  on held-out seeds 11 to 15 on 4 of 5 (+33.5, +28.1, +10.7, -6.1, +20.1; mean -298.9 against -316.2). No seed
  reaches -200 either way; greedy evaluation is unchanged (-170 ± 26 against -172 ± 27 on seeds 1 to 5, -153 against
  -154 on 11 to 15). A seed takes about 320 s against 272 s. An Actor-Critic checkpoint saved before this no longer
  loads with the committed config, which refuses a checkpoint saved with the other `normalize_observations`.
- **Code Health ideas measured and not adopted** (`feature/code-health-trials`; experiments, no code changed). Each
  through the benchmark, paired by seed against the committed config, with the deciding seeds fixed beforehand:
  - *Observation normalisation for Pendulum PPO*: held -200 on 2 of seeds 1 to 5 against 4, mean final -360.7
    against -186.5. Seed 5 learned and then collapsed (final -1005.6): rerun through `train.py`, bit-identical, its
    10-episode average reached -187 by episode 310 and fell to about -1000 by episode 900, with the entropy steady
    near 0.45 and a small value loss. The cause was not established. An earlier scratch variant that froze the
    statistics during each rollout collapsed on 3 of 5 (CHANGELOG.md, 2026-10-29).
  - *Continuous Actor-Critic with `n_steps: 50`* (four equal chunks to a 200-step episode, 20,000 updates over 1,000,000
    steps against 125,000; one environment, so not rl-zoo's 8 parallel ones, and the GAE horizon changes too): mean
    final -342.5 against -298.5, better on 2 of 5, seed 4 worse by 243; greedy evaluation -257 against -172.
  - *Faster to the threshold at a shorter budget* (the maintainer's rule, 2026-11-03: a shorter-budget run must hold
    as often). Pendulum PPO with gSDE and separate 64-64 tanh actor and critic (scratch code), both arms at 130,000
    steps with `anneal_steps` 130,000, on fresh seeds 21 to 25 (130,000 was chosen on seeds 1 to 5 and 11 to 15): it
    reached -200 sooner on all 5 (45,800 to 102,400 steps against 77,000 to 125,800) but held on 3 against 4, mean
    final -197.7 against -182.9, better on 1 of 5, at twice the time a seed. LunarLander DQN with `dueling: true`,
    both arms at 200,000 steps, held 200 on 4 of seeds 11 to 15 against 3 (mean final 206.7 against 187.4; seed 14
    worse by 20), and on 4 of seeds 1 to 5 against 2 (221.3 against 192.1; seed 3 worse by 16). It meets the win rule
    at that budget, but it holds less often than the committed 300,000 steps, which hold on 5 of seeds 1 to 5 and 4 of
    11 to 15 (CHANGELOG.md, 2026-10-27), and a 200,000-step dueling run took 874 to 937 s against 745 to 795 s for
    them; the config is unchanged (the maintainer's decision).
  - *Prioritised replay on LunarLander DQN*: rebuilt on develop (it combines with `double_q` and `dueling`), with
    `priority_alpha` 0.6, `priority_beta_start` 0.4 and `priority_beta_steps` 300,000, 300,000 steps, seeds 1 to 5.
    It held 200 on 1 of 5 against 5 of 5, mean final 166.6 against 235.3, worse on all 5. The baseline reproduced the
    committed config's figures (CHANGELOG.md, 2026-10-27 and 2026-10-29). With MountainCar (2026-10-27) that is two
    challenges where it did not help; it is not merged. The original branch `feature/dqn-prioritized-replay` stays
    as it was, and the rebuilt diff is kept as `~/rl-experiments/per/prioritized-replay-on-develop-6011d4b.patch` on
    the development machine.

## 2026-11-03

### Added

- **Robustness benchmark** (`feature/curriculum-robustness-benchmark`), the last of three PRs for the curriculum
  item, which is now complete: `robustness.py` plays a checkpoint greedily on the config's target and on environments
  perturbed one setting at a time (`--sweep SETTING=V1,V2,...`: a schedulable noise level, or a physics parameter
  pinned to one value), each built from the config, with paired start states. Each row says whether the value is
  inside the range the agent trained on (the config's value, or the curriculum's start to the difficulty the run
  reached, replayed from the trainer file; a config with another curriculum is refused, as a resume refuses it) and
  gives the greedy reward, length and share at the solve threshold;
  `--report` writes it as Markdown, and nothing is written without it (`rl_core.training.robustness`). The wrappers
  list their randomizable parameters in `RANDOMIZABLE_PARAMETERS` (Pendulum's `SUPPORTED_DR_PARAMS` renamed);
  `Trainer.evaluate_episodes` returns each greedy episode that `evaluate` averages; `src/benchmark.py`'s report names
  each agent that trained through a curriculum. Smoke runs only (CartPole and Pendulum PPO, 10 and 30 episodes, seed
  42): no learning claim; whether a curriculum helps is the trial under Changed below.
- **Curriculum scheduling** (`feature/curriculum-scheduler`), the second of three PRs for the curriculum item:
  `environment.curriculum` trains through easier noise and randomization first, with `linear`, `step` or
  `reward_threshold` pacing of one difficulty from 0 to 1 (`rl_core.training.curriculum.CurriculumScheduler`, built
  by `curriculum_from_config` in `train.py` and the benchmark). The `environment` section stays the target, which
  the evaluation environment plays; a curriculum without one is refused. The trainer applies the scheduled settings
  before each training episode, logs `curriculum_difficulty` and `curriculum_stage` in its 10-episode records, says
  when a stage advances and warns if a run ends below the target. The trainer file saves the curriculum; a resume
  replays the saved rewards and refuses a different curriculum. Without one, training is unchanged. CartPole now
  refuses a randomization range reaching 0, as Pendulum does (`checked_range(..., positive=True)`). Smoke runs only
  (CartPole and Pendulum PPO, 30 episodes, seed 42): no committed config has a curriculum, and whether one helps
  has not been measured.
- **Schedulable wrapper settings** (`feature/curriculum-wrapper-settings`), the first of three PRs for the curriculum
  item: each challenge wrapper lists the settings a curriculum may change between episodes in
  `SCHEDULABLE_SETTINGS` (observation noise everywhere; action noise and domain randomization on CartPole and
  Pendulum), reports them with `scheduled_settings()` and changes them with `apply_settings()`, all or nothing,
  checked as the constructor checks them, only for settings on at construction, and keeping a randomization's
  parameters and order of draws. `SchedulableEnv` is the Protocol. Seeded episodes are unchanged:
  `tests/test_env_golden.py` compares CartPole and Pendulum with noise and randomization against arrays recorded
  before the change. Pendulum now refuses a randomization parameter given twice under two names (`g` and
  `gravity`), which it used to collapse; CartPole's factories check randomization with the wrapper's rule, so an
  unsupported parameter now fails before any environment or worker process starts.

### Changed

- **A curriculum did not help CartPole PPO at the trial's target** (`feature/curriculum-trial`, an experiment; no
  code or config changed): `configs/cartpole_ppo.yaml` with `obs_noise_std: 0.05` and `domain_randomization: {length:
  [0.25, 1.0]}`, 200,000 steps through `train.py`, seeds 1 to 5, flat (the target from the first episode) against
  `curriculum: {mode: reward_threshold, stages: 4, threshold: 400, window: 20, min_episodes_per_stage: 20, start:
  {obs_noise_std: 0.0, domain_randomization: {length: [0.5, 0.5]}}}`. The deciding measure, fixed beforehand, was the
  greedy evaluation at the target (`robustness.py`'s nominal row, 20 episodes on the same start states for both arms,
  on the last checkpoint, at most 10 episodes before the end):

  | Seed | Flat | Curriculum | Curriculum at difficulty 1 from episode |
  | :---: | :---: | :---: | :---: |
  | 1 | 119 | 99 | 456 |
  | 2 | 111 | 110 | 383 |
  | 3 | 105 | 110 | 381 |
  | 4 | 115 | 102 | 431 |
  | 5 | 109 | 112 | 429 |

  Better on 2 of 5 (by 5 and 3), worse on seeds 1 and 4 (by 20 and 13): it fails the win rule. No run of either arm
  reached 475; final 100-episode training averages were 102 to 117 flat and 93 to 108 with the curriculum, which
  reached the easy settings' 400 in stage 0 and fell to about 100 once the target's noise came in. Beyond the
  training range (`--sweep length=0.125,...,2.0 --sweep obs_noise_std=0,...,0.15`) neither arm was more robust: at
  length 2 the flat arm scored 115 to 140 and the curriculum 36 to 123; at noise 0.15, 80 to 105 against 61 to 98.
  The one run that held 475 was the calibration run at seed 42 (curriculum, final average 497.7; flat 141.4), not
  part of the trial. Untried: a target the committed config can learn flat (length randomization alone, or less
  noise), where a curriculum could change how fast or how robustly it learns rather than whether.

- **Config readers moved to `rl_core.utils.config_values`** (`feature/neutral-config-readers`), from
  `rl_core.agents.config_values`: the strict integer and boolean readers are shared by the agents, the trainer, the
  environment wrappers and the challenges' environment factories, so the environments no longer import the agents
  package. `tests/test_source_conventions.py` now fails if any `environments` package, `rl_core`'s or a challenge's,
  imports `rl_core.agents`, relative imports included. No behaviour change. Code that imported the old module path
  must use the new one.

## 2026-11-02

### Added

- **ONNX export** (`feature/onnx-export`), completing the embedded-export item: `python export.py --format onnx
  --config <config> --checkpoint <checkpoint> --output policy.onnx` writes the same greedy policy as the C header as
  one ONNX model (opset 17, IR version 8): input `observation` (float32, any batch size), outputs `outputs` and
  `action`, normalisation and action head included, float32 weights, and the agent, checkpoint and observation
  settings in its metadata. It is checked in onnxruntime as the header is checked in C (same rollout, tolerance and
  near-tie rule) and written only if it passes. `onnx` and `onnxruntime` are the new optional `onnx` extra (the
  maintainer's decision), imported only by `rl_core/export/onnx_model.py`; the setup scripts and CI's test job
  install it. On this machine's checkpoints for seven runs over five challenges every model passed with no action
  mismatch; the worst error was 0.58 of the tolerance, on LunarLanderContinuous TD3 (400x300; TESTING.md `EXP-2`).
  Along the way: `export.py --format c` stays the default and writes the same header (now always with LF line
  endings); the tests every format must pass are shared (`tests/export_cases.py`), and the onnxruntime tests run
  last, because importing onnxruntime starts a thread and a fork after it may deadlock.

- **C header export** (`feature/c-header-export`), the first of two PRs for the embedded-export item (ONNX
  follows; it needs `onnx` as a new dependency): `python export.py --config <config> --checkpoint <checkpoint>
  --output policy_weights.h` writes a trained agent's greedy policy as one C99 header with the weights as exact
  float32 hexadecimal literals and `policy_outputs` and `policy_act` (an `int` action for discrete agents, an array
  for continuous ones), observation normalisation included. Every agent with a network exports (REINFORCE,
  Actor-Critic and PPO in both kinds, DQN plain and dueling, DDPG, TD3), PyTorch-backend checkpoints through the
  NumPy agent; the random agents are refused. Each agent describes its policy through a new
  `BaseAgent.inference_policy()` (`rl_core/agents/inference.py`), and dueling DQN is folded into one network whose
  outputs are its Q-values. `export.py` then plays 3 greedy episodes (`--verify-episodes`), compiles the header
  with gcc or clang and compares it with NumPy (1e-5 absolute plus 1e-5 relative; discrete near-ties counted
  apart). On this machine's existing checkpoints of 11 runs over six of the seven challenges, every export passed
  with no action mismatch (the MountainCar DQN checkpoint did not load under `configs/mountaincar_dqn.yaml`: it
  holds a 100,000-transition replay buffer where the config sets 10,000); the largest absolute error was 7.3e-5,
  on LunarLander DQN Q-values near 100 (TESTING.md `EXP-1`). The header takes the observation after
  `mask_velocity`, `frame_stack` and `frame_deltas` and names them; it does not apply them.

- **Observation normalisation and frame differences** (`feature/observation-normalization`,
  `feature/frame-deltas`): masked `frame_stack: 4` CartPole PPO solved 0 of 5 (CHANGELOG.md, 2026-11-01). The
  `training-doctor` diagnosis: the policy kept its entropy (about 0.41) and nothing went non-finite, but the value loss
  stayed at 1 to 10 all run (unmasked: about 0.0002). Velocity survives only as the change between consecutive frames,
  about 50 times smaller than the velocity itself (cart: std 0.014 against 0.73), so the network has to learn large,
  nearly cancelling weights along a direction its inputs barely vary in. The same 16 numbers re-expressed as the
  newest frame and the differences divided by CartPole's 0.02 s time step solved 5 of 5, so the stack held the
  information and lacked only its scale. Two opt-in settings came of it, neither changing a committed config:
  - `agent.config.normalize_observations: true` (REINFORCE, Actor-Critic, PPO): a running mean and variance of the
    inputs as SB3's `VecNormalize` keeps them (`rl_core/agents/observation_normalizer.py`), updated on every training
    observation and frozen in evaluation; `Trainer` passes each observation through `BaseAgent.normalize_observation`
    once, and the checkpoint holds the statistics as `obs_norm_*`. DQN, DDPG and TD3 refuse it: their replay buffers
    would hold inputs normalised by statistics that later change. Unlike the Pendulum attempt (CHANGELOG.md,
    2026-10-29), the statistics update on every observation.
  - `environment.frame_deltas: true` (needs `frame_stack` of 2 or more): the stack becomes its newest frame followed by
    the differences between consecutive frames, newest first, an invertible map of the same entries; every challenge
    takes it and checkpoints record it.

  Through the benchmark, masked `frame_stack: 4` CartPole PPO, the committed config otherwise, seeds 1 to 5, 200,000
  steps:

  | Inputs | Solved 475 | Held | Steps to 475 | Final 100-episode averages | Greedy evaluation |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | Raw stack | 0 of 5 | 0 of 5 | | 192 to 328 | 337 ± 74 |
  | `normalize_observations` | 4 of 5 | 2 of 5 | 161,582 to 192,019 | 461.5 to 486.3 | 500 ± 0 |
  | `normalize_observations` and `frame_deltas` | 5 of 5 | 5 of 5 | 62,890 to 78,077 | 500 on every seed | 500 ± 0 |

  The last row solves as fast as unmasked CartPole PPO (72k to 90k steps). Scratch runs on the same seeds: frame
  differences without normalisation solved 2 of 5 and held 1. The benchmark reproduced the scratch prototypes' outcomes;
  the normalisation run matched it episode for episode, the frame-differences run solved at different episodes
  (450, 431, 403, 403, 407 against 456, 431, 328, 620, 333) because the wrapper differences float32 frames in float32
  where the prototype used float64 (inputs differ by at most 9.3e-10). Not run: other challenges, other agents, other
  stack sizes, seeds beyond 1 to 5.

- **Observation settings in checkpoints** (`feature/checkpoint-observation-settings`): an agent built by a
  challenge's `make_agent` (as `train.py`, `record.py` and the benchmark build it) records its environment's
  `frame_stack` and `mask_velocity`, and `BaseAgent.save` stores them as `observation_setting_<name>`. `load` refuses,
  before changing anything, a checkpoint saved with other values, naming each differing key with its saved and current
  value, so `record.py` playback and `train.py --resume` under a different `environment` section stop at once. Before,
  a stack mismatch failed later with a shape error and a mask mismatch ran on the wrong inputs without one. A key a
  checkpoint lacks counts as its default, so a setting added later does not refuse today's checkpoints; `True` and `1`
  differ. Checkpoints saved before this change hold no record and load unchecked (a stack mismatch among them still
  fails late), and an agent built directly records nothing, so golden checkpoints are unchanged. The maintainer chose
  recording the settings over checking only the saved input size, which would not catch masking.

## 2026-11-01

### Added

- **Velocity masking and frame stacking** (`feature/frame-stacking`, `feature/frame-stacking-learning-run`): every
  challenge's `environment` section takes `mask_velocity` (default false), which zeroes the velocity entries as
  rl-zoo's `MaskVelocityWrapper` does, and `frame_stack: K` (default 1, off), which joins the last K observations,
  oldest first, into one vector. `BaseEnvWrapper` applies them after observation noise
  (`rl_core/environments/observation.py`); with neither key set nothing changes. Measured through the benchmark on
  CartPole PPO, `configs/cartpole_ppo.yaml` otherwise unchanged, 200,000 steps, seeds 1 to 5: unmasked solves and
  holds 475 on 5 of 5 (greedy evaluation 500 ± 0); masked solves 0 of 5 (final 100-episode averages 45 to 50,
  evaluation 42 ± 3); masked with `frame_stack: 4` solves 0 of 5 (final averages 192 to 328, evaluation 337 ± 74).
  With `total_steps` and `anneal_steps` doubled to 400,000 it still solves 0 of 5 and ends lower (final averages 161
  to 214, evaluation 233 ± 52), so the budget is not what holds it back; the cause is an open roadmap item. No
  masked config was committed.

- **Gradient clipping with plain SGD** (`feature/sgd-gradient-clip`): `max_grad_norm` used to be refused unless
  `optimizer: adam`. REINFORCE, the discrete Actor-Critic, discrete PPO and DQN now accept it with SGD too, and clip
  the global norm of the mean gradient, as Adam does (`ClippedSGD` in `rl_core/agents/optim.py`, which shares Adam's
  clip). With SGD, DQN's `max_grad_norm` replaces its element-wise `grad_clip_value`, and setting both is refused. A
  NaN `max_grad_norm`, which silently turned the clip off, is now refused with Adam too. SGD without a clip keeps its
  own step, so every committed config and golden test is unchanged; no committed config uses the new path, and no
  learning run was made.

### Changed

- **LunarLanderContinuous TD3 updates after every step** (`feature/td3-update-every`): the maintainer chose
  `training.update_every: 1` at 400,000 steps for `configs/lunarlander_continuous_td3.yaml`, as rl-zoo does
  (`train_freq: 1`), over the earlier per-episode updates (evidence: CHANGELOG.md, 2026-10-30). Republished through the
  benchmark on seeds 1 to 5 in the first `docs/benchmarks/lunarlander_continuous_results.md`: every seed reaches 200 (at
  174,000 to 294,000 steps) and holds it, with final 100-episode training averages 202 to 248 (mean 225.9), against 3
  of 5 holding before. This reproduces the 2026-10-30 run on the same seeds. Greedy evaluation is 182 ± 69 (58 to 251;
  2 of 5 at 200 or more). A seed took 53 to 54 minutes alone on the Jetson's GPU, against 28 before. Held-out seeds 11
  to 15 were not rerun (5 of 5 held, 2026-10-30), nor was seed 42 through `train.py`. The random agent ends at -214 to
  -228.

## 2026-10-31

### Fixed

- **`async_envs` read strictly** (`fix/strict-async-envs`): the CartPole and Pendulum environment factories turned a
  quoted `"false"` or `"no"` into parallel processes. `async_envs` now takes true, false or null (null still means
  synchronous, unset still asynchronous), and anything else, `0` and `1` included, is refused by both the vectorized
  and the single-environment factory with a message naming the key. The check is `async_envs_setting` in
  `rl_core.agents.config_values`, which shares its boolean test with `flag`. No committed config sets the key.

## 2026-10-30

### Added

- **True A2C for the continuous Actor-Critic, opt-in** (`feature/a2c-n-step-actor-critic`): with
  `training.update_every` equal to its `n_steps`, the trainer hands it each chunk as it is collected, so the next
  chunk is sampled by the updated policy, instead of replaying the finished episode in chunks sampled by a policy up
  to 24 updates old. The chunk maths is unchanged (GAE from the current critic, bootstrapped from the chunk's next
  state unless the episode truly terminated, so a time-limit cut bootstraps). Any other `update_every`, or `n_steps` 0,
  is refused at construction; for that the trainer now passes its value to
  `enable_mid_episode_updates(update_every)`. Without `update_every` nothing on the update path changed, the baseline
  below reproduces 2026-10-19's figures, and checkpoints keep their keys. Pendulum-v1,
  `configs/pendulum_actor_critic.yaml` plus `update_every: 8`, 1,000,000 steps, seeds 1 to 5, through the benchmark,
  paired against the committed config: no seed solves either way; mean final 100-episode average -295.0 against
  -298.5, better on 3 of 5 (+2.5, +9.0, +13.7; -7.0 and -0.8 on seeds 2 and 3); greedy evaluation -168 ± 23 against
  -172 ± 27; about 20% slower (319 s against 269 s a seed). It fails the project's win rule and the config is
  unchanged: the stale chunks were not what keeps the training average near -300.

- **Off-policy updates during the episode, opt-in** (`feature/off-policy-mid-episode-updates`): `training.update_every:
  N` makes the trainer call `agent.update` every N steps inside an episode, then once more on what is left at its end,
  as SB3 does with `train_freq`; without it the trainer updates once per episode as before, and every committed config
  and golden test is unchanged. Only DQN, DDPG and TD3 accept it (`enable_mid_episode_updates`; the random agents
  accept it and learn nothing); REINFORCE, Actor-Critic and PPO train on whole episodes and refuse it. With it, DQN,
  DDPG and TD3 count gradient steps over every transition trained on (`MidEpisodeUpdates`, carried in the checkpoint
  as `update_transitions`), so a fractional `updates_per_step` gives the same steps however the episode is cut up;
  DQN also needs `updates_per_step` and `exploration_steps`. The PyTorch backend now copies only the actor back to
  NumPy after an update and the rest before a checkpoint, which took a per-step-updating seed from 5.7 times slower
  to about 1.5 times. Metrics are averaged per update call, so with `update_every: 1` the logged losses are means of
  per-step values. LunarLanderContinuous TD3 with `update_every: 1`, 400,000 steps, against the committed config
  (which updates per episode), on the screen seeds and on held-out seeds:

  | Seeds | Held (committed → every step) | Mean final 100-episode average | Better / clearly worse (by over 20) | Reached 200 by 300,000 |
  | :--- | :--- | :--- | :--- | :--- |
  | 1 to 5 | 3 → 5 | 202.8 → 225.9 | 3 / 1 (seed 1, by 27) | 3 → 5 |
  | 11 to 15 | 4 → 5 | 214.1 → 242.8 | 4 / 1 (seed 14, by 26) | 4 → 4 |

  Both sets meet the project's win rule through more seeds holding, and neither through the paired averages, each
  having one seed clearly worse. Greedy evaluation on seeds 11 to 15 went from 87 to 269 (mean 191) to 191 to 259
  (mean 235); on seeds 1 to 5 it was 58 to 251 with the option. A seed took 54 to 78 minutes with two sharing the
  Jetson's GPU, against 36 to 47 updating per episode. The seed 1 to 5 baseline is the 2026-10-28 screen; the held-out
  baseline was run fresh and reproduces 2026-10-28's figures. The runs used a driver that patched the same chunking
  into the trainer, which matches `update_every: 1` exactly over 12,000 steps of seed 1. Adopting it in
  `configs/lunarlander_continuous_td3.yaml`, and whether 300,000 steps would then do, is the maintainer's decision
  (docs/roadmap.md, Training).

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

- **Dueling DQN, opt-in** (`feature/dueling-dqn`, the fourth of the five Code Health pull requests): `dueling: true`
  adds a value stream beside the Q-network, which then gives the advantages, and `Q = V + A - mean_a A` (Wang et al.
  2016, mean aggregation). As in the paper the streams share only their input: the value stream is a second
  `hidden_dim`-unit ReLU layer and a one-output head on the observation, so the network keeps plain DQN's depth and
  has twice its hidden units. It combines with `double_q`. Off by default, and a plain run is bit-for-bit unchanged.
  The **checkpoint format changes** for a dueling agent (`W1_value` … `b2_value` and their targets); a checkpoint
  loads only into an agent with the same setting, and a mismatch is rejected before anything is restored (a dueling
  checkpoint used to be able to load into a plain agent with its advantages read as Q). The target sync, Polyak
  average, SGD step and checkpoint now all read one parameter list. Through the benchmark on seeds 1 to 5, against
  the committed configs (item 3's runs, reused after Acrobot seed 1 reproduced exactly):

  | Config | Held (committed → dueling) | Mean final 100-episode average | Better on | Seeds clearly worse |
  | :--- | :--- | :--- | :--- | :--- |
  | CartPole | 5 → 4 | 497.5 → 474.6 | 1 of 5 | 1 (by 118) |
  | Acrobot | 5 → 5 | -92.6 → -90.5 | 4 of 5 | 0 |
  | MountainCar | 2 → 0 | -117.8 → -125.0 | 2 of 5 | 2 (by 18 and 19) |
  | LunarLander | 5 → 5 | 235.3 → 234.8 | 4 of 5 | 1 (by 34) |

  Acrobot met the rule on seeds 1 to 5 but not on held-out seeds 11 to 15, both arms run fresh (5 of 5 hold in each;
  mean -92.1 → -91.5, better on 3 of 5: +1.3, -5.2, +0.4, -4.5, +11.3), so no config changes. Not part of the rule: on
  LunarLander dueling reached 200 sooner on all five seeds (100,000 to 148,000 steps against 152,000 to 228,000). A
  dueling run takes about 1.6 times as long (1.9 on CartPole). With twice the hidden units, any gain could come from
  capacity; the paper's comparison, dueling with double against double alone, was not run.

### Changed

- **Pendulum PPO and CartPole DQN experiments, nothing adopted** (`feature/ppo-dqn-experiments`, the last of the five
  Code Health pull requests; no code or config change). Each idea ran through the benchmark on seeds 1 to 5 against
  the committed config, with its other settings as committed. The PPO ideas ran as scratch subclasses of
  `ContinuousPPOAgent`. Each first passed a finite-difference gradient check and a direction check, and a control
  with every option off reproduced the agent to 1e-12. The code and results are in `~/rl-experiments/item5/` on the
  development machine, outside the repository.

  | Pendulum PPO | Held | Mean final 100-episode average | Better on | Seeds clearly worse (by > 25) |
  | :--- | :--- | :--- | :--- | :--- |
  | Committed: one shared 128-unit ReLU layer | 4 of 5 | -186.5 | | |
  | Separate actor and critic, two 64-unit tanh layers each (SB3's default) | 4 of 5 | -183.3 | 3 of 5 | 0 |
  | gSDE on the shared layer | 4 of 5 | -186.8 | 3 of 5 | 1 (by 31) |
  | gSDE with separate tanh networks (rl-zoo's recipe) | 5 of 5 | -180.1 | 3 of 5 | 0 |
  | Observation normalisation | 0 of 5 | -544.8 | 0 of 5 | 5 |

  - gSDE follows SB3: a full standard-deviation matrix, the latent detached, and noise redrawn every 4 actions and at
    each rollout. Its initial log standard deviation was calibrated per seed so that the initial action spread on
    reset states matches the committed 0.61.
  - rl-zoo's recipe passed on seeds 1 to 5 (one more seed held) and failed on held-out seeds 11 to 15. There it held
    on 3 against 4, the mean final average fell from -186.7 to -191.4, and seed 12 ended 64 lower. It reached -200
    sooner on 9 of the 10 seeds (46,400 to 127,200 steps against 72,400 to 193,000).
  - Observation normalisation collapsed on three seeds, which never solved; its statistics were frozen while a
    rollout was collected and updated after its epochs. The cause was not diagnosed. rl-zoo does not normalise
    Pendulum.

  | CartPole DQN `max_updates_per_call` | Held | Mean final 100-episode average | Seeds clearly worse (by > 25) | Steps to solve |
  | :--- | :--- | :--- | :--- | :--- |
  | 64, committed (0.13 gradient steps per transition in a 500-step episode) | 5 of 5 | 497.5 | | 144,000–183,000 |
  | 128 | 2 of 5 | 420.3 | 3 | sooner on all 5 |
  | 256 (rl-zoo's 0.5 per transition) | 3 of 5 | 462.6 | 2 | sooner on 3 (87,000–103,000), later on 1 |

  The committed runs end at 500, 500, 494, 497 and 496, so no CartPole change could pass the win rule, and these ran
  as information. A higher cap also syncs the target network more often per episode; that effect was not separated
  out.

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
- **Continuous Actor-Critic and PPO read `bounded_mean` and `normalize_advantages` strictly**
  (`fix/actor-critic-strict-flags`): both were read with `bool(config.get(...))`, so a quoted `"false"` in a YAML
  file switched them on. They now go through `rl_core.agents.config_values.flag`, as `double_q` and `dueling` do,
  and anything but true or false is rejected. A config with real booleans, as every committed one has, runs
  unchanged.

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
