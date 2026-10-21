# Changelog

What changed in the framework, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
There are no version tags: work merges to `develop`. Each pull request adds its own entry under the day it was
handed off for merge, so it merges complete; sections before 2026-10-21 use the day the pull requests merged (a
month, before October 2026). Each entry is the condensed form of its roadmap item; the full discussion is in the
pull requests and the git history. Open work is in [docs/roadmap.md](docs/roadmap.md).

Results quote the committed config and seeds 1 to 5 unless they say otherwise. They use the 100-episode solve rule
from 2026-10-16; earlier results use the 10-episode rule.
"Solves" is the rolling training average reaching the threshold (475 on CartPole-v1, -200 on Pendulum-v1) and
"holds" is the final 100-episode average still at or above it.

## 2026-10-21

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
