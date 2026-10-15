# AI Reinforcement Learning Framework — Roadmap

Status labels: `[ ]` Not Started · `[~]` In Progress · `[x]` Complete

---

## Completed Releases & Milestones

### Phase 0 — Framework Foundation & CI/CD Pipeline *(Completed)*

- [x] **Repository Infrastructure & Tooling**: Configured `pyproject.toml` with Ruff, Mypy, and Pytest.
- [x] **Automated CI Workflows**: Linting (`lint.yml`), testing with coverage (`test.yml`), type-checking
  (`type-check.yml`), CodeQL security scanning (`codeql.yml`), and security auditing (`security.yml`).
- [x] **Nightly Maintenance Pipeline**: Scheduled daily automated scans and doc synchronization
  (`nightly-maintenance.yml`, `scripts/sync_workflow_docs.py`).

### Phase 1 — Gymnasium Environment Wrappers & Randomization *(Completed)*

- [x] **CartPole Gymnasium Wrapper**: Standardized `CartPoleEnv` wrapper adhering to Gymnasium v0.29+
  APIs (`terminated`, `truncated`, episode tracking, stats collection).
- [x] **Noise Injections**: Gaussian observation noise (`obs_noise_std`) and stochastic action flip
  probabilities (`action_noise_prob`).
- [x] **Domain Randomization**: Configurable physical parameters (gravity, cart mass, pole mass, pole length)
  with dynamic per-episode resampling and automatic derived parameter calculation.
- [x] **Vectorized Environments**: Synchronous and asynchronous multi-environment rollout factories
  (`make_vec_env`) with independent seeds.

### Phase 2 — Policy Gradient & Baseline Agents *(Completed)*

- [x] **Abstract Agent Architecture**: `BaseAgent` abstraction with standardized `select_action`, `update`,
  `save`, and `load` interfaces.
- [x] **Baseline Random Policy**: `RandomAgent` for environment sanity checks and performance baselining.
- [x] **REINFORCE Agent**: Monte Carlo policy gradient algorithm with discounted return returns and baseline
  subtraction.
- [x] **Actor-Critic (A2C) Agent**: Actor policy with value critic estimating state-value advantages.
- [x] **Proximal Policy Optimization (PPO)**: Clipped surrogate objective policy optimization with value
  clipping, entropy bonus, and pure NumPy feedforward neural network implementations.

### Phase 3 — Training Orchestration & Metrics Pipeline *(Completed)*

- [x] **Unified Trainer**: Trajectory collection, rollout batching, periodic evaluation, and model
  checkpoint persistence (`src/rl_cartpole/training/trainer.py`).
- [x] **Configuration Management**: Modular YAML/JSON hyperparameter parsing and merging
  (`src/rl_cartpole/utils/config.py`).
- [x] **Structured Logging**: Dual-stream console and machine-readable JSONL training metrics logging
  (`src/rl_cartpole/utils/logger.py`).
- [x] **Visualization Helpers**: Episode reward and length trajectory plotting (`visualization.py`).

---

### Phase 4 — Value-Based Learning: Deep Q-Network (DQN) *(Completed)*

- [x] **`ReplayBuffer` Component**: Fixed-capacity pure NumPy ring buffer storing transition tuples with
  O(1) insertion, vectorized batch addition, and uniform random sampling (`src/rl_cartpole/agents/replay_buffer.py`).
- [x] **`DQNAgent` Implementation**: Pure NumPy DQN agent with separate online and target Q-networks, epsilon-greedy
  decay, Bellman TD loss backpropagation, and hard/soft target synchronization (`src/rl_cartpole/agents/dqn_agent.py`).
- [x] **DQN Hyperparameter Tuning & Integration**: Calibrated `configs/cartpole_dqn.yaml`, CLI integration in
  `train.py` / `record.py`, multi-step updates, and verification solving CartPole-v1.

### Training & Workflow Tooling *(Completed)*

- [x] **Training Video Interval & Trained-Agent Recording**: `--video-every N` training clip cadence, `record.py`
  playback generator with clip concatenation, and test coverage.
- [x] **Per-Agent Output Directories**
  - **Description**: Training outputs from different agents currently share `checkpoints/`, `videos/` and `plots/`
    with identical file names, so training one agent overwrites another's checkpoints and videos. Default each
    output location to a per-agent subfolder named after the agent type.
  - **Acceptance criteria**:
    - `train.py` saves checkpoints to `<training.checkpoint_dir>/<agent_type>/` (e.g. `checkpoints/ppo/`).
    - `train.py --video-dir` and `--plot-dir` default to `videos/<agent_type>/` and `plots/<agent_type>/`;
      `record.py --video-dir` defaults to `videos/<agent_type>/`.
    - The agent type is the effective one, so `--agent-type` overrides choose the folder too.
    - An explicit `--video-dir` / `--plot-dir` is used exactly as given (no agent subfolder appended).
    - Logs are unchanged (already timestamped per run in `logs/`).
    - One shared helper builds the per-agent path for both scripts; no duplicated logic.
    - README and TESTING.md updated, including the wrong `agent_episode_1500.pt` example in the README.
  - **Known edge cases**: `training.checkpoint_dir` missing from config (falls back to `./checkpoints/<agent_type>`);
    re-training the same agent overwrites that agent's previous outputs (intended; copy the folder to keep a run).
  - **Out of scope**: Per-run (timestamped) output folders; `record.py` picking the latest checkpoint
    automatically; migrating existing files in the old flat folders.

### Phase 4 — Value-Based Learning: Deep Q-Network (DQN)

Targeting value-learning baseline for direct comparison against policy-gradient methods (A2C/PPO).

- [x] **`ReplayBuffer` Component**
  - **Description**: Fixed-capacity ring buffer storing transition tuples `(state, action, reward, next_state, done)`.
  - **Acceptance criteria**:
    - Pure NumPy ring buffer implementation supporting O(1) insertion and O(batch_size) uniform random sampling.
    - Support vectorized state/action ingestion.
    - Comprehensive unit tests in `tests/test_replay_buffer.py`.
  - **Known edge cases**: Buffer sampling before reaching minimum sample size; terminal state transitions.

- [x] **`DqnAgent` Implementation**
  - **Description**: Pure NumPy DQN agent with separate online and target Q-networks, epsilon-greedy exploration,
    and Bellman TD error optimization.
  - **Acceptance criteria**:
    - Online Q-network and target Q-network parameterized with pure NumPy matrices and backward pass autodiff.
    - Decaying epsilon-greedy policy (`epsilon_start`, `epsilon_end`, `epsilon_decay`).
    - Target network weight synchronization via periodic hard copy or Polyak soft update.
    - Unit tests in `tests/test_dqn_agent.py` achieving >85% coverage.
  - **Known edge cases**: Bellman target clipping, NaN/Inf gradient explosion, terminal state value zeroing.

- [x] **DQN Hyperparameter Configuration & Integration**
  - **Description**: Add `configs/cartpole_dqn.yaml`, register `dqn` in `train.py`, and run benchmark comparisons.
  - **Acceptance criteria**:
    - `configs/cartpole_dqn.yaml` with calibrated learning rate, replay capacity, batch size, and discount factor.
    - `train.py --config configs/cartpole_dqn.yaml` trains successfully to solve CartPole-v1 (avg reward >= 475).

---

## Upcoming Milestones

**Agreed order (2026-10-13, extended 2026-10-14):** Seed the Continuous PPO Mini-Batch Shuffle → Seeded Environments
Replay One Start State → the test-gap items (Discounted-Return Tests That Can Fail, Softmax Stability Tests, Single
`_compute_returns` Implementation) → Split `update` into Loss-and-Gradients and Apply → CartPole Agents Learn Within
Their Budgets → Actor-Critic Trains with Adam → Standard PPO for Discrete Actions → Break the Benchmark Import Cycle →
Pendulum PPO Learns and Solves Pendulum-v1 → DDPG.

### Workflow Tooling

- [x] **Dev Team Skills for Claude Code**
  - **Delivered in**: merged in #64.
  - **Description**: Port the dev-team skills from the sibling `eternal-descent` repo to this Python and RL
    codebase (`dev-loop`, `architect`, `test-crafter`, `cleanup`, `pr-reviewer`, `doc-sync`), add the RL-specific
    skills `algorithm-crafter`, `experiment-runner` and `training-doctor`, and add the tooling they rely on.
  - **Acceptance criteria**:
    - Nine skills under `.claude/skills/`, each naming this repository's real packages, helpers, commands and gates.
    - `.claude/settings.json` pins the brooks-lint plugin and pre-approves the quality-gate commands;
      `.brooks-lint.yaml` holds project review rules C1 to C6.
    - `tests/helpers.py` provides `numerical_grad` and `assert_gradients_match`, covered by `tests/test_helpers.py`.
    - `mutation_check.py` and `check_md_width.py` ship with the skills that use them and pass Ruff.
    - CLAUDE.md, CONTRIBUTING.md §4, GEMINI.md §5 and the README describe the workflow and its setup.
  - **Known edge cases**: Machines without Node.js cannot run `markdownlint`; the skills fall back to the
    line-length script. Without the brooks-lint plugin, the skills run with the project checks only.
  - **Out of scope**: A `release` skill (there is no `main` branch, tag or changelog to release from); adding
    gradient checks to the existing agents; the gaps found while building this (the four items below).

- [x] **Training Diagnostics in Logs**
  - **Delivered in**: merged in #65.
  - **Description**: `Trainer` discards the dictionary `agent.update()` returns, so losses, entropy and epsilon never
    reach the metrics log and a failing run can only be diagnosed from a scratch script. Keep what `update` returns,
    write it with the periodic reward metrics, and warn the moment a value stops being finite.
  - **Acceptance criteria**:
    - The periodic metrics record (every 10 episodes, alongside `avg_reward` and `avg_length`) gains an `update`
      mapping holding the mean of each metric `agent.update()` returned over those episodes.
    - An agent whose `update` returns nothing (the random agents) produces records without an `update` key.
    - The first non-finite value of each update metric is logged at WARNING level, naming the metric, the value and
      the episode; the trainer records that episode in `first_non_finite_episode`. Training continues.
    - Every metrics record is strict JSON: a non-finite mean is written as `null` and its name is listed under
      `non_finite`.
    - Evaluation episodes contribute nothing to the update metrics.
    - TESTING.md `LOG-1` describes the file as it is written.
  - **Known edge cases**: Metrics present in only some episodes of a window; non-numeric values in the dictionary;
    runs shorter than one logging interval (no periodic record is written, as today).
  - **Out of scope**: TensorBoard or other external logging backends; stopping a run on a non-finite value; plotting
    update metrics; making the 10-episode logging interval configurable; changing what any agent returns.

- [x] **Discounted-Return Tests That Can Fail**
  - **Delivered in**: open PR #68 (stack #70).
  - **Description**: The `*_compute_returns_discounting` tests in `tests/test_ppo_agent.py`,
    `tests/test_actor_critic_agent.py` and `tests/test_reinforce_agent.py` asserted only that returns decrease over
    five rewards of 1. They still passed with the discount removed, because 5, 4, 3, 2, 1 also decreases; only the
    exact-value tests in `tests/test_agents_extensive.py` noticed.
  - **Acceptance criteria**:
    - The three tests are removed in favour of hand-computed tests of the one shared function
      (`tests/test_returns.py`).
    - Dropping the discount from the shared function fails a named test, and a test per agent (REINFORCE,
      Actor-Critic, PPO and continuous Actor-Critic) fails if that agent stops passing its own `gamma`
      (checked with `mutation_check.py`).
  - **Out of scope**: Nothing further; this ships together with the single implementation below.

- [x] **Softmax Stability Tests**
  - **Delivered in**: open PR #68 (stack #70).
  - **Description**: Removing the `logits - np.max(logits)` line from the softmax in
    `rl_core/agents/actor_critic_base.py` or `rl_core/agents/reinforce_agent.py` failed no test in the suite, so the
    guard against overflow required by CONTRIBUTING.md §1 was not locked in.
  - **Acceptance criteria**:
    - Tests force logits of ±1000 on each forward pass and assert, with warnings treated as errors, that the
      probabilities are finite and sum to one and that `select_action` works in both modes. The logits are set
      through the policy head's bias: scaling the weights does not reliably overflow.
    - Removing the stabilisation line in either file fails a named test (checked with `mutation_check.py`).
  - **Known edge cases**: All logits equal; one logit far above the rest.
  - **Out of scope**: Gradient checks for the existing agents (see "Split `update` into Loss-and-Gradients and
    Apply"); sharing the softmax (see "One Softmax for the Discrete Policies").

- [x] **Single `_compute_returns` Implementation**
  - **Delivered in**: open PR #68 (stack #70).
  - **Description**: `ReinforceAgent._compute_returns` and `ActorCriticBase._compute_returns` were the same
    discounted-return loop written twice, with a third copy inline in `ContinuousActorCriticAgent.update`, so a fix
    to one could miss the others.
  - **Acceptance criteria**:
    - One implementation, `rl_core.agents.returns.discounted_returns`, used by REINFORCE, Actor-Critic, PPO and
      continuous Actor-Critic; no `_compute_returns` method remains.
    - Seeded runs give bit-identical weights before and after the change.
    - The duplicate-code audit reports nothing new.
  - **Known edge cases**: Empty reward array; `gamma` of 0 and of 1.
  - **Out of scope**: Sharing the softmax or the network code between REINFORCE and the actor-critic family (see
    "One Softmax for the Discrete Policies").

- [ ] **One Softmax for the Discrete Policies**
  - **Description**: The numerically stable softmax is written inline twice, in `ReinforceAgent._policy` and in
    `ActorCriticBase._forward`. Both copies are now covered by the same stability tests, so they cannot silently
    diverge, but a change to one still has to be made in the other.
  - **Acceptance criteria**:
    - One module-level `stable_softmax(logits)` under `src/rl_core/agents/`, with its own hand-computed tests, used
      by both forward passes.
    - Seeded runs give bit-identical weights before and after the change.
    - `tests/test_discrete_policy_numerics.py` passes unchanged.
  - **Known edge cases**: Batched logits (a 2-D input) if a later agent needs them; all logits equal.
  - **Out of scope**: Sharing the network or forward pass between REINFORCE and the actor-critic family.

### Code Health

Defects found while working on other items, then findings from the brooks-lint health pass of 2026-10-12
(dashboard 82/100), most severe first.

- [x] **Seeded Environments Replay One Start State**
  - **Delivered in**: merged in #67.
  - **Description**: `CartPoleEnv.reset()` and `PendulumEnv.reset()` reuse the constructor seed whenever they are
    called without one, and `Trainer` never passes one. With a seed in the config (every committed config has one)
    each training and evaluation episode starts from the identical initial state, so agents train on a single
    start state and the ten greedy evaluation episodes are the same episode ten times. Found on 2026-10-13.
  - **Acceptance criteria**:
    - A seeded environment gives a reproducible sequence of different start states: the seed is applied on the
      first reset, and later resets continue the same generator.
    - Two seeded runs are still identical to each other; the wrapper's noise generator follows the same rule.
    - Tests cover both wrappers: consecutive resets differ, and two environments with the same seed agree episode
      by episode.
    - Evaluation rewards across episodes are no longer identical for a deterministic policy.
  - **Known edge cases**: An explicit `reset(seed=...)` must still restart the sequence; vectorized environments;
    domain randomization resampling per episode.
  - **Out of scope**: Regenerating `docs/benchmarks/` (the published numbers were produced with the repeated start
    state and need a maintainer-approved rerun afterwards); retuning configs.

- [x] **CartPole Agents Learn Within Their Budgets**
  - **Delivered in**: open PRs #73, #74, #75 and the results PR that follows them (stack #70).
  - **Description**: After the seeding fixes, the benchmark of 2026-10-13 (750 episodes, seeds 42 and 123; see
    `docs/benchmarks/cartpole_results.md`) shows Actor-Critic and DQN solving CartPole-v1, REINFORCE improving slowly
    (mean evaluation reward 121) and PPO at the random level (27, against 25 for the random agent). The benchmark
    also trains every agent for the same number of episodes (500 by default), while the agents' own configs ask for
    500 to 2000. Find out which agents have a learning defect and which only lack episodes, fix the defects, and
    make the benchmark's budget and solve rule say what they mean.
  - **Acceptance criteria**:
    - Each learning agent is run at its own config's budget on at least 5 seeds, and the result is recorded: solve
      rate, evaluation reward, episodes to solve.
    - PPO gets the `training-doctor` ladder: a finite-difference gradient check, a direction test, overfitting a
      fixed batch and a read of one real batch. Each cause found is confirmed by a check that fails before the fix
      and passes after, and is fixed test-first. REINFORCE gets the same if it does not solve at its own budget.
    - Any change to a committed `configs/cartpole_*.yaml` is proposed with its evidence and approved by the
      maintainer.
    - The benchmark's default budget per agent is a deliberate choice: either each agent's config budget, or one
      shared number that every agent expected to solve the task can solve it in. The report states which.
    - The solve rule is documented where the report is read. Today a run counts as solved if the rolling training
      average reaches the threshold, or if only the final evaluation does, in which case the last episode is
      reported as the solve episode.
    - With the committed configs, every agent the README lists as solving CartPole-v1 reaches the threshold on at
      least 4 of 5 seeds, and `docs/benchmarks/cartpole_results.md` and TESTING.md `TR-1` to `TR-4` say what was
      measured.
  - **Known edge cases**: Things to rule out, none confirmed: PPO updating once per episode on that episode alone,
    with a learning rate of 0.0003; DQN reaching the threshold and then losing it (seed 42 solved at episode 274
    and finished at 334); epsilon never reaching its floor within 500 episodes.
  - **Ruled out (2026-10-13)**: The hand-written gradients of REINFORCE, Actor-Critic, PPO and DQN agree with
    finite differences for every parameter, and each `update` applies them as it should
    (`tests/test_policy_gradient_maths.py`, `tests/test_dqn_maths.py`). PPO's score is not a backward-pass error.
    Direction and overfit checks for REINFORCE, Actor-Critic and PPO (2026-10-14, same file): one update on a
    one-state batch raises the probability of the rewarded action, and 300 updates on it bring that probability above
    0.9 while the critic's loss halves. A read of real batches (`diag_step`, Measured below) found the step size.
  - **Measured (2026-10-14)**: Committed configs, each agent at its own config's budget, seeds 1 to 5: Actor-Critic
    solves 5 of 5 (evaluation 494 ± 11, at episodes 906 to 1269), DQN 3 of 5 in 500 episodes (406 ± 77), PPO 0 of 5
    in 1500 (160 ± 91), REINFORCE 0 of 5 in 2000 (239 ± 120), random 21 ± 3. On seed 1, one PPO update moves
    log π(a|s) 3 to 6 times less than one Actor-Critic update: PPO and REINFORCE normalise their advantages to unit
    spread (Actor-Critic's raw advantages have a spread of about 6), and PPO's learning rate of 0.0003 is the usual
    value for Adam, used here with plain SGD. PPO's clip never binds. With only PPO's learning rate raised to 0.0015,
    evaluation rose on every seed (356 ± 99) and 1 of 5 solved: step size is a cause, not the only one. At 0.003,
    evaluation is 449 ± 81 but the training average reaches 475 on no seed. REINFORCE at 0.005 (2000 episodes):
    evaluation 404 ± 35, training average 160 to 200, still rising. PPO with Adam (no clipping) at the committed
    0.0003: the training average reaches 475 on 3 of 5 seeds at episodes 586 to 652, faster than Actor-Critic, but
    is not held (one seed ends at 186). Adam with global-norm clipping at 0.5: 3 of 5 reach 475, later (episodes
    1265 to 1420), evaluation 442 ± 73, and the training average still falls back afterwards (final 189 to 456).
    REINFORCE with `optimizer: adam` and its committed learning rate and budget (0.001, 2000 episodes): 5 of 5
    reach 475 (episodes 645 to 1012, mean 830), evaluation 449 ± 59; one seed's final training average is 173.
    PPO with `optimizer: adam` at its committed 0.0003, 3000 episodes: 5 of 5 reach 475 (episodes 530, 651, 968,
    1066 and 1626), so 4 of 5 within the 1500 committed at the time; evaluation after 3000 episodes 442 (280 to
    500), against 495 after 2000 in the learning gate below. DQN at 750 episodes: 4 of 5 reach 475 (episodes 378,
    380, 463 and 576; seed 3 ends at 404), evaluation 450 ± 46. DQN with
    `epsilon_decay: 0.99` (floor by episode 389) at 500 episodes: 2 of 5, evaluation 294 ± 60, worse than the
    committed 0.995.
  - **Committed configs (2026-10-14)**: REINFORCE and PPO use `optimizer: adam`; PPO trains for 2000 episodes
    and DQN for 750. Learning gate with those configs, seeds 1 to 5: REINFORCE 5 of 5, Actor-Critic 5 of 5, PPO
    5 of 5, DQN 4 of 5. Published in `docs/benchmarks/cartpole_results.md` (seeds 1 to 5, each agent's config
    budget; the same per-seed results) and in the README's "Which agents solve which challenge".
  - **Decided (2026-10-14)**: A run counts as solved only when the rolling training average reaches the
    threshold; a final evaluation at or above it is reported separately. Without `--episodes`, every agent in every
    challenge trains for its own config's `training.num_episodes`, the random agents included. The README claims
    CartPole-v1 for each of REINFORCE, Actor-Critic, PPO and DQN that reaches the threshold on 4 of 5 seeds. Budgets
    may be raised where an agent is still improving. Tuning goes in `configs/cartpole_*.yaml`; a shared optimiser
    option defaults to the current behaviour.
  - **Plan**: four stacked PRs. A: the benchmark's budget and solve rule. B: an `optimizer` option (Adam, with
    optional global-norm clipping) for the discrete agents, then PPO and REINFORCE configs with 5-seed evidence.
    C: DQN to 4 of 5 (budget, epsilon schedule). D: regenerate `docs/benchmarks/cartpole_results.md`, README and
    TESTING.md `TR-1` to `TR-4`.
  - **DQN behaviours to decide on (2026-10-13; pinned by tests as they are, none changed)**:
    - Two comments in `configs/cartpole_dqn.yaml` do not match the code: epsilon decays once per `update` call,
      which is once per episode, not "per update"; and `updates_per_step: 1.0` is capped at 64 gradient steps per
      episode by `max_updates_per_call`, so a 500-step episode gets about 0.13 steps per transition.
    - `max_grad_norm` clips each gradient element, not the gradient's norm, and is absent from the config
      (default 10.0).
    - While the replay buffer is warming up, `update` returns a loss and mean Q-value of 0, which the training log
      averages in.
    - Not validated at construction: `target_update_frequency: 0` raises `ZeroDivisionError` during training;
      `tau` of 1 or more silently falls back to hard sync, 0 freezes the target network, and a negative value
      extrapolates; `min_buffer_size` below `batch_size` fails at the first sample; `buffer_capacity` below
      `batch_size` fails there too if `min_buffer_size` fits in the buffer, and otherwise (as with the default
      `min_buffer_size`) the buffer never counts as warm, so the agent silently never trains and epsilon never
      decays.
    - A batch without `dones` is not stored, but gradient steps still run.
    - A time-limit cut is stored as terminal, so the bootstrap is zeroed at CartPole's 500-step limit.
  - **Out of scope**: The Pendulum agents (their own item below); new algorithms; changing the solve threshold.

- [ ] **Separate Random Streams for Agents and Buffers**
  - **Description**: Agents and `ReplayBuffer` create their generators with `np.random.default_rng(seed)`, and the
    committed configs and both benchmark adapters give the agent the same seed as the environment. Weight
    initialisation, exploration, replay sampling and the environment's start states therefore draw from one
    identical stream. The environment wrappers already use `rl_core.seeding.derive_rng`.
  - **Acceptance criteria**:
    - Each agent and `ReplayBuffer` takes a named stream from `derive_rng`; no two components built from one seed
      draw the same numbers, and a test fails if they do.
    - Runs stay reproducible for a seed.
  - **Known edge cases**: Every seeded result changes once more; checkpoints that store generator state.
  - **Out of scope**: Regenerating `docs/benchmarks/`.

- [ ] **Separate Evaluation Environment**
  - **Description**: `Trainer.evaluate` runs on the training environment, so evaluation episodes advance the same
    random sequence and `eval_frequency` changes which start states later training episodes see. Runs stay
    reproducible, but evaluation is not independent of training.
  - **Acceptance criteria**:
    - Evaluation uses its own environment with its own seeded sequence; changing `eval_frequency` leaves the
      training episodes unchanged for a given seed.
    - Every evaluation uses the same fixed set of start states, so scores are comparable across a run.
  - **Out of scope**: Vectorized evaluation.

- [ ] **Pendulum PPO Learns and Solves Pendulum-v1**
  - **Description**: `ContinuousPPOAgent` with `configs/pendulum_ppo.yaml` does not learn. On 2026-10-13 the
    benchmark at its default 200 episodes gave evaluation rewards of -1632, -1673 and -1584 on seeds 1, 2 and 3
    (the same order of magnitude before the shuffle fix), against a solve threshold of -200 and a random agent
    between -892 and -1582. Fixing the repeated start state did not change that: the published run
    (`docs/benchmarks/pendulum_results.md`, 150 episodes, seeds 42 and 123) has PPO at -1626 and Actor-Critic at
    -1426, both below the random agent's -1323. The Phase 6 entry above says this configuration solves the task.
    Find out why it does not, fix it, and show that it solves Pendulum-v1 reproducibly.
  - **Acceptance criteria**:
    - A written diagnosis (the `training-doctor` ladder) names the cause or causes, each confirmed by a check that
      fails before the fix and passes after: finite-difference gradient checks of the continuous actor and critic,
      a direction test, overfitting a fixed batch, and a read of one real batch.
    - Each code defect found is fixed test-first; any change to `configs/pendulum_ppo.yaml` is proposed with its
      evidence and approved by the maintainer.
    - With the committed config, the rolling training average reaches -200 on at least 4 of 5 seeds within the
      configured episode budget, and two runs with the same seeds agree.
    - `ContinuousActorCriticAgent` is assessed the same way and either learns or has its own roadmap item.
    - TESTING.md `TR-6` states the measured result, and `docs/benchmarks/pendulum_results.md` is regenerated from a
      maintainer-approved run.
  - **Known edge cases**: Things to rule out, none confirmed yet: every episode starting from the same state (the
    item above); the 200-episode benchmark budget against the 500 the config asks for (the benchmark now uses each
    config's budget unless `--episodes` is given, so the published report predates that); Pendulum episodes end only
    by time limit, which the batch reports as `done`, so a bootstrapping critic may treat step 200 as terminal;
    rewards between about -16 and 0 per step with no scaling (a value loss near 7000 was logged at episode 20);
    one update per 200-step episode; action clipping against the Gaussian log-probability.
  - **Ruled out (2026-10-13)**: The hand-written gradients of both continuous agents agree with finite differences
    for every parameter, and so do Adam and PPO's advantage estimation
    (`tests/test_continuous_policy_gradient_maths.py`). The repeated start state is fixed and the agents still do
    not learn. So a backward-pass error is not the cause at the settings tested; the clip of `log_std` to
    [-20, 2] is the one piece of that path with no test.
  - **Observations to start from (2026-10-13, untrained agents, no training run; hypotheses, not findings)**:
    - The initial policy mean is often outside the action bounds of ±2. Two separate checks of untrained agents
      gave a median `|mu|` between about 1 and 9 depending on the seed, with 30% to 99% of sampled actions clipped.
      The log-density is evaluated on the clipped action, so for those steps `a − mu` is a fixed offset, not
      noise. A candidate cause; measure it per seed with the states sampled as training samples them.
    - Continuous Actor-Critic takes one Adam step per episode. Adam moves each weight by at most about the
      learning rate (3e-4) per step, so 300 episodes allow roughly 0.09 of travel, while returns of several
      hundred below zero were seen with a value loss of the order of 1e5.
    - Continuous PPO takes 32 Adam steps per 200-step episode (mini-batches of 64, 64, 64 and 8, for 8 epochs); the
      8-sample mini-batch gets a full step.
    - The value loss dominates the shared gradient: its norm is thousands against a clip of 0.5, and inside the
      shared trunk the critic term swamps the actor term.
    - `configs/pendulum_ppo.yaml` sets `clip_epsilon`, but `ContinuousPPOAgent` reads `clip_ratio`, so the config
      value is ignored (both are 0.2 today). The discrete `PPOAgent` reads `clip_epsilon`.
  - **Out of scope**: DDPG and other new algorithms; changing the solve threshold; the discrete agents.
  - **Depends on**: Seeded Environments Replay One Start State, so the diagnosis is not confounded by it.

- [x] **Seed the Continuous PPO Mini-Batch Shuffle**
  - **Delivered in**: merged in #66.
  - **Description**: `ContinuousPPOAgent.update` shuffles mini-batch indices with the global
    `np.random.shuffle` (`rl_core/agents/continuous_ppo.py`), so two runs with the same seed give different
    results. Found on 2026-10-12 while verifying the training diagnostics change: the Pendulum PPO benchmark with
    seeds 1 and 3 differed between two runs of the same code.
  - **Acceptance criteria**:
    - The shuffle draws from the agent's own seeded generator; no bare `np.random.*` call remains under `src/`,
      and a test fails if one is added.
    - An update neither reads nor advances NumPy's global random state.
    - A test trains two agents with the same seed and gets identical weights; a different seed differs.
    - Two benchmark runs with the same seeds report identical rewards for every agent on both challenges.
  - **Verified (2026-10-14)**: Two Pendulum benchmark runs (seeds 1 and 2, 15 episodes) gave identical rewards for
    every agent; on CartPole, the learning gate and the published run (seeds 1 to 5, full budgets) gave identical
    per-seed results for every agent.
  - **Known edge cases**: Seeded Pendulum PPO results change once, when the fix lands.
  - **Out of scope**: Retuning `configs/pendulum_ppo.yaml`; regenerating the Pendulum PPO row of
    `docs/benchmarks/pendulum_results.md`, which was produced with the unseeded shuffle and needs a
    maintainer-approved rerun.

- [x] **Actor-Critic Trains with Adam**
  - **Delivered in**: open PR #77, results in the PR that follows #78 (stack #70).
  - **Description**: On CartPole, Actor-Critic first reached the solve threshold later than REINFORCE (episodes 906 to
    1269 against 645 to 1012), the opposite of what a critic should give, because REINFORCE trains with Adam and
    Actor-Critic with plain SGD.
  - **Acceptance criteria**:
    - `configs/cartpole_actor_critic.yaml` uses `optimizer: "adam"` (maintainer-approved), and a test pins it.
    - With the committed config, Actor-Critic reaches 475 on at least 4 of 5 seeds within 1500 episodes.
  - **Measured (2026-10-14)**: Adam at the committed learning rate (0.001) and budget, seeds 1 to 5: 5 of 5, first
    reaching 475 at episodes 490 to 839 (SGD: 906 to 1269); final training averages 446 to 500; evaluation 476 ± 32.
  - **Learning gate (2026-10-15)**: the committed config gives the same per-seed results; published in
    `docs/benchmarks/cartpole_results.md` on 2026-10-15.
  - **Out of scope**: Bootstrapped (TD) critic targets.

- [x] **Standard PPO for Discrete Actions**
  - **Delivered in**: open PR #78 (code) and the config and results PR that follows it (stack #70).
  - **Description**: `PPOAgent` updated once per episode on that episode alone, full batch, with Monte-Carlo
    advantages. Standard PPO (Schulman et al., 2017) collects a rollout across episodes, uses GAE, and takes several
    epochs of shuffled mini-batch steps.
  - **Acceptance criteria**:
    - `PPOAgent` buffers whole episodes until `rollout_steps`, computes GAE (`gae_lambda`) over the rollout, and trains
      `ppo_epochs` passes of `minibatch_size` steps; without those keys it keeps its earlier per-episode method.
    - Gradient check, hand-computed loss and GAE, replay against an independent implementation, buffering and
      checkpoint tests; `ContinuousPPOAgent` shares the GAE function with bit-identical weights.
    - Optional linear annealing of the learning rate and clip range (`anneal_steps`).
    - The committed config (maintainer-approved) reaches 475 on at least 4 of 5 seeds and holds it.
  - **Measured (2026-10-14)**, seeds 1 to 5, 2000 episodes: CleanRL settings (learning rate 0.00025, 4 epochs) solve
    4 of 5 at either 256 or 512 steps per rollout, later than the per-episode PPO. rl-baselines3-zoo settings (256
    steps, 20 epochs, learning rate 0.001, λ 0.8, γ 0.98, no entropy bonus) solve 5 of 5 at episodes 353 to 449 but
    3 seeds collapse afterwards (evaluation 341). Adding annealing over 200,000 trained transitions: 5 of 5 at
    episodes 323 to 427, final training averages 453 to 500, evaluation 500 on every seed.
  - **Learning gate (2026-10-15)**: the committed config (those settings, 1000 episodes): 5 of 5 at episodes 323 to
    427, evaluation 500 on every seed, final training averages 360 to 500; about 85 seconds per run. "Holds it" is
    judged by the greedy evaluation after training (500 on every seed); seed 3's stochastic training average ends at
    360, below the threshold. Published in `docs/benchmarks/cartpole_results.md` on 2026-10-15.
  - **Known edge cases**: Steps still buffered when training ends are never trained on; checkpoints do not hold the
    buffer; CartPole's 500-step time limit is treated as terminal, as for every agent.
  - **Out of scope**: Value-loss clipping, KL early stopping, parallel environments.

- [~] **Break the Benchmark Import Cycle**
  - **Description**: `src/benchmark.py` imports `rl_cartpole.benchmark` and `rl_pendulum.benchmark` inside functions
    in two places, both import `benchmark` back, and a `sys.modules.setdefault("benchmark", ...)` alias holds the
    cycle together. A third challenge means editing the runner twice.
  - **Acceptance criteria**:
    - `BenchmarkChallenge` and the challenge registry live in `rl_core`; challenge packages import only `rl_core`.
    - The runner discovers registered challenges without naming any of them, and the `sys.modules` alias is gone.
    - `python src/benchmark.py` and the tests in `tests/test_benchmark.py` behave as before.
  - **Known edge cases**: Running `src/benchmark.py` directly as `__main__`; a challenge package that fails to import.
  - **Out of scope**: Changing benchmark metrics or the report format.

- [ ] **Select the Challenge Through the Registry**
  - **Description**: `train.py` and `record.py` choose the environment with `"pendulum" in env_name.lower()` in three
    places and hardcode `choices=["cartpole", "pendulum"]`, so an unknown challenge silently becomes CartPole.
  - **Acceptance criteria**:
    - Environment and agent construction go through the challenge registry; no string matching on the name.
    - `--challenge` choices come from the registry, and an unknown environment name is an error naming the options.
    - Adding a challenge requires no edit to `train.py` or `record.py`.
  - **Known edge cases**: Configs with no `environment.name`; `--challenge` disagreeing with the config.
  - **Out of scope**: Merging the discrete and continuous agent maps (depends on the cycle item above).

- [x] **Split `update` into Loss-and-Gradients and Apply**
  - **Delivered in**: open PRs #69, #71 and #72 (stack #70).
  - **Description**: The agents' `update` methods run 59 to 141 lines and mix batch parsing, forward pass, loss,
    backward pass and optimizer step, so no gradient check can be written against them.
  - **Progress**: Delivered in three parts. Part 1 (2026-10-13) covers REINFORCE, Actor-Critic and PPO
    (`tests/test_policy_gradient_maths.py`). Part 2 (2026-10-13) covers the two continuous agents
    (`tests/test_continuous_policy_gradient_maths.py`), including Adam with gradient clipping and PPO's generalised
    advantage estimation. Part 3 (2026-10-13) covers DQN (`tests/test_dqn_maths.py`), including its Bellman
    targets, element-wise gradient clip and target-network sync. The hand-written gradients of all six learning
    agents agree with finite differences. Confirmed complete by the maintainer on 2026-10-14.
  - **Acceptance criteria**:
    - Each learning agent has a method that returns the loss and gradients for a batch and changes no state, and a
      separate step that applies them.
    - Each has a finite-difference test using `assert_gradients_match` from `tests/helpers.py`.
    - The loss each agent computes is checked against a value worked out by hand, so a return that is misaligned
      with its step is caught. Today the tests prove only that each `update` depends on `gamma`: passing the
      rewards reversed, reversing the returns, or squaring the discount at a call site all survive the suite
      (found on 2026-10-13).
    - Seeded runs give bit-identical weights before and after the change.
  - **Known edge cases**: PPO's multiple epochs over one batch; DQN's several mini-batches per call; ReLU kinks and
    clip boundaries in the gradient check.
  - **Out of scope**: Shortening `train.main` and `BenchmarkRunner.run`; changing any update rule.

- [ ] **Test Agents Through Their Public Behaviour**
  - **Description**: Tests reach into private attributes 274 times (124 in `tests/test_agents_extensive.py`), so a
    refactor with no behaviour change breaks many of them. (The 11 tests that called `_compute_returns` went with
    that method on 2026-10-13; the count above predates that.)
  - **Acceptance criteria**:
    - Agent tests assert through `select_action`, the metrics `update` returns, and save/load round trips.
    - Private access remains only where a test checks the maths directly (gradient checks, exact returns), through
      one helper per agent family.
    - The five "weights change after update" tests are replaced by checks that could fail for a wrong gradient.
  - **Known edge cases**: Tests that need a specific weight state to reach an edge case.
  - **Out of scope**: The integration tests in `tests/test_trainer.py` and the script tests.

- [ ] **Shared Environment Wrapper Base**
  - **Description**: `rl_cartpole/environments/cartpole_env.py` and `rl_pendulum/environments/pendulum_env.py` repeat
    four near-identical blocks (the largest about 50 lines) for noise, statistics and wrapping, so a fix in one
    challenge misses the other. The seeding logic in `__init__` and `reset()` is a fifth, kept in step today only by
    the shared tests in `tests/test_env_seeding.py`.
  - **Acceptance criteria**:
    - The shared behaviour lives once in `rl_core`; each challenge keeps only what differs.
    - The duplicate-code audit at 8 similar lines reports no pair between the two files.
    - `tests/test_environment.py`, `tests/test_pendulum.py` and `tests/test_env_seeding.py` pass unchanged.
  - **Known edge cases**: Discrete action flips against continuous action noise; per-challenge randomized parameters.
  - **Out of scope**: The `make_env` factories; new wrappers such as frame stacking.

- [ ] **Complete Optimizer State in Checkpoints**
  - **Description**: The continuous agents save the Adam step count `adam_t` but not the moment estimates, so a run
    resumed from a checkpoint takes mis-scaled first steps. DQN does not save its target-sync counter.
  - **Acceptance criteria**:
    - Everything an agent needs to resume training round-trips through `save` and `load`.
    - A test trains, saves, loads into a fresh agent, and gets the same weights after one more seeded update.
    - Checkpoints written before the change still load.
  - **Out of scope**: A resume option for `train.py`.

### Phase 5 — Multi-Agent Benchmarking & Comparative Evaluation

- [x] **Cross-Agent Benchmark Suite & Comparative Visualizer**
  - **Description**: Add an automated benchmark framework (`src/benchmark.py`) with environment challenge adapters
    (`src/rl_cartpole/benchmark.py`) to train, evaluate, and compare all supported agents under standardized seeds,
    measuring sample efficiency, convergence speed, evaluation stability, and wall-clock execution time.
  - **Acceptance criteria**:
    - `src/benchmark.py` running multi-agent rollouts with reproducible seed sweeps across all algorithms.
    - Generates unified comparison plots (`plots/benchmarks/<challenge>/benchmark_comparison.png`) showing mean reward
      trajectories with shaded variance bands across runs.
    - Generates markdown report (`docs/benchmarks/<challenge>_results.md`) with summary metrics table (episodes to
      solve, final evaluation reward, sample efficiency).
    - CLI arguments: `--challenge`, `--agents`, `--episodes`, `--seeds`, `--output-dir`, `--report`, and `--no-plot`.
    - Unit and integration tests in `tests/test_benchmark.py` achieving >90% coverage.
  - **Known edge cases**: Interrupted benchmark runs; mismatched agent configuration files; varying episode lengths
    affecting rollout timing.
  - **Out of scope**: Distributed multi-machine training; automated Bayesian hyperparameter optimization sweeps.

---

### Phase 6 — Advanced RL & Robotics Extensions

- [x] **Continuous Action Space Support & Core Decoupling (Gaussian Policies & Pendulum-v1)**
  - **Description**: Decoupled algorithm-agnostic RL framework into `src/rl_core/` and extended policy gradient
    agents (`ContinuousActorCriticAgent`, `ContinuousPPOAgent`, `ContinuousRandomAgent`) to support continuous control
    benchmarks, introducing the `src/rl_pendulum/` challenge package.
  - **Acceptance criteria**:
    - Gaussian policy heads predicting state-dependent action mean $\mu(s)$ and trainable log standard deviation
      $\log \sigma$.
    - Reparameterized action sampling ($a = \mu + \sigma \odot \epsilon$) and Gaussian log-likelihood backprop.
    - Support for `gymnasium.spaces.Box` action spaces in `PendulumEnv` wrapper and vectorization factories.
    - Calibrated configuration `configs/pendulum_ppo.yaml` solving Pendulum-v1.
    - Unit tests in `tests/test_continuous_agent.py` and `tests/test_pendulum.py` achieving >95% coverage.
  - **Known edge cases**: Action squashing/tanh bounding; standard deviation vanishing or numerical overflow.
  - **Out of scope**: Off-policy continuous algorithms (DDPG/SAC); multi-discrete action spaces.

- [ ] **Continuous Off-Policy Learning & DDPG (Deep Deterministic Policy Gradient)**
  - **Description**: Implement a pure NumPy `DDPGAgent` utilizing a deterministic actor network $\mu_\theta(s)$,
    an action-value critic $Q_\phi(s, a)$, continuous transition experience replay via `ReplayBuffer`, exploration
    noise (Gaussian or Ornstein-Uhlenbeck), and Polyak soft target updates to solve `Pendulum-v1`
    sample-efficiently.
  - **Acceptance criteria**:
    - Pure NumPy deterministic actor and Q-value critic network architectures.
    - Continuous transition ingestion and random mini-batch sampling from `ReplayBuffer`.
    - Critic TD loss optimization and deterministic policy gradient backpropagation
      ($\nabla_\theta Q(s, \mu_\theta(s))$).
    - Polyak soft target weight synchronization ($\theta' \leftarrow \tau \theta + (1-\tau) \theta'$ with
      $\tau \approx 0.005$).
    - Reference configuration `configs/pendulum_ddpg.yaml` solving `Pendulum-v1` (mean reward $\ge -200$)
      in $\le 100$ episodes.
    - CLI dispatch support in `train.py`, `record.py`, and `src/benchmark.py`.
    - Unit and integration tests in `tests/test_ddpg_agent.py` achieving $>90\%$ coverage.
  - **Known edge cases**: Q-value overestimation; divergence from non-smooth actor gradients; torque saturation.
  - **Out of scope**: Twin Delayed DDPG (TD3) double Q-learning; stochastic entropy auto-tuning (SAC).

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
