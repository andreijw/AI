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

**Agreed order (2026-10-13, extended 2026-10-14 and 2026-10-15, reordered 2026-10-16):** Seed the Continuous PPO
Mini-Batch Shuffle → Seeded Environments Replay One Start State → the test-gap items (Discounted-Return Tests That
Can Fail, Softmax Stability Tests, Single `_compute_returns` Implementation) → Split `update` into Loss-and-Gradients
and Apply → CartPole Agents Learn Within Their Budgets → Actor-Critic Trains with Adam → Standard PPO for Discrete
Actions → Break the Benchmark Import Cycle → Pendulum PPO Learns and Solves Pendulum-v1 → Benchmark Budgets and Solve
Rule in Environment Steps → Select the Challenge Through the Registry → DQN Reaches the rl-zoo Reference → DDPG →
Pendulum PPO Holds -200 Under the 100-Episode Rule → Continuous Actor-Critic Learns Pendulum-v1.

Why this order: the registry item is small and fixes a silent fallback to CartPole before DDPG adds another agent to
`train.py` and `record.py`. DQN's fixes (time-limit bootstrapping, a per-step exploration schedule, replay and
target-network settings) carry straight into DDPG, which reuses its replay buffer and target network. DDPG is the
off-policy method that solves Pendulum-v1 cheaply in the reference setups, so it comes before the two on-policy
Pendulum items; continuous Actor-Critic is last because even the reference needs 1,000,000 steps for it.

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
      (default 10.0). Resolved 2026-10-16: the element-wise clip is `grad_clip_value` (plain SGD only), and
      `max_grad_norm` is Adam's global-norm clip, as in the other agents.
    - While the replay buffer is warming up, `update` returns a loss and mean Q-value of 0, which the training log
      averages in.
    - Not validated at construction: `target_update_frequency: 0` raises `ZeroDivisionError` during training;
      `tau` of 1 or more silently falls back to hard sync, 0 freezes the target network, and a negative value
      extrapolates; `min_buffer_size` below `batch_size` fails at the first sample; `buffer_capacity` below
      `batch_size` fails there too if `min_buffer_size` fits in the buffer, and otherwise (as with the default
      `min_buffer_size`) the buffer never counts as warm, so the agent silently never trains and epsilon never
      decays.
    - A batch without `dones` is not stored, but gradient steps still run. (Since 2026-10-16 a batch with
      `terminated` and no `dones` is stored; one with neither still is not.)
    - A time-limit cut is stored as terminal, so the bootstrap is zeroed at CartPole's 500-step limit. Resolved
      2026-10-16: DQN stores the trainer's `terminated` flag ("DQN Reaches the rl-zoo Reference").
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
  - **Progress (2026-10-16)**: `DDPGAgent` already draws weights, exploration and replay sampling from named
    streams (`AGENT_WEIGHTS_STREAM`, `AGENT_EXPLORATION_STREAM`, `AGENT_REPLAY_STREAM` in `rl_core.seeding`); the
    other agents remain.
  - **Out of scope**: Regenerating `docs/benchmarks/`.

- [x] **Separate Evaluation Environment**
  - **Delivered in**: #96, with the benchmarks republished afterwards. Confirmed by the maintainer on 2026-10-18.
  - **Republished (2026-10-18)**: both benchmarks on seeds 1 to 5, approved by the maintainer. Every seed now trains
    on a different start-state sequence, so seed-level results moved; every agent still meets the at-least-4-of-5
    bar. CartPole: REINFORCE 5 of 5 (was 4), Actor-Critic 5 of 5 holding on 2 (was 4), PPO 4 of 5 (was 5; seed 2 ends
    at 423), DQN 5 of 5. Pendulum: DDPG 5 of 5, PPO 4 of 5 (was 5; seed 4 ends at -415), Actor-Critic beats random
    on 5 of 5 (evaluation -168 ± 30).
  - **Description**: `Trainer.evaluate` runs on the training environment, so evaluation episodes advance the same
    random sequence and `eval_frequency` changes which start states later training episodes see. Runs stay
    reproducible, but evaluation is not independent of training.
  - **Acceptance criteria**:
    - Evaluation uses its own environment with its own seeded sequence; changing `eval_frequency` leaves the
      training episodes unchanged for a given seed.
    - Every evaluation uses the same fixed set of start states, so scores are comparable across a run.
  - **Plan (2026-10-18, reviewed)**: `Trainer` takes `eval_env` and `eval_seed`; evaluation episode i resets the
    evaluation environment with `eval_seed + i` (start state, wrapper noise and domain randomization all fixed).
    `eval_seed` comes from the run's seed on a dedicated stream (`EVAL_STREAM`), drawn once and logged when the
    run has no seed. record.py passes its playback environment as `eval_env` with `eval_seed_for(seed)`, so a
    playback shows the training run's evaluation start states. Without `eval_env` (direct `Trainer` callers,
    tests), evaluation stays on the training environment.
    Challenges gain a non-abstract `setup_eval_env`, so `setup_run` and custom challenges are unchanged. The random
    agents act from a separate evaluation stream, so evaluation does not shift their training actions either
    (maintainer's choice, option A). Afterwards: a learning check of every published claim, then both benchmarks
    rerun on seeds 1 to 5 and republished, in a separate PR.
  - **Out of scope**: Vectorized evaluation.

- [x] **Pendulum PPO Learns and Solves Pendulum-v1**
  - **Delivered in**: open PRs #82 (defect), #83 (code options) and the config and results PR that follows them.
  - **Result (2026-10-15)**: published in `docs/benchmarks/pendulum_results.md`: the committed config's training
    average reaches -200 on 5 of 5 seeds (1 to 5, episodes 211 to 318) but does not hold it: final training averages
    -196 to -386 (1 of 5 still at -200), evaluation -285 ± 82 (1 of 5 at -200 or better). Scratch runs, not
    published: 4 of 5 on seeds 6 to 10, and a separate gate run on seeds 1 to 5 that agrees with the published one
    seed for seed. Continuous Actor-Critic does not learn with these settings and has its own item below.
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
  - **Diagnosis (2026-10-15)**, committed configs, seeds 1 to 5, scratch outputs. Baseline at each config's budget:
    final training averages: PPO -1314 to -1806 with NaN on seed 2 (episode 489); Actor-Critic -1031 to -1509; the
    random agent -1155 to -1325. Five causes, each needed (γ and the learning rate are one change):
    1. **Defect: log-probability of the clipped action.** `select_action` clipped the sample, so the batch held the
       clipped value and its log-density. A negative advantage then pushed the mean further outside the bounds: on
       seed 2 the median |mean| reached 268, every action clipped, PPO's ratio overflowed (losses up to 1e111), NaN.
       Storing the unclipped sample removes the NaN on every seed but alone does not make either agent learn.
    2. **Initial mean far outside the bounds**: 96 to 100% of actions clipped on seeds 1, 2 and 5 before training.
    3. **Unscaled rewards**: the critic's gradient (norm 458 to 3349) swamps the actor's (0.2 to 65) in the shared
       trunk; the value loss stays 4000 to 10000 and the policy settles on a constant one-sided torque.
    4. **The 200-step time limit treated as terminal**: every Pendulum episode ends that way.
    5. **γ 0.99 and learning rate 0.0003**, changed together to γ 0.9 and 0.001. γ is the essential half: with the
       other four changes, γ 0.9 at the old learning rate gives 2 of 5, the new learning rate at γ 0.99 gives 0.
    With all five changes (scratch patches): 5 of 5 seeds reach -200 (first at episodes 206 to 363) but drift back
    (evaluation -210 to -766); leaving any one of the five out drops to at most 1 of 5 (counting γ and the
    learning rate as one); on seeds 6 to 10, 3 of 5. Adding linear
    annealing of the learning rate and clip range over 100,000 transitions: 4 of 5 reach -200 and stay near it
    (evaluation -216 to -251 on those four), in those scratch patches; with the committed code (Result above) the
    final training averages are -196 to -386. Gradients, direction (the mean moves toward rewarded actions) and Adam
    were already ruled out.
  - **Progress**: Part 1 (unclipped training samples; the Pendulum wrapper clips torque before action noise) fixes
    the NaN. Part 2 adds, as code with tests: `mean_init_scale` (default 0.01) on both continuous agents,
    `reward_scale` (default 1), a `terminated` array in the Trainer's batch so that continuous PPO bootstraps a final
    step cut by the time limit, `anneal_steps` for continuous PPO, `clip_epsilon` as its clip key (`clip_ratio` kept
    as an alias) and a clamp on the log-ratio. Part 3: the measured config, learning gate and published results for
    both challenges.
  - **Out of scope**: DDPG and other new algorithms; changing the solve threshold; the discrete agents.
  - **Depends on**: Seeded Environments Replay One Start State, so the diagnosis is not confounded by it.

- [x] **Pendulum PPO Holds -200 Under the 100-Episode Rule**
  - **Delivered in**: `feature/pendulum-ppo-rollouts`. Confirmed by the maintainer on 2026-10-18.
  - **Description**: Under the 10-episode rule Pendulum PPO reached -200 on 5 of 5 seeds; under the 100-episode
    rule ("Benchmark Budgets and Solve Rule in Environment Steps", 2026-10-15) it reaches it on none: final 100-episode
    training averages -241 to -313 at 100,000 steps (rl-zoo's budget), evaluation -285 ± 82. Doubling the budget and
    the annealing to 200,000 steps does not help (0 of 5; two seeds end at -529 and -841). rl-zoo's Pendulum PPO
    collects 1024-step rollouts on 4 parallel environments, uses state-dependent exploration (gSDE) and 10 epochs,
    where ours trains on each 200-step episode with 8 epochs and no observation normalisation.
  - **Acceptance criteria**:
    - A diagnosis names what separates ours from the reference; the causes needed to meet the bar are checked,
      and those left unchecked are listed (reworded 2026-10-18 with the maintainer).
    - With a maintainer-approved config, the 100-episode training average reaches -200 on at least 4 of 5 seeds
      within the budget, and the report says whether it held.
  - **Diagnosis (2026-10-18)**: scratch harness patching `ContinuousPPOAgent`, benchmark solve rule. The cause is
    training on one 200-step episode at a time. Buffering whole episodes into multi-episode rollouts, with each
    time-limit cut bootstrapped by adding γ·V(next state) to its last (scaled) reward as SB3 does, fixes it at
    200,000 steps: rollouts of 1024 / 2048 / 4096 transitions (10 epochs, annealing over 200k) solve 8 / 9 / 10 of
    seeds 1 to 10 and hold 7 / 8 / 4. At 100,000 steps the best is 2 of 5. Removing the annealing or the reward
    scale of 0.1 makes it worse. Not checked (the bar is met without them): gSDE exploration, separate tanh
    actor and critic networks, observation normalisation. Discrete PPO's rollouts treat every episode end as
    terminal, including CartPole's 500-step cut; left as it is.
  - **Decided (2026-10-18)**: `rollout_steps` for `ContinuousPPOAgent` (default 1, unchanged behaviour);
    `configs/pendulum_ppo.yaml` with rollouts of 2048, 10 epochs, annealing over and a budget of 200,000 steps
    (twice rl-zoo's 100,000, as CartPole PPO).
  - **Learning gate (2026-10-18)**: the real implementation with the committed config, benchmark seeds 1 to 5: 5 of 5
    reach -200 (steps 78,200 to 155,200) and 4 hold (seed 3 ends at -216); final averages -169 to -216, evaluation
    -150 ± 28 across seed means, identical to the scratch harness seed by seed. `rollout_steps: 1` reproduces the
    previous weights bit for bit.
  - **Follow-up noted (2026-10-18)**: discrete `PPOAgent` reads `rollout_steps` with `int(...)`, so 2.5 becomes 2,
    where continuous PPO rejects it; worth aligning when a third rollout user appears.
  - **Out of scope**: Changing the threshold or the window; SAC/TD3.

- [x] **Continuous Actor-Critic Learns Pendulum-v1**
  - **Delivered in**: #94 (fallback criterion: beats random on 5 of 5, does not solve). Confirmed by the maintainer
    on 2026-10-18.
  - **Description**: Split out of "Pendulum PPO Learns and Solves Pendulum-v1". `ContinuousActorCriticAgent` takes one
    Adam step per 200-step episode (about 5000 in its 1,000,000-step budget, against continuous PPO's 32 per episode),
    and it does not yet use time-limit bootstrapping or annealing. With the new defaults (small initial mean head) and γ
    0.9, learning rate 0.001 and `reward_scale` 0.1 (a scratch run, not published; the committed config still has γ 0.99
    and learning rate 0.0003), its final training average on seeds 1 to 5 was -1429 to -1530 after 300 episodes
    (2026-10-15), at or below the random agent. The published committed config, at its 1,000,000-step budget, scores
    -1422 ± 134 in evaluation (2026-10-16).
  - **Acceptance criteria**:
    - A diagnosis names why it does not learn, each cause confirmed by a check that fails before the fix.
    - It uses the batch's `terminated` flag like continuous PPO, with a test.
    - With a maintainer-approved config, the training average reaches -200 on at least 4 of 5 seeds within its
      budget. If it cannot, its evaluation at least beats the random agent's on 4 of 5 seeds, and the item records
      why one update per episode falls short and what would close the gap.
    - Two runs with the same seeds agree; TESTING.md `TR-5` states the measured result, and
      `docs/benchmarks/pendulum_results.md` is regenerated from a maintainer-approved run.
  - **Known edge cases**: Its `update` takes the whole episode in one step; a larger budget may be needed.
  - **Diagnosis (2026-10-18)**: traces and a scratch harness, seeds 1 to 5, 1,000,000 steps, random agent -1224.
    (1) The Gaussian mean is an unbounded linear head: it leaves the ±2 torque bounds (9 to 12 in the committed
    config, about 9,000 with rl-zoo's A2C settings), every executed torque is then the same, and nothing pulls it
    back; without a bounded mean even the best variant scores -1593. (2) Without an entropy bonus the log std
    collapses (from -2 to -17.6); entropy 0.003 and 0.001 give -589 and -967 against -152 at 0.01. (3) One
    Monte Carlo step per 200-step episode is weaker than short bootstrapped updates (-421 ± 246 against -152 ± 45).
    (4) Scale: rewards × 0.05 (-152 ± 45) match a running reward normaliser (-151 ± 45); × 0.1 gives -496 ± 374.
    Observation normalisation adds nothing measurable; annealing the entropy hurts.
  - **Decided (2026-10-18)**: the fallback criterion. Bounded mean, 8-step chunked updates with GAE 0.9 and
    bootstrapping (not strictly on-policy: later chunks were collected before the episode's first update), γ 0.9,
    lr 7e-4 annealed over 1,000,000 steps, no advantage normalisation, value coefficient 0.4, entropy 0.01, log
    std -2, rewards × 0.05. Harness result: evaluation -152 ± 45 (rl-zoo A2C -163), beating random on 5 of 5 seeds;
    the training average stays at -286 to -341 and does not reach -200. What might close the gap (not attempted):
    true A2C with updates every n steps, which needs the trainer to call `update` mid-episode for every agent; gSDE;
    a longer budget.
  - **Learning gate (2026-10-18)**: the real implementation with the committed config, benchmark seeds 1 to 5,
    1,000,000 steps: evaluation -152 ± 45 across seed means (-136, -157, -213, -78, -175), beating the random agent
    on 5 of 5; 0 of 5 solve (final 100-episode averages -286 to -341). Identical to the scratch harness seed by seed,
    and the default settings reproduce the previous weights of both continuous agents bit for bit.
  - **Out of scope**: Turning it into PPO; DDPG.

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

- [x] **Break the Benchmark Import Cycle**
  - **Delivered in**: merged in #81.
  - **Description**: `src/benchmark.py` imports `rl_cartpole.benchmark` and `rl_pendulum.benchmark` inside functions
    in two places, both import `benchmark` back, and a `sys.modules.setdefault("benchmark", ...)` alias holds the
    cycle together. A third challenge means editing the runner twice.
  - **Acceptance criteria**:
    - `BenchmarkChallenge` and the challenge registry live in `rl_core`; challenge packages import only `rl_core`.
    - The runner discovers registered challenges without naming any of them, and the `sys.modules` alias is gone.
    - `python src/benchmark.py` and the tests in `tests/test_benchmark.py` behave as before.
  - **Known edge cases**: Running `src/benchmark.py` directly as `__main__`; a challenge package that fails to import.
  - **Out of scope**: Changing benchmark metrics or the report format.

- [x] **Select the Challenge Through the Registry**
  - **Delivered in**: `feature/challenge-registry-cli` (stacked on #86). Confirmed by the maintainer on 2026-10-16.
  - **Description**: `train.py` and `record.py` choose the environment with `"pendulum" in env_name.lower()` in three
    places and hardcode `choices=["cartpole", "pendulum"]`, so an unknown challenge silently becomes CartPole.
  - **Acceptance criteria**:
    - Environment and agent construction go through the challenge registry; no string matching on the name.
    - `--challenge` choices come from the registry, and an unknown environment name is an error naming the options.
    - Adding a challenge requires no edit to `train.py` or `record.py`.
  - **Known edge cases**: Configs with no `environment.name`; `--challenge` disagreeing with the config.
  - **Built (2026-10-16)**: `BenchmarkChallenge` gains abstract `make_env` and `make_agent` and a concrete
    `setup_run` built from them; each challenge's `make_env` calls its package's `make_env_from_config`, where a YAML
    `null` now takes the default (`rl_core.utils.config.config_get`). `rl_core.challenges.select_challenge` picks the
    challenge whose environment id equals `environment.name` exactly, or the one `--challenge` names; it rejects an
    unknown id (naming the options), a `--challenge` that disagrees, a config with neither, and an id shared by two
    challenges without `--challenge`. `train.py` lost `AGENT_CLASSES`, `CONTINUOUS_AGENT_CLASSES`, `create_env` and
    `create_agent`; its `--challenge` and `--agent-type` choices come from the registry, and `record.py` reuses the
    same helpers. Seeded runs of all eight committed configs give identical metrics and checkpoints before and after.
  - **Behaviour changes**: A config without `environment.name` needs `--challenge` (it used to default to
    CartPole); ids the wrappers accept but no challenge registers, such as `CartPole-v0`, are rejected; an agent the
    challenge does not support fails before any output folder is created.
  - **Out of scope**: Merging the discrete and continuous agent maps (each challenge keeps its own, and the scripts
    no longer hold copies); moving the shared `make_agent` steps into the base class.

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
  - **Progress (2026-10-16)**: DQN now saves and restores its target-sync counter, its exploration step and its
    Adam state, with a resume test (`tests/test_dqn_options.py`); the continuous agents remain.
  - **Acceptance criteria**:
    - Everything an agent needs to resume training round-trips through `save` and `load`.
    - A test trains, saves, loads into a fresh agent, and gets the same weights after one more seeded update.
    - Checkpoints written before the change still load.
  - **Out of scope**: A resume option for `train.py`.

- [ ] **PPO Episode Ends and Validation**
  - **Description**: Found 2026-10-18 while giving continuous PPO multi-episode rollouts. Discrete `PPOAgent`'s
    rollouts end every episode's GAE trace without a bootstrap, so CartPole's 500-step time limit counts as a true
    end; continuous PPO adds γ·V(next state) to the last reward of a cut episode. Discrete PPO also reads
    `rollout_steps` with `int(...)`, so 2.5 becomes 2, where continuous PPO rejects it.
  - **Acceptance criteria**:
    - Discrete PPO bootstraps a time-limit cut like continuous PPO, with a hand-computed test; a true termination
      does not bootstrap.
    - `rollout_steps` is validated the same way in both PPO agents.
    - CartPole PPO is rerun on seeds 1 to 5 and still meets its published claim (5 of 5 solved).
  - **Out of scope**: Sharing one rollout buffer between the two PPO agents (until a third user appears).

- [ ] **DQN Config Validation**
  - **Description**: Listed under "CartPole Agents Learn Within Their Budgets" → DQN behaviours and never given
    its own item. Not validated at construction: `target_update_frequency: 0` raises `ZeroDivisionError` during
    training; `tau` of 1 or more silently falls back to hard sync, 0 freezes the target network and a negative
    value extrapolates; `min_buffer_size` below `batch_size` fails at the first sample; a `buffer_capacity` below
    `batch_size` can leave the agent never training. While the buffer warms up, `update` returns a loss and mean
    Q-value of 0, which the training log averages in.
  - **Acceptance criteria**:
    - Each invalid setting raises a `ValueError` naming the key at construction, with a test per case.
    - During warm-up `update` returns no loss or Q-value (as DDPG does), so the log averages only real updates.
    - The committed DQN config and its seeded results are unchanged.
  - **Out of scope**: Changing DQN's defaults.

- [ ] **Quiet Video Recorder Cleanup in Tests**
  - **Description**: `TrainingVideoRecorder` validates its arguments before calling Gymnasium's `RecordVideo`
    constructor, so a rejected recorder has no `recorded_frames` and Gymnasium's `__del__` raises an
    `AttributeError` that pytest reports as an unraisable-exception warning (`tests/test_video.py`). Harmless, but it
    hides real warnings in the test output.
  - **Acceptance criteria**: The test suite prints no unraisable-exception warning from the recorder; invalid
    arguments still raise the same errors.
  - **Out of scope**: Other third-party deprecation warnings.

- [ ] **Ideas: Close the Actor-Critic Gap; Larger Networks on a GPU**
  - **Description**: Lower-priority ideas recorded 2026-10-18, to be turned into items when picked.
    - The continuous Actor-Critic's greedy policy matches rl-zoo's A2C, but its 100-episode training average stays
      at about -300 ("Continuous Actor-Critic Learns Pendulum-v1"). True A2C updates every n steps, which needs the
      trainer to call `update` mid-episode for every agent; gSDE exploration is the other untried difference.
    - The maintainer expects to use the Jetson's GPU later, so PyTorch should be allowed then. That would make
      rl-zoo's 400×300 DDPG networks practical (about 16 times slower than 64×64 in NumPy here) and pixel-based or
      larger tasks possible. It would be an optional backend beside the NumPy agents, with the NumPy versions kept
      as the gradient reference, and needs the pure-NumPy rule in CLAUDE.md and CONTRIBUTING.md updated.

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

- [x] **Benchmark Budgets and Solve Rule in Environment Steps**
  - **Delivered in**: #85 (step budgets, 100-episode window, Held) and the follow-up PR with the approved budgets
    and both republished reports.
  - **Description**: Budgets are counted in episodes, and CartPole episodes run from about 10 to 500 steps, so an
    episode count says little about how much an agent trained; the reference setups everyone compares against
    (rl-baselines3-zoo) set budgets in environment steps. A run also counts as solved on a 10-episode average the
    first time it touches the threshold, where the usual CartPole convention averages 100 episodes, and nothing reports
    whether the agent held it. Measured 2026-10-15 (committed configs, seeds 1 to 5, steps to first solve): CartPole PPO
    35k to 86k (rl-zoo budget 100k), DQN 73k to 117k with one seed never (rl-zoo 50k), Actor-Critic 78k to 174k (rl-zoo
    A2C 500k), REINFORCE 157k to 247k (no rl-zoo entry); Pendulum PPO 42k to 64k (rl-zoo 100k), Actor-Critic never in
    60k (rl-zoo A2C 1,000,000).
  - **Acceptance criteria**:
    - Training can be budgeted in environment steps; the benchmark reports steps to solve beside episodes.
    - The solve window is 100 episodes by default, and the report has a "held" measure (for example the final
      100-episode training average at or above the threshold), with the rule written in the report.
    - Each agent's budget is chosen against its rl-zoo reference where one exists, recorded in the config comment;
      the values are the maintainer's to approve.
    - Both published reports and the README solve table are regenerated under the new rule.
  - **Known edge cases**: An episode that crosses the step budget (finish it, or cut it); evaluation episodes are not
    counted as training steps; agents that train per episode keep doing so.
  - **Out of scope**: Changing thresholds; tuning agents that fall short (DQN has its own item).
  - **Results** (2026-10-16, committed configs, seeds 1 to 5, 100-episode window): CartPole PPO 200k steps solves
    5 of 5 (73k to 199k steps), holds 4 of 5; Actor-Critic 500k solves 5 of 5 (184k to 465k), holds 4 of 5;
    REINFORCE 500k solves 4 of 5 (242k to 446k), holds 2 of 5; DQN 50k solves 0 of 5 (final averages 247 to 312).
    Pendulum PPO 100k solves 0 of 5 (final -241 to -313); Actor-Critic 1M solves 0 of 5, evaluation -1422 ± 134
    against the random agent's -1225. The default configs' budgets (50k CartPole, 40k Pendulum) only set the
    random baseline's length.

- [x] **DQN Reaches the rl-zoo Reference**
  - **Description**: Under the old 10-episode rule (2026-10-15, 750 episodes) CartPole DQN needed 73k to 117k steps to
    first reach 475 and one seed of five never did, against rl-baselines3-zoo's 50k-step budget. rl-zoo's DQN uses a
    256×256 network, 128 gradient steps every 256 environment steps, a 1000-step warm-up, a target update every 10 steps
    and exploration decayed over the first 16% of training. Ours: one 128-unit hidden layer, at most 64 gradient steps
    per episode, epsilon decayed per episode, time-limit cuts treated as terminal (see "CartPole Agents Learn Within
    Their Budgets" → DQN behaviours). Under the 100-episode rule at 50k steps (2026-10-16) it solves 0 of 5: final
    averages 247 to 312, evaluation 368 ± 94.
  - **Acceptance criteria**:
    - A diagnosis names what separates our DQN from the reference, each cause checked.
    - Fixes are test-first; config changes are measured and maintainer-approved.
    - Under the 100-episode rule, DQN solves CartPole-v1 on at least 4 of 5 seeds within its config's step budget.
      (Reworded 2026-10-16 with the maintainer: 50k steps cannot satisfy the rule, see the diagnosis; rl-zoo's 500 at
      50k is a greedy evaluation after training, from one run.)
  - **Diagnosis (2026-10-16)**: seeds 1 to 5, CartPole, benchmark solve rule, causes checked one at a time by patching
    `DQNAgent` in a scratch script.
    - The 50k budget cannot satisfy the rule: 100 episodes averaging 475 take at least 47,500 steps, so the agent would
      have to be near-perfect by step 2,500, while rl-zoo itself explores until step 8,000.
    - Exploration decays ×0.995 per episode: epsilon is 0.50 at episode 150 and 0.19 at the end of 50k steps (seed 1).
      rl-zoo decays linearly per environment step to 0.04 over the first 8,000.
    - Time-limit cuts are stored as terminal, zeroing the bootstrap at step 500; rl-zoo bootstraps through them.
    - Learning is unstable with plain SGD at 0.005 and a target sync every 50 gradient steps: the mean Q-value reaches
      146 to 190 against a true maximum of 100, and doubling the gradient steps sends it to about 970.
    - Each alone does not explain it: a 256-unit layer, a 1,000-step warm-up and a 100k buffer change little; Huber
      loss with SGD diverges. rl-zoo's full setup (two 256-unit layers, Adam, Huber, both fixes) at 50k reaches a
      greedy evaluation of 475 on 3 of 5 seeds. `max_updates_per_call: 64` gives about 0.13 gradient steps per
      transition on 500-step episodes against rl-zoo's 0.5; not tuned here.
  - **Measured at 200k steps unless stated** (solved / held at the end; final 100-episode average; greedy evaluation):
    committed config 1 / 0, 431, 407 ± 57; both fixes with SGD 0.005, 0 / 0, 333; both fixes with Adam (global-norm
    clip 10) at 0.001 and a target sync every 128 steps, 3 / 3, 462, 500 ± 0; the same at 0.0007, 4 / 4 (steps 142k
    to 179k), 487, 500 ± 1; the same at 0.001 for 300k steps, 5 / 5 (steps 168k to 238k), 498, 500 ± 0.
  - **Decided (2026-10-16)**: keep the 100-episode rule for every agent (option B); fix exploration per step and
    time-limit bootstrapping; add the shared `optimizer` option to DQN. Committed config: Adam, learning rate 0.001,
    target sync every 128 gradient steps, epsilon to 0.04 over 8,000 steps, 300,000 steps.
  - **Plan**: two stacked PRs. 1: `exploration_steps`, bootstrapping through time-limit cuts, `optimizer` (with the
    element-wise clip renamed `grad_clip_value`), and complete DQN checkpoints. 2: the config and republished numbers.
  - **Delivered in**: #88 (agent options), then the config PR stacked on it. Confirmed by the maintainer on
    2026-10-16.
  - **Learning gate (2026-10-16)**: committed config, the real implementation, benchmark seeds 1 to 5: 5 of 5 solve
    (steps 145,401 to 238,149, episodes 548 to 1167) and hold, final averages 496 to 500, evaluation 500 ± 0. The full
    CartPole benchmark was then rerun on seeds 1 to 5 (2026-10-16): every agent's numbers matched the earlier run,
    only the timings changed, and `docs/benchmarks/cartpole_results.md` and its plot come from that one run.
  - **Out of scope**: Double, dueling and prioritised-replay DQN (each its own item if wanted); DDPG; a second hidden
    layer.

- [x] **Continuous Off-Policy Learning & DDPG (Deep Deterministic Policy Gradient)**
  - **Delivered in**: #90, #91 and #92. Confirmed by the maintainer on 2026-10-18.
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
    - Reference configuration `configs/pendulum_ddpg.yaml` solving `Pendulum-v1` under the 100-episode rule
      (average training reward $\ge -200$ over 100 episodes) on at least 4 of 5 seeds within its step budget, with
      the greedy evaluation reported against rl-zoo's -152 ± 94. (Reworded 2026-10-16 with the maintainer: "in
      ≤ 100 episodes" would need the first 100 episodes, including 50 of random warm-up, to average -200.)
    - CLI dispatch support in `train.py`, `record.py`, and `src/benchmark.py`.
    - Unit and integration tests in `tests/test_ddpg_agent.py` achieving $>90\%$ coverage.
  - **Known edge cases**: Q-value overestimation; divergence from non-smooth actor gradients; torque saturation.
  - **Plan (2026-10-16)**: rl-zoo's DDPG for Pendulum-v1 (SB3 DDPG: tanh actor scaled to the bounds, critic on
    `[s, a]`, two 400×300 ReLU layers, Adam 1e-3, batch 256, γ 0.98, τ 0.005, Gaussian noise 0.1 in scaled action
    space, 10,000 uniform warm-up steps, bootstrapping through time-limit cuts). Three stacked PRs: 1, `ReplayBuffer`
    with continuous actions (`action_dim`); 2, `DDPGAgent` with a shared MLP helper (`rl_core/agents/mlp.py`), a
    shared `action_bounds` helper, prefixed Adam checkpoint keys, a continuous agent contract test and named random
    streams; 3, the config, a timed network size, the learning gate and docs.
  - **Progress (2026-10-16)**: PR 1 (#90) stores continuous actions. PR 2 adds `DDPGAgent` with finite-difference
    checks of the critic, the actor (through the critic's action input) and the shared MLP, hand-computed losses,
    targets and Polyak updates, a replay of one `update`, and a continuous contract test for every continuous agent.
    Not yet registered with the Pendulum challenge and not yet shown to learn: that is PR 3.
  - **Network size (2026-10-16)**: one Pendulum episode (200 gradient steps, batch 256) takes 19.6 s with
    rl-zoo's 400×300 networks, 11.0 s with 256×256 and 1.2 s with 64×64 in NumPy on this machine; 400×300 would
    need about 1.6 hours per seed. With 64×64 and rl-zoo's other settings, seeds 1 to 5 all solve at 31,000 to
    31,600 steps and hold, at 60,000 and at 40,000 steps. Decided with the maintainer: 64×64, 40,000 steps (twice
    rl-zoo's 20,000); 400×300 not measured. A GPU backend (PyTorch) for larger networks is a later idea.
  - **Learning gate (2026-10-16)**: committed config, full Pendulum benchmark on seeds 1 to 5: 5 of 5 solve (steps
    31,000 to 31,600, episodes 155 to 158) and hold, final averages -146 to -168, evaluation averaging -150 (± 27
    across seed means) against rl-zoo's -152 (± 94 across episodes of one run); the other agents' rows are
    unchanged. Seed 42 with `train.py`: final average -159, and its recorded video averages -196 over 5 greedy
    episodes.
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
