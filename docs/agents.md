# How Each Agent Works

A short technical guide to every agent in `packages/rl-core/src/rl_core/agents/`: the idea, the update as this code
computes it, what this repo's version adds or leaves out, what was measured, and what was learned the hard way. It is
kept current: a new agent adds its section, and a change to an agent's update, options, committed config or measured
result updates its section, in the same pull request (the `doc-sync` skill).

Terms used throughout:

- **Solved** means the training reward averaged over 100 consecutive episodes reached the challenge's threshold (475 on
  CartPole-v1, -200 on Pendulum-v1, -100 on Acrobot-v1, -110 on MountainCar-v0, 200 on LunarLander, 90 on
  MountainCarContinuous, 300 on BipedalWalker); **held** means the last 100 episodes still average at or above it.
  **Greedy evaluation** plays the policy without exploration on a separate environment.
- **Terminated** is a true end (the pole fell, the lander crashed); **truncated** is a time-limit cut. A value target
  bootstraps from the next state after a cut and not after a termination, unless a section says otherwise.
- Results come from `benchmark.py` on seeds 1 to 5 unless they say otherwise; the dated `(CHANGELOG.md, <date>)`
  entry has the full numbers.

| Agent | Actions | Learns | Committed configs |
| :--- | :--- | :--- | :--- |
| REINFORCE | discrete | policy, from whole-episode returns | CartPole |
| Actor-Critic | discrete, continuous | policy and state value | CartPole, Pendulum |
| PPO | discrete, continuous | policy and state value, clipped reuse of rollouts | CartPole, Pendulum, Acrobot, Navigation (both) |
| DQN | discrete | action value, from replay | CartPole, Acrobot, MountainCar, LunarLander, Navigation |
| DDPG | continuous | deterministic policy and action value, from replay | Pendulum, MountainCarContinuous |
| TD3 | continuous | as DDPG, twin critics and delayed actor | Pendulum, MountainCarContinuous, LunarLanderContinuous, BipedalWalker, NavigationContinuous |
| SAC | continuous | stochastic policy, twin action values and an entropy temperature, from replay | Pendulum, LunarLanderContinuous, BipedalWalker |

## The agent classes (`BaseAgent` and the implemented agents)

The `BaseAgent` abstract class defines the interface for all RL agents:

- `select_action()`: Choose actions based on observations
- `update()`: Update policy based on collected experience
- `_checkpoint_arrays()` / `_load_checkpoint_arrays(data)`: what a checkpoint holds; `BaseAgent.save()` and
  `load()` write and read it, with every generator. Before assigning anything, `load()` checks every entry the agent
  itself saves against its own shape (`_check_checkpoint_arrays`), so a checkpoint saved with another `hidden_dim`,
  `hidden_sizes`, `observation_dim` or `action_dim` raises a ValueError naming the entry and both shapes and leaves
  the agent as it was. A missing entry is refused the same way ("is not a DQNAgent checkpoint: it has no target_W2"),
  except those an older format or another setting did not save, which each agent lists in
  `_OPTIONAL_CHECKPOINT_PREFIXES` (the Adam state of an SGD or older continuous file, DQN's counters, any replay buffer)

Implemented agents (in `packages/rl-core/src/rl_core/agents/`; the sections below say how each one works, its update and
what it measured):

- **Discrete Control Agents**:
  - `RandomAgent`: Random action baseline policy
  - `ReinforceAgent`: Monte Carlo policy-gradient agent
  - `ActorCriticAgent`: REINFORCE with a learned value baseline (policy + value network; not A2C)
  - `PPOAgent`: Clipped-policy Proximal Policy Optimization agent
  - `DQNAgent`: Deep Q-Network with replay buffer and target network; double DQN targets (`double_q: true`) and a
    dueling network (`dueling: true`) and a Huber TD loss (`td_loss: huber`) are opt-in
- **Continuous Control Agents**:
  - `ContinuousRandomAgent`: Uniform random continuous torque baseline policy
  - `ContinuousActorCriticAgent`: Continuous Gaussian policy A2C with state-value critic
  - `ContinuousPPOAgent`: Continuous Gaussian policy PPO with clipped surrogate objective
  - `DDPGAgent`: Deep Deterministic Policy Gradient with replay, a Q critic and Polyak target networks; it explores
    with Gaussian noise or, with `noise_type: ornstein_uhlenbeck`, temporally correlated Ornstein-Uhlenbeck noise
    (`rl_core/agents/exploration_noise.py`)
  - `TD3Agent`: Twin Delayed DDPG: two critics with the smaller target, smoothed target actions and delayed actor
    and target updates (`configs/pendulum_td3.yaml`)
  - `SACAgent`: Soft Actor-Critic: a squashed Gaussian actor, twin critics with Polyak targets and a learned
    entropy temperature (`ent_coef`, `target_entropy`; `configs/pendulum_sac.yaml`, and on the PyTorch backend
    `configs/lunarlander_continuous_sac.yaml` and `configs/bipedalwalker_sac.yaml`)

## REINFORCE (`ReinforceAgent`, `rl_core/agents/reinforce_agent.py`)

**Idea.** Learns a softmax policy directly from sampled episodes (Williams 1992): after each episode it raises the
probability of actions followed by above-average return and lowers the others. No value function is needed. The
estimate is unbiased but noisy, since every action is credited with everything that happened after it.

**Update.** Once per finished episode, one optimizer step:

```text
G_t    = r_t + gamma * G_{t+1}                    (Monte Carlo return; the episode's last step ends it)
Ghat_t = (G_t - mean(G)) / std(G)                 (a constant baseline; mean only if std <= 1e-8)
L      = -(1/T) * sum_t Ghat_t * log pi(a_t | s_t)    (Ghat held constant)
```

A time-limit cut is not bootstrapped: the returns simply stop at the last step.

**In this repo.** `obs -> hidden_dim (ReLU) -> action_dim -> softmax` (`hidden_dim` 128), sampled in training and
argmax in evaluation. `optimizer` defaults to plain SGD; `adam` and `max_grad_norm` are options; there is no entropy
term and no annealing. `normalize_observations` (running mean and variance of the inputs) is available to every agent.

**What was measured.** `configs/cartpole_reinforce.yaml` (Adam, 500,000 steps): 4 of 5 seeds solve and 3 hold;
greedy evaluation 497 ± 5 (`docs/benchmarks/cartpole_results.md`).

**Lessons.** Plain SGD solved 0 of 5 seeds in 2,000 episodes where Adam solved 5 of 5 (CHANGELOG.md, 2026-10-15).
Seeding moved to named random streams per concern after a seed was found sharing numbers with Gymnasium's
`reset(seed=)`, and the benchmarks were republished (CHANGELOG.md, 2026-10-19).

## Actor-Critic, discrete (`ActorCriticAgent`, `rl_core/agents/actor_critic_agent.py`)

**Idea.** REINFORCE plus a learned critic V(s), subtracted from the return as a state-dependent baseline, which cuts
the gradient's variance far more than a single mean. Actor and critic share a hidden layer.

**Update.** Once per episode, one optimizer step on the episode-averaged gradient:

```text
G_t = Monte Carlo return (no GAE);   A_t = G_t - V(s_t)   (held constant in the actor term)
L   = (1/T) * sum_t [ -A_t * log pi(a_t|s_t) + c_v * 0.5 * (V(s_t) - G_t)^2 - c_e * H(pi(.|s_t)) ]
```

`c_v` is `value_coef` (0.5) and `c_e` is `entropy_coef` (0.01); advantages are not normalised. The critic term
carries a 1/2, as in every loss below, where SB3's `vf_coef` multiplies the plain mean squared error: `value_coef`
0.5 here equals SB3 `vf_coef` 0.25. By default, like REINFORCE, it treats a time-limit cut as the end of the episode.
With `bootstrap_truncation: true` (default false; on in `configs/cartpole_actor_critic.yaml`) a cut episode's
returns become G_t + gamma^(T-t) * V(s_T), with V taken at the true next state `next_observations[-1]` and held
constant, as PPO does; a termination is unchanged.

**In this repo.** A shared `obs -> hidden_dim (ReLU)` trunk (`ActorCriticBase`) with a softmax policy head and a
scalar value head; the hand-written backward pass meets in the shared layer. Only the policy is exported.

**What was measured.** `configs/cartpole_actor_critic.yaml` (Adam, `bootstrap_truncation`, 500,000 steps): 5 of 5
seeds solve, at 156k to 282k steps, and 5 hold; greedy evaluation 500 (`docs/benchmarks/cartpole_results.md`). Without
the bootstrap, on the same seeds, they solved at 209k to 456k steps and 4 held (CHANGELOG.md, 2026-11-12).

**Lessons.** Adam reached 475 at episodes 490 to 839 where SGD needed 906 to 1,269 (CHANGELOG.md, 2026-10-15). A
"collapse" in a test was the test's fault: a fixed batch kept an action the policy had stopped choosing, so the
-A·log π term had no lower bound. Tests now sample each batch from the current policy (CHANGELOG.md, 2026-10-29).

## Actor-Critic, continuous (`ContinuousActorCriticAgent`, `rl_core/agents/continuous_actor_critic.py`)

**Idea.** The same actor-critic with a diagonal Gaussian policy for continuous actions (Pendulum's torque): the
network gives a state-dependent mean, and a separate learned log standard deviation sets the exploration noise.
With `n_steps` it learns from short bootstrapped chunks, as A2C does.

**Update.** With `n_steps: 0`, once per episode on Monte Carlo returns with normalised advantages; a time-limit cut is
then treated as an end unless `bootstrap_truncation: true` adds gamma^(T-t) * V(s_T) to the returns, after
`reward_scale`. With `n_steps: N` the episode is replayed in chunks of N steps, one Adam step per chunk:

```text
A = gae(reward_scale * r, V, gamma, lambda, V(next state)),   R = A + V    (A, R held constant)
L = -mean(log N(a | mu, sigma^2) * A) + c_v * 0.5 * mean((V - R)^2) - c_e * H,   H = sum(log_std + 0.5*log(2*pi*e))
```

A chunk bootstraps from the next state unless the episode truly terminated. With `training.update_every` equal to
`n_steps` each chunk is trained on as it is collected (true A2C); any other value is refused.

**In this repo.** A `hidden_dim` ReLU trunk, a mean head and a value head, and a state-independent `log_std`
(started at `init_log_std`, kept within [-20, 2]). `bounded_mean: true` makes the mean `bias + scale * tanh(z)`
inside the action bounds. Training actions are the unclipped sample (the environment clips), so the
log-probability matches what was sampled; evaluation plays the mean. Always Adam with a global-norm clip
(`max_grad_norm` 0.5); `anneal_steps` decays the learning rate linearly to 0; `reward_scale` multiplies the rewards.

**What was measured.** `configs/pendulum_actor_critic.yaml` follows rl-zoo's A2C (`n_steps` 8, `gae_lambda` 0.9,
`gamma` 0.9, lr 7e-4 annealed over 1,000,000 steps, `init_log_std` -2, `reward_scale` 0.05) with observation
normalisation since 2026-11-04 and, since 2026-11-12, rl-zoo's 8 environments (`training.num_envs: 8`,
`update_every: 8`: one Adam step over 64 transitions). Paired against one environment on seeds 1 to 5, the final
training average is -244.1 against -282.7, better on all 5 seeds; two seeds reach -200 in a 100-episode window but
none holds it, and the greedy evaluation is worse (-184.0 ± 42.7 against -170.0 ± 25.9); it still beats the random
agent's -1,293 (CHANGELOG.md, 2026-11-10 and 2026-11-12; `docs/benchmarks/pendulum_results.md`).

**Lessons.** A bounded mean and 8-step chunks made it beat random (CHANGELOG.md, 2026-10-18). Without the entropy
bonus the log std collapses. True A2C (`update_every: 8`) and 50-step chunks did not close the gap to -200; rl-zoo's
8 parallel environments narrowed it but did not close it either (CHANGELOG.md, 2026-10-30, 2026-11-04 and
2026-11-10).

## PPO, discrete and continuous (`PPOAgent`, `ppo_agent.py`; `ContinuousPPOAgent`, `continuous_ppo.py`)

**Idea.** Proximal Policy Optimization (Schulman et al. 2017) reuses each rollout for several passes, and clips the
probability ratio pi/pi_old so that no pass moves the policy far from the one that collected the data. That makes it
far more sample-efficient and stable than a plain policy gradient. Both variants use the Actor-Critic networks above.

**Update.** Episodes are buffered until the rollout holds at least `rollout_steps` transitions, then:

```text
A_t = gae(r, V_old, gamma, lambda),   R_t = A_t + V_old(s_t)        (once per rollout, then held constant)
A_t <- (A_t - mean(A)) / (std(A) + 1e-8)
ratio_t = exp(log pi(a_t|s_t) - log pi_old(a_t|s_t))
L = -mean( min(ratio_t * A_t, clip(ratio_t, 1 - eps, 1 + eps) * A_t) ) + c_v * 0.5 * mean((V - R)^2) - c_e * mean(H)
```

repeated for `ppo_epochs` passes over shuffled mini-batches. Each episode ends its own GAE trace; a time-limit cut
adds gamma·V(next state) to its last reward, a termination does not. `anneal_steps` decays the learning rate and the
clip range together, linearly to 0.

**In this repo.** Discrete: softmax policy, `ppo_epochs` 4, `gae_lambda` 1.0 by default, SGD or Adam, `minibatch_size`
(the whole rollout by default). Continuous: the Gaussian policy above, always Adam, `ppo_epochs` 8, `gae_lambda` 0.95,
`batch_size` 64, `reward_scale`; a log-ratio beyond ±20 is treated as stale and gets no gradient. The checkpoint holds
the partly filled rollout and the annealing counter, so a resumed run continues exactly.

**What was measured.**

- CartPole, `configs/cartpole_ppo.yaml` (rl-zoo's settings on one environment: 256-step rollouts, 20 epochs, gamma
  0.98, lambda 0.8, Adam with annealing, 200,000 steps): 5 of 5 solve at 72k to 90k steps and hold; greedy 500.
- Pendulum, `configs/pendulum_ppo.yaml` (2,048-step rollouts, 10 epochs, gamma 0.9, `reward_scale` 0.1, 200,000
  steps): 5 of 5 solve and 4 hold; greedy evaluation -188 ± 35 (`docs/benchmarks/pendulum_results.md`).
- Acrobot, `configs/acrobot_ppo.yaml` (rl-zoo's settings, 500,000 steps): 5 of 5 solve and hold, greedy -86 ± 9
  (CHANGELOG.md, 2026-10-22).
- Navigation, `configs/navigation_ppo.yaml` and `navigation_continuous_ppo.yaml` (SB3's defaults, 2,048-step rollouts,
  200,000 steps; entropy 0.01 for the five discrete actions, 0.0 continuous): greedy evaluation over 100 held-out
  layouts reaches the goal 88% of the time with either action space (82 to 97% discrete, 82 to 91% continuous),
  rewards 6.75 ± 0.85 and 6.66 ± 0.43; about 5 minutes a seed with three runs sharing the CPU. Against the 7.0
  threshold both solve on 5 of 5 seeds but hold it on 1 (discrete) and 0 (continuous) (2026-11-07,
  `docs/benchmarks/navigation_results.md` and `navigation_continuous_results.md`).

**Lessons.**

- Pendulum PPO needed five fixes at once: the log-probability of the clipped action gave NaN, the initial mean sat
  outside the bounds, rewards were unscaled, the time limit was treated as terminal, and gamma 0.99 became 0.9
  (CHANGELOG.md, 2026-10-15). Multi-episode rollouts then took it from 0 of 5 to 5 of 5 (2026-10-18).
- Bootstrapping CartPole's 500-step cuts cut the mean steps to solve from 108,325 to 79,556 (2026-10-20).
- On CartPole with the velocities masked, PPO solved 0 of 5 until `normalize_observations` and `frame_deltas` were
  on, then 5 of 5 (2026-11-02). On Pendulum, normalisation made it worse (held 2 of 5 against 4): one seed learned and
  then slid to about -1,000, cause open (2026-11-04; roadmap, "Why Pendulum PPO collapses").

## DQN (`DQNAgent`, `rl_core/agents/dqn_agent.py`)

**Idea.** Learns the action value Q(s, a), the discounted return of taking a in s and then acting greedily, with a
small network trained off-policy by regression onto a one-step Bellman target. A replay buffer (random mini-batches
break the correlation between consecutive steps) and a slowly moving target network (the target does not chase the
weights being trained) keep that regression stable. It acts epsilon-greedily on Q.

**Update.** For a mini-batch of B transitions sampled uniformly from the replay buffer:

```text
y = r + gamma * (1 - terminated) * max_a' Q_target(s', a')                     (default)
y = r + gamma * (1 - terminated) * Q_target(s', argmax_a' Q_online(s', a'))    (double_q: true)
L = mean_B( 0.5 * (Q_online(s, a) - y)^2 )                                     (y held constant)
L = mean_B( h(Q_online(s, a) - y) )                                            (td_loss: huber)
h(e) = 0.5 * e^2 if |e| <= delta, else delta * (|e| - 0.5 * delta)   (gradient: clip(e, -delta, delta))
```

Only a true termination stops the bootstrap: a time-limit cut is stored as not terminal. The loss is squared error
by default; `td_loss: huber` switches to the Huber loss (below). The trainer calls `update` once per episode (or
every `training.update_every` steps, below). Each
call stores the new transitions and, once the buffer holds `min_buffer_size`, takes one gradient step, or
`max(1, int(n_stored * updates_per_step))` capped at `max_updates_per_call` (64) with `updates_per_step`, each on a
fresh mini-batch.

**In this repo.**

- Network: `obs -> hidden_dim (ReLU) -> action_dim`, He-initialised, with a target copy. `hidden_dim` defaults to
  128; the Acrobot, MountainCar and LunarLander configs use one layer of 256 where rl-zoo has two.
- Target network: a hard copy every `target_update_frequency` gradient steps (default 100; gradient steps, not
  environment steps), or with `tau` in (0, 1] a Polyak average after every step.
- Exploration, two exclusive schedules: `exploration_steps` (epsilon falls linearly per training action, from
  `epsilon_start` 1.0 to `epsilon_end` 0.02) or `epsilon_decay` (0.995, multiplied once per `update` call that
  trained, so once per episode). Evaluation is greedy and leaves epsilon alone.
- Opt-in options: `td_loss: huber` with `huber_delta` (default 1.0; must be finite and positive, and is refused
  without `td_loss: huber`): the Huber loss above, PyTorch's `huber_loss(delta)`, which at delta 1 equals SB3's
  `smooth_l1_loss`, the loss SB3 and rl-zoo use; the logged `loss` is then the Huber value, and a squared-error
  checkpoint loads into a Huber agent; `double_q` (targets above); `dueling` (a second `hidden_dim` value stream on
  the observation and `Q = V + A - mean_a A`, Wang et al. 2016; twice the hidden units, the same depth);
  `normalize_observations` (the buffer stores raw observations and normalises each sampled batch with the current
  statistics); `anneal_steps` (the step size decays linearly to 0 over that many transitions).
- Optimizer: plain SGD with an element-wise clip (`grad_clip_value` 10.0) by default; every committed config uses
  `optimizer: adam` with a global-norm clip (`max_grad_norm` 10.0).
- Checkpoint: both networks, epsilon and the step counters, the optimizer state, the replay buffer and every
  generator, so a resumed run continues exactly. A dueling checkpoint loads only into a dueling agent.

**What was measured** (through the benchmark, 100-episode rule):

- CartPole (`configs/cartpole_dqn.yaml`, Adam, 300,000 steps): solves and holds on 5 of 5 seeds, at 144k to 183k
  steps, greedy evaluation 500 (CHANGELOG.md, 2026-10-18 and 2026-10-19).
- Acrobot (`acrobot_dqn.yaml`, 100,000 steps): solves and holds on 5 of 5 (final -90 to -96, greedy -85 ± 6) with an
  epsilon floor of 0.02; rl-zoo's 0.1 held on 1 of 5 (CHANGELOG.md, 2026-10-22).
- MountainCar (`mountaincar_dqn.yaml`, 200,000 steps, a 100,000 buffer, a decaying rate and observation
  normalisation): reaches -110 on 10 of seeds 1 to 5 and 11 to 15 but holds it on 3; final averages -106 to -119,
  greedy -110 ± 7 (CHANGELOG.md, 2026-11-04). It learns the task but does not hold the threshold reliably.
- LunarLander (`lunarlander_dqn.yaml`, 300,000 steps, epsilon floor 0.1): reaches 200 and holds it on 5 of seeds 1 to
  5 and 4 of 11 to 15, at 152k to 297k steps (CHANGELOG.md, 2026-10-27).
- Navigation (`navigation_dqn.yaml`, LunarLander's settings, 300,000 steps): with squared error and a fixed rate it
  failed (2026-11-06): greedy evaluation over 100 held-out layouts reached the goal 11% of the time (0% on three
  seeds, 50% on the best; counted by a scratch harness), reward -0.58 ± 2.61, where PPO reaches 88% with 200,000
  steps; the greedy policy mostly drove until the 400-step limit. The training-doctor (2026-11-06) found two stacked
  causes. First, the squared-error loss: a collision (-10) is about 200 times a step's reward std (0.046), so the
  rare collisions dominated the gradient, Q was overestimated (mean Q 12 to 18 where no return exceeds about 10) and
  the robot parked and dithered until the timeout. Second, a fixed learning rate on tiny action gaps (about 0.07):
  the greedy policy kept flipping. The fix is `td_loss: huber` with `anneal_steps: 300000` (the rate decays to 0
  over the whole budget). Result on seeds 1 to 5: 94% at the goal over 100 held-out layouts (89 to 96% by seed),
  against 0 to 50% before; each change alone falls short (Huber with a fixed rate 0 to 18%, mean 4%; squared error
  with the decaying rate 14 to 86%, mean 64%). Double DQN, dueling and an epsilon floor of 0.02 did not help
  (seed 1) (CHANGELOG.md, 2026-11-07). It solves and holds the 7.0 threshold on 5 of 5 seeds
  (`docs/benchmarks/navigation_results.md`).
- Double DQN and dueling stay opt-in: neither won on any of the four configs (CHANGELOG.md, 2026-10-29). The
  committed configs barely overestimate (logged mean Q within about 4 of the policy's own discounted return).

**Lessons.**

- Walk the numbers of a schedule through a real run (`.claude/skills/dev-loop/SKILL.md`; 0.995^500 ≈ 0.08):
  `epsilon_decay: 0.995` "per update" was applied once per episode, so in a 500-episode run epsilon ended near 0.08
  and never reached its floor. Read as per gradient step it would have ended exploration within 40 episodes. Hence
  `exploration_steps`.
- A small buffer forgets (2026-10-25): on MountainCar a 10,000-transition buffer holds about 85 episodes, and a good
  greedy policy found mid-run drifted away; 100,000 helped the training average.
- Huber helps when one rare reward is far larger than the typical step reward (Navigation's -10 collision against a
  0.046 step std); on MountainCar, where every step pays -1, it was among the add-ons that did not help (2026-10-23).
  Pair it with learning-rate annealing when the gaps between actions' values are small, or the greedy policy keeps
  flipping.
- Plausible add-ons often lose against a tuned config: n-step returns, two layers, Huber loss and scaled inputs
  on MountainCar (2026-10-23), prioritised replay on MountainCar and LunarLander (2026-10-27, 2026-11-04). Compare
  paired by seed, then confirm on held-out seeds.

## Replay buffer (`ReplayBuffer`, `rl_core/agents/replay_buffer.py`)

A fixed-capacity ring buffer of (observation, action, reward, next observation, done) in pre-allocated NumPy arrays,
shared by DQN, DDPG, TD3 and SAC. When full it overwrites the oldest transition; `sample` draws indices uniformly
without replacement, on its own named stream of the seed. The stored `done` means "do not bootstrap": `add_transitions`
takes the trainer's `terminated` flags, so a time-limit cut stays non-terminal. Actions are int64 (discrete) or float64
vectors (`action_dim`). With an agent's observation statistics it stores raw observations and normalises each sampled
batch (CHANGELOG.md, 2026-11-04). The checkpoint holds the filled part, the write index and the generator state, at
about 40 bytes a transition compressed (CHANGELOG.md, 2026-10-21).

## Updating during the episode (`MidEpisodeUpdates`, `rl_core/agents/mid_episode.py`)

By default the trainer hands an off-policy agent its transitions once per episode. With `training.update_every: N` it
calls `update` every N steps and once more at the episode's end, as SB3's `train_freq` does, so the data comes from a
policy updated moments before (CHANGELOG.md, 2026-10-30). The gradient steps owed are counted over all transitions,
`floor((counted + n) * updates_per_step) - floor(counted * updates_per_step)`, so the total does not depend on how the
episode is cut. DQN, DDPG, TD3 and SAC accept it (DQN needs `updates_per_step` and `exploration_steps`); REINFORCE and
the discrete Actor-Critic train on whole episodes and refuse it; the continuous Actor-Critic accepts it only with
`update_every` equal to its `n_steps` (true A2C); PPO accepts it only with several training environments
(`training.num_envs`), where every rollout is whole windows of `update_every` steps from each (docs/usage.md,
"Several training environments").

## Random agents (`RandomAgent`, `ContinuousRandomAgent`)

The floor every learner is compared against: a uniform action (an integer below `action_dim`, or a vector inside
the action bounds), drawn on their own streams, and no learning (`update` returns `{}`). Floors through the benchmark:
CartPole about 23, Pendulum -1293, Acrobot -500, MountainCar -200, LunarLander -178 to -194, Navigation about -11
with 1% of episodes at the goal (`docs/benchmarks/` and CHANGELOG.md).

## DDPG (`DDPGAgent`, `rl_core/agents/ddpg_agent.py`)

**Idea.** Deep Deterministic Policy Gradient (Lillicrap et al. 2016) learns a deterministic actor mu(s) that outputs
one action and a critic Q(s, a) that scores it. The critic is trained by bootstrapped regression on replayed
transitions; the actor climbs the critic along dQ/da. Replacing DQN's max over actions with the target actor's action
is what makes off-policy learning work for continuous actions.

**Update.** One gradient step on a replayed mini-batch, in this order:

```text
y      = r + gamma * (1 - terminated) * Q'(s', mu'(s'))        (y held constant)
L_crit = mean( 0.5 * (Q(s, a) - y)^2 )
L_act  = -mean( Q(s, mu(s)) )                                    (stepped after the critic, critic held constant)
target <- tau * online + (1 - tau) * target                      (Polyak, actor and critic, every step)
mu(s)  = bias + scale * tanh(net(s)),  scale = (high - low) / 2, bias = (high + low) / 2
```

The first `learning_starts` training actions are uniform in the bounds; after that `update` takes `updates_per_step`
gradient steps per stored transition (1.0). A time-limit cut bootstraps.

**In this repo.** ReLU networks of `hidden_sizes` ([400, 300] by default, rl-zoo's; the committed configs use [64, 64],
since 400×300 is about 16 times slower in NumPy). The critic takes `[obs, action]`. Adam only, one per network.
Exploration is added in the scaled [-1, 1] space: Gaussian (`noise_std` 0.1) or, with `noise_type: ornstein_uhlenbeck`,
SB3's OU process (`ou_theta` 0.15, `ou_dt` 0.01), whose state decays by only 0.15% a step, so a push persists like a
held throttle; it resets at each episode's end. Evaluation plays mu(s). The checkpoint holds the four networks, both
Adams, the counters and the replay buffer. The replay buffer, warm-up, update scheduling, critic loss, Polyak averaging
and checkpoints live in `OffPolicyAgent` (`packages/rl-core/src/rl_core/agents/off_policy.py`), which TD3 shares; DDPG
adds the actor, its target and the exploration noise.

**What was measured.**

- Pendulum, `configs/pendulum_ddpg.yaml` (64×64, 40,000 steps): 5 of 5 solve at about 31,200 steps and hold; greedy
  -158 ± 25 against rl-zoo's -152 (CHANGELOG.md, 2026-10-19).
- MountainCarContinuous, `configs/mountaincar_continuous_ddpg.yaml` (OU noise 0.5, 64×64, 300,000 steps): 5 of 5 reach
  90 at 25k to 69k steps and hold; greedy 93.4 ± 0.2 against rl-zoo's 93.5 (CHANGELOG.md, 2026-10-23).

**Lessons.** Exploration noise must fit the task: on MountainCarContinuous, Gaussian noise of the same 0.5 solved 0 of
5, four seeds learning to stand still; temporally correlated OU noise solved 5 of 5 (CHANGELOG.md, 2026-10-23).

## TD3 (`TD3Agent`, `rl_core/agents/td3_agent.py`)

**Idea.** Twin Delayed DDPG (Fujimoto et al. 2018) fixes DDPG's overestimation and brittleness three ways: two
critics with the smaller target taken, clipped noise on the target action so the critic cannot be exploited at a
sharp peak, and an actor (and targets) updated only every `policy_delay` critic steps.

**Update.** With n counting gradient steps over the agent's life:

```text
eps  = clip(N(0, target_policy_noise^2), -target_noise_clip, target_noise_clip)      (scaled space, as SB3)
a'   = clip(mu'(s') + scale * eps, low, high)
y    = r + gamma * (1 - terminated) * min(Q1'(s', a'), Q2'(s', a'))                  (held constant)
L_Qi = mean( 0.5 * (Qi(s, a) - y)^2 ),  i = 1, 2                                      (every step)
every policy_delay steps:  L_act = -mean(Q1(s, mu(s))), then Polyak on the three targets
```

**In this repo.** A subclass of `DDPGAgent` (exploration is DDPG's; warm-up and `update` are `OffPolicyAgent`'s).
Defaults are SB3's: `policy_delay` 2, `target_policy_noise` 0.2, `target_noise_clip` 0.5. The second critic is drawn
after DDPG's networks and the target noise has its own random stream, so a seed's actor, first critic, exploration and
replay match DDPG's. One Adam per critic. A TD3 agent refuses a DDPG checkpoint; a DDPG agent can load a TD3 one.

**What was measured.**

- Pendulum, `configs/pendulum_td3.yaml` (NumPy, 64×64, 40,000 steps): 5 of 5 solve at 32,600 to 33,400 steps and hold,
  greedy -157 ± 24, about 1,700 steps after DDPG since the actor updates half as often (CHANGELOG.md, 2026-10-25).
- MountainCarContinuous, `mountaincar_continuous_td3.yaml` (400×300, PyTorch backend, 300,000 steps): 5 of 5 reach 90
  and hold, greedy 93.8 ± 0.2; about 20 minutes a seed on the Jetson (CHANGELOG.md, 2026-10-27).
- LunarLanderContinuous, `lunarlander_continuous_td3.yaml` (400×300, PyTorch backend, 400,000 steps, updates after
  every step): 5 of 5 reach 200 and hold, final averages 202 to 248, greedy 182 ± 69; 53 to 54 minutes a seed
  (CHANGELOG.md, 2026-11-01).
- BipedalWalker, `bipedalwalker_td3.yaml` (400×300, PyTorch backend, 500,000 steps, half rl-zoo's 1,000,000 so it
  shares SAC's budget, updates after every step): 2 of 5 reach 300 (seeds 3 and 4, at 425,277 and 398,519 steps),
  none holds it (final averages 242 to 295), greedy 300.1 ± 12.3 (281.6 to 317.7), 2 of 5 at 300 or more; about 2.1
  hours a seed with three runs sharing the Jetson's GPU (CHANGELOG.md, 2026-11-05;
  `docs/benchmarks/bipedalwalker_results.md`). SAC beats it on every seed (below).
- NavigationContinuous, `navigation_continuous_td3.yaml` (NumPy, 64×64, 100,000 steps, updates after every step):
  greedy evaluation over 100 held-out layouts reaches the goal 96% of the time (93 to 100% by seed), reward 8.08 ±
  0.38, final training averages 7.4 to 8.4, ahead of continuous PPO on every seed with half its steps but about 2.9
  times its wall time; about 15 minutes a seed with three runs sharing the CPU; solves and holds the 7.0 threshold on
  5 of 5 seeds (2026-11-07, `docs/benchmarks/navigation_continuous_results.md`). Its success stays at 87 to 98% with
  sensor noise from none to 0.4 m and dropout from none to 40% (TESTING.md `ROB-2`, seeds 42, 1, 2).

**Lessons.** LunarLanderContinuous needed SB3's update after every environment step (`training.update_every: 1`):
once per episode it held on 3 of 5 seeds, every step on 5 of 5 (CHANGELOG.md, 2026-10-30; adopted 2026-11-01). TD3
is not faster on an easy task; its gain shows on the long ones.

## SAC (`SACAgent`, `rl_core/agents/sac_agent.py`)

**Idea.** Soft Actor-Critic (Haarnoja et al. 2018, with the learned temperature of their follow-up paper) maximises
reward plus `alpha` times the policy's entropy. The actor is a Gaussian squashed by tanh into the bounds, trained
through a reparameterised sample rather than a likelihood ratio; twin critics as TD3's keep the value honest; and
`alpha` is tuned so the entropy stays near a target, exploring a lot early and little once the policy is confident.

**Update.** Per gradient step, in Stable-Baselines3's order (t = tanh u, log pi of t in [-1, 1]):

```text
m, l     = actor(s) split in two; l = clip(l, -20, 2); u = m + e^l * eps; a = bias + scale * tanh(u)
log pi   = sum(-eps^2/2 - l - log(2 pi)/2) - sum 2 (log 2 - u - softplus(-2u))    (the stable log(1 - t^2))
L_alpha  = -log(alpha) * mean(log pi + target_entropy)     (log pi constant; alpha from before this step below)
y        = r + gamma * (1 - terminated) * (min(Q1'(s', a'), Q2'(s', a')) - alpha * log pi(a'|s'))   (a' fresh)
L_Qi     = mean( 0.5 * (Qi(s, a) - y)^2 ),  i = 1, 2
L_act    = mean(alpha * log pi(a|s) - min(Q1, Q2)(s, a))    (on the critics just updated), then Polyak on the critics
```

**In this repo.** An `OffPolicyAgent` with SB3's SAC defaults (256×256, learning rate 3e-4, `gamma` 0.99, `tau`
0.005, `learning_starts` 100, buffer 1,000,000). One actor MLP with 2A outputs (mean, then log std): the same
initial distribution as SB3's two heads, and export cuts the last layer to its first A outputs for a `tanh_scale`
head, so the greedy action is `bias + scale * tanh(m)`. `ent_coef: auto` learns `log alpha` from 0 (`auto_<x>`
from log x; a number fixes alpha) by an unclipped Adam at the learning rate; `target_entropy: auto` is minus the
action size. `log alpha` is a parameter group of shape (1,) saved with its Adam in every checkpoint, so DDPG and
TD3 files are refused and they refuse SAC's. Training actions are policy samples (in the bounds, stored as executed)
after uniform warm-up; the update's noise comes from its own seed stream (`AGENT_POLICY_NOISE_STREAM`). Logged:
`ent_coef`, `ent_coef_loss`, `entropy` (-mean log pi), `saturation` and the losses and gradient norms. Deliberate
differences from SB3: the exact stable tanh correction instead of `log(1 - t^2 + 1e-6)`; the critics see actions
in the environment's scale, not SB3's [-1, 1]; training starts once the buffer can fill a batch, not at
`learning_starts` (both as DDPG and TD3 here); left out: gSDE, n-step
returns, learning-rate schedules, `target_update_interval` other than 1. `backend: torch` runs the step in PyTorch
(`TorchSACAgent`; see the PyTorch backend below), as `configs/lunarlander_continuous_sac.yaml` does: rl-zoo's
LunarLanderContinuous settings (400×300, learning rate 7.3e-4 held constant where rl-zoo decays it linearly, `tau`
0.01, 10,000 warm-up actions, 500,000 steps, updates after every step), and `configs/bipedalwalker_sac.yaml`:
rl-zoo's BipedalWalker settings (400×300, learning rate 7.3e-4, `gamma` 0.98, `tau` 0.02, 300,000-transition buffer,
10,000 warm-up actions, 500,000 steps) without its gSDE exploration, updating in bursts of 64 gradient steps every 64
transitions (`training.update_every: 64`) as rl-zoo's `train_freq` 64 does.

**What was measured.**

- Pendulum, `configs/pendulum_sac.yaml` (NumPy, 64×64, 40,000 steps, updates after every step), seeds 1 to 5
  through the benchmark (2026-11-04): all 5 solve -200 at 21,800 to 23,200 steps (episodes 109 to 116) and hold,
  final 100-episode training averages -135.9 to -163.8, greedy evaluation -157 ± 20; about 9 minutes a seed (11 ms a
  gradient step). TD3's committed 64×64 config solves at 32,600 to 33,400 steps with the same greedy score.
- At SB3's 256×256 a gradient step takes about 88 ms (TD3 at that width with `policy_delay` 1: 73 ms), about an hour
  a seed in NumPy; not measured for learning.
- LunarLanderContinuous, `lunarlander_continuous_sac.yaml` (PyTorch backend): see the PyTorch backend below.
- BipedalWalker, `bipedalwalker_sac.yaml` (400×300, PyTorch backend, 500,000 steps), 2026-11-05: 4 of 5 reach 300 at
  210,235 to 387,254 steps (mean 307,090; seed 3 does not), 1 of 5 holds it (seed 1, final 306.8; the others end at
  251 to 287); greedy evaluation 324.4 ± 1.9 (321.3 to 326.9), all 5 at 300 or more, above TD3's on every paired seed
  (TD3 on the same budget: 300.1 ± 12.3, 2 of 5). About 1.7 hours a seed with three runs sharing the Jetson's GPU
  (alone about 0.9 hours, estimated from a 13,000-step probe) (CHANGELOG.md, 2026-11-05;
  `docs/benchmarks/bipedalwalker_results.md`). Seed 42 through `train.py` ends at a 100-episode average of 302.4; its
  `agent_episode_630.pt` walks the course in `record.py` at 325.2 ± 0.2 and exports to C and ONNX within 0.5 of the
  tolerance on 1,993 observations.

**Lessons.** The plan's 30 ms a step, scaled from TD3 at 400×300, was three times too low: a SAC step runs eight
forward and five backward passes, all of them every step. Time a short run before planning a sweep. On
BipedalWalker the 100-episode training average and the greedy evaluation part: seeds whose greedy policy scores
above 320 (10 episodes) end their training average at 251 to 287. Exploration noise is the likely cause, not yet
tested (roadmap, parked ideas). Read the greedy evaluation beside the solve and hold counts before calling a run a
failure.

## The PyTorch backend (`TorchDDPGAgent`, `TorchTD3Agent`, `TorchSACAgent`)

`backend: torch` in a DDPG, TD3 or SAC config runs the same gradient step in PyTorch (imported only then; PyTorch is
allowed only in `rl_core/agents/torch_*.py`). Acting, exploration and the replay buffer stay in NumPy on the CPU;
batches are sampled in the NumPy agent's order and moved to the device in chunks, with any noise the step needs (TD3's
target noise, SAC's two policy noises) drawn on the host from the NumPy agent's own generator. On CUDA the whole step
is captured as a CUDA graph and replayed as one launch, which is what pays: plain PyTorch on the GPU is no faster than
NumPy for these small networks. Options: `device` (auto, cpu, cuda), `dtype` (float32, or float64, which matches the
NumPy agent to rounding: every case of `tests/test_ddpg_td3_golden.py` replays to a relative 1e-9), `cuda_graph`.
To use it, add `backend: torch` to a DDPG, TD3 or SAC config's `agent.config` (optional: `device: auto|cpu|cuda`,
`dtype: float32|float64`, `cuda_graph: true|false`). It needs PyTorch, which `.venv` and CI do not have: build a
separate environment with `./setup.sh --profile gpu` (`.venv-gpu`), as in
[`pytorch_jetson_timing.md`](pytorch_jetson_timing.md), and run `train.py` from it. Its checkpoints are the NumPy
agent's format, so `record.py`, `export.py` and `robustness.py` replay a GPU-trained checkpoint without PyTorch:
they build the NumPy agent whatever `backend` the run's config names. Measured speed: see the timing document.

Checkpoints are the NumPy agent's format both ways. The three share `TorchOffPolicyAgent`
(`rl_core/agents/torch_off_policy.py`: device copies, staging, capture, the mirror, the critic step, the minimum
target Q); each adds only its step. A group is clipped on the device exactly when its NumPy Adam clips it, so SAC's
temperature never is. Measured on the Jetson: a 400×300 DDPG step 2.5 ms against NumPy's 44.7 ms; Pendulum DDPG 34 s
a run against 144 s (`docs/pytorch_jetson_timing.md`); Pendulum SAC (64×64, seed 1) solves at 22,000 steps, final
average -134.2, greedy -179.9, in 4.5 minutes, against NumPy's 21,800, -135.9 and about 9 minutes (2026-11-04).
LunarLanderContinuous SAC at 400×300 costs about 10 ms an environment step once training starts, alone on the GPU
(TD3's committed run: about 8 ms), and about 16 ms each with three runs at once (2026-11-04, 20,000-step timing runs).
Through the benchmark on seeds 1 to 5 (500,000 steps): every seed reaches 200 at 79,587 to 244,525 steps and holds
it (final 100-episode averages 266 to 284), greedy evaluation 282 ± 5, all 5 at 200 or more; TD3's published
run reaches 200 at 174,000 to 294,000 steps, with greedy evaluation 182 ± 69 (2026-11-04).

## Exporting a policy (`rl_core/agents/inference.py`, `rl_core/export/`, `export.py`)

Every agent with a network describes its greedy policy as an `InferencePolicy`: dense ReLU layers, the observation
statistics when it normalises, and a head (`argmax` for discrete agents, `clip` for a Gaussian mean, or `tanh_scale` for
a bounded one and for DDPG, TD3 and SAC's mean). `export.py` writes it as a C99 header or an ONNX model, and writes
nothing unless the exported policy matches the NumPy agent on every observation of a few greedy episodes (CHANGELOG.md,
2026-11-02). It is the path to running a trained policy on a microcontroller or a robot's computer.
