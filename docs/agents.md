# How Each Agent Works

The reference for the agent classes in `packages/rl-core/src/rl_core/agents/`: what they share, the replay buffer,
updates during an episode, the random baselines, the PyTorch backend and policy export. How each algorithm works, its
update as this code computes it, the settings that matter and the fixes it needed are taught in the algorithm pages
under [`learn/algorithms/`](learn/algorithms/), one per family; which agent solves which challenge is in
[results.md](results.md). A change to an agent's update, options or committed config updates its algorithm page, and a
change to what this page covers updates it, in the same pull request (the `doc-sync` skill).

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

Implemented agents (in `packages/rl-core/src/rl_core/agents/`; the algorithm pages listed in the next section say how
each one works and what it needed):

- **Discrete Control Agents**:
  - `RandomAgent`: Random action baseline policy
  - `ReinforceAgent`: Monte Carlo policy-gradient agent
  - `ActorCriticAgent`: REINFORCE with a learned value baseline (policy + value network; not A2C)
  - `PPOAgent`: Clipped-policy Proximal Policy Optimization agent
  - `DQNAgent`: Deep Q-Network with replay buffer and target network; double DQN targets (`double_q: true`) and a
    dueling network (`dueling: true`) and a Huber TD loss (`td_loss: huber`) are opt-in
- **Continuous Control Agents**:
  - `ContinuousRandomAgent`: Uniform random continuous torque baseline policy
  - `ContinuousActorCriticAgent`: Gaussian-policy actor-critic with a state-value critic; with `n_steps` and a matching
    `training.update_every`, true A2C
  - `ContinuousPPOAgent`: Continuous Gaussian policy PPO with clipped surrogate objective
  - `DDPGAgent`: Deep Deterministic Policy Gradient with replay, a Q critic and Polyak target networks; it explores
    with Gaussian noise or, with `noise_type: ornstein_uhlenbeck`, temporally correlated Ornstein-Uhlenbeck noise
    (`rl_core/agents/exploration_noise.py`)
  - `TD3Agent`: Twin Delayed DDPG: two critics with the smaller target, smoothed target actions and delayed actor
    and target updates (`configs/pendulum_td3.yaml`)
  - `SACAgent`: Soft Actor-Critic: a squashed Gaussian actor, twin critics with Polyak targets and a learned
    entropy temperature (`ent_coef`, `target_entropy`; `configs/pendulum_sac.yaml`, and on the PyTorch backend
    `configs/lunarlander_continuous_sac.yaml` and `configs/bipedalwalker_sac.yaml`)

## Where each agent is explained

| Agent | Algorithm page |
| :--- | :--- |
| `ReinforceAgent`, `ActorCriticAgent`, `ContinuousActorCriticAgent` | [Policy gradients](learn/algorithms/policy-gradient.md) |
| `PPOAgent`, `ContinuousPPOAgent` | [PPO](learn/algorithms/ppo.md) |
| `DQNAgent` (double, dueling, Huber, decaying rate) | [DQN and its variants](learn/algorithms/dqn.md) |
| `DDPGAgent`, `TD3Agent` (and `OffPolicyAgent`, their shared base) | [DDPG and TD3](learn/algorithms/ddpg-td3.md) |
| `SACAgent` | [SAC](learn/algorithms/sac.md) |

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

## The PyTorch backend (`TorchDDPGAgent`, `TorchTD3Agent`, `TorchSACAgent`)

`backend: torch` in a DDPG, TD3 or SAC config runs the same gradient step in PyTorch (imported only then; PyTorch is
allowed only in `rl_core/agents/torch_*.py`). Acting, exploration and the replay buffer stay in NumPy on the CPU;
batches are sampled in the NumPy agent's order and moved to the device in chunks, with any noise the step needs (TD3's
target noise, SAC's two policy noises) drawn on the host from the NumPy agent's own generator. On CUDA the whole step is
captured as a CUDA graph and replayed as one launch, which is what pays: without graphs, one DDPG step on the GPU took
about 22 ms at both 64×64 and 400×300, about twice as fast as NumPy's 44.7 ms at 400×300 but slower than its 5.46 ms at
64×64, while the graph replayed it in 0.84 and 2.50 ms (CHANGELOG.md, 2026-10-25). Options: `device` (auto, cpu, cuda),
`dtype` (float32, or float64, which matches the NumPy agent to rounding: every case of `tests/test_ddpg_td3_golden.py`
replays to a relative 1e-9), `cuda_graph`. To use it, add `backend: torch` to a DDPG, TD3 or SAC config's `agent.config`
(optional: `device: auto|cpu|cuda`, `dtype: float32|float64`, `cuda_graph: true|false`). It needs PyTorch, which `.venv`
and CI do not have: build a separate environment with `./setup.sh --profile gpu` (`.venv-gpu`), as in
[`pytorch_jetson_timing.md`](pytorch_jetson_timing.md), and run `train.py` from it. Its checkpoints are the NumPy
agent's format, so `record.py`, `export.py` and `robustness.py` replay a GPU-trained checkpoint without PyTorch: they
build the NumPy agent whatever `backend` the run's config names. Measured speed: see the timing document.

Checkpoints are the NumPy agent's format both ways. The three share `TorchOffPolicyAgent`
(`rl_core/agents/torch_off_policy.py`: device copies, staging, capture, the mirror, the critic step, the minimum
target Q); each adds only its step. A group is clipped on the device exactly when its NumPy Adam clips it, so SAC's
temperature never is. Measured on the Jetson: a 400×300 DDPG step 2.5 ms against NumPy's 44.7 ms; Pendulum DDPG 34 s
a run against 144 s (`docs/pytorch_jetson_timing.md`); Pendulum SAC (64×64, seed 1) solves at 22,000 steps, final
average -134.2, greedy -179.9, in 4.5 minutes, against NumPy's 21,800, -135.9 and about 9 minutes (2026-11-04).
LunarLanderContinuous SAC at 400×300 runs at about 10 ms an environment step once training starts (TESTING.md
`TR-29`).
Through the benchmark on seeds 1 to 5 (500,000 steps): every seed reaches 200 at 79,587 to 244,525 steps and holds
it (final 100-episode averages 266 to 284), greedy evaluation 282 ± 5, all 5 at 200 or more; TD3's published
run reaches 200 at 174,000 to 294,000 steps, with greedy evaluation 182 ± 69 (2026-11-04).

### Fixes the backend needed

- **A measurement first.** PyTorch's own aarch64 wheel drives the Orin's GPU; one DDPG step replayed as a CUDA graph was
  6.5 and 18 times faster than NumPy at 64×64 and 400×300, while plain PyTorch on the GPU, without graphs, was limited
  by kernel launches (CHANGELOG.md, 2026-10-25; `docs/pytorch_jetson_timing.md`).
- **Same training, same checkpoints.** The DDPG backend trains as `DDPGAgent` does to rounding in float64 and reads and
  writes its checkpoints both ways; no committed config switched backend, since `.venv` and CI have no PyTorch
  (CHANGELOG.md, 2026-10-25).
- **The layout read from the NumPy agent.** The backend stopped naming DDPG's networks, targets and metrics and reads
  them from the NumPy agent's hooks, so a TD3 backend only supplies its step (CHANGELOG.md, 2026-10-26). TD3 followed as
  two CUDA graphs, a critic-only step and a full step, chosen by TD3's own delay rule (CHANGELOG.md, 2026-10-26).
- **Copy back only what acting needs.** Copying every network back to NumPy after each update made a per-step-updating
  seed 5.7 times slower; copying only the actor after an update, and the rest before a checkpoint, brought it to about
  1.5 times (CHANGELOG.md, 2026-10-30).
- **One base, clipped where NumPy clips.** `TorchOffPolicyAgent` took what the DDPG and TD3 backends shared when SAC
  joined; each parameter group is clipped on the device exactly when its NumPy Adam clips it (CHANGELOG.md, 2026-11-04).
- **Replay without PyTorch.** `record.py` builds the NumPy agent, as `export.py` and `robustness.py` do, so a `backend:
  torch` run records without PyTorch, and a test pins SAC's PyTorch gradient at the log-std bounds (CHANGELOG.md,
  2026-11-11).

## Exporting a policy (`rl_core/agents/inference.py`, `rl_core/export/`, `export.py`)

Every agent with a network describes its greedy policy as an `InferencePolicy`: dense ReLU layers, the observation
statistics when it normalises, and a head (`argmax` for discrete agents, `clip` for a Gaussian mean, or `tanh_scale` for
a bounded one and for DDPG, TD3 and SAC's mean). `export.py` writes it as a C99 header or an ONNX model, and writes
nothing unless the exported policy matches the NumPy agent on every observation of a few greedy episodes (CHANGELOG.md,
2026-11-02). It is the path to running a trained policy on a microcontroller or a robot's computer.
