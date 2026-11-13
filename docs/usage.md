# Usage

Everything about running the code: install, train, record, export, benchmark, configure, and the development
commands. The [README](../README.md) is the front page; this page holds the details.

- [Installation](#installation): prerequisites, Windows, Linux and macOS, Box2D, manual setup
- [Training](#training): first run, one command per committed config, quick overrides, outputs, resuming
- [Recording](#recording): training videos, recording a trained agent
- [Export](#export): a trained policy to C or ONNX
- [Experiments](#experiments): benchmarking, configuration, curriculum, robustness
- [Environments and checkpoints](#environments-and-checkpoints): the wrapper, observation settings, the pipeline
- [Development](#development): tests, formatting, linting, types, security, Markdown, CI, pre-commit

Where the rest lives: [navigation.md](navigation.md) (the simulator and state estimation),
[results.md](results.md) (what each agent measured), [agents.md](agents.md) (how each agent works),
[ai-assistants.md](ai-assistants.md) (Claude Code setup).

## Installation

### Prerequisites

- Python 3.12+ (the version CI tests)
- pip or conda package manager

### Windows Quick Setup

From PowerShell in the repository root, run:

```powershell
.\setup.ps1
```

The script installs Git and Python 3.12 via `winget` if they are missing, creates a `.venv` virtual environment,
installs `requirements-dev.txt` (the three distributions, editable, with the dev tools) and the video extras, and runs
the same checks as CI (Ruff, mypy, pytest). Flags: `-NonInteractive` (no prompts), `-SkipChecks`, `-Update` (upgrade Git
and Python with winget, and re-sync the packages to `requirements-lock.txt`), and `-InstallHooks` (`pre-commit
install`). Activate the environment afterwards with `.\.venv\Scripts\Activate.ps1`.

If script execution is disabled, run it once with `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.

### Linux and macOS Setup

```bash
./setup.sh                  # .venv, the CPU profile: requirements-dev.txt, video and box2d extras, no PyTorch
./setup.sh --profile gpu    # .venv-gpu, adds PyTorch for the DDPG, TD3 and SAC backend (backend: torch); ~5 GiB
```

Each creates its environment if missing, brings its packages to the versions in `requirements-lock.txt` and runs the
CI checks (`--skip-checks` to skip). Both
build Box2D for LunarLander and BipedalWalker first (see [LunarLander (Box2D)](#lunarlander-box2d) below); CI
installs neither PyTorch nor Box2D and skips their
tests. Both profiles, `setup.ps1` and CI's test job install the `onnx` extra (`onnx` and `onnxruntime`) for the ONNX
export.

### LunarLander (Box2D)

LunarLander and BipedalWalker need Box2D. Gymnasium's own extra (`pip install "gymnasium[box2d]"`) pins `box2d==2.3.10`,
which has no build for the Jetson's aarch64, so this project's `box2d` extra names `box2d-py`, which builds from source
with `swig` (about 90 seconds; it needs a C++ compiler and the Python headers, on Debian and Ubuntu `sudo apt install
build-essential python3.12-dev`). `./setup.sh` does it; by hand, inside the virtual environment:

```bash
pip install swig setuptools wheel -c requirements-lock.txt   # the lock holds the versions it was verified with
PATH="$PWD/.venv/bin:$PATH" pip install --no-build-isolation --no-binary box2d-py box2d-py -c requirements-lock.txt
pip install -r requirements-dev.txt -c requirements-lock.txt -e "./packages/rl-core[video,onnx]" \
    -e "./packages/rl-challenges[box2d]"
```

`setup.ps1` does not install it on Windows. Without Box2D the LunarLander and BipedalWalker challenges are still
listed, and building their environment fails at once with these steps: `BaseEnvWrapper` checks every Gymnasium Box2D
environment (`rl_core.environments.uses_box2d`) before `gym.make`, so a new Box2D challenge gets the same message.

### Manual Setup (any OS)

1. Clone the repository:

```bash
git clone https://github.com/andreijw/AI.git
cd AI
```

2. Install the three distributions under `packages/` in development mode, with the dev tools, from the
   repository root (pip reads the `./packages/...` paths relative to the current directory):

```bash
pip install -r requirements-dev.txt -c requirements-lock.txt
```

`requirements-lock.txt` pins every package's version (the lock); `requirements-dev.txt` names what to install.

## Training

### Quick Start: Train Your First Agent

The fastest way to try the framework is to run the training script with the default
random-agent baseline configuration:

```bash
python train.py --config configs/cartpole_default.yaml
```

By default, `configs/cartpole_default.yaml` trains for `total_steps: 50000` environment steps.

For a quick smoke-test pass `--num-episodes 50` (or `--total-steps 2000`), or create a minimal config:

```yaml
# configs/quick_test.yaml
environment:
  name: "CartPole-v1"
  max_episode_steps: 500
  seed: 42            # reproducible sequence of start states (each episode differs)

agent:
  type: "random"
  config:
    seed: 42          # makes action selection reproducible

training:
  num_episodes: 50
  max_steps_per_episode: 500
  eval_frequency: 10
  save_frequency: 50
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

Then run:

```bash
python train.py --config configs/quick_test.yaml
```

To generate and save learning-curve plots during training, include `--plot` on your initial run:

```bash
python train.py --config configs/quick_test.yaml --plot
```

> Note: Running `train.py` again with `--plot` will start a new training run and then save plots,
> it does **not** only load and plot previous results.

### Train specific agent configurations

```bash
# Random baseline
python train.py --config configs/cartpole_default.yaml

# REINFORCE
python train.py --config configs/cartpole_reinforce.yaml

# Actor-Critic
python train.py --config configs/cartpole_actor_critic.yaml

# PPO
python train.py --config configs/cartpole_ppo.yaml

# DQN
python train.py --config configs/cartpole_dqn.yaml

# Pendulum Continuous Actor-Critic
python train.py --config configs/pendulum_actor_critic.yaml

# Pendulum Continuous PPO
python train.py --config configs/pendulum_ppo.yaml

# Pendulum DDPG
python train.py --config configs/pendulum_ddpg.yaml

# Pendulum TD3
python train.py --config configs/pendulum_td3.yaml

# Pendulum SAC (64x64 networks, updating after every step; about 9 minutes)
python train.py --config configs/pendulum_sac.yaml

# Acrobot DQN and PPO
python train.py --config configs/acrobot_dqn.yaml
python train.py --config configs/acrobot_ppo.yaml

# MountainCar DQN (learns, does not reliably solve)
python train.py --config configs/mountaincar_dqn.yaml

# MountainCarContinuous DDPG with Ornstein-Uhlenbeck exploration noise
python train.py --config configs/mountaincar_continuous_ddpg.yaml

# MountainCarContinuous TD3 at 400x300 on the PyTorch backend (needs ./setup.sh --profile gpu)
.venv-gpu/bin/python train.py --config configs/mountaincar_continuous_td3.yaml

# LunarLander random baselines, discrete and continuous (need Box2D: see LunarLander (Box2D) under Installation)
python train.py --config configs/lunarlander_default.yaml
python train.py --config configs/lunarlander_continuous_default.yaml

# LunarLander DQN (rl-zoo settings; solves 200)
python train.py --config configs/lunarlander_dqn.yaml

# LunarLanderContinuous TD3 at 400x300 on the PyTorch backend (needs ./setup.sh --profile gpu; solves 200)
.venv-gpu/bin/python train.py --config configs/lunarlander_continuous_td3.yaml

# LunarLanderContinuous SAC at 400x300 on the PyTorch backend (needs ./setup.sh --profile gpu; solves 200)
.venv-gpu/bin/python train.py --config configs/lunarlander_continuous_sac.yaml

# BipedalWalker random baseline (needs Box2D: see LunarLander (Box2D) under Installation)
python train.py --config configs/bipedalwalker_default.yaml

# BipedalWalker SAC and TD3 at 400x300 on the PyTorch backend, 500,000 steps each (needs ./setup.sh --profile gpu;
# about 0.9 and 1.3 hours alone on the Jetson's GPU, estimated; SAC reaches 300 on 4 of 5 seeds, TD3 on 2 of 5)
.venv-gpu/bin/python train.py --config configs/bipedalwalker_sac.yaml
.venv-gpu/bin/python train.py --config configs/bipedalwalker_td3.yaml

# Navigation random baselines, discrete and continuous (threshold 7.0, far above random; see docs/navigation.md)
python train.py --config configs/navigation_default.yaml
python train.py --config configs/navigation_continuous_default.yaml

# Navigation PPO and DQN (Navigation-v0), PPO and TD3 (NavigationContinuous-v0); about 5 minutes for PPO, 15 for DQN
# and TD3 a seed with three runs sharing the CPU (DQN's result: see docs/navigation.md)
python train.py --config configs/navigation_ppo.yaml
python train.py --config configs/navigation_dqn.yaml
python train.py --config configs/navigation_continuous_ppo.yaml
python train.py --config configs/navigation_continuous_td3.yaml
```

### Run any agent from one base config (quick overrides)

You can override both the agent type and number of episodes directly from CLI without
editing YAML files:

```bash
# quick random agent run
python train.py --config configs/cartpole_default.yaml --agent-type random --num-episodes 50

# quick REINFORCE run
python train.py --config configs/cartpole_default.yaml --agent-type reinforce --num-episodes 200

# quick Actor-Critic run
python train.py --config configs/cartpole_default.yaml --agent-type actor_critic --num-episodes 200

# quick PPO run
python train.py --config configs/cartpole_default.yaml --agent-type ppo --num-episodes 200

# quick DQN run
python train.py --config configs/cartpole_default.yaml --agent-type dqn --num-episodes 200
```

The challenge comes from the config's `environment.name`, which must be a registered challenge's environment id
exactly (`Acrobot-v1`, `BipedalWalker-v3`, `CartPole-v1`, `LunarLander-v3`, `LunarLanderContinuous-v3`,
`MountainCar-v0`, `MountainCarContinuous-v0`, `Navigation-v0`, `NavigationContinuous-v0` or `Pendulum-v1`); any other
name is an error that lists the options. `--challenge acrobot`, `bipedalwalker`, `cartpole`, `lunarlander`,
`lunarlander_continuous`, `mountaincar`, `mountaincar_continuous`, `navigation`, `navigation_continuous` or `pendulum`
must agree with that name, and is required for a config that has no `environment.name`.
An agent the challenge does not support (for example `--agent-type dqn` on Pendulum) is rejected before training.

### Visualization and video recording

Outputs are partitioned into per-challenge and per-agent folders so runs never overwrite one another:
`checkpoints/<challenge>/<agent_type>/`, `videos/<challenge>/<agent_type>/`, `plots/<challenge>/<agent_type>/`,
and `logs/<challenge>/<agent_type>/` (for example `checkpoints/cartpole/ppo/agent_episode_1400.pt` or
`checkpoints/pendulum/ppo/agent_episode_200.pt`). Re-training the same agent replaces that agent's previous
outputs; copy its folders first to keep a run. Logs in `logs/` are timestamped per run.

### Resuming a run

`train.py` saves a checkpoint every `save_frequency` episodes and once more after its last episode, so the newest
checkpoint holds the final weights (an interrupted run keeps only the periodic ones; the
benchmark writes none). Every
checkpoint is two files: `agent_episode_<n>.pt` (the agent, its optimizer state, generators and any replay buffer) and
`trainer_episode_<n>.npz` beside it (the episode and step counts, reward history, evaluation seed and the training
environment's generators, the curriculum if any, and a hash of the agent file, so a mismatched pair is refused).
Resuming from one continues the run so that it ends exactly as it would have without the stop; the budget counts what
was already trained, and checkpoint numbering continues (a run ending on a multiple of `save_frequency` writes no
extra file, and resuming with the budget already met writes none):

```bash
python train.py --config configs/cartpole_ppo.yaml --resume checkpoints/cartpole/ppo/agent_episode_300.pt
```

Use the same config (and `--agent-type` or budget flags) as the original run; hyperparameter changes are not
detected, except network sizes: a checkpoint saved with another `hidden_dim` or `hidden_sizes` is refused, naming the
entry and both shapes, before anything is loaded (as by `record.py`, `export.py` and `robustness.py`). With
`--record-video` the resumed run records the same clips, under the same episode numbers, as an
uninterrupted run would; into the original video folder Gymnasium warns "Overwriting existing videos", but the
earlier clips are kept, as the new ones have later numbers. A resumed run writes new timestamped log files, and
checkpoints saved before 2026-10-21 hold no trainer state and cannot be resumed.

For a full run with saved learning-curve plots (saved to `plots/ppo/`):

```bash
python train.py --config configs/cartpole_ppo.yaml --plot
```

To visualize live training locally (requires display):

```bash
python train.py --config configs/cartpole_ppo.yaml --render
```

For headless visualization (saved MP4 videos):

```bash
python train.py --config configs/cartpole_ppo.yaml --record-video
```

### Basic/default run

Run training with defaults (`configs/cartpole_default.yaml`):

```bash
python train.py
```

## Recording

### Record Training Videos (Headless / SSH)

If you are running on a headless machine such as an **NVIDIA Orin Nano accessed via SSH**,
there is no display available so `--render` will not work. Use `--record-video` instead to
save MP4 videos of training episodes to disk. No display or X server is required.

First, install the optional video dependencies:

```bash
pip install -e "./packages/rl-core[video]"
# or: pip install moviepy "gymnasium[classic-control]"
```

Then run training with video recording:

```bash
python train.py --record-video
```

Videos are saved to `./videos/<agent_type>/` by default. Use `--video-dir` to choose a different
location (used exactly as given):

```bash
python train.py --record-video --video-dir /path/to/videos
```

By default a clip is recorded every `eval_frequency` training episodes (configured in the YAML
file, default 100). Clips are named after the training episode they show
(`cartpole-training-episode-<N>.mp4`); evaluation episodes are never recorded. Use
`--video-every` to change the interval:

```bash
# Record every 25th training episode as its own clip
python train.py --record-video --video-every 25
```

Every recorded episode is rendered and encoded frame by frame, so small intervals slow training
down (recording all 1000 random-agent episodes takes ~6 minutes instead of ~8 seconds).

The localization video (`python -m rl_navigation.localization_video`, see
[State estimation](navigation.md#state-estimation)) does not record an
agent: it takes a required `--output` path and never writes into `videos/` on its own. Recordings kept by hand go in
`videos/navigation/localization/`, git-ignored like the rest of `videos/`.

### Record a Trained Agent

To get one longer video of a trained agent, record it from a checkpoint after training. This
does not slow training down:

```bash
python record.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_1400.pt --episodes 5
```

`record.py` loads the agent type from the config (or `--agent-type`), plays `--episodes`
evaluation episodes (default 5) and saves them as one
`videos/<challenge>/<agent_type>/<challenge>-playback-full.mp4` (`--video-dir` to change).
A checkpoint trained on the PyTorch backend records without PyTorch, through the NumPy agent.
A well-trained agent balances for the full 500 steps, i.e. 10 seconds per episode.

After training, copy the videos to your local machine and open them in any video player:

```bash
# From your local machine
scp -r user@orin-nano:/path/to/AI/videos .
```

## Export

### Export a Trained Agent to C

To run a trained policy without Python, on a microcontroller or a robot, export it as one self-contained C99
header:

```bash
python export.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_600.pt --output policy_weights.h
```

The header holds the weights as `float` arrays (hexadecimal literals, exact in float32) and two functions,
`policy_outputs(obs, out)` and `policy_act`: `int policy_act(const float *obs)` for a discrete agent, `void
policy_act(const float *obs, float *action)` for a continuous one. It acts as `select_action(obs, training=False)`
does, including the agent's observation normalisation, and needs only `<math.h>` (link with `-lm`). Its input is
the observation after the environment's `mask_velocity`, `frame_stack` and `frame_deltas`, which the header's
comment names but does not apply.

Every agent with a network exports: REINFORCE, Actor-Critic and PPO (discrete and continuous), DQN (plain and dueling),
DDPG, TD3 and SAC (its mean). A checkpoint trained on the PyTorch backend exports without PyTorch, through the NumPy
agent. The random agents are refused ("has no network to export") before the checkpoint is read. Before writing the
header, `export.py` plays `--verify-episodes` greedy episodes (default 3; 0 skips this), compiles the header with `$CC`,
`cc`, `gcc` or `clang`, and checks its outputs against NumPy's on every observation, to within 1e-5 absolute plus 1e-5
relative; the summary gives the worst error as a share of that tolerance. If they disagree it writes no header and exits
with status 1; without a compiler it writes the header unchecked and exits with status 1. A discrete action may differ
only where NumPy's two best outputs are that close; these are reported as near-ties.

### Export a Trained Agent to ONNX

For a device with an ONNX runtime (onnxruntime, TensorRT, OpenVINO and others), export the same policy as an ONNX
model. It needs the `onnx` extra (`pip install -e "./packages/rl-core[onnx]"`; the setup scripts install it):

```bash
python export.py --format onnx --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_600.pt --output policy.onnx
```

The model takes `observation` (float32, `[batch, observation size]`; one observation is a batch of one) and gives
`outputs` (the network's outputs) and `action`: an int64 index per row for a discrete agent, a float32 action vector
for a continuous one. Like the header it includes the observation normalisation and the action head, and takes the
observation after `mask_velocity`, `frame_stack` and `frame_deltas`. The weights are float32; the model uses opset 17
and IR version 8, which onnxruntime has loaded since 1.14; its metadata names the agent, the checkpoint (with its
SHA-256), the action head and the observation settings.

The same agents export, and the check is the same, run in onnxruntime instead of a compiler. The model is written only
when it passes: without `onnx`, or without `onnxruntime` when checking, `export.py` stops before loading anything,
names the install command and exits with status 1 (`--verify-episodes 0` writes it unchecked, needing only `onnx`).

## Experiments

### Multi-Agent Benchmarking

Run an automated comparative benchmark across all algorithms over reproducible seeds:

```bash
# Benchmark all agents on CartPole, each for its own config's budget, across seeds 42 and 123
python benchmark.py --challenge cartpole \
    --agents random reinforce actor_critic ppo dqn --seeds 42 123

# Quick comparison between PPO and DQN, both for 150 episodes
python benchmark.py --challenge cartpole --agents ppo dqn --episodes 150 --seeds 42
```

Without `--episodes` or `--steps`, each agent trains for its own config's budget (`configs/<challenge>_<agent>.yaml`,
or `<challenge>_default.yaml` for the random agent): `training.total_steps` environment steps when set, else
`training.num_episodes` episodes. A step budget finishes the episode that crosses it. A run counts as solved when its
average training reward over `--window` consecutive episodes (default 100) reaches the challenge's threshold (475 for
CartPole-v1, -200 for Pendulum-v1); the report gives the episode and the environment steps it took, and **Held**: the
share of runs whose last `--window` episodes still average at or above the threshold. The final greedy evaluation is
reported in its own column and does not count as a solve. `train.py --total-steps N` trains for a step budget too.
Without `--agents`, every agent the challenge supports runs. Before any agent trains, the benchmark checks that each
one's config can be built, so a backend that is not installed stops the run at once: the MountainCarContinuous TD3
config needs PyTorch, so outside `.venv-gpu` name the others (`--challenge mountaincar_continuous --agents random
ddpg`).

Outputs:

- Comparative learning curve plot: `plots/benchmarks/cartpole/benchmark_comparison.png`
- Markdown summary table: [`docs/benchmarks/cartpole_results.md`](benchmarks/cartpole_results.md)
- Raw metrics JSON: `plots/benchmarks/cartpole/benchmark_metrics.json`

### Configuration

Configuration files are in YAML format. Here's an example:

```yaml
# Environment settings
environment:
  name: "CartPole-v1"
  render_mode: null
  max_episode_steps: 500
  seed: 42
  # frame_stack: 4        # optional, every challenge: the last 4 observations, oldest first, as one vector
  # mask_velocity: true   # optional, every challenge but Navigation: zero the velocity entries (partially observed)
  # frame_deltas: true    # optional, with frame_stack >= 2: the newest frame, then the differences between frames
  # obs_noise_std: 0.05   # optional: noise and domain randomization, the settings evaluation always plays
  # curriculum:           # optional: train from easier settings up to the ones above (see Curriculum below)
  #   mode: step          #   linear | step | reward_threshold
  #   episodes: 400       #   linear and step: training episodes to reach the target
  #   stages: 4           #   step and reward_threshold
  #   start: {obs_noise_std: 0.0}

# Agent settings
agent:
  type: "random"  # Options: "random", "reinforce", "actor_critic", "ppo", "dqn"
  config:
    seed: 42
    # normalize_observations: true  # optional: running mean and variance of the inputs, saved with the
    #                               #   checkpoint (every agent; DQN, DDPG, TD3 and SAC store raw observations
    #                               #   and normalise each sampled batch)

# Training settings
training:
  num_episodes: 1000
  # total_steps: 100000  # optional: train for this many environment steps instead (num_episodes is then ignored)
  # update_every: 1      # optional: update every N steps inside the episode, not after it (DQN, DDPG, TD3, SAC;
  #                      #   the continuous Actor-Critic with n_steps equal to it, as A2C)
  # num_envs: 1          # optional: training environments stepped in lockstep (REINFORCE, both Actor-Critics, PPO with
  #                      #   update_every, and the random agents; see "Several training environments")
  max_steps_per_episode: 500
  eval_frequency: 100
  save_frequency: 100
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

### Several training environments

`training.num_envs: N` (default 1) trains from N copies of the `environment` section's environment, stepped in
lockstep in one process. Without the key, or with 1, training is exactly what it was (`tests/test_trainer_golden.py`
holds it to a recording). With N above 1:

- Environment 0 keeps the section's seed; environment i gets `rl_core.seeding.env_seed(seed, i)`, its own stream
  of the run's seed (not `seed + i`, under which seed 1's second environment would replay seed 2's first).
- Each vector step chooses every playing environment's action in environment order, then steps them all. An
  episode that ends (terminated, truncated, or cut at `max_steps_per_episode`) goes to the agent as one batch,
  exactly as a single environment's would, and is then recorded. The records, the solve rule and the 100-episode
  window count finished episodes in the order they finish. A time-limit cut still bootstraps and a termination
  does not, for the agents that bootstrap.
- Evaluation and a checkpoint run once, at the end of the vector step in which their frequency was crossed, and the
  checkpoint is named by the episodes recorded then (`agent_episode_<n>` need not be a multiple of `save_frequency`).
- Once the budget is met no new episode starts; episodes in flight finish and are trained on, so `total_steps` can
  be overshot by up to `(N - 1) × max_steps_per_episode` more than with one environment.
- The observation normaliser sees every environment's observations; the curriculum gives each environment its
  settings before each of its resets. An episode that began under an earlier curriculum stage is recorded with the
  difficulty at its end.
- `--resume` restores every environment's generators and refuses a different `num_envs`. The episodes in flight at
  the checkpoint are lost, so a resumed run is reproducible but is not the uninterrupted run.
- Without `update_every`, REINFORCE, both Actor-Critics and the random agents accept it. Each update is one
  finished episode, so the other environments' episodes in flight were partly played by an earlier policy (up to
  N - 1 updates old).
- With `update_every: k`, one clock runs across the environments: every k vector steps the agent gets every
  environment's steps of that window in one call (`update_segments`), each a run of one environment's steps that
  ends where its episode or the window ended (a window cut bootstraps from its next state). An environment whose
  episode ended inside the window gives two: the end of the old episode and the start of the new. What is left
  when training ends goes in a last call. With one environment `update_every` keeps restarting its chunks with each
  episode, as before. An episode that ends off a window boundary is recorded, and any evaluation or checkpoint
  of that vector step runs, before its last steps (up to `update_every - 1`) are trained on at the window's end,
  so such a checkpoint's agent has not seen them; with one environment the update comes first.
  - The continuous Actor-Critic (`n_steps` equal to `update_every`) makes one Adam step over the window's N × k
    transitions: every segment's GAE targets from the same weights, each bootstrapped by its own end, advantages
    normalised over the window when `normalize_advantages` is on, the loss averaged over all of them. That is
    rl-zoo's A2C layout (8 environments × 8 steps on Pendulum), which `configs/pendulum_actor_critic.yaml` uses
    since 2026-11-12.
  - PPO (both kinds) needs `update_every` with several environments, and `rollout_steps` a multiple of
    `update_every × num_envs` (SB3's `n_steps × n_envs`; CartPole with 4 environments: `update_every: 64` for
    256). It buffers every segment of a window and trains only at the end of a call, so each rollout is data its
    current weights collected and its probability ratio starts at 1. Whole episodes handed over one at a time
    would be trained on after updates their steps did not see, with the ratio blind to it, so PPO refuses
    several environments without `update_every`; with one environment it still refuses `update_every`.
- Refused, with an error naming the reason: DQN, DDPG, TD3 and SAC; REINFORCE and the discrete Actor-Critic with
  `update_every`; a training environment that renders (`--render`, `--record-video` or a `render_mode`); and a run
  without an evaluation environment (direct `Trainer` callers). `record.py` and `robustness.py` evaluate one
  environment from such a config.

### Curriculum

`environment.curriculum` trains through easier settings first. The `environment` section stays the target: the
evaluation environment is built from it as always, so evaluation and its greedy episodes play the hardest settings.
`start` gives the easiest value of any schedulable setting the section turns on (`obs_noise_std` on every
challenge; `action_noise_prob` or `action_noise_std` and `domain_randomization` on CartPole, Pendulum and the two
Navigation ids). A start
randomization may name some of the target's parameters; the others stay at their target range. Before each
training episode one difficulty d, from 0 to 1, sets every scheduled setting to `(1 - d) * start + d * target`, and
moves a randomization range's two ends separately.

| `mode` | Keys | Difficulty |
| :--- | :--- | :--- |
| `linear` | `episodes` | episodes done / `episodes`, capped at 1 |
| `step` | `episodes`, `stages` | `stages` equal steps over `episodes` |
| `reward_threshold` | `stages`, `threshold`, `window`, `min_episodes_per_stage` | one stage up when the average training reward of the stage's last `window` episodes reaches `threshold`, after at least `min_episodes_per_stage` (no fewer than `window`) at that stage; it never falls |

The 10-episode metrics records carry `curriculum_difficulty` (and `curriculum_stage`) of the last episode trained,
the log says when a stage advances, and a run that ends below difficulty 1 warns. The solve rule is unchanged, but
its training average then mixes difficulties, so quote a curriculum run's greedy evaluation at the target as well.
A run with a curriculum resumes only with the same one. `record.py`, `export.py` and the benchmark's evaluation
build the target and ignore the curriculum; `benchmark.py` trains each run through its config's curriculum, and
its report names each agent that did.

### Robustness of a Trained Agent

`robustness.py` plays a checkpoint greedily with one setting changed at a time, to show how far outside its training
the policy holds up:

```bash
python robustness.py --config configs/cartpole_ppo.yaml --checkpoint checkpoints/cartpole/ppo/agent_episode_600.pt --sweep length=0.25,0.5,1.0,2.0 --sweep obs_noise_std=0,0.05,0.1 --report <scratch>/robustness.md
```

Each `--sweep SETTING=V1,V2,...` names a noise setting the challenge's wrapper can schedule (`obs_noise_std`;
`action_noise_prob` on CartPole and Navigation-v0, `action_noise_std` on Pendulum and NavigationContinuous-v0) or a
parameter it can randomize (CartPole `gravity`, `masscart`, `masspole`, `length`; Pendulum `g`, `m`, `l` or their long
names; Navigation `wheel_slip`, `sensor_noise_std`, `dropout_prob`, `num_obstacles`), one `--sweep` per setting.
Every value gets a new evaluation environment built from the config's `environment` section, with the noise level
set or the parameter pinned to that value (other randomized parameters keep their ranges), and plays `--episodes`
greedy episodes (default 20). Episode i of every row resets with the same seed, so the start states are paired;
noise draws are not once a pinned parameter adds a draw at each reset. The first row is the config's own target.

Each row says whether the value is inside the range the agent trained on: the config's value without a curriculum,
or from the curriculum's start to the difficulty the run reached, replayed from the trainer file beside the
checkpoint (the configured start to target when there is none). A config whose curriculum differs from the one in
the trainer file, or that has none where the run had one, is refused, as a resume refuses it. The table gives mean
and spread of the greedy reward, mean length, the share of episodes whose reward reached the solve threshold (the
"Reward ≥ threshold" column) and, for an environment that reports `is_success` (Navigation), the share that succeeded
("Success"). It is printed; `--report` also writes it as Markdown, and nothing is written without it. A checkpoint
from the PyTorch backend loads into the NumPy agent, so no PyTorch is needed.

## Environments and checkpoints

### Environment Wrapper

The `CartPoleEnv` class wraps Gymnasium's CartPole-v1 environment, `PendulumEnv` wraps Pendulum-v1, `AcrobotEnv`
wraps Acrobot-v1, `MountainCarEnv` wraps MountainCar-v0 and MountainCarContinuous-v0, `LunarLanderEnv` wraps
LunarLander-v3 and LunarLanderContinuous-v3 and `BipedalWalkerEnv` wraps BipedalWalker-v3 (observation noise only for
those four), and `NavigationEnv` wraps this project's Navigation-v0 and NavigationContinuous-v0 ([Range-Sensor
Navigation](navigation.md#range-sensor-navigation)). All seven subclass
`rl_core.environments.BaseEnvWrapper`, which owns seeding, observation noise and episode statistics. They provide:

- Episode tracking and statistics
- Standardized interface for training pipeline
- Easy monitoring and logging integration

Every challenge's `environment` section also takes `frame_stack`, `mask_velocity` and `frame_deltas`, applied by the
base after the observation noise; Navigation refuses `mask_velocity: true`, as its observation holds no velocity of
the robot's state. `mask_velocity: true` zeroes the velocity entries and keeps the observation's length,
as rl-zoo's `MaskVelocityWrapper` does. `frame_stack: K` joins the last K observations, oldest first, into one vector;
at a reset the first observation fills every slot. `frame_deltas: true` gives that stack as its newest frame followed by
the differences between consecutive frames, newest first: the same entries, but a velocity hidden in the small change
between frames becomes an input of its own. Agents size themselves from the stacked space, so a checkpoint trained with
these keys must be played back (`record.py`) or resumed with the same `environment` section. Checkpoints record the
keys, and loading one under other values stops with an error naming each key with its saved and current value;
checkpoints saved before 2026-11-02 hold no record and load unchecked. On masked CartPole, PPO with `frame_stack: 4`
recovers much of what masking takes away but does not solve (CHANGELOG.md, 2026-11-01). With `normalize_observations:
true` in the agent's config it solves on 4 of 5 seeds and holds on 2, and with `frame_deltas: true` as well on 5 of 5,
holding on all 5 (CHANGELOG.md, 2026-11-02).

### Training Pipeline

The `Trainer` class orchestrates the training process:

- Episode collection and management
- Agent policy updates
- Evaluation and checkpointing
- Metrics logging

## Development

### Running Tests

Run the test suite:

```bash
pytest -q
```

Run tests with coverage (CI-aligned):

```bash
pytest --cov-report=xml --cov-branch --cov-fail-under=90
```

### Code Formatting

Format code with Ruff:

```bash
ruff format .
```

### Linting

Check code quality with Ruff:

```bash
ruff check .
```

Fix auto-fixable issues:

```bash
ruff check --fix .
```

### Type Checking

Run type checking with mypy:

```bash
mypy
```

### Security Scanning

Run security checks with Bandit:

```bash
bandit -r packages/ benchmark.py -ll
```

### Markdown Linting

Lint markdown documentation:

```bash
npx --yes markdownlint-cli@0.39.0 '**/*.md' --ignore node_modules   # the version in ci.yml's MARKDOWNLINT_CLI_VERSION
```

### CI/CD

Every pull request runs two jobs in `ci.yml`, which are the required status checks, plus CodeQL:

- **Checks**: Ruff lint and format, MyPy, Bandit, the pylint duplicate-code audit, markdownlint, the Markdown link
  check (`scripts/check_doc_links.py`), the teaching-page source and docs-media check
  (`scripts/check_doc_sources.py`), and a check that the workflow list in the README is current
- **Unit Tests**: pytest with branch coverage (90% minimum) and the Codecov upload
- **Packages** (not required): builds the three distributions, installs each wheel in a fresh environment and
  imports it (`scripts/release.py smoke-all`; `rl-navigation` may load only NumPy), and runs the robot-side tests
  from the rl-navigation wheel alone on its NumPy floor (1.26)
- **CodeQL**: security analysis, also on every push to `develop` and weekly

A pushed tag `<dist>-v<version>` runs `release.yml`, which drafts that distribution's GitHub Release
([releasing.md](releasing.md)).

A pull request that changes only `docs/**` or Markdown files (outside `packages/` and `tests/`) skips the Python
steps but still reports both checks; markdownlint, the link check and the workflow-list check always run. A new
push to a pull request cancels its older run. Weekly Maintenance re-runs `ci.yml` on `develop` every Monday.

Every action is pinned to a commit SHA, every workflow has a read-only token (CodeQL's results upload and
`release.yml`'s draft release are the two writes), and no workflow reads a secret other than `CODECOV_TOKEN` or calls a
paid LLM API.
`tests/test_ci_policy.py` fails when a workflow breaks any of these rules. Dependabot opens one grouped pull
request per week for pip and one for GitHub Actions.

### Local Development Tools

Pre-commit hooks are available for local development. Install them with:

```bash
pip install -r requirements-dev.txt -c requirements-lock.txt
pre-commit install
```

The Ruff, mypy and Bandit hooks run this environment's tools (`language: system`), at the lock's versions, as CI does:
commit from the activated `.venv`.

The pre-commit hooks will automatically run:

- Code formatting (Ruff)
- Linting checks (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit)
- Markdown linting (markdownlint)
- File validation (trailing whitespace, YAML/JSON syntax, etc.)
