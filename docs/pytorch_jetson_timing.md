# PyTorch on the Jetson: DDPG Gradient-Step Timing

Measured 2026-10-25 on the development board, to decide whether an optional PyTorch backend is worth adding beside
the NumPy agents. Nothing in the framework uses PyTorch; the scripts live in
[scripts/pytorch_timing/](../scripts/pytorch_timing/), and PyTorch was installed into a separate virtual environment,
not `.venv`.

## Answer in Brief

- **The GPU is usable.** PyTorch's own `torch 2.14.1+cu132` aarch64 wheel from `download.pytorch.org` runs on the
  Orin's GPU. No NVIDIA-specific build was needed.
- **Plain PyTorch on the GPU barely helps.** A DDPG step costs about 22 ms whatever the network size, apparently
  because launching dozens of tiny kernels takes longer than the arithmetic: four times slower than NumPy at 64×64,
  twice as fast at 400×300, and no faster than PyTorch on the CPU there.
- **With CUDA graphs, the GPU is 6.5 times faster than NumPy at 64×64 and 18 times faster at 400×300.** A CUDA
  graph records the whole gradient step once and replays it as a single launch. These GPU figures are float32
  against today's float64 NumPy agent; NumPy in float32 was not timed, so part of the gain may be precision.
- **Acting on the GPU costs about 1.1 ms per environment step**, against 0.04 ms in NumPy. Actions should be chosen on
  the CPU.

## Machine

| Item | Value |
| :--- | :--- |
| Board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB, 6 CPU cores |
| Software | JetPack 7.2.1 (L4T R39.2.1), driver 595.78, CUDA 13.2 under `/usr/local/cuda`, cuDNN 9.20 |
| GPU | Orin (nvgpu), compute capability 8.7 |
| Power mode | **25W** (`nvpmodel -q`), not MAXN SUPER; clocks left to the governor. MAXN SUPER would likely raise both sides; not measured |
| Python | 3.12.3; NumPy agent from this repository at commit `2a60b82` (develop) |

## Results

Pendulum shapes (3 observations, 1 action), batch 256, Adam with learning rate 0.001, Polyak rate 0.005, no
gradient clipping. **Step** is one full DDPG gradient step: sample the batch from the replay buffer, Bellman targets
from the target networks, a critic Adam step, an actor Adam step and the Polyak update. **Act** is one greedy action
for a single observation that starts on the CPU and returns to it. Each figure is the median in milliseconds of 300
steps or 2,000 actions after a warm-up, from the last of three full runs; the earlier two agreed within a few percent.

| Backend | Step, 64×64 | Step, 400×300 | Act, 64×64 | Act, 400×300 |
| :--- | ---: | ---: | ---: | ---: |
| NumPy, float64 (today's agent) | 5.46 | 44.7 | 0.041 | 0.123 |
| PyTorch CPU, float64 | 10.5 | 31.0 | 0.297 | 0.365 |
| PyTorch CPU, float32 | 11.3 | 22.2 | 0.656 | 0.758 |
| PyTorch GPU, float32 | 22.1 | 22.5 | 1.09 | 1.12 |
| PyTorch GPU, float64 | 22.6 | 38.5 | 1.13 | 1.12 |
| **PyTorch GPU, float32, CUDA graph** | **0.84** | **2.50** | 1.11 | 1.12 |

What the table shows:

- **NumPy is the fastest CPU option at 64×64.** PyTorch's per-call overhead outweighs its faster maths. At 400×300,
  PyTorch on the CPU is 1.4 times (float64) to 2 times (float32) faster.
- **Without graphs, the GPU's time does not depend on the network size.** Both sizes take about 22 ms, which
  suggests the cost is launching kernels rather than computing (inferred from the timings, not profiled).
- **The CUDA-graph step is 6.5 times faster than NumPy at 64×64 and 18 times faster at 400×300.**
- **Training cost per environment step**, at one gradient step per transition as the committed DDPG configs use
  (estimated from the table, not measured end to end):
  - 64×64: NumPy 5.5 ms. On the GPU, about 2.0 ms when acting on the GPU, or about 0.9 ms when acting with
    NumPy on the CPU, plus copying the actor's weights back after each update, which was not measured.
  - 400×300: NumPy 45 ms. On the GPU, about 3.6 ms, or 2.6 ms plus that copy when acting on the CPU.

## Is It the Same Computation?

When this was measured, the PyTorch step was a script, `scripts/pytorch_timing/torch_ddpg.py`, checked against the
NumPy agent by `tests/test_ddpg_torch_reference.py`: one step on one batch in float64 on the CPU agreed with the NumPy
agent's `update` to a relative 1e-10, Adam and Polyak included. Both are in git history at commit `8afafc5`, and the
tables above were measured with them. That step is now `TorchDDPGAgent` (`src/rl_core/agents/torch_ddpg_agent.py`),
and `tests/test_torch_ddpg_agent.py` checks it more widely: several episodes of training against the NumPy agent in
float64, with and without gradient clipping; checkpoints in both directions; and on CUDA, the graph against the same
step run eagerly, and loading into an agent whose graph is captured.

## Through the Agent (2026-10-25)

The same measurement through `TorchDDPGAgent` and `DDPGAgent`'s public `update`, with
[time_ddpg.py](../scripts/pytorch_timing/time_ddpg.py), 25W mode. **Per call** is an `update` with one transition
and one gradient step: for the PyTorch agent it includes the fixed cost of every call (staging the batches, mirroring
all weights and Adam states back to NumPy, reading the metrics). **Per step in an episode** is an `update` with a
200-step Pendulum episode, so 200 gradient steps, divided by 200: what training pays. Medians in milliseconds:

| Backend | Per call, 64×64 | Per call, 400×300 | Per step in an episode, 64×64 | Per step in an episode, 400×300 |
| :--- | ---: | ---: | ---: | ---: |
| NumPy, float64 | 5.42 | 44.0 | 5.43 | 36.6 |
| PyTorch CPU, float32 | 11.5 | 26.1 | 10.4 | 21.3 |
| PyTorch GPU, float32, no graph | 25.8 | 30.4 | 17.8 | 17.9 |
| **PyTorch GPU, float32, CUDA graph** | 11.0 | 22.9 | **0.86** | **2.80** |

- **The fixed cost per call is about 10 ms.** Spread over a 200-step episode, it leaves the step at 0.86 ms and
  2.80 ms: 6.3 and 13 times faster than NumPy.
- **NumPy's 400×300 step is faster here than in the table above** (36.6 against 44 ms), probably because repeated
  steps keep its arrays in cache. The ratio through the agent is therefore lower than the 18 times of the bare step.
- **Acting costs the same as in NumPy** (0.04 and 0.12 ms), because the agent acts on the CPU.

## TD3's Two Graphs (2026-10-26)

`TorchTD3Agent` captures two CUDA graphs, a critic-only step and a full one, where DDPG captures one. Measured with
[gpu_memory.py](../scripts/pytorch_timing/gpu_memory.py) at MountainCarContinuous shapes, 400×300, batch 256,
float32: after training, DDPG holds 37.6 MiB on the device (peak 44.3, 56 MiB reserved) and TD3 56.3 MiB (peak 65.3,
104 MiB reserved), of the board's 7.3 GiB shared memory. An `update` of 999 gradient steps, acting included, takes
2.8 s for DDPG and 3.3 s for TD3, so a 300,000-step run spends roughly 15 to 17 minutes in updates.
End to end, `configs/mountaincar_continuous_td3.yaml` (300,000 steps) took about 20 minutes per seed through the
benchmark with the GPU to itself, and about 35 with two benchmark processes sharing it (2026-10-26; CHANGELOG.md,
2026-10-27).

## End-to-End Training (2026-10-25)

`configs/pendulum_ddpg.yaml` through the benchmark: 40,000 steps, of which the first 10,000 are random warm-up
actions, then one gradient step per transition, and 10 greedy evaluation episodes at the end. Same seeds, same
interpreter (`.venv-gpu`) and one run at a time for both backends, 25W mode. The PyTorch runs add only
`backend: torch` to the config; the 400×300 runs change only `hidden_sizes`. Wall-clock is the benchmark's
`duration`: the whole run, environment steps, warm-up and evaluation included.

| Networks | Backend | Seeds | Wall-clock per run (mean) | Solved / held | Solve step | Final training average | Greedy evaluation |
| :--- | :--- | :---: | ---: | :---: | :---: | :---: | :---: |
| 64×64 | NumPy | 1 to 5 | 144 s | 5 / 5 | 31,200 | -144 to -166 | -122 to -185 |
| 64×64 | **PyTorch, GPU** | 1 to 5 | **34 s** | 5 / 5 | 31,200 to 31,600 | -152 to -166 | -127 to -179 |
| 400×300 | NumPy | 1 to 3 | 1,074 s | 3 / 3 | 31,000 to 31,200 | -139 to -167 | -119 to -176 |
| 400×300 | **PyTorch, GPU** | 1 to 3 | **98 s** | 3 / 3 | 31,000 to 31,200 | -140 to -166 | -122 to -180 |

- **Learning is the same.** Every run solved -200 and held it. Seed by seed, the solve step matches to within one
  episode and the final averages and evaluations lie within each other's spread.
- **Whole runs are 4.2 times faster at 64×64 and 11 times faster at 400×300** (18 minutes to 98 seconds). The gain
  is below the per-step 6.3 and 13 times because environment steps, warm-up and evaluation do not speed up.
- **Not measured:** other challenges (MountainCarContinuous DDPG takes 300,000 steps), seeds 4 and 5 at 400×300,
  and MAXN SUPER mode.

## Install Steps That Worked

```bash
python3 -m venv <torch-venv>                     # anywhere outside the repository; 5.2 GiB once installed
<torch-venv>/bin/pip install --upgrade pip
<torch-venv>/bin/pip install "torch==2.14.1+cu132" numpy \
    --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple
<torch-venv>/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line prints `True Orin`. The wheel ships kernels for `sm_80` and newer, not `sm_87`. Kernels built for
compute capability 8.0 run on 8.7, so this works without a warning. Jetson AI Lab's index
(`pypi.jetson-ai-lab.io`) had no JetPack 7 PyTorch wheels on 2026-10-25, so it was not needed. The wheel brings its
own CUDA 13.2 libraries (cuBLAS, cuDNN 9.24), so the system toolkit is not used.

In the repository, `./setup.sh --profile gpu` does all of this: it builds `.venv-gpu` with `.[dev,video,torch]`
from that index and runs the checks. `.venv` stays the CPU profile, without PyTorch, as CI is.

To reproduce the agent table (the first table's scripts are at commit `8afafc5`):

```bash
.venv/bin/python scripts/pytorch_timing/time_ddpg.py
.venv-gpu/bin/python scripts/pytorch_timing/time_ddpg.py torch cuda float32   # or: ... eager, torch cpu float32
.venv-gpu/bin/python -m pytest tests/test_torch_ddpg_agent.py --no-cov
```

## Recommendation (Approved 2026-10-25)

**Add an optional PyTorch backend, for the off-policy agents only, built around CUDA graphs from the start.** The
maintainer approved it on 2026-10-25; DDPG's is `TorchDDPGAgent`, TD3's `TorchTD3Agent` (2026-10-26).

- **Where it pays.** DDPG, and later TD3 and SAC, take one or more gradient steps per environment step, so the
  update dominates training time. At 400×300, the 300,000 steps of `configs/mountaincar_continuous_ddpg.yaml` would
  spend about 3.7 hours on gradient steps in NumPy and about 13 minutes on the GPU. That should make rl-zoo's network
  sizes and the Tier 2 and Tier 3 challenges more practical here, though only DDPG with Pendulum shapes was measured.
  DQN here uses small networks, so it would gain less.
- **Where it does not.** PPO, Actor-Critic and REINFORCE on CartPole or Pendulum use small networks and few updates,
  and NumPy is already fast for them.
- **Shape.** A `[torch]` extra in `pyproject.toml`, chosen per agent in the config. The NumPy agents stay the default
  and the gradient reference, with a test that both take the same step. That test has to run the PyTorch agent in
  float64, or compare with a stated tolerance, because float32 is not bit-exact. (Built: `TorchDDPGAgent` and
  `TorchTD3Agent`, tested by `tests/test_torch_ddpg_agent.py` on the board; CI never installs PyTorch.)
- **Rules to keep.**
  - Act on the CPU: copy the actor's weights to NumPy after each update, or keep a CPU copy of the actor.
  - Keep the replay buffer on the GPU, so sampling stays inside the graph. (Built otherwise: the agent samples from
    the NumPy buffer, so a seed draws the same batches in both backends, and stages them in chunks; the agent table
    shows the cost.)
  - Float32 on the GPU; float64 is slower there.
  - Accumulate the per-step metrics that `update` returns (losses, `mean_q`, gradient norms) on the GPU and read
    them once per `update` call. A `.item()` on every step forces a synchronisation, and the 0.84 ms excludes it.
- **Costs.**
  - A 5.2 GiB dependency.
  - A second implementation of each ported agent to keep matching.
  - CUDA graphs restrict the step: fixed shapes, Adam with `capturable=True`, no CPU synchronisation inside it.
  - The pure-NumPy rule in CLAUDE.md and CONTRIBUTING.md would need rewriting.
  - CI has no GPU, so GPU paths are tested here only.
- **A cheaper first step.** Port DDPG alone and time a full 400×300 Pendulum run end to end, before committing to
  the rest.

**Not yet** is also reasonable: every committed config solves in NumPy today, and only the 400×300 networks and
Tier 2 and later would gain.
