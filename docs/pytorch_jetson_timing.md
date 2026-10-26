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

[tests/test_ddpg_torch_reference.py](../tests/test_ddpg_torch_reference.py) checks it. The step lives in
[torch_ddpg.py](../scripts/pytorch_timing/torch_ddpg.py), which `time_torch.py` also times. The test starts from the
NumPy agent's weights, with a buffer that holds exactly one batch (some steps terminal). It runs one public `update`
of the NumPy agent and one PyTorch step on the same batch, in float64 on the CPU. All four networks then agree to a
relative 1e-10: the critic and actor Adam steps, their order and the Polyak update included. Five deliberate breaks,
on either side, all fail it. The test skips unless PyTorch is installed, so CI and `.venv` never run it; to run it:

```bash
<torch-venv>/bin/pip install -e ".[dev]"
<torch-venv>/bin/python -m pytest tests/test_ddpg_torch_reference.py --no-cov
```

What it does not cover: float32, the GPU and CUDA-graph replay. By hand on 2026-10-25, three replays of the graph
each moved the actor's weights by about the learning rate, moved the target networks and advanced Adam's step
count, so the graph trains rather than replaying a stale step; no committed check repeats that. Sampling differs
slightly in the timings: NumPy draws the batch without replacement, PyTorch with replacement.

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

To reproduce the tables (`time_numpy.py` uses only the agent's public `update` and `select_action`):

```bash
.venv/bin/python scripts/pytorch_timing/time_numpy.py
<torch-venv>/bin/python scripts/pytorch_timing/time_torch.py cuda float32 graph   # or: cpu float64, cuda float32
```

## Recommendation (for the Maintainer to Decide)

**Add an optional PyTorch backend, for the off-policy agents only, built around CUDA graphs from the start.**

- **Where it pays.** DDPG, and later TD3 and SAC, take one or more gradient steps per environment step, so the
  update dominates training time. At 400×300, the 300,000 steps of `configs/mountaincar_continuous_ddpg.yaml` would
  spend about 3.7 hours on gradient steps in NumPy and about 13 minutes on the GPU. That should make rl-zoo's network
  sizes and the Tier 2 and Tier 3 challenges more practical here, though only DDPG with Pendulum shapes was measured.
  DQN here uses small networks, so it would gain less.
- **Where it does not.** PPO, Actor-Critic and REINFORCE on CartPole or Pendulum use small networks and few updates,
  and NumPy is already fast for them.
- **Shape.** A `[torch]` extra in `pyproject.toml`, chosen per agent in the config. The NumPy agents stay the default
  and the gradient reference, with a test that both take the same step, as `test_ddpg_torch_reference.py` does.
  That test has to run the PyTorch agent in float64, or compare with a stated tolerance, because float32 is not
  bit-exact, and it should run in CI once PyTorch is a dependency.
- **Rules to keep.**
  - Act on the CPU: copy the actor's weights to NumPy after each update, or keep a CPU copy of the actor.
  - Keep the replay buffer on the GPU, so sampling stays inside the graph.
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
