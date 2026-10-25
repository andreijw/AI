"""Time one DDPG gradient step in PyTorch, matching the NumPy agent's step (Pendulum shapes).

The step is torch_ddpg.ddpg_step (checked against the NumPy agent by
tests/test_ddpg_torch_reference.py), on a batch of 256 sampled from a replay buffer held on the
device. Also times one greedy action for a single observation coming from the CPU, the way the
trainer would call it. Needs a separate virtual environment with PyTorch
(docs/pytorch_jetson_timing.md); prints JSON as time_numpy.py does.

    <torch-venv>/bin/python scripts/pytorch_timing/time_torch.py cuda float32 graph
    <torch-venv>/bin/python scripts/pytorch_timing/time_torch.py cpu float64
"""

import json
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial

import torch
import torch_ddpg

OBS, ACT, BATCH, BUFFER = 3, 1, 256, 5000


def make_buffer(device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(BUFFER, OBS, device=device, dtype=dtype),
        "act": torch.rand(BUFFER, ACT, device=device, dtype=dtype) * 4 - 2,
        "rew": torch.randn(BUFFER, device=device, dtype=dtype),
        "next_obs": torch.randn(BUFFER, OBS, device=device, dtype=dtype),
        "done": torch.zeros(BUFFER, device=device, dtype=dtype),
    }


def sample_and_step(agent: torch_ddpg.TorchDDPG, buffer: dict[str, torch.Tensor]) -> None:
    """Sample a batch on the device (inside the graph when captured), then take one step."""
    idx = torch.randint(0, BUFFER, (BATCH,), device=buffer["obs"].device)
    torch_ddpg.ddpg_step(agent, {key: value[idx] for key, value in buffer.items()})


def graphed(step: Callable[[], None]) -> Callable[[], None]:
    """Warm up on a side stream, capture one step, and return a function that replays it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph.replay


def act(agent: torch_ddpg.TorchDDPG, obs_cpu, device: str, dtype: torch.dtype):
    with torch.no_grad():
        obs = torch.as_tensor(obs_cpu, dtype=dtype).to(device).unsqueeze(0)
        return torch_ddpg.policy(agent, agent.actor, obs)[0].cpu().numpy()


def median_ms(fn: Callable[[], object], device: str, warmup: int, repeats: int):
    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), statistics.quantiles(times, n=10)[-1]


def main() -> None:
    device = sys.argv[1]
    dtype = {"float32": torch.float32, "float64": torch.float64}[sys.argv[2]]
    graph = len(sys.argv) > 3 and sys.argv[3] == "graph"  # replay the whole step as one CUDA graph
    torch.manual_seed(0)
    results = {}
    for hidden in ([64, 64], [400, 300]):
        agent = torch_ddpg.build(OBS, ACT, hidden, device, dtype, capturable=graph)
        step = partial(sample_and_step, agent, make_buffer(device, dtype))
        obs = torch.randn(OBS).numpy()
        results["x".join(map(str, hidden))] = {
            "step_ms": median_ms(graphed(step) if graph else step, device, 50, 300),
            "act_ms": median_ms(partial(act, agent, obs, device, dtype), device, 50, 2000),
        }
    info = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = torch.cuda.get_device_capability(0)
    backend = f"torch-{device}-{sys.argv[2]}{'-graph' if graph else ''}"
    print(json.dumps({"backend": backend, "info": info, "results": results}))


if __name__ == "__main__":
    main()
