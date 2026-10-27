"""Device memory and per-update time of DDPG and TD3 on the PyTorch backend, at 400x300.

MountainCarContinuous shapes (2 observations, 1 action), batch 256, float32 on CUDA with graphs:
DDPG captures one graph, TD3 two (a critic-only and a full step). Each agent fills its buffer with
one 999-step episode, trains on a second (capturing its graphs), then times a third:
``update_s`` is that call, 999 gradient steps with acting included. ``allocated_mib`` is what the
agent holds on the device after training, ``peak_mib`` the most it held, ``reserved_mib`` what
PyTorch's caching allocator had reserved. Results: docs/pytorch_jetson_timing.md.

    .venv-gpu/bin/python scripts/pytorch_timing/gpu_memory.py
"""

import json
import time

import numpy as np
import torch

from rl_core.agents import BaseAgent, DDPGAgent, TD3Agent

OBS, ACT, EPISODE = 2, 1, 999


def play(agent: BaseAgent, rng: np.random.Generator) -> dict[str, float]:
    obs = rng.normal(size=(EPISODE + 1, OBS))
    actions = np.array([agent.select_action(o, training=True) for o in obs[:-1]])
    return agent.update(
        {
            "observations": obs[:-1],
            "actions": actions,
            "rewards": rng.normal(size=EPISODE),
            "next_observations": obs[1:],
            "terminated": np.zeros(EPISODE, dtype=bool),
        }
    )


def measure(agent_cls: type[BaseAgent]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    config = {
        "seed": 0,
        "hidden_sizes": [400, 300],
        "batch_size": 256,
        "buffer_capacity": 100_000,
        "learning_starts": EPISODE,
        "action_low": [-1.0],
        "action_high": [1.0],
        "backend": "torch",
        "device": "cuda",
    }
    agent = agent_cls.class_for_config(config)(OBS, ACT, config)
    rng = np.random.default_rng(0)
    play(agent, rng)  # warm-up only
    play(agent, rng)  # trains, capturing the graphs
    torch.cuda.synchronize()
    allocated, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    start = time.perf_counter()
    metrics = play(agent, rng)  # an update reads its metrics back, so it has finished
    elapsed = time.perf_counter() - start
    assert all(np.isfinite(v) for v in metrics.values())
    mib = 2**20
    return {
        "allocated_mib": round((allocated - base) / mib, 1),
        "peak_mib": round((peak - base) / mib, 1),
        "reserved_mib": round(reserved / mib, 1),
        "update_s": round(elapsed, 2),
    }


def main() -> None:
    results = {cls.__name__: measure(cls) for cls in (DDPGAgent, TD3Agent)}
    total = torch.cuda.mem_get_info()[1]
    print(json.dumps({"device_total_gib": round(total / 2**30, 2), "results": results}))


if __name__ == "__main__":
    main()
