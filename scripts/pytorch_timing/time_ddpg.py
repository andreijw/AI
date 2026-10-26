"""Time one DDPG gradient step of either backend (Pendulum shapes, batch 256).

Uses only the agent's public interface, as the trainer does. ``step_ms``: an ``update`` call with
one transition and one gradient step (sample, Bellman targets, critic Adam step, actor Adam step,
Polyak); for the torch backend that includes the fixed cost of every call (staging, mirroring the
weights back to NumPy, reading the metrics). ``episode_step_ms``: an ``update`` call with a
200-step Pendulum episode, so 200 gradient steps at one per transition, divided by 200: what
training pays per gradient step. ``act_ms``: one greedy action for one observation. Each is the
median and 90th percentile in milliseconds.
Results and method: docs/pytorch_jetson_timing.md.

    python scripts/pytorch_timing/time_ddpg.py                          # NumPy, the project's .venv
    .venv-gpu/bin/python scripts/pytorch_timing/time_ddpg.py torch cuda float32
    .venv-gpu/bin/python scripts/pytorch_timing/time_ddpg.py torch cuda float32 eager
    .venv-gpu/bin/python scripts/pytorch_timing/time_ddpg.py torch cpu float64
"""

import json
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from rl_core.agents import BaseAgent, DDPGAgent


def transitions(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "observations": rng.normal(size=(n, 3)),
        "actions": rng.uniform(-2, 2, size=(n, 1)),
        "rewards": rng.normal(size=n),
        "next_observations": rng.normal(size=(n, 3)),
        "terminated": np.zeros(n, dtype=bool),
    }


def make_agent(hidden: list[int], backend: dict[str, Any], per_call: int | None) -> BaseAgent:
    config = {
        "hidden_sizes": hidden,
        "learning_rate": 0.001,
        "gamma": 0.98,
        "tau": 0.005,
        "batch_size": 256,
        "buffer_capacity": 20000,
        "learning_starts": 0,
        "max_updates_per_call": per_call,
        "seed": 0,
        "action_low": [-2.0],
        "action_high": [2.0],
        **backend,
    }
    agent = DDPGAgent.class_for_config(config)(3, 1, config)
    agent.update(transitions(np.random.default_rng(0), 5000))  # fill the buffer
    return agent


def median_ms(fn: Callable[[], object], warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()  # a torch update reads its metrics back, so it has finished on the device
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), statistics.quantiles(times, n=10)[-1]


def backend_settings(argv: list[str]) -> tuple[str, dict[str, Any]]:
    if not argv or argv[0] == "numpy":
        return "numpy-float64", {}
    _, device, dtype, *mode = argv
    graph = None if not mode else mode[0] == "graph"
    name = f"torch-{device}-{dtype}{'' if not mode else '-' + mode[0]}"
    return name, {"backend": "torch", "device": device, "dtype": dtype, "cuda_graph": graph}


def main() -> None:
    name, backend = backend_settings(sys.argv[1:])
    results = {}
    for hidden in ([64, 64], [400, 300]):
        agent = make_agent(hidden, backend, per_call=1)
        one = transitions(np.random.default_rng(1), 1)
        obs = np.random.default_rng(1).normal(size=3)
        step = median_ms(partial(agent.update, one), 20, 300)
        act = median_ms(partial(agent.select_action, obs, training=False), 50, 2000)
        episode_agent = make_agent(hidden, backend, per_call=200)  # caps the fill as well
        episode = transitions(np.random.default_rng(2), 200)
        per_episode = median_ms(partial(episode_agent.update, episode), 2, 15)
        results["x".join(map(str, hidden))] = {
            "step_ms": step,
            "episode_step_ms": tuple(ms / 200 for ms in per_episode),
            "act_ms": act,
        }
    print(json.dumps({"backend": name, "results": results}))


if __name__ == "__main__":
    main()
