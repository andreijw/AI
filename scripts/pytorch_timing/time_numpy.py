"""Time one DDPG gradient step of the repository's NumPy agent (Pendulum shapes, batch 256).

Uses only the agent's public interface, as the trainer does: with ``max_updates_per_call: 1``, each
``update`` call stores one transition, resets the exploration noise and takes exactly one gradient
step (sample, Bellman targets, critic Adam step, actor Adam step, Polyak); the store and the reset
cost microseconds. Also times a single-observation greedy action. Prints JSON: the median and 90th
percentile in milliseconds. Results and method: docs/pytorch_jetson_timing.md.

    python scripts/pytorch_timing/time_numpy.py        # the project's .venv
"""

import json
import statistics
import time
from collections.abc import Callable
from functools import partial

import numpy as np

from rl_core.agents.ddpg_agent import DDPGAgent


def transitions(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "observations": rng.normal(size=(n, 3)),
        "actions": rng.uniform(-2, 2, size=(n, 1)),
        "rewards": rng.normal(size=n),
        "next_observations": rng.normal(size=(n, 3)),
        "terminated": np.zeros(n, dtype=bool),
    }


def make_agent(hidden: list[int]) -> DDPGAgent:
    agent = DDPGAgent(
        3,
        1,
        {
            "hidden_sizes": hidden,
            "learning_rate": 0.001,
            "gamma": 0.98,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_capacity": 20000,
            "learning_starts": 0,
            "max_updates_per_call": 1,
            "seed": 0,
            "action_low": [-2.0],
            "action_high": [2.0],
        },
    )
    agent.update(transitions(np.random.default_rng(0), 5000))  # fill the buffer
    return agent


def median_ms(fn: Callable[[], object], warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), statistics.quantiles(times, n=10)[-1]


def main() -> None:
    results = {}
    for hidden in ([64, 64], [400, 300]):
        agent = make_agent(hidden)
        one = transitions(np.random.default_rng(1), 1)
        obs = np.random.default_rng(1).normal(size=3)
        step = median_ms(partial(agent.update, one), 20, 300)
        act = median_ms(partial(agent.select_action, obs, training=False), 50, 2000)
        results["x".join(map(str, hidden))] = {"step_ms": step, "act_ms": act}
    print(json.dumps({"backend": "numpy-float64", "results": results}))


if __name__ == "__main__":
    main()
