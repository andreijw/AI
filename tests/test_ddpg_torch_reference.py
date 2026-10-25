"""The PyTorch DDPG step behind docs/pytorch_jetson_timing.md does what the NumPy agent does.

Skipped unless PyTorch is installed, which it is not in .venv or CI. To run it, install the project
into the PyTorch environment of docs/pytorch_jetson_timing.md and run this file there:

    <torch-venv>/bin/pip install -e ".[dev]"
    <torch-venv>/bin/python -m pytest tests/test_ddpg_torch_reference.py --no-cov
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "pytorch_timing"))

import torch_ddpg  # noqa: E402

from rl_core.agents.ddpg_agent import DDPGAgent  # noqa: E402
from tests.agent_internals import weights  # noqa: E402

BATCH = 32
CONFIG = {
    "hidden_sizes": [16, 12],
    "learning_rate": 0.001,
    "gamma": 0.98,
    "tau": 0.005,
    "batch_size": BATCH,
    "buffer_capacity": 100,
    "learning_starts": 0,
    "max_updates_per_call": 1,
    "seed": 3,
    "action_low": [-2.0],
    "action_high": [2.0],
}


def _transitions(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "observations": rng.normal(size=(n, 3)),
        "actions": rng.uniform(-2, 2, size=(n, 1)),
        "rewards": rng.normal(size=n),
        "next_observations": rng.normal(size=(n, 3)),
        "terminated": rng.uniform(size=n) < 0.2,  # some terminal steps, so the mask matters
    }


def test_torch_ddpg_step_matches_the_numpy_agents_update():
    """One full step, Adam and Polyak included, leaves all four networks with the same weights.

    The buffer holds exactly one batch when the agent trains, so its sample is every transition
    (in another order, which changes only the summation order: hence rtol 1e-10, not 0).
    """
    rng = np.random.default_rng(5)
    agent = DDPGAgent(3, 1, CONFIG)
    first, last = _transitions(rng, BATCH - 1), _transitions(rng, 1)
    assert agent.update(first) == {}, "One transition short of a batch, the agent should not train."

    reference = torch_ddpg.build(3, 1, CONFIG["hidden_sizes"], "cpu", torch.float64)
    torch_ddpg.load_weights(reference, weights(agent))
    before = weights(agent)

    assert agent.update(last), "With a full batch stored, the agent should take one step."
    every = {key: np.concatenate([first[key], last[key]]) for key in first}
    torch_ddpg.ddpg_step(
        reference,
        {
            "obs": torch.from_numpy(every["observations"]),
            "act": torch.from_numpy(every["actions"]),
            "rew": torch.from_numpy(every["rewards"]),
            "next_obs": torch.from_numpy(every["next_observations"]),
            "done": torch.from_numpy(every["terminated"].astype(np.float64)),
        },
    )

    expected, actual = weights(agent), torch_ddpg.numpy_weights(reference)
    for name, value in expected.items():
        assert not np.array_equal(value, before[name]), f"{name} should change in one step."
        np.testing.assert_allclose(
            actual[name],
            value,
            rtol=1e-10,
            atol=1e-14,
            err_msg=f"{name} differs between the PyTorch step and the NumPy agent's update.",
        )
