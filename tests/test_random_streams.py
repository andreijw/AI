"""Every agent's random generators come from named streams of its seed (CHANGELOG.md,
2026-10-19).

Committed configs give an agent the environment's seed, and Gymnasium's ``reset(seed=s)`` draws start states from a
generator equal to ``np.random.default_rng(s)``. An agent that seeded its own generators the same way drew the very
numbers the environment did, and its weight init, exploration and sampling drew them from each other.

These tests catch two generators that repeat each other's numbers. They cannot see one generator object read by
two roles (say, a shuffle drawing from the exploration generator); the replay tests that copy `_shuffle_rng` and
`_explore_rng` by name catch that.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
    TD3Agent,
)
from rl_core.seeding import WRAPPER_STREAM, derive_rng
from tests.helpers import torch_backend_cases

SEED = 7
DISCRETE = [
    (RandomAgent, {}),
    (ReinforceAgent, {}),
    (ActorCriticAgent, {}),
    (PPOAgent, {"rollout_steps": 4, "minibatch_size": 2, "gae_lambda": 0.95}),
    (DQNAgent, {"batch_size": 2, "min_buffer_size": 2}),
]
CONTINUOUS = [
    (ContinuousRandomAgent, {}),
    (ContinuousActorCriticAgent, {"n_steps": 2}),
    (ContinuousPPOAgent, {"batch_size": 2}),
    (DDPGAgent, {"batch_size": 2, "learning_starts": 0, "hidden_sizes": [4, 4]}),
    (TD3Agent, {"batch_size": 2, "learning_starts": 0, "hidden_sizes": [4, 4]}),
]
# The PyTorch backend, where it is installed.
CONTINUOUS += torch_backend_cases({"batch_size": 2, "learning_starts": 0, "hidden_sizes": [4, 4]})


def _batch(continuous: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(100)  # test data, not compared
    n = 5
    actions = rng.uniform(-1, 1, (n, 1)) if continuous else rng.integers(0, 2, n)
    return {
        "observations": rng.standard_normal((n, 3 if continuous else 4)),
        "actions": actions,
        "rewards": -np.ones(n),
        "next_observations": rng.standard_normal((n, 3 if continuous else 4)),
        "dones": np.array([False] * (n - 1) + [True]),
        "terminated": np.zeros(n, dtype=bool),
    }


def _generators_used(monkeypatch, agent_cls, config, continuous: bool) -> list[np.random.Generator]:
    """Every generator the agent creates while it is built, acts and updates (kept or thrown away)."""
    created: list[np.random.Generator] = []
    original = np.random.default_rng

    def recording(*args, **kwargs):
        generator = original(*args, **kwargs)
        created.append(copy.deepcopy(generator))  # its state at creation, before any draw
        return generator

    monkeypatch.setattr(np.random, "default_rng", recording)
    obs_dim = 3 if continuous else 4
    agent = agent_cls(obs_dim, 1 if continuous else 2, config={"seed": SEED, **config})
    for _ in range(3):
        agent.select_action(np.zeros(obs_dim), training=True)
    agent.update(_batch(continuous))
    monkeypatch.undo()
    return created


@pytest.mark.parametrize(
    ("agent_cls", "config", "continuous"),
    [(cls, cfg, False) for cls, cfg in DISCRETE] + [(cls, cfg, True) for cls, cfg in CONTINUOUS],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_no_two_generators_from_one_seed_draw_the_same_numbers(
    monkeypatch, agent_cls, config, continuous
):
    generators = _generators_used(monkeypatch, agent_cls, config, continuous)
    assert generators, "the agent should draw randomness from generators of its own"
    environment = [np.random.default_rng(SEED), derive_rng(SEED, WRAPPER_STREAM)]
    first_draws = [tuple(g.random(4)) for g in generators + environment]
    duplicates = len(first_draws) - len(set(first_draws))
    assert duplicates == 0, (
        f"{agent_cls.__name__}: {duplicates} generator(s) repeat another's numbers "
        "(the environment's start states included)"
    )


@pytest.mark.parametrize(
    ("agent_cls", "config", "continuous"),
    [(cls, cfg, False) for cls, cfg in DISCRETE] + [(cls, cfg, True) for cls, cfg in CONTINUOUS],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_runs_stay_reproducible_for_a_seed(agent_cls, config, continuous):
    obs_dim = 3 if continuous else 4
    runs = []
    for _ in range(2):
        agent = agent_cls(obs_dim, 1 if continuous else 2, config={"seed": SEED, **config})
        actions = [agent.select_action(np.full(obs_dim, 0.1), training=True) for _ in range(5)]
        agent.update(_batch(continuous))
        actions += [agent.select_action(np.full(obs_dim, 0.1), training=True) for _ in range(5)]
        runs.append(np.array(actions, dtype=np.float64))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_a_random_agent_checkpoint_saved_before_named_streams_still_loads(tmp_path):
    """Old checkpoints hold only the training generator's state (seeded the old way); loading restores it, so the
    agent continues that old sequence."""
    import json

    old = np.random.default_rng(SEED)
    path = str(tmp_path / "old_random.pt")
    with open(path, "wb") as f:
        np.savez(
            f,
            observation_dim=np.int64(4),
            action_dim=np.int64(2),
            rng_state=json.dumps(old.bit_generator.state),
        )
    agent = RandomAgent(4, 2, config={"seed": 99})
    agent.load(path)
    expected = [int(old.integers(0, 2)) for _ in range(10)]
    assert [agent.select_action(np.zeros(4), training=True) for _ in range(10)] == expected
