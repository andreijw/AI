"""Shared test harness validating universal agent contracts across implementations."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.agents import (
    ActorCriticAgent,
    BaseAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)

ALL_AGENTS: list[tuple[type[BaseAgent], dict[str, object]]] = [
    (RandomAgent, {"seed": 42}),
    (ReinforceAgent, {"seed": 42, "learning_rate": 0.001}),
    (ActorCriticAgent, {"seed": 42, "learning_rate": 0.001}),
    (PPOAgent, {"seed": 42, "learning_rate": 0.0003}),
]


@pytest.mark.parametrize("agent_cls,config", ALL_AGENTS)
def test_agent_select_action_contract(
    agent_cls: type[BaseAgent], config: dict[str, object]
) -> None:
    """Verify select_action returns a valid discrete action in both training and eval modes."""
    agent = agent_cls(observation_dim=4, action_dim=2, config=config)
    obs = np.array([0.05, -0.1, 0.02, 0.08])

    for training_mode in (True, False):
        action = agent.select_action(obs, training=training_mode)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 2


@pytest.mark.parametrize("agent_cls,config", ALL_AGENTS)
def test_agent_update_batch_contract(agent_cls: type[BaseAgent], config: dict[str, object]) -> None:
    """Verify update accepts the unified batch with next_observations and dones."""
    agent = agent_cls(observation_dim=4, action_dim=2, config=config)
    n = 5
    batch = {
        "observations": np.random.randn(n, 4),
        "actions": np.random.randint(0, 2, size=n),
        "rewards": np.ones(n, dtype=np.float64),
        "next_observations": np.random.randn(n, 4),
        "dones": np.array([False, False, False, False, True]),
    }

    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert isinstance(k, str)
        assert isinstance(v, (float, int, np.floating, np.integer))


@pytest.mark.parametrize("agent_cls,config", ALL_AGENTS)
def test_agent_save_load_roundtrip_contract(
    agent_cls: type[BaseAgent], config: dict[str, object], tmp_path
) -> None:
    """Verify save and load restore state losslessly and preserve greedy action selection."""
    agent = agent_cls(observation_dim=4, action_dim=2, config=config)
    obs = np.array([0.1, -0.2, 0.3, -0.4])

    checkpoint_path = str(tmp_path / f"{agent_cls.__name__}.pt")
    agent.save(checkpoint_path)

    restored = agent_cls(observation_dim=4, action_dim=2, config={})
    restored.load(checkpoint_path)

    # Deterministic evaluation check
    orig_action = agent.select_action(obs, training=False)
    restored_action = restored.select_action(obs, training=False)
    assert orig_action == restored_action


@pytest.mark.parametrize("agent_cls,config", ALL_AGENTS)
def test_agent_dimension_validation(agent_cls: type[BaseAgent], config: dict[str, object]) -> None:
    """Verify non-positive observation_dim or action_dim is rejected."""
    if agent_cls is RandomAgent:
        # RandomAgent does not perform network dimension checks
        return

    with pytest.raises(ValueError):
        agent_cls(observation_dim=0, action_dim=2, config=config)

    with pytest.raises(ValueError):
        agent_cls(observation_dim=4, action_dim=-1, config=config)
