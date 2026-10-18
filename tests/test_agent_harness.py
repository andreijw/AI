"""Shared test harness validating universal agent contracts across implementations."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import (
    ActorCriticAgent,
    BaseAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)

ALL_AGENTS: list[tuple[type[BaseAgent], dict[str, object]]] = [
    (RandomAgent, {"seed": 42}),
    (ReinforceAgent, {"seed": 42, "learning_rate": 0.001}),
    (ActorCriticAgent, {"seed": 42, "learning_rate": 0.001}),
    (PPOAgent, {"seed": 42, "learning_rate": 0.0003}),
    (
        DQNAgent,
        {"seed": 42, "learning_rate": 0.001, "batch_size": 2, "min_buffer_size": 2},
    ),
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


# ---------------------------------------------------------------------------
# Continuous-action agents: float action vectors inside the bounds
# ---------------------------------------------------------------------------

OBS_DIM, ACT_DIM = 3, 2
CONTINUOUS_AGENTS: list[tuple[type[BaseAgent], dict[str, object]]] = [
    (ContinuousRandomAgent, {"seed": 42}),
    (ContinuousActorCriticAgent, {"seed": 42}),
    (ContinuousPPOAgent, {"seed": 42}),
    (DDPGAgent, {"seed": 42, "batch_size": 2, "learning_starts": 0}),
]


@pytest.mark.parametrize("agent_cls,config", CONTINUOUS_AGENTS)
def test_continuous_select_action_contract(
    agent_cls: type[BaseAgent], config: dict[str, object]
) -> None:
    """A continuous agent returns a float vector of length action_dim inside its bounds."""
    agent = agent_cls(observation_dim=OBS_DIM, action_dim=ACT_DIM, config=config)
    obs = np.array([0.3, -0.2, 0.1])
    for training_mode in (True, False):
        action = agent.select_action(obs, training=training_mode)
        assert isinstance(action, np.ndarray) and action.shape == (ACT_DIM,)
        assert np.issubdtype(action.dtype, np.floating)
        if not training_mode or agent_cls in (ContinuousRandomAgent, DDPGAgent):
            # Gaussian agents train on the unclipped sample; the wrapper clips what it applies.
            assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("agent_cls,config", CONTINUOUS_AGENTS)
def test_continuous_update_batch_contract(
    agent_cls: type[BaseAgent], config: dict[str, object]
) -> None:
    """update accepts the trainer's batch: float (T, A) actions, dones and terminated."""
    agent = agent_cls(observation_dim=OBS_DIM, action_dim=ACT_DIM, config=config)
    rng = np.random.default_rng(0)
    n = 5
    batch = {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.uniform(-1.0, 1.0, size=(n, ACT_DIM)).astype(np.float32),
        "rewards": -np.ones(n, dtype=np.float64),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "dones": np.array([False] * (n - 1) + [True]),
        "terminated": np.zeros(n, dtype=bool),
    }
    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    for name, value in metrics.items():
        assert isinstance(name, str)
        assert isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(value)


@pytest.mark.parametrize("agent_cls,config", CONTINUOUS_AGENTS)
def test_continuous_save_load_roundtrip_contract(
    agent_cls: type[BaseAgent], config: dict[str, object], tmp_path
) -> None:
    """A checkpoint loaded into a fresh agent built with default settings acts the same."""
    agent = agent_cls(observation_dim=OBS_DIM, action_dim=ACT_DIM, config=config)
    path = str(tmp_path / f"{agent_cls.__name__}.pt")
    agent.save(path)
    restored = agent_cls(observation_dim=OBS_DIM, action_dim=ACT_DIM, config={})
    restored.load(path)
    obs = np.array([0.1, -0.2, 0.3])
    if agent_cls is not ContinuousRandomAgent:  # its action is random even when evaluating
        np.testing.assert_array_equal(
            agent.select_action(obs, training=False), restored.select_action(obs, training=False)
        )


@pytest.mark.parametrize("agent_cls,config", CONTINUOUS_AGENTS)
def test_continuous_dimension_validation(
    agent_cls: type[BaseAgent], config: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        agent_cls(observation_dim=0, action_dim=ACT_DIM, config=config)
    with pytest.raises(ValueError):
        agent_cls(observation_dim=OBS_DIM, action_dim=-1, config=config)
