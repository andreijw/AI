"""Unit and integration tests for continuous action space agents (Gaussian policies)."""

import os

import numpy as np
import pytest

from rl_core.agents.continuous_actor_critic import ContinuousActorCriticAgent
from rl_core.agents.continuous_ppo import ContinuousPPOAgent


def test_continuous_base_dimension_validation():
    """Validates observation_dim, action_dim, and hidden_dim must be positive."""
    with pytest.raises(ValueError, match="observation_dim must be positive"):
        ContinuousActorCriticAgent(observation_dim=0, action_dim=1, config={})

    with pytest.raises(ValueError, match="action_dim must be positive"):
        ContinuousActorCriticAgent(observation_dim=3, action_dim=0, config={})

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        ContinuousActorCriticAgent(observation_dim=3, action_dim=1, config={"hidden_dim": 0})


def test_continuous_base_action_bounds():
    """Supports both scalar and array action bounds."""
    # Scalar bounds
    agent = ContinuousActorCriticAgent(
        observation_dim=3,
        action_dim=2,
        config={"action_low": -2.0, "action_high": 2.0},
    )
    assert np.allclose(agent.action_low, [-2.0, -2.0])
    assert np.allclose(agent.action_high, [2.0, 2.0])

    # Array bounds
    agent_arr = ContinuousActorCriticAgent(
        observation_dim=3,
        action_dim=2,
        config={"action_low": [-1.0, -2.0], "action_high": [1.0, 2.0]},
    )
    assert np.allclose(agent_arr.action_low, [-1.0, -2.0])
    assert np.allclose(agent_arr.action_high, [1.0, 2.0])


def test_continuous_action_selection():
    """Action selection respects bounds, deterministic in evaluate mode, stochastic in training."""
    agent = ContinuousActorCriticAgent(
        observation_dim=3,
        action_dim=1,
        config={"action_low": -2.0, "action_high": 2.0, "seed": 42},
    )
    state = np.array([0.5, -0.2, 1.0])

    # Deterministic evaluation action
    act_eval1 = agent.select_action(state, evaluate=True)
    act_eval2 = agent.select_action(state, evaluate=True)
    assert np.allclose(act_eval1, act_eval2)
    assert -2.0 <= act_eval1[0] <= 2.0

    # Stochastic exploration actions
    actions = [agent.select_action(state, evaluate=False) for _ in range(20)]
    assert any(not np.allclose(actions[0], a) for a in actions[1:])
    for a in actions:
        assert -2.0 <= a[0] <= 2.0


def test_continuous_save_load(tmp_path):
    """Model weights roundtrip faithfully through NumPy .npz checkpoints."""
    checkpoint_file = str(tmp_path / "continuous_agent.pt")
    agent = ContinuousActorCriticAgent(
        observation_dim=4,
        action_dim=2,
        config={"hidden_dim": 64, "seed": 7},
    )
    state = np.array([0.1, 0.2, 0.3, 0.4])
    orig_action = agent.select_action(state, evaluate=True)

    agent.save(checkpoint_file)
    assert os.path.isfile(checkpoint_file)

    loaded_agent = ContinuousActorCriticAgent(
        observation_dim=4,
        action_dim=2,
        config={"hidden_dim": 64, "seed": 999},
    )
    assert not np.allclose(loaded_agent.select_action(state, evaluate=True), orig_action)

    loaded_agent.load(checkpoint_file)
    assert np.allclose(loaded_agent.select_action(state, evaluate=True), orig_action)

    with pytest.raises(FileNotFoundError):
        loaded_agent.load(str(tmp_path / "non_existent.pt"))


def test_continuous_actor_critic_update():
    """ContinuousActorCriticAgent updates weights on trajectory and returns metrics."""
    agent = ContinuousActorCriticAgent(
        observation_dim=3,
        action_dim=1,
        config={"learning_rate": 1e-3, "seed": 42},
    )

    # Empty trajectory returns empty dict
    assert agent.update([]) == {}

    # Synthetic trajectory
    trajectory = [
        {
            "state": np.array([1.0, 0.0, 0.0]),
            "action": np.array([0.5]),
            "reward": 1.0,
            "next_state": np.array([0.9, 0.1, 0.1]),
            "done": False,
        },
        {
            "state": np.array([0.9, 0.1, 0.1]),
            "action": np.array([0.2]),
            "reward": 0.5,
            "next_state": np.array([0.8, 0.2, 0.2]),
            "done": True,
        },
    ]

    metrics = agent.update(trajectory)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert isinstance(metrics["loss"], float)


def test_continuous_ppo_update():
    """ContinuousPPOAgent updates weights using clipped surrogate objective across epochs."""
    agent = ContinuousPPOAgent(
        observation_dim=3,
        action_dim=1,
        config={
            "learning_rate": 1e-3,
            "ppo_epochs": 4,
            "batch_size": 2,
            "clip_ratio": 0.2,
            "seed": 42,
        },
    )

    # Empty trajectory
    assert agent.update([]) == {}

    # Synthetic multi-step trajectory
    trajectory = [
        {
            "state": np.array([1.0, 0.0, 0.0]),
            "action": np.array([0.5]),
            "reward": 1.0,
            "next_state": np.array([0.9, 0.1, 0.1]),
            "done": False,
        },
        {
            "state": np.array([0.9, 0.1, 0.1]),
            "action": np.array([0.2]),
            "reward": 0.5,
            "next_state": np.array([0.8, 0.2, 0.2]),
            "done": False,
        },
        {
            "state": np.array([0.8, 0.2, 0.2]),
            "action": np.array([-0.1]),
            "reward": 0.0,
            "next_state": np.array([0.0, 0.0, 0.0]),
            "done": True,
        },
    ]

    metrics = agent.update(trajectory)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert isinstance(metrics["policy_loss"], float)
