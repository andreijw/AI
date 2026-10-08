"""Unit tests for DQNAgent implementation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.agents.dqn_agent import DQNAgent


def test_dqn_init_valid():
    """Verify DQNAgent initialization with default and custom configs."""
    agent = DQNAgent(observation_dim=4, action_dim=2, config={"seed": 42})
    assert agent.observation_dim == 4
    assert agent.action_dim == 2
    assert agent.epsilon == 1.0
    assert agent.replay_buffer.capacity == 10000
    assert len(agent.replay_buffer) == 0


def test_dqn_dimension_validation():
    """Verify DQNAgent raises ValueError on non-positive dimensions."""
    with pytest.raises(ValueError, match="observation_dim"):
        DQNAgent(observation_dim=0, action_dim=2, config={})
    with pytest.raises(ValueError, match="action_dim"):
        DQNAgent(observation_dim=4, action_dim=-1, config={})
    with pytest.raises(ValueError, match="hidden_dim"):
        DQNAgent(observation_dim=4, action_dim=2, config={"hidden_dim": 0})


def test_dqn_hyperparameter_validation():
    """Verify validation of learning_rate, gamma, epsilon parameters."""
    with pytest.raises(ValueError, match="learning_rate"):
        DQNAgent(4, 2, config={"learning_rate": -0.01})
    with pytest.raises(ValueError, match="gamma"):
        DQNAgent(4, 2, config={"gamma": 0.0})
    with pytest.raises(ValueError, match="gamma"):
        DQNAgent(4, 2, config={"gamma": 1.5})
    with pytest.raises(ValueError, match="epsilon_start"):
        DQNAgent(4, 2, config={"epsilon_start": 1.5})
    with pytest.raises(ValueError, match="epsilon_end"):
        DQNAgent(4, 2, config={"epsilon_end": -0.1})
    with pytest.raises(ValueError, match="epsilon_decay"):
        DQNAgent(4, 2, config={"epsilon_decay": 0.0})
    with pytest.raises(ValueError, match="buffer_capacity"):
        DQNAgent(4, 2, config={"buffer_capacity": 0})
    with pytest.raises(ValueError, match="batch_size"):
        DQNAgent(4, 2, config={"batch_size": 0})


def test_dqn_select_action_greedy_in_eval():
    """Verify greedy action selection during evaluation (training=False)."""
    agent = DQNAgent(observation_dim=4, action_dim=2, config={"seed": 42, "epsilon_start": 1.0})
    obs = np.array([0.1, -0.2, 0.3, -0.4])

    # In eval mode, action must be argmax Q even if epsilon is 1.0
    q_vals, _ = agent._forward_q(obs, target=False)
    expected_action = int(np.argmax(q_vals))

    for _ in range(20):
        action = agent.select_action(obs, training=False)
        assert action == expected_action


def test_dqn_select_action_exploratory_in_training():
    """Verify stochastic exploration during training when epsilon is high."""
    agent = DQNAgent(observation_dim=4, action_dim=2, config={"seed": 42, "epsilon_start": 1.0})
    obs = np.array([0.1, -0.2, 0.3, -0.4])

    # With epsilon=1.0, training mode should explore both actions
    actions = {agent.select_action(obs, training=True) for _ in range(50)}
    assert actions == {0, 1}


def test_dqn_epsilon_decay():
    """Verify epsilon decays towards epsilon_end upon updates."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={
            "seed": 42,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.9,
            "batch_size": 2,
            "min_buffer_size": 2,
        },
    )

    batch = {
        "observations": np.random.randn(5, 4),
        "actions": np.array([0, 1, 0, 1, 0]),
        "rewards": np.ones(5),
        "next_observations": np.random.randn(5, 4),
        "dones": np.zeros(5, dtype=bool),
    }

    initial_eps = agent.epsilon
    metrics = agent.update(batch)
    assert agent.epsilon < initial_eps
    assert metrics["epsilon"] == agent.epsilon
    assert "loss" in metrics
    assert "mean_q" in metrics


def test_dqn_insufficient_buffer_no_update():
    """Verify update returns zero loss if buffer does not meet min_buffer_size."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={"batch_size": 32, "min_buffer_size": 32},
    )

    # Ingest only 5 samples (less than 32)
    batch = {
        "observations": np.random.randn(5, 4),
        "actions": np.array([0, 1, 0, 1, 0]),
        "rewards": np.ones(5),
        "next_observations": np.random.randn(5, 4),
        "dones": np.zeros(5, dtype=bool),
    }

    metrics = agent.update(batch)
    assert metrics["loss"] == 0.0
    assert metrics["mean_q"] == 0.0


def test_dqn_target_network_hard_sync():
    """Verify target network weights are updated after target_update_frequency updates."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={
            "seed": 42,
            "batch_size": 2,
            "min_buffer_size": 2,
            "target_update_frequency": 3,
            "tau": None,
        },
    )

    batch = {
        "observations": np.random.randn(10, 4),
        "actions": np.random.randint(0, 2, size=10),
        "rewards": np.ones(10),
        "next_observations": np.random.randn(10, 4),
        "dones": np.zeros(10, dtype=bool),
    }

    # Initial target and online weights are identical
    np.testing.assert_allclose(agent._W1, agent._target_W1)

    # Update 1: online changes, target remains untouched
    agent.update(batch)
    assert not np.allclose(agent._W1, agent._target_W1)
    saved_target_w1 = np.copy(agent._target_W1)

    # Update 2: target still untouched
    agent.update(batch)
    np.testing.assert_allclose(agent._target_W1, saved_target_w1)

    # Update 3: hard update synchronizes target with online
    agent.update(batch)
    np.testing.assert_allclose(agent._W1, agent._target_W1)


def test_dqn_target_network_soft_sync():
    """Verify Polyak soft update (tau) updates target network smoothly."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={
            "seed": 42,
            "batch_size": 2,
            "min_buffer_size": 2,
            "tau": 0.5,
        },
    )

    batch = {
        "observations": np.random.randn(10, 4),
        "actions": np.random.randint(0, 2, size=10),
        "rewards": np.ones(10),
        "next_observations": np.random.randn(10, 4),
        "dones": np.zeros(10, dtype=bool),
    }

    initial_target_w1 = np.copy(agent._target_W1)
    agent.update(batch)

    # Target should move towards online but not be identical
    assert not np.allclose(agent._target_W1, initial_target_w1)
    assert not np.allclose(agent._target_W1, agent._W1)


def test_dqn_terminal_state_value_zeroed():
    """Verify terminal next_states do not contribute future Q-value to Bellman target."""
    agent = DQNAgent(observation_dim=2, action_dim=2, config={"seed": 42, "gamma": 0.99})

    s_next = np.array([[1.0, 2.0]])
    rewards = np.array([5.0])
    dones_terminal = np.array([True])
    dones_non_terminal = np.array([False])

    target_terminal = agent._compute_bellman_targets(rewards, s_next, dones_terminal)
    target_non_terminal = agent._compute_bellman_targets(rewards, s_next, dones_non_terminal)

    # Terminal target is exactly reward (5.0)
    assert target_terminal[0] == 5.0
    # Non-terminal target includes discounted max next Q
    assert target_non_terminal[0] > 5.0 or target_non_terminal[0] < 5.0


def test_dqn_save_load_checkpoint(tmp_path):
    """Verify saving and loading restores weights and exploration state."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={"seed": 42, "epsilon_start": 0.5},
    )
    checkpoint_path = str(tmp_path / "dqn_checkpoint.pt")
    agent.save(checkpoint_path)

    restored = DQNAgent(observation_dim=4, action_dim=2, config={})
    restored.load(checkpoint_path)

    np.testing.assert_allclose(agent._W1, restored._W1)
    np.testing.assert_allclose(agent._b1, restored._b1)
    np.testing.assert_allclose(agent._W2, restored._W2)
    np.testing.assert_allclose(agent._b2, restored._b2)
    np.testing.assert_allclose(agent._target_W1, restored._target_W1)
    assert agent.epsilon == restored.epsilon


def test_dqn_updates_per_step_multiple_updates():
    """Verify updates_per_step triggers proportional mini-batch gradient updates."""
    agent = DQNAgent(
        observation_dim=4,
        action_dim=2,
        config={
            "seed": 42,
            "batch_size": 2,
            "min_buffer_size": 2,
            "updates_per_step": 0.5,
            "max_updates_per_call": 10,
        },
    )
    batch = {
        "observations": np.random.randn(8, 4),
        "actions": np.random.randint(0, 2, size=8),
        "rewards": np.ones(8),
        "next_observations": np.random.randn(8, 4),
        "dones": np.zeros(8, dtype=bool),
    }
    agent.update(batch)
    # 8 transitions * 0.5 updates_per_step = 4 gradient updates
    assert agent._update_counter == 4


def test_dqn_updates_per_step_validation():
    """Verify invalid updates_per_step and max_updates_per_call raise ValueError."""
    with pytest.raises(ValueError, match="updates_per_step"):
        DQNAgent(observation_dim=4, action_dim=2, config={"updates_per_step": -0.5})

    with pytest.raises(ValueError, match="max_updates_per_call"):
        DQNAgent(observation_dim=4, action_dim=2, config={"max_updates_per_call": 0})
