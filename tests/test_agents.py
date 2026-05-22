"""Tests for agent implementations."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_cartpole.agents import RandomAgent


def test_random_agent_creation():
    """Test random agent creation."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})
    assert agent is not None
    assert agent.observation_dim == 4
    assert agent.action_dim == 2


def test_random_agent_select_action():
    """Test random agent action selection."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})
    obs = np.array([0.1, 0.2, 0.3, 0.4])

    # Test multiple action selections
    actions = [agent.select_action(obs) for _ in range(100)]

    # Check actions are valid
    assert all(action in [0, 1] for action in actions)

    # Check both actions are selected at least once (with high probability)
    assert len(set(actions)) > 1


def test_random_agent_update():
    """Test random agent update (should be no-op)."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})
    batch = {
        "observations": np.random.randn(10, 4),
        "actions": np.random.randint(0, 2, size=10),
        "rewards": np.random.randn(10),
    }

    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    assert len(metrics) == 0  # No metrics for random agent


def test_random_agent_save_load(tmp_path):
    """Test random agent save/load preserves RNG progression."""
    obs = np.zeros(4)
    agent = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 123})

    _ = [agent.select_action(obs) for _ in range(10)]
    checkpoint_path = str(tmp_path / "random_agent")
    agent.save(checkpoint_path)
    assert os.path.exists(f"{checkpoint_path}.npz")

    expected_future = [agent.select_action(obs) for _ in range(10)]
    restored = RandomAgent(observation_dim=4, action_dim=2, config={})
    restored.load(checkpoint_path)
    restored_future = [restored.select_action(obs) for _ in range(10)]

    assert restored_future == expected_future


def test_random_agent_select_action_training_false():
    """Test that training=False does not change action selection behaviour."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})
    obs = np.array([0.1, 0.2, 0.3, 0.4])

    actions = [agent.select_action(obs, training=False) for _ in range(50)]
    assert all(action in [0, 1] for action in actions)


def test_random_agent_larger_action_dim():
    """Test random agent with more than 2 actions."""
    agent = RandomAgent(observation_dim=4, action_dim=5, config={})
    obs = np.zeros(4)

    actions = [agent.select_action(obs) for _ in range(200)]
    assert all(0 <= a < 5 for a in actions)
    # With 200 samples across 5 actions, every action should appear
    assert len(set(actions)) == 5


def test_random_agent_stores_config():
    """Config should be stored on the agent."""
    config = {"lr": 0.01, "gamma": 0.99}
    agent = RandomAgent(observation_dim=4, action_dim=2, config=config)
    assert agent.config == config


def test_random_agent_update_ignores_batch_content():
    """update() must return an empty dict regardless of batch content."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})

    # Empty batch
    assert agent.update({}) == {}

    # Batch with extra keys
    batch = {"observations": np.zeros((5, 4)), "bogus_key": "ignored"}
    assert agent.update(batch) == {}


def test_random_agent_seeded_reproducibility():
    """The same seed should produce the same action sequence."""
    obs = np.zeros(4)

    agent_a = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 99})
    agent_b = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 99})

    actions_a = [agent_a.select_action(obs) for _ in range(20)]
    actions_b = [agent_b.select_action(obs) for _ in range(20)]

    assert actions_a == actions_b


def test_random_agent_different_seeds_differ():
    """Different seeds should parameterize independent, deterministic RNG streams."""
    obs = np.zeros(4)

    # Two agents with the same seed=1 should produce identical sequences
    agent_a1 = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 1})
    agent_a2 = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 1})

    # Two agents with the same seed=2 should also produce identical sequences
    agent_b1 = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 2})
    agent_b2 = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 2})

    actions_a1 = [agent_a1.select_action(obs) for _ in range(50)]
    actions_a2 = [agent_a2.select_action(obs) for _ in range(50)]
    actions_b1 = [agent_b1.select_action(obs) for _ in range(50)]
    actions_b2 = [agent_b2.select_action(obs) for _ in range(50)]

    # For a given seed, sequences must be reproducible across agent instances
    assert actions_a1 == actions_a2
    assert actions_b1 == actions_b2

    # Different seeds should, with overwhelming probability, produce different sequences
    assert actions_a1 != actions_b1


def test_random_agent_no_seed_does_not_raise():
    """RandomAgent with no seed (seed=None) should work without raising."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})
    obs = np.zeros(4)
    action = agent.select_action(obs)
    assert action in [0, 1]


def test_random_agent_load_dimension_mismatch_raises(tmp_path):
    """Loading checkpoint with mismatched dimensions should fail fast."""
    path = str(tmp_path / "random_checkpoint")
    source = RandomAgent(observation_dim=4, action_dim=2, config={"seed": 1})
    source.save(path)

    target = RandomAgent(observation_dim=4, action_dim=3, config={})
    with pytest.raises(ValueError, match="Checkpoint dimensions do not match"):
        target.load(path)
