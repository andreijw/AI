"""Tests for agent implementations."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

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


def test_random_agent_save_load():
    """Test random agent save/load (should be no-op)."""
    agent = RandomAgent(observation_dim=4, action_dim=2, config={})

    # These should not raise errors
    agent.save("/tmp/test_agent.pt")
    agent.load("/tmp/test_agent.pt")
