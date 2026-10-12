"""Unit tests for ContinuousRandomAgent."""

import numpy as np
import pytest

from rl_core.agents.continuous_random_agent import ContinuousRandomAgent


def test_continuous_random_agent_dimension_validation():
    """Verify validation on observation_dim and action_dim."""
    with pytest.raises(ValueError, match="observation_dim must be positive"):
        ContinuousRandomAgent(observation_dim=0, action_dim=1, config={})
    with pytest.raises(ValueError, match="action_dim must be positive"):
        ContinuousRandomAgent(observation_dim=3, action_dim=0, config={})


def test_continuous_random_agent_action_bounds_and_sampling():
    """Verify continuous random actions stay within configured bounds."""
    config = {
        "action_low": -2.0,
        "action_high": 2.0,
        "seed": 42,
    }
    agent = ContinuousRandomAgent(observation_dim=3, action_dim=1, config=config)
    obs = np.array([0.5, -0.2, 1.0], dtype=np.float32)

    for _ in range(50):
        action = agent.select_action(obs)
        assert action.shape == (1,)
        assert -2.0 <= action[0] <= 2.0

    # Test update is no-op
    metrics = agent.update({"state": obs})
    assert metrics == {}


def test_continuous_random_agent_save_load(tmp_path):
    """Verify save and load roundtrip."""
    config = {"action_low": -3.0, "action_high": 3.0, "seed": 123}
    agent = ContinuousRandomAgent(observation_dim=4, action_dim=2, config=config)

    save_file = str(tmp_path / "random_agent.npz")
    agent.save(save_file)

    agent2 = ContinuousRandomAgent(observation_dim=1, action_dim=1, config={})
    agent2.load(save_file)

    assert agent2.observation_dim == 4
    assert agent2.action_dim == 2
    assert np.allclose(agent2.action_low, [-3.0, -3.0])
    assert np.allclose(agent2.action_high, [3.0, 3.0])
