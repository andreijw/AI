"""Tests for ReinforceAgent implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_core.agents import ReinforceAgent
from tests.agent_internals import probabilities
from tests.helpers import assert_same_checkpoint

_REINFORCE_CONFIG = {"learning_rate": 1e-3, "gamma": 0.99, "hidden_dim": 64, "seed": 0}


def test_reinforce_agent_creation():
    """ReinforceAgent stores dimensions and config correctly."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    assert agent.observation_dim == 4
    assert agent.action_dim == 2
    assert agent.learning_rate == 1e-3
    assert agent.gamma == 0.99
    assert agent.hidden_dim == 64


def test_reinforce_agent_select_action_valid():
    """select_action returns a valid action index during training."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    obs = np.array([0.1, -0.2, 0.05, 0.3])
    actions = [agent.select_action(obs, training=True) for _ in range(100)]
    assert all(a in (0, 1) for a in actions)


def test_reinforce_agent_select_action_stochastic():
    """During training the agent should explore (both actions sampled over many draws)."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    obs = np.zeros(4)
    actions = {agent.select_action(obs, training=True) for _ in range(200)}
    assert len(actions) == 2, "Expected both actions to be sampled during training"


def test_reinforce_agent_select_action_greedy():
    """During evaluation the agent returns a deterministic (greedy) action."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    obs = np.array([1.0, 0.0, -1.0, 0.5])
    actions = [agent.select_action(obs, training=False) for _ in range(20)]
    assert len(set(actions)) == 1


def test_reinforce_agent_update_returns_loss():
    """update() returns a dict with a scalar 'policy_loss' key."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    batch = {
        "observations": np.random.randn(10, 4),
        "actions": np.random.randint(0, 2, size=10),
        "rewards": np.ones(10),
    }
    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    assert "policy_loss" in metrics
    assert isinstance(metrics["policy_loss"], float)


def test_reinforce_agent_update_empty_batch():
    """update() with an empty trajectory returns policy_loss of 0.0."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
    metrics = agent.update(batch)
    assert metrics == {"policy_loss": 0.0}


def test_reinforce_agent_save_load(tmp_path):
    """A loaded agent holds the saved weights (extension-less path)."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    agent.update(
        {
            "observations": np.random.default_rng(1).standard_normal((8, 4)),
            "actions": np.array([0, 1] * 4),
            "rewards": np.arange(8.0),
        }
    )  # trained, so no weight (the zero-initialised biases included) is at its initial value
    save_path = str(tmp_path / "reinforce_test")
    agent.save(save_path)

    agent2 = ReinforceAgent(
        observation_dim=4, action_dim=2, config={**_REINFORCE_CONFIG, "seed": 9}
    )
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_reinforce_agent_save_load_pt_extension(tmp_path):
    """When an explicit .pt path is provided, save/load should use that exact file."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)

    save_path = str(tmp_path / "reinforce_test.pt")
    agent.save(save_path)

    assert os.path.exists(save_path)
    assert not os.path.exists(f"{save_path}.npz")

    agent2 = ReinforceAgent(
        observation_dim=4, action_dim=2, config={**_REINFORCE_CONFIG, "seed": 9}
    )
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_reinforce_agent_save_load_npz_extension(tmp_path):
    """When an explicit .npz path is provided, save/load should use that exact file."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)

    save_path = str(tmp_path / "reinforce_test.npz")
    agent.save(save_path)

    assert os.path.exists(save_path)

    agent2 = ReinforceAgent(
        observation_dim=4, action_dim=2, config={**_REINFORCE_CONFIG, "seed": 9}
    )
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_reinforce_agent_load_pt_legacy_npz_fallback(tmp_path):
    """Loading a .pt path supports legacy '<name>.pt.npz' checkpoints."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    agent.save(str(tmp_path / "reinforce_legacy.pt.npz"))

    agent2 = ReinforceAgent(
        observation_dim=4, action_dim=2, config={**_REINFORCE_CONFIG, "seed": 9}
    )
    agent2.load(str(tmp_path / "reinforce_legacy.pt"))

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_reinforce_agent_action_probabilities_sum_to_one():
    """Policy outputs a valid probability distribution."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    obs = np.array([0.5, -0.3, 0.1, 0.8])
    probs = probabilities(agent, obs)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0)


def test_reinforce_agent_stores_config():
    """Config dictionary is accessible via agent.config."""
    config = {"learning_rate": 5e-4, "gamma": 0.95, "hidden_dim": 32}
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=config)
    assert agent.config == config
