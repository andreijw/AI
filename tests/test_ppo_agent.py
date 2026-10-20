"""Tests for PPOAgent implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_core.agents import PPOAgent
from tests.agent_internals import probabilities, state_value

_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "hidden_dim": 64,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "clip_epsilon": 0.2,
    "ppo_epochs": 3,
    "seed": 0,
}


def test_ppo_agent_creation():
    """PPOAgent stores dimensions and config correctly."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    assert agent.observation_dim == 4
    assert agent.action_dim == 2
    assert agent.learning_rate == 3e-4
    assert agent.gamma == 0.99
    assert agent.hidden_dim == 64
    assert agent.value_coef == 0.5
    assert agent.entropy_coef == 0.01
    assert agent.clip_epsilon == 0.2
    assert agent.ppo_epochs == 3


def test_ppo_agent_select_action_valid():
    """select_action returns a valid action index during training."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    obs = np.array([0.1, -0.2, 0.05, 0.3])
    actions = [agent.select_action(obs, training=True) for _ in range(100)]
    assert all(a in (0, 1) for a in actions)


def test_ppo_agent_select_action_stochastic():
    """During training the agent should explore."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    obs = np.zeros(4)
    actions = {agent.select_action(obs, training=True) for _ in range(200)}
    assert len(actions) == 2


def test_ppo_agent_select_action_greedy():
    """During evaluation the agent returns a deterministic (greedy) action."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    obs = np.array([1.0, 0.0, -1.0, 0.5])
    actions = [agent.select_action(obs, training=False) for _ in range(20)]
    assert len(set(actions)) == 1


def test_ppo_agent_update_returns_metrics():
    """update() returns policy/value/entropy/clip metrics."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    batch = {
        "observations": np.random.randn(20, 4),
        "actions": np.random.randint(0, 2, size=20),
        "rewards": np.ones(20),
    }
    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_bonus" in metrics
    assert "clip_fraction" in metrics
    assert isinstance(metrics["policy_loss"], float)
    assert isinstance(metrics["value_loss"], float)
    assert isinstance(metrics["entropy_bonus"], float)
    assert isinstance(metrics["clip_fraction"], float)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0


def test_ppo_agent_update_empty_batch():
    """update() with an empty trajectory trains on nothing and reports no metrics."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
    assert agent.update(batch) == {}


def test_ppo_agent_save_load(tmp_path):
    """A loaded agent holds the saved weights: saving it again writes the same arrays."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    agent.update(
        {
            "observations": np.random.default_rng(1).standard_normal((8, 4)),
            "actions": np.array([0, 1] * 4),
            "rewards": np.arange(8.0),
        }
    )  # trained, so no weight (the zero-initialised biases included) is at its initial value
    save_path = str(tmp_path / "ppo_test")
    agent.save(save_path)

    agent2 = PPOAgent(observation_dim=4, action_dim=2, config={**_PPO_CONFIG, "seed": 99})
    agent2.load(save_path)
    agent2.save(str(tmp_path / "resaved"))

    with np.load(f"{save_path}.npz") as saved, np.load(tmp_path / "resaved.npz") as resaved:
        assert saved.files == resaved.files
        for key in saved.files:
            np.testing.assert_array_equal(resaved[key], saved[key], err_msg=f"{key} differs.")
    observations = np.random.default_rng(0).standard_normal((20, 4))
    assert [agent2.select_action(o, training=False) for o in observations] == [
        agent.select_action(o, training=False) for o in observations
    ], "The loaded agent must act greedily exactly as the saved one."


def test_ppo_agent_action_probabilities_sum_to_one():
    """Policy outputs a valid probability distribution and a scalar value."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    obs = np.array([0.5, -0.3, 0.1, 0.8])
    probs = probabilities(agent, obs)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0)
    assert isinstance(state_value(agent, obs), float)


def test_ppo_agent_invalid_config():
    """Invalid core hyperparameters raise ValueError."""
    with pytest.raises(ValueError):
        PPOAgent(observation_dim=4, action_dim=2, config={**_PPO_CONFIG, "clip_epsilon": 0.0})
    with pytest.raises(ValueError):
        PPOAgent(observation_dim=4, action_dim=2, config={**_PPO_CONFIG, "ppo_epochs": 0})
