"""Tests for PPOAgent implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_cartpole.agents import PPOAgent

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
    """update() with an empty trajectory returns zero losses."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
    metrics = agent.update(batch)
    assert metrics == {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_bonus": 0.0,
        "clip_fraction": 0.0,
    }


def test_ppo_agent_weights_change_after_update():
    """Network weights should change after a non-trivial update."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    w1_before = agent._W1.copy()
    w_pi_before = agent._W_pi.copy()
    w_v_before = agent._W_v.copy()
    batch = {
        "observations": np.random.randn(20, 4),
        "actions": np.random.randint(0, 2, size=20),
        "rewards": np.ones(20),
    }
    agent.update(batch)
    assert not np.allclose(agent._W1, w1_before)
    assert not np.allclose(agent._W_pi, w_pi_before)
    assert not np.allclose(agent._W_v, w_v_before)


def test_ppo_agent_save_load(tmp_path):
    """Saved weights are restored faithfully on load (extension-less path)."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    save_path = str(tmp_path / "ppo_test")
    agent.save(save_path)

    agent2 = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    agent2._W1 = np.zeros_like(agent2._W1)
    agent2.load(save_path)

    np.testing.assert_array_equal(agent._W1, agent2._W1)
    np.testing.assert_array_equal(agent._W_pi, agent2._W_pi)
    np.testing.assert_array_equal(agent._W_v, agent2._W_v)
    assert agent._b_v == pytest.approx(agent2._b_v)


def test_ppo_agent_action_probabilities_sum_to_one():
    """Policy outputs a valid probability distribution."""
    agent = PPOAgent(observation_dim=4, action_dim=2, config=_PPO_CONFIG)
    obs = np.array([0.5, -0.3, 0.1, 0.8])
    probs, value, h = agent._forward(obs)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0)
    assert isinstance(value, float)
    assert h.shape == (64,)


def test_ppo_agent_compute_returns_discounting():
    """Discounted returns decrease over time with γ < 1 and constant rewards."""
    config = {**_PPO_CONFIG, "gamma": 0.99}
    agent = PPOAgent(observation_dim=4, action_dim=2, config=config)
    rewards = np.ones(5)
    returns = agent._compute_returns(rewards)
    for t in range(len(returns) - 1):
        assert returns[t] > returns[t + 1], f"G[{t}] should be > G[{t + 1}]"


def test_ppo_agent_invalid_config():
    """Invalid core hyperparameters raise ValueError."""
    with pytest.raises(ValueError):
        PPOAgent(observation_dim=4, action_dim=2, config={**_PPO_CONFIG, "clip_epsilon": 0.0})
    with pytest.raises(ValueError):
        PPOAgent(observation_dim=4, action_dim=2, config={**_PPO_CONFIG, "ppo_epochs": 0})
