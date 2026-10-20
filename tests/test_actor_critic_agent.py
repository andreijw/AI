"""Tests for ActorCriticAgent implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_core.agents import ActorCriticAgent
from tests.agent_internals import probabilities, state_value
from tests.helpers import assert_same_checkpoint

_AC_CONFIG = {
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 64,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "seed": 0,
}


def test_actor_critic_agent_creation():
    """ActorCriticAgent stores dimensions and config correctly."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    assert agent.observation_dim == 4
    assert agent.action_dim == 2
    assert agent.learning_rate == 1e-3
    assert agent.gamma == 0.99
    assert agent.hidden_dim == 64
    assert agent.value_coef == 0.5
    assert agent.entropy_coef == 0.01


def test_actor_critic_agent_select_action_valid():
    """select_action returns a valid action index during training."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    obs = np.array([0.1, -0.2, 0.05, 0.3])
    actions = [agent.select_action(obs, training=True) for _ in range(100)]
    assert all(a in (0, 1) for a in actions)


def test_actor_critic_agent_select_action_stochastic():
    """During training the agent should explore (both actions sampled over many draws)."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    obs = np.zeros(4)
    actions = {agent.select_action(obs, training=True) for _ in range(200)}
    assert len(actions) == 2, "Expected both actions to be sampled during training"


def test_actor_critic_agent_select_action_greedy():
    """During evaluation the agent returns a deterministic (greedy) action."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    obs = np.array([1.0, 0.0, -1.0, 0.5])
    actions = [agent.select_action(obs, training=False) for _ in range(20)]
    assert len(set(actions)) == 1


def test_actor_critic_agent_update_returns_metrics():
    """update() returns a dict with scalar 'policy_loss', 'value_loss', and 'entropy_bonus' keys."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    batch = {
        "observations": np.random.randn(10, 4),
        "actions": np.random.randint(0, 2, size=10),
        "rewards": np.ones(10),
    }
    metrics = agent.update(batch)
    assert isinstance(metrics, dict)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_bonus" in metrics
    assert isinstance(metrics["policy_loss"], float)
    assert isinstance(metrics["value_loss"], float)
    assert isinstance(metrics["entropy_bonus"], float)
    # Entropy should be non-negative (it's the mean policy entropy, not weighted)
    assert metrics["entropy_bonus"] >= 0.0


def test_actor_critic_agent_update_empty_batch():
    """update() with an empty trajectory returns zero losses."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    batch = {"observations": np.empty((0, 4)), "actions": np.array([]), "rewards": np.array([])}
    metrics = agent.update(batch)
    assert metrics == {"policy_loss": 0.0, "value_loss": 0.0, "entropy_bonus": 0.0}


def test_actor_critic_agent_save_load(tmp_path):
    """A loaded agent holds the saved weights (extension-less path)."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    agent.update(
        {
            "observations": np.random.default_rng(1).standard_normal((8, 4)),
            "actions": np.array([0, 1] * 4),
            "rewards": np.arange(8.0),
        }
    )  # trained, so no weight (the zero-initialised biases included) is at its initial value

    save_path = str(tmp_path / "ac_test")
    agent.save(save_path)

    agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CONFIG, "seed": 9})
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_actor_critic_agent_save_load_pt_extension(tmp_path):
    """When an explicit .pt path is provided, save/load should use that exact file."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)

    save_path = str(tmp_path / "ac_test.pt")
    agent.save(save_path)

    assert os.path.exists(save_path)
    assert not os.path.exists(f"{save_path}.npz")

    agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CONFIG, "seed": 9})
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_actor_critic_agent_save_load_npz_extension(tmp_path):
    """When an explicit .npz path is provided, save/load should use that exact file."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)

    save_path = str(tmp_path / "ac_test.npz")
    agent.save(save_path)

    assert os.path.exists(save_path)

    agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CONFIG, "seed": 9})
    agent2.load(save_path)

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_actor_critic_agent_load_pt_legacy_npz_fallback(tmp_path):
    """Loading a .pt path supports legacy '<name>.pt.npz' checkpoints."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    agent.save(str(tmp_path / "ac_legacy.pt.npz"))

    agent2 = ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CONFIG, "seed": 9})
    agent2.load(str(tmp_path / "ac_legacy.pt"))

    assert_same_checkpoint(agent, agent2, tmp_path)


def test_actor_critic_agent_action_probabilities_sum_to_one():
    """Policy outputs a valid probability distribution and a scalar value."""
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=_AC_CONFIG)
    obs = np.array([0.5, -0.3, 0.1, 0.8])
    probs = probabilities(agent, obs)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0)
    assert isinstance(state_value(agent, obs), float)


def test_actor_critic_agent_stores_config():
    """Config dictionary is accessible via agent.config."""
    config = {"learning_rate": 5e-4, "gamma": 0.95, "hidden_dim": 32}
    agent = ActorCriticAgent(observation_dim=4, action_dim=2, config=config)
    assert agent.config == config


def test_actor_critic_agent_invalid_observation_dim():
    """Non-positive observation_dim should raise ValueError."""
    import pytest

    with pytest.raises(ValueError):
        ActorCriticAgent(observation_dim=0, action_dim=2, config=_AC_CONFIG)


def test_actor_critic_agent_invalid_gamma():
    """Out-of-range gamma should raise ValueError."""
    import pytest

    with pytest.raises(ValueError):
        ActorCriticAgent(observation_dim=4, action_dim=2, config={**_AC_CONFIG, "gamma": 1.5})


def test_actor_critic_agent_value_loss_decreases():
    """Value loss should decrease (on average) after repeated updates on a fixed batch."""
    agent = ActorCriticAgent(
        observation_dim=4,
        action_dim=2,
        config={**_AC_CONFIG, "learning_rate": 1e-2, "entropy_coef": 0.0},
    )
    rng = np.random.default_rng(42)
    batch = {
        "observations": rng.standard_normal((50, 4)),
        "actions": rng.integers(0, 2, size=50),
        "rewards": np.ones(50),
    }

    losses = [agent.update(batch)["value_loss"] for _ in range(50)]
    # Average loss in the last 10 updates should be lower than in the first 10
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (
        "Value loss should trend downward with repeated updates on the same batch"
    )
