"""Tests for agent implementations."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_cartpole.agents import RandomAgent, ReinforceAgent


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


# ---------------------------------------------------------------------------
# ReinforceAgent tests
# ---------------------------------------------------------------------------

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
    # All greedy calls must return the same action
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


def test_reinforce_agent_weights_change_after_update():
    """Policy weights must change after a non-trivial update."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    w1_before = agent._W1.copy()
    batch = {
        "observations": np.random.randn(20, 4),
        "actions": np.random.randint(0, 2, size=20),
        "rewards": np.ones(20),
    }
    agent.update(batch)
    assert not np.allclose(agent._W1, w1_before), "W1 should change after update"


def test_reinforce_agent_save_load(tmp_path):
    """Saved weights are restored faithfully on load."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)

    save_path = str(tmp_path / "reinforce_test")
    agent.save(save_path)

    agent2 = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    # Mutate agent2 weights so we can verify load overwrites them
    agent2._W1 = np.zeros_like(agent2._W1)

    agent2.load(save_path)

    np.testing.assert_array_equal(agent._W1, agent2._W1)
    np.testing.assert_array_equal(agent._W2, agent2._W2)
    np.testing.assert_array_equal(agent._b1, agent2._b1)
    np.testing.assert_array_equal(agent._b2, agent2._b2)


def test_reinforce_agent_action_probabilities_sum_to_one():
    """Policy outputs a valid probability distribution."""
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=_REINFORCE_CONFIG)
    obs = np.array([0.5, -0.3, 0.1, 0.8])
    probs, _ = agent._policy(obs)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0)


def test_reinforce_agent_compute_returns_discounting():
    """Discounted returns decrease for earlier timesteps with rewards=1 and γ<1."""
    config = {**_REINFORCE_CONFIG, "gamma": 0.99}
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=config)
    assert agent.gamma < 1.0, "Test requires gamma < 1.0 for strict monotone ordering"
    rewards = np.ones(5)
    returns = agent._compute_returns(rewards)
    # Each G_t = 1 + γ*G_{t+1}, so G_0 > G_1 > ... > G_4
    for t in range(len(returns) - 1):
        assert returns[t] > returns[t + 1], f"G[{t}] should be > G[{t+1}]"


def test_reinforce_agent_stores_config():
    """Config dictionary is accessible via agent.config."""
    config = {"learning_rate": 5e-4, "gamma": 0.95, "hidden_dim": 32}
    agent = ReinforceAgent(observation_dim=4, action_dim=2, config=config)
    assert agent.config == config
