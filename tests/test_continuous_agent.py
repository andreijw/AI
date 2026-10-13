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


# ---------------------------------------------------------------------------
# Reproducibility of the PPO mini-batch shuffle
# ---------------------------------------------------------------------------

_PPO_PARAMS = ("_W1", "_b1", "_W_mu", "_b_mu", "_log_std", "_W_v", "_b_v")


@pytest.fixture
def preserve_global_numpy_random_state():
    """Let a test seed NumPy's global generator without leaking that into other tests."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _shuffle_sensitive_trajectory() -> dict[str, np.ndarray]:
    """Sixteen fixed transitions: with batch_size 4 the mini-batch order changes the result."""
    rng = np.random.default_rng(1234)
    n_steps = 16
    return {
        "observations": rng.standard_normal((n_steps, 3)),
        "actions": rng.uniform(-2.0, 2.0, size=(n_steps, 1)),
        "rewards": rng.standard_normal(n_steps),
        "next_observations": rng.standard_normal((n_steps, 3)),
        "dones": np.array([False] * (n_steps - 1) + [True]),
    }


def _ppo_agent(seed: int) -> ContinuousPPOAgent:
    config = {"hidden_dim": 8, "ppo_epochs": 3, "batch_size": 4, "seed": seed}
    return ContinuousPPOAgent(observation_dim=3, action_dim=1, config=config)


def _ppo_params(agent: ContinuousPPOAgent) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(agent, name), dtype=np.float64) for name in _PPO_PARAMS}


def _assert_same_params(first: ContinuousPPOAgent, second: ContinuousPPOAgent) -> None:
    a, b = _ppo_params(first), _ppo_params(second)
    for name in _PPO_PARAMS:
        np.testing.assert_array_equal(
            a[name], b[name], err_msg=f"{name} differs between two agents with the same seed."
        )


def _params_differ(first: ContinuousPPOAgent, second: ContinuousPPOAgent) -> bool:
    a, b = _ppo_params(first), _ppo_params(second)
    return any(not np.array_equal(a[name], b[name]) for name in _PPO_PARAMS)


def test_continuous_ppo_update_is_deterministic_for_a_seed(preserve_global_numpy_random_state):
    """Same seed and data give identical weights, whatever NumPy's global generator holds."""
    trajectory = _shuffle_sensitive_trajectory()
    first, second = _ppo_agent(seed=7), _ppo_agent(seed=7)

    np.random.seed(0)
    first.update(trajectory)
    np.random.seed(1)
    second.update(trajectory)

    _assert_same_params(first, second)


def test_continuous_ppo_update_leaves_global_random_state_untouched():
    """An update neither reads nor advances NumPy's global random state."""
    agent = _ppo_agent(seed=7)
    before = np.random.get_state()

    agent.update(_shuffle_sensitive_trajectory())

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1]), "update() advanced the global NumPy generator."
    assert before[2:] == after[2:], "update() advanced the global NumPy generator."


def test_continuous_ppo_shuffle_draws_from_the_agents_own_generator():
    """Replacing the agent's generator changes the mini-batch order, and so the weights."""
    trajectory = _shuffle_sensitive_trajectory()
    first, second = _ppo_agent(seed=7), _ppo_agent(seed=7)
    second._rng = np.random.default_rng(999)

    first.update(trajectory)
    second.update(trajectory)

    assert _params_differ(first, second), (
        "Weights are identical with a different generator: the shuffle is not using self._rng."
    )


def test_continuous_ppo_update_differs_across_seeds():
    """Different seeds give different weights (initialisation alone already ensures this)."""
    trajectory = _shuffle_sensitive_trajectory()
    first, second = _ppo_agent(seed=1), _ppo_agent(seed=2)

    first.update(trajectory)
    second.update(trajectory)

    assert _params_differ(first, second)


# ---------------------------------------------------------------------------
# The update uses the agent's own discount factor
# ---------------------------------------------------------------------------


def _actor_critic_weights_after_update(gamma: float) -> dict[str, np.ndarray]:
    """Train a continuous actor-critic agent on a fixed batch with uneven, late rewards.

    Three updates, not one: the first Adam step moves every weight by about the learning rate
    whatever the gradient's size, so after a single step two runs differ only where a gradient
    changed sign.
    """
    rng = np.random.default_rng(11)
    batch = {
        "observations": rng.standard_normal((6, 3)),
        "actions": rng.uniform(-2.0, 2.0, size=(6, 1)),
        "rewards": np.array([-1.0, 0.0, -2.0, -1.0, 0.0, -3.0]),
        "next_observations": rng.standard_normal((6, 3)),
        "dones": np.array([False] * 5 + [True]),
    }
    agent = ContinuousActorCriticAgent(
        observation_dim=3, action_dim=1, config={"hidden_dim": 8, "seed": 5, "gamma": gamma}
    )
    for _ in range(3):
        agent.update(batch)
    return {name: np.array(getattr(agent, name), dtype=np.float64) for name in _PPO_PARAMS}


def test_continuous_actor_critic_update_depends_on_the_discount_factor():
    """A different gamma must lead to different weights, well beyond rounding noise."""
    short_sighted = _actor_critic_weights_after_update(gamma=0.5)
    far_sighted = _actor_critic_weights_after_update(gamma=0.99)

    difference = max(
        float(np.max(np.abs(short_sighted[name] - far_sighted[name]))) for name in _PPO_PARAMS
    )
    assert difference > 1e-6, (
        f"Weights match for gamma 0.5 and 0.99 (largest difference {difference:.2e}): "
        "the update is ignoring gamma."
    )


def test_continuous_actor_critic_update_is_identical_for_the_same_discount_factor():
    """Guards the test above: with gamma held fixed, nothing else makes the weights differ."""
    first = _actor_critic_weights_after_update(gamma=0.9)
    second = _actor_critic_weights_after_update(gamma=0.9)

    for name in _PPO_PARAMS:
        np.testing.assert_array_equal(first[name], second[name])
