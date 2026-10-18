"""The shared parsing of continuous action bounds from an agent config."""

import numpy as np
import pytest

from rl_core.agents import ContinuousActorCriticAgent, ContinuousRandomAgent
from rl_core.agents.bounds import action_bounds


def test_bounds_default_to_minus_one_and_one():
    low, high = action_bounds({}, 2)
    np.testing.assert_array_equal(low, [-1.0, -1.0])
    np.testing.assert_array_equal(high, [1.0, 1.0])
    assert low.dtype == high.dtype == np.float32


def test_scalar_bounds_are_broadcast_and_arrays_kept():
    low, high = action_bounds({"action_low": -2.0, "action_high": [1.0, 3.0]}, 2)
    np.testing.assert_array_equal(low, [-2.0, -2.0])
    np.testing.assert_array_equal(high, [1.0, 3.0])


def test_a_zero_dimensional_array_counts_as_a_scalar():
    low, high = action_bounds({"action_low": np.array(-2.0), "action_high": np.float32(2.0)}, 2)
    np.testing.assert_array_equal(low, [-2.0, -2.0])
    np.testing.assert_array_equal(high, [2.0, 2.0])


def test_bounds_of_the_wrong_length_are_rejected():
    with pytest.raises(ValueError, match="action_high"):
        action_bounds({"action_high": [1.0, 2.0, 3.0]}, 2)


@pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (2.0, 1.0), (-np.inf, 1.0), (-1.0, np.nan)])
def test_finite_bounds_with_low_below_high_can_be_required(low, high):
    """DDPG scales tanh by (high - low) / 2, which needs finite bounds with room between them."""
    config = {"action_low": low, "action_high": high}
    with pytest.raises(ValueError, match="action_low|action_high"):
        action_bounds(config, 1, require_finite=True)


def test_without_the_requirement_any_bounds_are_parsed_as_before():
    low, high = action_bounds({"action_low": -np.inf, "action_high": np.inf}, 1)
    assert np.isneginf(low[0]) and np.isposinf(high[0])


@pytest.mark.parametrize("agent_cls", [ContinuousRandomAgent, ContinuousActorCriticAgent])
def test_the_continuous_agents_read_their_bounds_through_the_helper(agent_cls):
    config = {"action_low": -2.0, "action_high": [1.0, 3.0]}
    agent = agent_cls(observation_dim=3, action_dim=2, config=config)
    expected_low, expected_high = action_bounds(config, 2)
    np.testing.assert_array_equal(agent.action_low, expected_low)
    np.testing.assert_array_equal(agent.action_high, expected_high)
