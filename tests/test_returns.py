"""Tests for the shared discounted-return function."""

import numpy as np
import pytest

from rl_core.agents.returns import discounted_returns


@pytest.mark.parametrize(
    "rewards,gamma,expected",
    [
        # G[2] = 1; G[1] = 1 + 0.5 * 1 = 1.5; G[0] = 1 + 0.5 * 1.5 = 1.75
        ([1.0, 1.0, 1.0], 0.5, [1.75, 1.5, 1.0]),
        # G[2] = 2; G[1] = 0 + 0.9 * 2 = 1.8; G[0] = 1 + 0.9 * 1.8 = 2.62
        ([1.0, 0.0, 2.0], 0.9, [2.62, 1.8, 2.0]),
        # gamma 1: plain suffix sums
        ([1.0, 2.0, 3.0], 1.0, [6.0, 5.0, 3.0]),
        # gamma 0: nothing after the current step counts
        ([4.0, -1.0, 2.5], 0.0, [4.0, -1.0, 2.5]),
        # a single step is its own return
        ([3.0], 0.99, [3.0]),
        # no reward, no return
        ([0.0, 0.0, 0.0, 0.0], 0.99, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_discounted_returns_match_hand_computed_values(rewards, gamma, expected):
    returns = discounted_returns(np.array(rewards), gamma)

    np.testing.assert_allclose(returns, expected, rtol=0, atol=1e-12)


def test_discounted_returns_weight_later_rewards_less():
    """One late reward is discounted once per step back from where it was earned."""
    returns = discounted_returns(np.array([0.0, 0.0, 0.0, 8.0]), 0.5)

    np.testing.assert_allclose(returns, [1.0, 2.0, 4.0, 8.0], rtol=0, atol=1e-12)


def test_discounted_returns_of_empty_episode_is_empty():
    returns = discounted_returns(np.array([]), 0.99)

    assert returns.shape == (0,)
    assert returns.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_discounted_returns_are_float64_whatever_the_reward_dtype(dtype):
    rewards = np.array([1, 2, 3], dtype=dtype)

    returns = discounted_returns(rewards, 0.5)

    assert returns.dtype == np.float64
    assert returns.shape == rewards.shape
    # G[2] = 3; G[1] = 2 + 1.5 = 3.5; G[0] = 1 + 1.75 = 2.75
    np.testing.assert_allclose(returns, [2.75, 3.5, 3.0], rtol=0, atol=1e-12)


def test_discounted_returns_do_not_modify_the_rewards():
    rewards = np.array([1.0, 2.0, 3.0])
    before = rewards.copy()

    discounted_returns(rewards, 0.9)

    np.testing.assert_array_equal(rewards, before)


@pytest.mark.parametrize("length", [1, 5, 50])
def test_discounted_returns_have_one_entry_per_step(length):
    assert len(discounted_returns(np.ones(length), 0.99)) == length
