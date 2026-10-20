"""Tests for the shared discounted-return function."""

import numpy as np
import pytest

from rl_core.agents.returns import discounted_returns, episode_end, gae


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


# ---------------------------------------------------------------------------
# Generalised advantage estimation
# ---------------------------------------------------------------------------
# Two episodes back to back, gamma 0.5 and lambda 0.5 (so gamma * lambda = 0.25). Worked backwards:
#   episode 2, t=4 (done):  delta = 1 - 0.1                 = 0.9    A = 0.9
#              t=3:         delta = 1 + 0.5*0.1 - 0.2       = 0.85   A = 0.85 + 0.25*0.9   = 1.075
#   episode 1, t=2 (done):  delta = 1 - 0.3                 = 0.7    A = 0.7 (trace reset)
#              t=1:         delta = 1 + 0.5*0.3 - 0.4       = 0.75   A = 0.75 + 0.25*0.7   = 0.925
#              t=0:         delta = 1 + 0.5*0.4 - 0.5       = 0.7    A = 0.7 + 0.25*0.925  = 0.93125

GAE_REWARDS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
GAE_VALUES = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
GAE_DONES = np.array([False, False, True, False, True])


def test_gae_matches_hand_computed_values_across_an_episode_boundary():
    advantages = gae(GAE_REWARDS, GAE_VALUES, GAE_DONES, gamma=0.5, lam=0.5)
    np.testing.assert_allclose(advantages, [0.93125, 0.925, 0.7, 1.075, 0.9], rtol=1e-12)


def test_gae_bootstraps_an_unfinished_last_step_from_last_value():
    # delta = 1 + 0.5 * 2.0 - 0.5 = 1.5
    advantages = gae(np.array([1.0]), np.array([0.5]), np.array([False]), 0.5, 0.5, last_value=2.0)
    np.testing.assert_allclose(advantages, [1.5], rtol=1e-12)


def test_gae_ignores_last_value_after_a_finished_episode():
    advantages = gae(np.array([1.0]), np.array([0.5]), np.array([True]), 0.5, 0.5, last_value=2.0)
    np.testing.assert_allclose(advantages, [0.5], rtol=1e-12)


def test_gae_with_lambda_one_is_the_return_minus_the_value():
    """With lambda 1 and episodes that finish, GAE is the Monte-Carlo return minus V(s)."""
    advantages = gae(GAE_REWARDS[:3], GAE_VALUES[:3], GAE_DONES[:3], gamma=0.9, lam=1.0)
    expected = discounted_returns(GAE_REWARDS[:3], 0.9) - GAE_VALUES[:3]
    np.testing.assert_allclose(advantages, expected, rtol=1e-12)


def test_gae_of_an_empty_rollout_is_empty():
    assert gae(np.array([]), np.array([]), np.array([], dtype=bool), 0.9, 0.95).shape == (0,)


# ---------------------------------------------------------------------------
# episode_end
# ---------------------------------------------------------------------------

NEXT = np.arange(6, dtype=np.float32).reshape(3, 2)  # float32, as Pendulum returns


@pytest.mark.parametrize(
    "dones,terminated,bootstraps",
    [
        ([False, False, True], [False, False, False], True),  # time-limit cut
        ([False, False, False], [False, False, False], True),  # cut by max_steps_per_episode
        ([False, False, True], [False, False, True], False),  # true termination
        ([False, False, False], None, True),  # no terminated flags: the dones decide
        ([False, False, True], None, False),
        (None, None, True),  # no flags at all: a cut
    ],
)
def test_episode_end_bootstraps_only_a_cut(dones, terminated, bootstraps):
    ends, bootstrap_obs = episode_end(3, dones, terminated, NEXT)

    np.testing.assert_array_equal(ends, [False, False, True])
    if bootstraps:
        assert bootstrap_obs.dtype == np.float64
        np.testing.assert_array_equal(bootstrap_obs, [4.0, 5.0])
    else:
        assert bootstrap_obs is None


def test_episode_end_without_next_observations_treats_a_cut_as_terminal():
    ends, bootstrap_obs = episode_end(3, [False, False, True], [False, False, False], None)
    np.testing.assert_array_equal(ends, [False, False, True])
    assert bootstrap_obs is None


def test_episode_end_keeps_earlier_episode_ends():
    ends, _ = episode_end(4, [False, True, False, False], None, None)
    np.testing.assert_array_equal(ends, [False, True, False, True])


def test_episode_end_does_not_modify_its_inputs():
    dones = np.array([False, False, False])
    terminated = np.array([False, False, False])
    ends, _ = episode_end(3, dones, terminated, NEXT)
    np.testing.assert_array_equal(dones, [False, False, False])
    np.testing.assert_array_equal(terminated, [False, False, False])
    assert ends is not dones


def test_episode_end_of_an_empty_episode():
    ends, bootstrap_obs = episode_end(0, [], [], np.empty((0, 2)))
    assert ends.shape == (0,)
    assert bootstrap_obs is None


@pytest.mark.parametrize("dones", [[False, True], [False, False, False, True]])
def test_episode_end_rejects_dones_of_the_wrong_length(dones):
    with pytest.raises(ValueError, match="one flag per step"):
        episode_end(3, dones, None, None)
