"""Exploration noise for deterministic policies: Gaussian and Ornstein-Uhlenbeck.

The OU process is SB3's ``OrnsteinUhlenbeckActionNoise`` discretisation, with mean 0 and a zero start:
x_{t+1} = x_t - theta * x_t * dt + sigma * sqrt(dt) * eps, eps ~ N(0, I).
"""

import numpy as np
import pytest

from rl_core.agents.exploration_noise import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    make_exploration_noise,
)


def test_gaussian_noise_is_the_plain_normal_draw():
    """The Gaussian path must draw exactly what DDPG drew before, so Pendulum runs repeat."""
    noise = GaussianNoise(std=0.3, action_dim=2)
    expected = np.random.default_rng(4).normal(0.0, 0.3, 2)
    np.testing.assert_array_equal(noise.sample(np.random.default_rng(4)), expected)


def test_ou_noise_follows_the_recursion_from_zero():
    sigma, theta, dt = 0.5, 0.15, 0.01
    noise = OrnsteinUhlenbeckNoise(sigma=sigma, action_dim=2, theta=theta, dt=dt)
    eps = np.random.default_rng(9).standard_normal((3, 2))
    # x1 = 0.05·eps1; x2 = x1·(1 - 0.0015) + 0.05·eps2; x3 likewise.
    x = np.zeros(2)
    expected = []
    for e in eps:
        x = x - theta * x * dt + sigma * np.sqrt(dt) * e
        expected.append(x.copy())

    rng = np.random.default_rng(9)
    samples = [noise.sample(rng) for _ in range(3)]
    np.testing.assert_allclose(samples, expected, rtol=1e-12, err_msg="OU recursion")
    np.testing.assert_allclose(samples[0], 0.05 * eps[0], rtol=1e-12, err_msg="starts at zero")


def test_ou_noise_reset_returns_the_state_to_zero():
    noise = OrnsteinUhlenbeckNoise(sigma=0.5, action_dim=1)
    rng = np.random.default_rng(2)
    for _ in range(50):
        noise.sample(rng)
    noise.reset()
    eps = np.random.default_rng(5).standard_normal(1)
    np.testing.assert_allclose(
        noise.sample(np.random.default_rng(5)),
        0.5 * np.sqrt(0.01) * eps,
        rtol=1e-12,
        err_msg="after reset the first sample is one increment from zero",
    )


def test_ou_noise_has_the_stationary_spread_and_correlation_of_the_process():
    """With theta·dt = 0.05 the stationary std is sigma·sqrt(dt / (1 - (1 - theta·dt)²)) and the
    lag-1 autocorrelation is 1 - theta·dt; white noise would have autocorrelation 0."""
    sigma, theta, dt = 0.4, 0.5, 0.1
    noise = OrnsteinUhlenbeckNoise(sigma=sigma, action_dim=1, theta=theta, dt=dt)
    rng = np.random.default_rng(0)
    for _ in range(200):  # burn in: 10 correlation times
        noise.sample(rng)
    xs = np.array([noise.sample(rng)[0] for _ in range(40_000)])
    decay = 1.0 - theta * dt
    np.testing.assert_allclose(xs.std(), sigma * np.sqrt(dt / (1.0 - decay**2)), rtol=0.05)
    np.testing.assert_allclose(np.corrcoef(xs[:-1], xs[1:])[0, 1], decay, atol=0.01)


def test_gaussian_reset_is_a_no_op():
    noise = GaussianNoise(std=1.0, action_dim=1)
    noise.reset()
    np.testing.assert_array_equal(
        noise.sample(np.random.default_rng(1)), np.random.default_rng(1).normal(0.0, 1.0, 1)
    )


@pytest.mark.parametrize(
    ("config", "kind"),
    [
        ({}, GaussianNoise),
        ({"noise_type": "gaussian", "noise_std": 0.2}, GaussianNoise),
        ({"noise_type": "ornstein_uhlenbeck", "noise_std": 0.5}, OrnsteinUhlenbeckNoise),
    ],
)
def test_the_factory_builds_the_configured_noise(config, kind):
    assert type(make_exploration_noise(config, action_dim=2, default_std=0.1)) is kind


def test_the_factory_passes_the_ou_settings_through():
    config = {"noise_type": "ornstein_uhlenbeck", "noise_std": 0.5, "ou_theta": 0.3, "ou_dt": 0.1}
    noise = make_exploration_noise(config, action_dim=1, default_std=0.1)
    eps = np.random.default_rng(3).standard_normal((2, 1))
    rng = np.random.default_rng(3)
    noise.sample(rng)
    x1 = 0.5 * np.sqrt(0.1) * eps[0]
    np.testing.assert_allclose(noise.sample(rng), x1 * (1 - 0.03) + 0.5 * np.sqrt(0.1) * eps[1])


def test_the_factory_takes_the_agents_default_std():
    noise = make_exploration_noise({}, action_dim=1, default_std=0.25)
    np.testing.assert_array_equal(
        noise.sample(np.random.default_rng(8)), np.random.default_rng(8).normal(0.0, 0.25, 1)
    )


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"noise_type": "pink"}, "noise_type"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_type": "ornstein_uhlenbeck", "ou_theta": 0.0}, "ou_theta"),
        ({"noise_type": "ornstein_uhlenbeck", "ou_dt": -0.01}, "ou_dt"),
        ({"noise_type": "ornstein_uhlenbeck", "ou_theta": 20.0, "ou_dt": 0.1}, "theta.*dt"),
        ({"ou_theta": 0.15}, "ou_theta"),
        ({"noise_type": "gaussian", "ou_dt": 0.01}, "ou_dt"),
    ],
)
def test_the_factory_rejects_invalid_settings(bad, match):
    with pytest.raises(ValueError, match=match):
        make_exploration_noise(bad, action_dim=1, default_std=0.1)
