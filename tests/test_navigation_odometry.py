"""Simulated wheel encoders: hand-computed wheel travel, the noise model and its draws."""

import math

import numpy as np
import pytest

from rl_navigation.sim import WheelOdometry


@pytest.mark.parametrize(
    ("v", "omega", "expected"),
    [
        (0.3, 0.0, [0.03, 0.03]),  # straight: both wheels roll v dt
        (0.0, 2.0, [-0.02, 0.02]),  # in place: each wheel rolls omega dt b / 2 = 0.02 m
        (0.5, 1.0, [0.04, 0.06]),  # arc: v dt -+ omega dt b / 2
    ],
)
def test_wheel_travel_by_hand(v, omega, expected):
    travel = WheelOdometry(wheel_base=0.2).wheel_travel(v, omega, dt=0.1)
    np.testing.assert_allclose(
        travel, expected, rtol=1e-12, err_msg="Wheel travel should be (v dt -+ omega dt b/2)."
    )


def test_increments_invert_the_wheel_travel():
    distance, turn = WheelOdometry(wheel_base=0.2).increments([0.04, 0.06])
    assert distance == pytest.approx(0.05, rel=1e-12), "Distance is the mean of the wheels."
    assert turn == pytest.approx(0.1, rel=1e-12), "Turn is (right - left) / wheel base."


def test_wheel_covariance_grows_with_the_distance_each_wheel_rolls():
    covariance = WheelOdometry(noise_per_metre=2.5e-4).wheel_covariance([-0.04, 0.06])
    np.testing.assert_allclose(
        covariance,
        np.diag([1e-5, 1.5e-5]),
        rtol=1e-12,
        err_msg="Each wheel's variance should be k |travel|, the wheels independent.",
    )


def test_read_noise_matches_the_model_over_many_draws():
    odometry = WheelOdometry(wheel_base=0.2, noise_per_metre=2.5e-4)
    rng = np.random.default_rng(0)
    draws = np.array([odometry.read(0.5, 1.0, 0.1, rng) for _ in range(20_000)])
    expected_var = np.array([2.5e-4 * 0.04, 2.5e-4 * 0.06])
    standard_error = expected_var * math.sqrt(2.0 / (len(draws) - 1))
    assert np.all(np.abs(draws.var(axis=0, ddof=1) - expected_var) < 3 * standard_error), (
        f"Sample variances {draws.var(axis=0, ddof=1)} should be k |travel| = {expected_var}."
    )
    assert np.all(np.abs(draws.mean(axis=0) - [0.04, 0.06]) < 3 * np.sqrt(expected_var / 20_000)), (
        "The encoder noise should have zero mean."
    )


def test_a_still_robot_reads_exactly_zero_and_still_draws_twice():
    odometry = WheelOdometry()
    rng, reference = np.random.default_rng(5), np.random.default_rng(5)
    np.testing.assert_array_equal(
        odometry.read(0.0, 0.0, 0.1, rng), [0.0, 0.0], err_msg="No motion, no wheel travel."
    )
    reference.standard_normal(2)
    assert rng.random() == reference.random(), (
        "read should draw two normals whatever the motion, so the stream does not depend on it."
    )


@pytest.mark.parametrize(
    "kwargs", [{"wheel_base": 0.0}, {"wheel_base": -0.2}, {"noise_per_metre": -1e-4}]
)
def test_invalid_odometry_is_refused(kwargs):
    with pytest.raises(ValueError, match="WheelOdometry"):
        WheelOdometry(**kwargs)
