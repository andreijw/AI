"""rl_navigation.sim.sensors: range sensors, their noise and dropout, and the two presets."""

import math

import numpy as np
import pytest

from rl_navigation.sim.geometry import Arena
from rl_navigation.sim.sensors import RangeSensor, hc_sr04, lidar

EMPTY = Arena(width=5.0, height=5.0)
CENTRE = np.array([2.5, 2.5, 0.0])


def noiseless(sensor: RangeSensor, arena: Arena, pose) -> np.ndarray:
    readings, _ = sensor.read(arena, np.asarray(pose, float), np.random.default_rng(0), 0.0, 0.0)
    return readings


def test_hc_sr04_preset_matches_the_datasheet():
    sensor = hc_sr04()
    np.testing.assert_allclose(sensor.mount_angles, np.radians([-60, -30, 0, 30, 60]), atol=1e-15)
    assert sensor.cone_width == pytest.approx(math.radians(15.0))
    assert sensor.rays_per_cone == 5
    assert (sensor.min_range, sensor.max_range) == (0.02, 4.0)
    assert (sensor.noise_std, sensor.dropout_prob) == (0.01, 0.02)
    assert sensor.num_readings == 5


def test_hc_sr04_cone_rays_span_the_full_15_degrees():
    offsets = hc_sr04().ray_offsets
    assert offsets.shape == (5, 5)
    np.testing.assert_allclose(offsets[2], np.radians([-7.5, -3.75, 0.0, 3.75, 7.5]), atol=1e-15)


def test_lidar_preset_spreads_its_beams_evenly_around_the_robot():
    sensor = lidar()
    assert sensor.num_readings == 64 and sensor.ray_offsets.shape == (64, 1)
    assert (sensor.max_range, sensor.noise_std, sensor.cone_width) == (4.0, 0.01, 0.0)
    angles = np.sort(sensor.mount_angles)
    assert 0.0 in sensor.mount_angles, "One beam should point straight ahead."
    np.testing.assert_allclose(np.diff(angles), 2 * math.pi / 64, atol=1e-12)
    assert angles[0] >= -math.pi and angles[-1] < math.pi


def test_lidar_with_a_narrower_field_of_view_spans_it_end_to_end():
    sensor = lidar(num_beams=5, field_of_view=math.radians(90.0))
    np.testing.assert_allclose(
        sensor.mount_angles, np.radians([-45, -22.5, 0, 22.5, 45]), atol=1e-15
    )


def test_reading_is_the_minimum_over_the_cone():
    # One sensor, cone 15 deg, 3 rays at -7.5, 0 and +7.5 deg. A 0.1 m circle centred 2 m along the
    # +7.5 deg ray is hit by that ray alone, 2 - 0.1 = 1.9 m away; the centre ray reaches the
    # wall 4 m away.
    sensor = RangeSensor(mount_angles=[0.0], cone_width=math.radians(15.0), rays_per_cone=3)
    origin = np.array([1.0, 2.5])
    centre = origin + 2.0 * np.array([math.cos(math.radians(7.5)), math.sin(math.radians(7.5))])
    arena = Arena(5.0, 5.0, circle_centers=[centre], circle_radii=[0.1])
    assert noiseless(sensor, arena, [1.0, 2.5, 0.0])[0] == pytest.approx(1.9, abs=1e-12)


def test_readings_rotate_with_the_robot_heading():
    # Facing +90 deg from (2.5, 1): the forward sensor sees the y = 5 wall 4 m away, the +60 deg
    # sensor (at 150 deg) the x = 0 wall at 2.5 / cos(30 deg) m, nearer than the cone's other rays.
    sensor = RangeSensor(mount_angles=[0.0, math.radians(60.0)], max_range=10.0)
    np.testing.assert_allclose(
        noiseless(sensor, EMPTY, [2.5, 1.0, math.pi / 2]),
        [4.0, 2.5 / math.cos(math.radians(30.0))],
        rtol=1e-12,
    )


def test_readings_are_clipped_to_the_minimum_and_maximum_range():
    # From (4, 2.5): ahead, the circle at (4.51, 2.5) r 0.5 starts at x = 4.01, 0.01 m away, under
    # the 0.02 m minimum; behind, the x = 0 wall is 4 m away, exactly the maximum.
    sensor = RangeSensor(mount_angles=[0.0, math.pi], min_range=0.02, max_range=4.0)
    arena = Arena(5.0, 5.0, circle_centers=[[4.51, 2.5]], circle_radii=[0.5])
    readings, true_ranges = sensor.read(
        arena, np.array([4.0, 2.5, 0.0]), np.random.default_rng(0), 0, 0
    )
    np.testing.assert_allclose(true_ranges, [0.01, 4.0], atol=1e-12)
    assert readings[0] == 0.02, "A reading below the minimum range should read the minimum."
    assert readings[1] == pytest.approx(4.0, abs=1e-12), "A reading at the maximum is kept."
    # From (0.5, 2.5) facing +x the wall is 4.5 m away: the reading is the maximum, the true range not.
    readings, true_ranges = sensor.read(
        EMPTY, np.array([0.5, 2.5, 0.0]), np.random.default_rng(0), 0, 0
    )
    assert true_ranges[0] == pytest.approx(4.5, abs=1e-12), "True ranges are not clipped."
    assert readings[0] == 4.0, "A reading beyond the maximum range should read the maximum."


def test_dropout_reads_the_maximum_range_at_the_given_rate():
    # 100,000 beams from the centre: every true range is 2.5 to 3.54 m, under the 4 m maximum, so
    # a reading of exactly 4 m is a dropout. Binomial standard deviation sqrt(p (1 - p) / n).
    n, p = 100_000, 0.02
    sensor = lidar(num_beams=n, noise_std=0.0, dropout_prob=p)
    readings, true_ranges = sensor.read(EMPTY, CENTRE, np.random.default_rng(5))
    assert true_ranges.max() < 3.6
    rate = float(np.mean(readings == 4.0))
    assert abs(rate - p) < 3 * math.sqrt(p * (1 - p) / n), f"Dropout rate {rate} should be ~{p}."
    np.testing.assert_array_equal(readings[readings != 4.0], true_ranges[readings != 4.0])


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0.0), (1.0, 1.0)])
def test_dropout_probability_at_its_bounds(p, expected):
    sensor = lidar(num_beams=1000, noise_std=0.0)
    readings, _ = sensor.read(EMPTY, CENTRE, np.random.default_rng(1), dropout_prob=p)
    assert np.mean(readings == 4.0) == expected


def test_noise_has_the_given_standard_deviation():
    # The sensor's own noise is 0; the argument gives 0.05, as the environment passes its attribute.
    # Standard error of a sample standard deviation is about sigma / sqrt(2 n).
    n, sigma = 100_000, 0.05
    sensor = lidar(num_beams=n, noise_std=0.0, dropout_prob=0.0)
    readings, true_ranges = sensor.read(EMPTY, CENTRE, np.random.default_rng(7), noise_std=sigma)
    errors = readings - true_ranges
    assert abs(errors.mean()) < 3 * sigma / math.sqrt(n), "Noise should have zero mean."
    assert abs(errors.std() - sigma) < 3 * sigma / math.sqrt(2 * n), (
        f"Noise standard deviation {errors.std()} should be ~{sigma}."
    )


def test_read_falls_back_to_the_sensors_own_noise_and_dropout():
    quiet = lidar(num_beams=1000, noise_std=0.0, dropout_prob=0.0)
    readings, true_ranges = quiet.read(EMPTY, CENTRE, np.random.default_rng(2))
    np.testing.assert_array_equal(readings, true_ranges)
    noisy = lidar(num_beams=1000, noise_std=0.05, dropout_prob=0.0)
    readings, true_ranges = noisy.read(EMPTY, CENTRE, np.random.default_rng(2))
    assert 0.03 < np.std(readings - true_ranges) < 0.07


def test_same_generator_seed_gives_the_same_readings():
    sensor = hc_sr04()
    first, _ = sensor.read(EMPTY, CENTRE, np.random.default_rng(11))
    second, _ = sensor.read(EMPTY, CENTRE, np.random.default_rng(11))
    third, _ = sensor.read(EMPTY, CENTRE, np.random.default_rng(12))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_true_ranges_are_batched_over_poses():
    sensor = hc_sr04()
    poses = np.array([[1.0, 1.0, 0.0], [2.5, 2.5, 1.0], [4.0, 3.0, -2.0]])
    batched = sensor.true_ranges(EMPTY, poses)
    assert batched.shape == (3, 5)
    for i, pose in enumerate(poses):
        _, single = sensor.read(EMPTY, pose, np.random.default_rng(0))
        np.testing.assert_allclose(batched[i], single, rtol=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_range": 4.0, "max_range": 4.0}, "range"),
        ({"rays_per_cone": 0}, "rays_per_cone"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"dropout_prob": 1.5}, "dropout_prob"),
        ({"cone_width": -0.1}, "cone_width"),
    ],
)
def test_range_sensor_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RangeSensor(mount_angles=[0.0], **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"noise_std": -1.0}, "noise_std"), ({"dropout_prob": -0.1}, "dropout")]
)
def test_read_rejects_invalid_noise_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        hc_sr04().read(EMPTY, CENTRE, np.random.default_rng(0), **kwargs)


# --- review fixes -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [({"noise_std": 0.0}, {"noise_std": 0.05}), ({"dropout_prob": 0.0}, {"dropout_prob": 0.5})],
    ids=["noise", "dropout"],
)
def test_read_advances_the_generator_the_same_whatever_the_noise(first, second):
    states = []
    for kwargs in (first, second):
        rng = np.random.default_rng(21)
        hc_sr04().read(EMPTY, CENTRE, rng, **kwargs)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1], "A scan should draw the same numbers at any noise level."


def test_mount_radius_measures_from_the_robot_perimeter():
    # From (1, 2.5) heading +x with mounts 0.1 m out: the forward ray starts at (1.1, 2.5), 3.9 m
    # from x = 5; the +90 deg ray starts at (1, 2.6), 2.4 m from y = 5.
    sensor = RangeSensor(mount_angles=[0.0, math.pi / 2], mount_radius=0.1)
    np.testing.assert_allclose(noiseless(sensor, EMPTY, [1.0, 2.5, 0.0]), [3.9, 2.4], atol=1e-12)


def test_hc_sr04_mounts_sit_on_the_default_robot_perimeter_and_lidar_at_the_centre():
    assert hc_sr04().mount_radius == 0.1
    assert lidar().mount_radius == 0.0


def test_single_lidar_beam_with_a_narrow_field_of_view_points_ahead():
    np.testing.assert_array_equal(lidar(num_beams=1, field_of_view=1.0).mount_angles, [0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"rays_per_cone": 2.5}, "rays_per_cone"), ({"mount_radius": -0.1}, "mount_radius")],
)
def test_range_sensor_rejects_a_fractional_ray_count_and_a_negative_mount_radius(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RangeSensor(mount_angles=[0.0], **kwargs)


def test_lidar_needs_at_least_one_beam():
    with pytest.raises(ValueError, match="num_beams"):
        lidar(num_beams=0)
