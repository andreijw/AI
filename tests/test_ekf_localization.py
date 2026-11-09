"""EKF localization: the predict step and its Jacobians, the range linearization, masking,
gating and the iterated update, against hand-computed values and finite differences."""

import math
import warnings

import numpy as np
import pytest

from rl_navigation.estimation import (
    EkfLocalizer,
    SensorMap,
    kalman_update,
    linearize_ranges,
    motion_jacobians,
)
from rl_navigation.sim import (
    Arena,
    RangeSensor,
    WheelOdometry,
    hc_sr04,
    lidar,
    unicycle_step,
    wrap_angle,
)

ODOMETRY = WheelOdometry(wheel_base=0.2, noise_per_metre=2.5e-4)
ROOM = Arena(
    5.0,
    5.0,
    circle_centers=[[3.5, 3.5]],
    circle_radii=[0.4],
    boxes=[[1.0, 3.2, 1.8, 4.0]],
)


def _beam(angle=0.0, max_range=10.0):
    return RangeSensor(mount_angles=[angle], max_range=max_range, noise_std=0.01)


def _ekf(range_map, pose, covariance, **kwargs):
    return EkfLocalizer(range_map, ODOMETRY, pose, covariance, **kwargs)


def _fd_jacobian(f, point, eps=1e-6):
    """Central differences of f, whose output's third entry is an angle: each difference is
    wrapped, so a step across +-pi reads as the small change it is."""
    columns = []
    for k in range(len(point)):
        step = np.zeros(len(point))
        step[k] = eps
        diff = f(point + step) - f(point - step)
        diff[2] = wrap_angle(diff[2])
        columns.append(diff / (2 * eps))
    return np.stack(columns, axis=1)


# Predict ----------------------------------------------------------------------------------------


def test_predict_a_straight_move_by_hand():
    # Delta s = 0.1, Delta theta = 0 at theta = 0: F = [[1,0,0],[0,1,0.1],[0,0,1]], G A =
    # [[0.5,0.5],[-0.25,0.25],[-5,5]], Sigma_w = 2.5e-5 I, so Q = 2.5e-5 (G A)(G A)^T.
    P0 = 0.01 * np.eye(3)
    ekf = _ekf(SensorMap(ROOM, lidar()), [1.0, 2.0, 0.0], P0)
    ekf.predict([0.1, 0.1])
    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
    Q = np.array([[1.25e-5, 0.0, 0.0], [0.0, 3.125e-6, 6.25e-5], [0.0, 6.25e-5, 1.25e-3]])
    np.testing.assert_allclose(ekf.pose, [1.1, 2.0, 0.0], atol=1e-15, err_msg="Straight 0.1 m.")
    np.testing.assert_allclose(
        ekf.covariance, F @ P0 @ F.T + Q, rtol=1e-12, atol=1e-18, err_msg="P' = F P F^T + Q."
    )


def test_predict_a_quarter_turn_in_place_by_hand():
    # Wheels -+0.05 pi: Delta s = 0, Delta theta = pi/2, so F = I; G's first column is
    # sinc(pi/4) (cos, sin)(pi/4) = (2/pi, 2/pi), and with Sigma_w = 2.5e-4 * 0.05 pi I,
    # Q = 1.25e-5 pi [[2/pi^2, 2/pi^2, 0], [2/pi^2, 2/pi^2, 0], [0, 0, 50]].
    P0 = 1e-6 * np.eye(3)
    ekf = _ekf(SensorMap(ROOM, lidar()), [2.0, 2.0, 0.0], P0)
    ekf.predict([-0.05 * math.pi, 0.05 * math.pi])
    q = 2.5e-5 / math.pi
    Q = np.array([[q, q, 0.0], [q, q, 0.0], [0.0, 0.0, 6.25e-4 * math.pi]])
    np.testing.assert_allclose(ekf.pose, [2.0, 2.0, math.pi / 2], atol=1e-15)
    np.testing.assert_allclose(
        ekf.covariance,
        P0 + Q,
        rtol=1e-12,
        atol=1e-18,
        err_msg="Turning in place: P' = P + Q with F = I.",
    )


def test_predict_wraps_the_heading():
    ekf = _ekf(SensorMap(ROOM, lidar()), [2.0, 2.0, math.pi - 0.05], 0.01 * np.eye(3))
    ekf.predict([-0.01, 0.01])  # turn 0.1 rad
    assert ekf.pose[2] == pytest.approx(-math.pi + 0.05, abs=1e-12), "Heading should wrap at pi."


@pytest.mark.parametrize("turn", [0.0, 1e-6, 0.0199, 0.0201, 0.3])
@pytest.mark.parametrize("theta", [0.4, math.pi - 1e-3])
def test_motion_jacobians_match_finite_differences(turn, theta):
    """0.0199 and 0.0201 straddle the switch to the series for d sinc at Delta theta = 0.02."""
    pose, distance = np.array([1.0, -2.0, theta]), 0.5
    F, G = motion_jacobians(pose, distance, turn)
    np.testing.assert_allclose(
        F,
        _fd_jacobian(lambda p: unicycle_step(p, distance, turn, 1.0), pose),
        rtol=1e-6,
        atol=1e-9,
        err_msg="F = d pose' / d pose",
    )
    np.testing.assert_allclose(
        G,
        _fd_jacobian(lambda u: unicycle_step(pose, u[0], u[1], 1.0), np.array([distance, turn])),
        rtol=1e-6,
        atol=1e-9,
        err_msg="G = d pose' / d (Delta s, Delta theta)",
    )


def test_an_arcs_process_noise_follows_the_wheels():
    """On an arc the wheels roll different distances, so their noise correlates distance and
    turn. Q must equal J Sigma_w J^T with J = d pose' / d (left, right) by finite differences."""
    pose, travel = np.array([1.0, 2.0, 0.7]), np.array([0.02, 0.06])
    ekf = _ekf(SensorMap(ROOM, lidar()), pose, 1e-9 * np.eye(3))
    ekf.predict(travel)

    def moved(wheels):
        return unicycle_step(pose, *ODOMETRY.increments(wheels), 1.0)

    J = _fd_jacobian(moved, travel)
    F = _fd_jacobian(lambda p: unicycle_step(p, *ODOMETRY.increments(travel), 1.0), pose)
    expected = 1e-9 * F @ F.T + J @ np.diag(2.5e-4 * travel) @ J.T
    np.testing.assert_allclose(ekf.covariance, expected, rtol=1e-6, atol=1e-15)


@pytest.mark.parametrize("travel", [[0.1], [0.1, 0.1, 0.1], [0.1, np.nan], [np.inf, 0.0]])
def test_predict_refuses_a_bad_wheel_travel(travel):
    ekf = _ekf(SensorMap(ROOM, lidar()), [2.0, 2.0, 0.0], 0.01 * np.eye(3))
    with pytest.raises(ValueError, match="wheel_travel"):
        ekf.predict(travel)


# Linearization ----------------------------------------------------------------------------------


def test_range_jacobian_for_a_wall_by_hand():
    # A beam at phi = 0.3 from (5, 5) meets the wall x = 10 at t = 5 / cos phi:
    # dt/dx = -1 / cos phi, dt/dy = 0, dt/dtheta = 5 sin phi / cos^2 phi.
    phi = 0.3
    lin = linearize_ranges(SensorMap(Arena(10.0, 10.0), _beam()), [5.0, 5.0, phi])
    assert lin.h[0] == pytest.approx(5.0 / math.cos(phi), rel=1e-12)
    np.testing.assert_allclose(
        lin.H[0],
        [-1.0 / math.cos(phi), 0.0, 5.0 * math.sin(phi) / math.cos(phi) ** 2],
        rtol=1e-6,
        atol=1e-9,
        err_msg="Range Jacobian against a flat wall",
    )
    assert not lin.nonlinear[0], "A flat wall is linear within the window."


def test_range_jacobian_for_a_circle_by_hand():
    # From (5, 5.5) along +x to the circle of radius 1 at (8, 5): offset q = 0.5 from the centre
    # line, t = 3 - sqrt(1 - q^2); dt/dx = -1, dt/dy = q / sqrt(1 - q^2),
    # dt/dtheta = -0.5 + q * 3 / sqrt(1 - q^2).
    arena = Arena(10.0, 10.0, circle_centers=[[8.0, 5.0]], circle_radii=[1.0])
    lin = linearize_ranges(SensorMap(arena, _beam()), [5.0, 5.5, 0.0])
    root = math.sqrt(0.75)
    assert lin.h[0] == pytest.approx(3.0 - root, rel=1e-12)
    np.testing.assert_allclose(
        lin.H[0],
        [-1.0, 0.5 / root, -0.5 + 1.5 / root],
        rtol=1e-6,
        err_msg="Range Jacobian against a circle",
    )


def test_a_beam_on_a_box_corner_is_flagged_nonlinear():
    arena = Arena(10.0, 10.0, boxes=[[7.0, 4.0, 8.0, 6.0]])
    corner = linearize_ranges(SensorMap(arena, _beam()), [5.0, 6.0, 0.0])
    face = linearize_ranges(SensorMap(arena, _beam()), [5.0, 5.0, 0.0])
    assert corner.nonlinear[0], "A beam grazing the box's corner jumps within 3 cm: not linear."
    assert not face.nonlinear[0], "A beam on the middle of the box's face is linear."


def test_probes_past_a_wall_drop_the_beam_without_warnings():
    """At x = 0.025 the -0.03 probe is outside the arena, where the backward beam meets nothing:
    its cast is inf, the differences NaN, and the beam must drop out quietly."""
    sensor = RangeSensor(mount_angles=[math.pi], max_range=10.0, noise_std=0.01)
    range_map = SensorMap(Arena(10.0, 10.0), sensor)
    pose = [0.025, 5.0, 0.0]
    assert np.isinf(range_map.expected_ranges([[pose[0] - 0.03, 5.0, 0.0]])[0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lin = linearize_ranges(range_map, pose)
        ekf = _ekf(range_map, pose, 0.01 * np.eye(3))
        result = ekf.update([0.025])
    assert lin.nonlinear[0] and not result.used[0], "A beam with an inf probe must not be used."
    assert np.isfinite(ekf.pose).all() and np.isfinite(ekf.covariance).all()


# Update -----------------------------------------------------------------------------------------


def _scan(sensor, pose, seed=0, arena=ROOM):
    return sensor.read(arena, pose, np.random.default_rng(seed))[0]


TRUTH = np.array([2.5, 2.0, 0.3])


def test_a_still_robot_converges_from_a_large_prior():
    """Started 0.2 m and 0.1 rad off with P0 = diag(0.2^2, 0.2^2, 0.1^2), which the error lies
    within, so the gate passes the scan."""
    sensor = lidar()
    ekf = _ekf(SensorMap(ROOM, sensor), TRUTH + [0.2, 0.0, 0.1], np.diag([0.04, 0.04, 0.01]))
    traces = [np.trace(ekf.covariance)]
    for step in range(20):
        ekf.predict([0.0, 0.0])
        ekf.update(_scan(sensor, TRUTH, seed=step))
        traces.append(np.trace(ekf.covariance))
    error = ekf.pose - TRUTH
    assert np.hypot(error[0], error[1]) < 0.01 and abs(error[2]) < 0.01, (
        f"A still robot should converge to within 1 cm and 0.01 rad, ended {error} off."
    )
    assert np.all(np.diff(traces) <= 0), f"trace(P) should never grow when still: {traces}"


def _facing_wall():
    """Five sonars facing the wall x = 5 from x = 4: every beam sees the x error."""
    sensor = hc_sr04()
    truth = np.array([4.0, 2.5, 0.0])
    return SensorMap(Arena(5.0, 5.0), sensor), truth, sensor.true_ranges(Arena(5.0, 5.0), truth)[0]


def test_an_error_beyond_the_prior_is_gated_out_entirely():
    """An estimate 6 sigma off in x with a confident prior: every beam's innovation fails the
    3-sigma gate, and nothing changes."""
    range_map, truth, readings = _facing_wall()
    P0 = np.diag([0.01**2, 0.01**2, 0.005**2])
    ekf = _ekf(range_map, truth + [0.06, 0.0, 0.0], P0)
    result = ekf.update(readings)
    assert result.gated.all() and not result.used.any(), f"All beams should be gated: {result}"
    np.testing.assert_array_equal(ekf.pose, truth + [0.06, 0.0, 0.0])
    np.testing.assert_array_equal(ekf.covariance, P0)


def test_the_same_error_within_a_wide_prior_uses_every_beam():
    range_map, truth, readings = _facing_wall()
    ekf = _ekf(range_map, truth + [0.06, 0.0, 0.0], np.diag([0.1**2, 0.1**2, 0.05**2]))
    result = ekf.update(readings)
    assert result.used.all(), f"Within the prior, all five beams should be used: {result}"
    assert abs(ekf.pose[0] - truth[0]) < 0.01, "The update should pull x back to the wall's range."


@pytest.mark.parametrize("lost", ["max_range", "nan", "inf", "min_range"])
def test_a_scan_with_nothing_usable_changes_nothing(lost):
    sensor = lidar()
    fill = {"max_range": sensor.max_range, "nan": np.nan, "inf": np.inf}.get(lost, sensor.min_range)
    P0 = np.diag([0.01, 0.01, 0.004])
    ekf = _ekf(SensorMap(ROOM, sensor), TRUTH + [0.05, 0.0, 0.0], P0)
    result = ekf.update(np.full(sensor.num_readings, fill))
    assert not result.used.any(), f"A scan of {lost} readings should use no beam."
    np.testing.assert_array_equal(ekf.pose, TRUTH + [0.05, 0.0, 0.0])
    np.testing.assert_array_equal(ekf.covariance, P0)


def test_a_lost_reading_equals_a_sensor_without_that_beam():
    sensor = lidar()
    readings = _scan(sensor, TRUTH)
    lost = 7
    fewer = RangeSensor(
        mount_angles=np.delete(sensor.mount_angles, lost),
        min_range=sensor.min_range,
        max_range=sensor.max_range,
        noise_std=sensor.noise_std,
    )
    start, P0 = TRUTH + [0.05, -0.03, 0.02], np.diag([0.01, 0.01, 0.004])
    for spelling in (np.nan, sensor.max_range):
        with_dropout = readings.copy()
        with_dropout[lost] = spelling
        full = _ekf(SensorMap(ROOM, sensor), start, P0)
        full.update(with_dropout)
        reduced = _ekf(SensorMap(ROOM, fewer), start, P0)
        reduced.update(np.delete(readings, lost))
        np.testing.assert_allclose(full.pose, reduced.pose, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(full.covariance, reduced.covariance, rtol=1e-10, atol=1e-18)


def test_an_outlier_is_gated_and_ignored():
    sensor = lidar()
    readings = _scan(sensor, TRUTH)
    start, P0 = TRUTH + [0.02, 0.0, 0.0], np.diag([0.01, 0.01, 0.004])
    beam = int(np.argmin(np.abs(readings - 2.0)))
    with_outlier, without = readings.copy(), readings.copy()
    with_outlier[beam], without[beam] = 0.3, sensor.max_range
    first, second = (
        _ekf(SensorMap(ROOM, sensor), start, P0),
        _ekf(SensorMap(ROOM, sensor), start, P0),
    )
    result = first.update(with_outlier)
    second.update(without)
    assert result.gated[beam] and not result.used[beam], "A 0.3 m reading where ~2 m is expected."
    np.testing.assert_array_equal(first.pose, second.pose)
    np.testing.assert_array_equal(first.covariance, second.covariance)


def test_one_iteration_is_the_plain_ekf_update():
    sensor = lidar()
    range_map = SensorMap(ROOM, sensor)
    readings = _scan(sensor, TRUTH)
    start, P0 = TRUTH + [0.03, -0.02, 0.01], np.diag([0.01, 0.01, 0.004])
    ekf = _ekf(range_map, start, P0, iterations=1)
    result = ekf.update(readings)
    lin = linearize_ranges(range_map, start)
    used = result.used
    expected_x, expected_P = kalman_update(
        start,
        P0,
        readings[used] - lin.h[used],
        lin.H[used],
        sensor.noise_std**2 * np.eye(int(used.sum())),
    )
    expected_x[2] = wrap_angle(expected_x[2])
    np.testing.assert_allclose(ekf.pose, expected_x, rtol=1e-12)
    np.testing.assert_allclose(ekf.covariance, expected_P, rtol=1e-10)


def test_iterating_corrects_a_large_error_better_than_one_update():
    sensor = lidar()
    readings = _scan(sensor, TRUTH)
    start, P0 = TRUTH + [0.2, 0.0, 0.1], np.diag([0.04, 0.04, 0.01])
    errors = {}
    for iterations in (1, 3):
        ekf = _ekf(SensorMap(ROOM, sensor), start, P0, iterations=iterations)
        result = ekf.update(readings)
        errors[iterations] = np.linalg.norm(ekf.pose - TRUTH)
        assert result.iterations == iterations, f"Expected {iterations} accepted iterations."
    assert errors[3] < 0.5 * errors[1], f"Relinearizing should cut the error: {errors}"


def test_the_guard_refuses_an_iterate_that_leaves_the_window():
    """With a 1e-6 window every relinearized step is too long, so only the first update stands."""
    sensor = lidar()
    readings = _scan(sensor, TRUTH)
    start, P0 = TRUTH + [0.2, 0.0, 0.1], np.diag([0.04, 0.04, 0.01])
    guarded = _ekf(SensorMap(ROOM, sensor), start, P0, iterations=3, linearity_window=1e-6)
    plain = _ekf(SensorMap(ROOM, sensor), start, P0, iterations=1, linearity_window=1e-6)
    result = guarded.update(readings)
    plain.update(readings)
    assert result.iterations == 1, "The second iterate moved more than the window."
    np.testing.assert_array_equal(guarded.pose, plain.pose)
    np.testing.assert_array_equal(guarded.covariance, plain.covariance)


def test_a_scan_of_the_wrong_length_is_refused():
    ekf = _ekf(SensorMap(ROOM, lidar()), TRUTH, 0.01 * np.eye(3))
    with pytest.raises(ValueError, match="readings"):
        ekf.update(np.ones(5))


@pytest.mark.parametrize(
    ("pose", "covariance", "kwargs"),
    [
        ([1.0, 1.0], np.eye(3), {}),
        ([1.0, 1.0, np.nan], np.eye(3), {}),
        ([1.0, 1.0, 0.0], np.eye(2), {}),
        ([1.0, 1.0, 0.0], [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], {}),
        ([1.0, 1.0, 0.0], -np.eye(3), {}),
        ([1.0, 1.0, 0.0], np.eye(3), {"iterations": 0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"fd_step": 0.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"linearity_window": -1.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"linearity_tol": 0.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"gate": 0.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_after": 0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_after": True}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_after": 2.5}),
        ([1.0, 1.0, 0.0], np.eye(3), {"min_candidates": 0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_window": 0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_std": (0.2, 0.2)}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_std": (0.2, 0.0, 0.1)}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_std": (0.2, -0.2, 0.1)}),
        ([1.0, 1.0, 0.0], np.eye(3), {"reset_std": (0.2, np.nan, 0.1)}),
    ],
)
def test_invalid_filter_settings_are_refused(pose, covariance, kwargs):
    with pytest.raises(ValueError, match="EkfLocalizer"):
        _ekf(SensorMap(ROOM, lidar()), pose, covariance, **kwargs)


def test_a_noiseless_sensor_is_refused():
    with pytest.raises(ValueError, match="noise"):
        _ekf(SensorMap(ROOM, lidar(noise_std=0.0)), TRUTH, np.eye(3))


class _LinearMap:
    """A one-beam map whose range is ``offset + slope . pose``, and ``inf`` from poses at or
    beyond ``finite_below_x``: the protocol seam lets a test choose the measurement exactly."""

    min_range, max_range, noise_std = 0.02, 4.0, 0.01

    def __init__(self, offset, slope, finite_below_x=np.inf):
        self.offset, self.slope, self.finite_below_x = (
            offset,
            np.asarray(slope, float),
            finite_below_x,
        )

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        ranges = self.offset + poses @ self.slope
        return np.where(poses[:, 0] < self.finite_below_x, ranges, np.inf)[:, None]


def test_a_beam_expected_near_max_range_is_not_used():
    """Expected 3.99 m with a 4 m maximum: the real reading would often be censored."""
    near = _ekf(_LinearMap(3.99, [0.0, 0.0, 0.0]), [1.0, 1.0, 0.0], np.eye(3))
    clear = _ekf(_LinearMap(3.9, [0.0, 0.0, 0.0]), [1.0, 1.0, 0.0], np.eye(3))
    assert not near.update([3.98]).used[0], "Within 3 sigma of max_range: censored, unused."
    assert clear.update([3.89]).used[0], "Well inside the range the beam is used."


@pytest.mark.parametrize(
    ("expected", "reading"), [(3.9, 4.0), (0.05, 0.02), (2.0, np.nan), (2.0, np.inf)]
)
def test_lost_readings_are_dropped_not_gated(expected, reading):
    """A reading at max_range (no echo), at min_range (clipped) or not finite carries no range:
    it is dropped before the gate, even where a loose prior would let the gate pass it."""
    ekf = _ekf(_LinearMap(expected - 1.0, [1.0, 0.0, 0.0]), [1.0, 1.0, 0.0], np.eye(3))
    result = ekf.update([reading])
    assert not result.used[0] and not result.gated[0], f"{reading} should be dropped: {result}"


def test_an_iterate_whose_beams_cast_inf_is_refused():
    """The first update moves x from 0 to about 0.5, where the map casts inf: relinearizing there
    would put NaN in the pose, so the first iterate stands."""
    range_map = _LinearMap(2.0, [-1.0, 0.0, 0.0], finite_below_x=0.1)
    ekf = _ekf(range_map, [0.0, 1.0, 0.0], np.eye(3))
    result = ekf.update([1.5])
    assert result.iterations == 1, "The iterate at x ~ 0.5 casts inf and must be refused."
    assert np.isfinite(ekf.pose).all() and ekf.pose[0] > 0.4, f"Pose {ekf.pose}"


class _HeadingMap(_LinearMap):
    """One beam reading 3 + (theta - pi), wrapped: linear in the heading across +-pi."""

    def __init__(self):
        super().__init__(3.0, [0.0, 0.0, 0.0])

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        return (3.0 + wrap_angle(poses[:, 2] - math.pi))[:, None]


def test_an_update_across_pi_wraps_the_heading():
    """From theta = pi - 0.01 with prior variance 0.01, a reading of 3.0099 moves the heading by
    K nu = 0.0199 * 0.01 / 0.0101, past pi: it wraps to -pi + 0.0097. The map is linear, so every
    iteration must land on the same point, which needs its offset from the prior wrapped too."""
    ekf = _ekf(_HeadingMap(), [1.0, 1.0, math.pi - 0.01], np.diag([1.0, 1.0, 0.1**2]))
    result = ekf.update([3.0099])
    expected = -math.pi - 0.01 + 0.0199 * 0.01 / 0.0101
    assert result.iterations == 3, "Every iterate is a small step."
    assert ekf.pose[2] == pytest.approx(expected, abs=1e-9), (
        f"Heading {ekf.pose[2]} should wrap to {expected}."
    )


def test_an_initial_heading_out_of_range_is_wrapped():
    ekf = _ekf(SensorMap(ROOM, lidar()), [2.5, 2.0, 1.5 * math.pi], np.eye(3))
    assert ekf.pose[2] == pytest.approx(-0.5 * math.pi, abs=1e-12), "3 pi / 2 is -pi / 2."


class _SteppedMap(_LinearMap):
    """One beam reading 2 - x, which jumps by 1 m at x = ``jump_at`` (a corner coming into view)."""

    def __init__(self, jump_at):
        super().__init__(2.0, [-1.0, 0.0, 0.0])
        self.jump_at = jump_at

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        return (2.0 - poses[:, 0] + np.where(poses[:, 0] >= self.jump_at, 1.0, 0.0))[:, None]


def test_an_iterate_on_a_discontinuity_does_not_collapse_the_covariance():
    """The first update moves x from 0 to x1; the beam jumps 5e-5 m beyond x1, inside the
    Jacobian's +-1e-4 step. Relinearizing there would give dh/dx of about 5000 and a covariance
    near zero (lidar seed 18, step 96 of the evaluation, collapsed sigma_theta to 6e-7 this way),
    so the beam leaves the iterations and, with no beam left, x1 and its covariance stand."""
    P0, reading = np.eye(3), 1.5
    x1, P1 = kalman_update(np.zeros(3), P0, [reading - 2.0], [[-1.0, 0.0, 0.0]], [[1e-4]])
    ekf = _ekf(_SteppedMap(jump_at=x1[0] + 5e-5), [0.0, 0.0, 0.0], P0)
    result = ekf.update([reading])
    assert result.iterations == 1, "The iterate's only beam is not linear: stop there."
    assert result.used[0], "The beam did correct the pose, in the first (accepted) iteration."
    np.testing.assert_allclose(ekf.covariance, P1, rtol=1e-10, err_msg="Keep the first update's P.")


class _DiagonalStepMap(_LinearMap):
    """One beam reading 2 - x - y, which jumps by 1 m once x + y reaches ``jump_at``."""

    def __init__(self, jump_at):
        super().__init__(2.0, [-1.0, -1.0, 0.0])
        self.jump_at = jump_at

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        total = poses[:, 0] + poses[:, 1]
        return (2.0 - total + np.where(total >= self.jump_at, 1.0, 0.0))[:, None]


def test_an_iterate_that_loses_every_beam_keeps_the_last_update():
    """The first update moves (x, y) diagonally by 0.02 each, less than the window, to just short
    of the jump, which the prior's axis-aligned probes (x + y <= 0.03) never reached. At the
    iterate the beam fails the check; with no beam left, iterating on would fall back to the
    prior (a step within the window). The first update must stand."""
    P0, reading = np.eye(3), 1.96
    x1, P1 = kalman_update(np.zeros(3), P0, [reading - 2.0], [[-1.0, -1.0, 0.0]], [[1e-4]])
    ekf = _ekf(_DiagonalStepMap(jump_at=x1[0] + x1[1] + 5e-5), [0.0, 0.0, 0.0], P0)
    result = ekf.update([reading])
    assert result.iterations == 1
    np.testing.assert_allclose(ekf.pose, x1, rtol=1e-8, err_msg="Keep the first update's pose.")
    np.testing.assert_allclose(ekf.covariance, P1, rtol=1e-8)


def test_the_update_reports_its_normalised_innovation():
    """One beam reading 2 + x at x = 0 with P = I and sigma_r^2 = 1e-4: H = (1, 0, 0), so
    S = 1 + 1e-4 and the NIS of an innovation nu is nu^2 / (1 + 1e-4)."""
    ekf = _ekf(_LinearMap(2.0, [1.0, 0.0, 0.0]), [0.0, 0.0, 0.0], np.eye(3))
    result = ekf.update([2.3])
    assert result.nis == pytest.approx(0.09 / (1.0 + 1e-4), rel=1e-6)


class _TwoBeamMap(_LinearMap):
    """Beam 0 reads 2 - x everywhere; beam 1 reads 3 - x and jumps by 1 m once x reaches
    ``jump_at``: a corner that comes into view only near the corrected pose."""

    def __init__(self, jump_at):
        super().__init__(2.0, [-1.0, 0.0, 0.0])
        self.jump_at = jump_at

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        x = poses[:, 0]
        return np.stack([2.0 - x, 3.0 - x + np.where(x >= self.jump_at, 1.0, 0.0)], axis=1)


def test_a_beam_lost_at_an_iterate_is_reported_and_the_rest_iterate_on():
    """Both beams pass at the prior; at the first iterate beam 1 is on its corner and leaves, and
    beam 0 carries the later iterations. The result says which beam corrected the final pose."""
    readings = np.array([1.5, 2.5])  # both say x = 0.5
    x1, _ = kalman_update(
        np.zeros(3),
        np.eye(3),
        readings - [2.0, 3.0],
        [[-1.0, 0, 0], [-1.0, 0, 0]],
        1e-4 * np.eye(2),
    )
    ekf = _ekf(_TwoBeamMap(jump_at=x1[0] + 0.01), [0.0, 0.0, 0.0], np.eye(3))
    result = ekf.update(readings)
    assert result.iterations >= 2, f"Beam 0 should keep iterating: {result}"
    np.testing.assert_array_equal(result.used, [True, False])
    np.testing.assert_array_equal(result.nonlinear, [False, True])


# The still-robot reset --------------------------------------------------------------------------

STILL_P0 = np.diag([1e-4, 1e-4, 1e-4])
SETTLE_SCANS = 50


def _bumped_still_robot(sensor, bump, scans_after, **kwargs):
    """A still robot scanned SETTLE_SCANS times at TRUTH, then bumped by ``bump`` without the
    encoders noticing and scanned ``scans_after`` more times. Returns the filter, the bumped
    truth, and each post-bump update's result with the ``recent_resets`` that followed it."""
    ekf = _ekf(SensorMap(ROOM, sensor), TRUTH, STILL_P0, **kwargs)
    bumped = TRUTH + np.asarray(bump, dtype=float)
    after = []
    for step in range(SETTLE_SCANS + scans_after):
        ekf.predict([0.0, 0.0])
        result = ekf.update(_scan(sensor, TRUTH if step < SETTLE_SCANS else bumped, seed=step))
        if step >= SETTLE_SCANS:
            after.append((result, ekf.recent_resets))
    return ekf, bumped, after


def _pose_error(estimate, truth):
    error = estimate - truth
    return float(np.hypot(error[0], error[1])), abs(float(wrap_angle(error[2])))


@pytest.mark.parametrize("make_sensor", [lidar, hc_sr04])
def test_a_bumped_still_robot_recovers_by_resetting(make_sensor):
    """Bumped 0.1 m sideways after its covariance has collapsed, every beam is gated; the reset
    re-opens the covariance and the scans pull the estimate back within 20 scans."""
    ekf, bumped, after = _bumped_still_robot(make_sensor(), [0.0, 0.1, 0.0], scans_after=20)
    position, heading = _pose_error(ekf.pose, bumped)
    assert position < 0.01 and heading < 0.01, (
        f"20 scans after a 0.1 m bump the estimate should be within 1 cm and 0.01 rad, ended "
        f"{position:.4f} m and {heading:.4f} rad off."
    )
    assert any(result.reset for result, _ in after), "Recovering should have taken a reset."


def test_without_the_reset_a_bumped_still_robot_stays_lost():
    """``reset_after=None`` keeps the old behaviour: every beam gated, the estimate unmoved."""
    ekf, bumped, after = _bumped_still_robot(
        lidar(), [0.0, 0.1, 0.0], scans_after=20, reset_after=None
    )
    position, _ = _pose_error(ekf.pose, bumped)
    assert position > 0.09, f"Without the reset the 0.1 m error should stay; it is {position:.4f}."
    assert not any(result.reset for result, _ in after), "A disabled reset should never fire."
    assert ekf.recent_resets == 0, "A disabled reset should count no resets."


def _wall_scans(estimate_offset_x):
    """The five sonars facing the wall x = 5 from x = 4 (``_facing_wall``), and a filter whose
    estimate is ``estimate_offset_x`` metres off in x; the readings the estimate expects."""
    range_map, truth, readings = _facing_wall()
    estimate = truth + [estimate_offset_x, 0.0, 0.0]
    expected = hc_sr04().true_ranges(Arena(5.0, 5.0), estimate)[0]
    return range_map, estimate, readings, expected


def test_the_reset_fires_on_the_third_consecutive_counting_scan():
    """A scan counts when it has at least 2 candidate beams and at least half are gated.

    ``half`` has 2 candidates, 1 gated (true readings) and 1 used (the readings the estimate
    expects): the boundary of both rules. ``clean`` uses every beam; ``single`` has 1 candidate,
    gated. Each non-counting scan restarts the count."""
    range_map, estimate, gated_readings, expected = _wall_scans(0.06)
    nan = np.full(5, np.nan)
    half, single = nan.copy(), nan.copy()
    half[1], half[2] = gated_readings[1], expected[2]
    single[2] = gated_readings[2]
    scans = {"half": half, "clean": expected, "single": single, "gated": gated_readings}
    ekf = _ekf(range_map, estimate, np.diag([0.01**2, 0.01**2, 0.005**2]))
    sequence = ["half", "half", "clean", "half", "half", "single", "half", "half", "half"]
    fired = []
    for name in sequence:
        result = ekf.update(scans[name])
        if name == "half":
            assert result.gated.sum() == 1 and result.used.sum() == 1, (
                f"The half scan should gate one beam and use one: {result}"
            )
        if name == "single":
            assert result.gated.sum() == 1 and not result.used.any(), f"Single: {result}"
        fired.append(result.reset)
    assert fired == [False] * 8 + [True], (
        f"Only the 9th scan, the third counting one after the single, should reset: {fired}"
    )


def test_a_mostly_used_scan_does_not_count():
    """Five candidates with two gated is under half: no reset however often it repeats."""
    range_map, estimate, gated_readings, expected = _wall_scans(0.06)
    scan = expected.copy()
    scan[:2] = gated_readings[:2]
    ekf = _ekf(range_map, estimate, np.diag([0.01**2, 0.01**2, 0.005**2]))
    results = [ekf.update(scan) for _ in range(6)]
    assert all(r.gated.sum() == 2 and r.used.sum() == 3 for r in results), results[0]
    assert not any(r.reset for r in results), "Two gated of five should never reset."


def test_a_reset_fires_when_every_beam_is_gated():
    """The all-gated scan leaves the update early; the reset is counted there too."""
    range_map, estimate, gated_readings, _ = _wall_scans(0.06)
    P0 = np.diag([0.01**2, 0.01**2, 0.005**2])
    ekf = _ekf(range_map, estimate, P0)
    results = [ekf.update(gated_readings) for _ in range(3)]
    assert all(r.gated.all() and not r.used.any() for r in results), results[-1]
    assert [r.reset for r in results] == [False, False, True], [r.reset for r in results]
    np.testing.assert_array_equal(ekf.pose, estimate, err_msg="A reset should not move the pose.")
    np.testing.assert_array_equal(
        np.diag(ekf.covariance), [0.2**2, 0.2**2, 0.1**2], err_msg="The default reset_std."
    )


def test_the_reset_raises_the_diagonal_to_a_floor_once():
    """P[i, i] <- max(P[i, i], reset_std[i]^2): x and theta are raised, the larger y kept, the
    off-diagonals untouched; a second reset with the variances at the floor changes nothing.

    The estimate is 0.8 m off, 4 floor deviations, so the scans stay gated after a reset."""
    range_map, estimate, gated_readings, _ = _wall_scans(-0.8)
    P0 = np.array([[1e-4, 2e-5, 0.0], [2e-5, 1e-4, 0.0], [0.0, 0.0, 2.5e-5]])
    ekf = _ekf(range_map, estimate, P0, reset_std=(0.2, 0.005, 0.1))
    results = [ekf.update(gated_readings) for _ in range(6)]
    assert all(not r.used.any() for r in results), "The scans should stay gated after a reset."
    assert [r.reset for r in results] == [False, False, True] * 2, [r.reset for r in results]
    expected = P0.copy()
    expected[0, 0], expected[2, 2] = 0.2**2, 0.1**2  # y's 1e-4 exceeds 0.005^2 and is kept
    np.testing.assert_array_equal(
        ekf.covariance, expected, err_msg="Two resets should raise the diagonal to the floor once."
    )


def test_recent_resets_stay_up_while_lost_and_fall_after_a_recovery():
    """A 1.5 m bump cannot be recovered: the filter keeps resetting, and ``recent_resets`` (the
    resets in the last 20 scans) stays at 2 or more once the first window has filled. A 0.1 m
    bump is recovered on one reset, and the count is back to 0 once it leaves the window."""
    _, _, lost = _bumped_still_robot(lidar(), [1.5, 0.0, 0.0], scans_after=100)
    held = [count for _, count in lost[20:]]
    assert min(held) >= 2, f"Lost, recent_resets should hold at 2 or more: {held}"
    _, _, recovered = _bumped_still_robot(lidar(), [0.0, 0.1, 0.0], scans_after=40)
    counts = [count for _, count in recovered]
    assert max(counts) >= 1 and counts[-1] == 0, (
        f"Recovered, recent_resets should rise and return to 0: {counts}"
    )


def test_one_outlier_beam_does_not_reset():
    """One of 64 lidar beams reading 0.3 m for 10 scans is gated each time, but one gated beam of
    64 is far from half: no reset."""
    sensor = lidar()
    ekf = _ekf(SensorMap(ROOM, sensor), TRUTH, STILL_P0)
    results = []
    for step in range(10):
        ekf.predict([0.0, 0.0])
        readings = _scan(sensor, TRUTH, seed=step)
        readings[10] = 0.3
        results.append(ekf.update(readings))
    assert all(r.gated[10] for r in results), "The 0.3 m reading should be gated."
    assert not any(r.reset for r in results) and ekf.recent_resets == 0, "One outlier: no reset."
