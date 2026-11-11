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
from rl_navigation.estimation.ekf import chi2_quantile
from rl_navigation.estimation.evaluation import ekf as ekf_factory
from rl_navigation.estimation.evaluation import run_trial
from rl_navigation.sim import (
    Arena,
    RangeSensor,
    WheelOdometry,
    hc_sr04,
    lidar,
    unicycle_step,
    wrap_angle,
)
from rl_navigation.sim.layouts import LayoutParams, generate_layout

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
        ([1.0, 1.0, 0.0], np.eye(3), {"nis_quantile": 0.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"nis_quantile": 1.0}),
        ([1.0, 1.0, 0.0], np.eye(3), {"nis_quantile": np.nan}),
        ([1.0, 1.0, 0.0], np.eye(3), {"count_off_map": 1}),
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

    def __init__(self, offset, slope, finite_below_x=np.inf, on_map=True):
        self.offset, self.slope, self.finite_below_x = (
            offset,
            np.asarray(slope, float),
            finite_below_x,
        )
        self.on_map = on_map
        self.free_calls = 0

    def expected_ranges(self, poses):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        ranges = self.offset + poses @ self.slope
        return np.where(poses[:, 0] < self.finite_below_x, ranges, np.inf)[:, None]

    def free(self, points, margin):
        """Every point is free when ``on_map``, none otherwise; counts the calls."""
        self.free_calls += 1
        return np.full(len(np.asarray(points).reshape(-1, 2)), self.on_map)


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
    """With the reset off (``reset_after=None``) every beam is gated and the estimate unmoved."""
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


# Holding an update that did not converge or whose innovations contradict the prior ----------------
#
# Seed 131 of EST-3 (HC-SR04, 30 % dropout) was the one lost tracking run of 2026-11-09. The
# numbers below are the filter as it was before the hold (``nis_quantile=None``): its state after
# step 362, the wheel travels, readings and true poses of steps 363 to 377, and its prior at step
# 370. At step 363 sonar 2's cone is linearized on the wrong wall (the switch is 0.043 rad away,
# outside the 0.03 rad linearity window) and the heading goes 0.05 rad the wrong way; that update
# is consistent with its prior (NIS 4.7 on 2 beams) and no check can see it. At step 370 three
# beams give NIS 23.4, above chi2_0.999(3) = 16.3, yet the old filter shrank sigma_theta from 19.9
# to 4.0 mrad; it reset only at step 376, 0.21 rad off, and was lost by step 377.
_SEED131_POSE_362 = [1.9507638040143513, 4.009231944378659, -2.100934082626776]
_SEED131_COV_362 = [
    [9.907061084064041e-06, 1.1287398737856307e-05, -4.4806508384584167e-07],
    [1.1287398737856307e-05, 2.9778716223442555e-05, -1.827210876017049e-05],
    [-4.4806508384584167e-07, -1.827210876017049e-05, 0.00047906639249746134],
]
_SEED131_TRAVEL = [
    [0.031850363019846506, 0.025327657019296925],
    [0.033197225941967744, 0.033247022831314345],
    [0.029600471917171408, 0.02853671921424797],
    [0.026757916779304853, 0.030772748908694202],
    [0.025757187007001, 0.029356955946066455],
    [0.03394917380078697, 0.025048497364936376],
    [0.02760488377992021, 0.025791502938974566],
    [0.02541812094345377, 0.03524314549731354],
    [0.02877306881243227, 0.03333018429615962],
    [0.031707572532318776, 0.02976556806520715],
    [0.023032110647406727, 0.03066620101874188],
    [0.03097972476532617, 0.031200755417086966],
    [0.03111228320547451, 0.033932013773937834],
    [0.03100078628216524, 0.03454189300246952],
    [0.02958528124509988, 0.03310943530490044],
]
_SEED131_SCANS = [
    [0.794914849792851, 4.0, 2.7599095157161613, 2.573505038806481, 1.4817232070943662],
    [0.7725553117801495, 2.003188957621566, 4.0, 4.0, 1.459797701300991],
    [4.0, 4.0, 2.7043084538387965, 2.569164641443524, 1.451225482095205],
    [0.7605395906951327, 4.0, 2.6848374737399556, 2.474571180246643, 1.4566155505558431],
    [4.0, 1.951051856735986, 2.6516993096125914, 4.0, 1.423563579204403],
    [0.7235320876438347, 1.918754660795801, 4.0, 2.4401397197764423, 4.0],
    [4.0, 1.9120788853579493, 2.596004668521687, 2.4070132036576677, 1.3840972255818722],
    [0.7166889183917062, 4.0, 2.541586265981821, 2.3780761760776294, 1.3710410572995235],
    [4.0, 4.0, 4.0, 4.0, 1.3503343230913594],
    [0.7235755827888232, 1.9114273896221827, 4.0, 4.0, 4.0],
    [4.0, 4.0, 2.4061081206278123, 4.0, 1.3394124873840116],
    [4.0, 1.8866053746757803, 2.3999465747355084, 2.5783038920310064, 4.0],
    [0.7559576682953627, 1.8912808746257022, 2.342937763816361, 4.0, 1.2994489267660565],
    [
        0.7838278996462587,
        1.8618731590164106,
        2.321355794067121,
        2.647937599611295,
        1.2792205981570244,
    ],
    [
        1.6227485021598682,
        1.8794780073680422,
        2.265952161473758,
        3.5032361950301945,
        1.2663533895570058,
    ],
]
_SEED131_TRUTH = [
    [1.9342141366078673, 3.9799026297119733, -2.0903120620449207],
    [1.9193203628866198, 3.9538608266956676, -2.090310446215904],
    [1.9046601007019714, 3.927687314159252, -2.072438890232532],
    [1.8900294437322367, 3.901497132610909, -2.0880454047158388],
    [1.8754348324538983, 3.875286980809833, -2.0696916648872548],
    [1.8607500786471716, 3.8491276578332188, -2.0949445347408338],
    [1.8458100255193384, 3.823112424447173, -2.0892355830072855],
    [1.8314298244547216, 3.7967854974356023, -2.0522245649176445],
    [1.8178827941064137, 3.7700193228150316, -2.0264392904920925],
    [1.8045057508233915, 3.7431670999106, -2.0395226988714286],
    [1.7912220151091796, 3.7162689157188744, -2.0195050185550936],
    [1.7781231022734287, 3.6892797415731353, -2.0257888138632683],
    [1.7650157317509905, 3.6622946635186144, -2.0201315100331256],
    [1.75205599857514, 3.635238366052072, -2.0148609443939423],
    [1.7396404485050274, 3.6079296915832075, -1.9801394153151821],
]
_SEED131_PRIOR_370 = [1.8162149397723593, 3.8171289960987282, -2.144467506544944]
_SEED131_PRIOR_COV_370 = [
    [1.4650473455653426e-05, 1.8148983299483967e-05, -4.971250238741822e-06],
    [1.8148983299483967e-05, 3.614486568836422e-05, -1.70434100371686e-05],
    [-4.971250238741822e-06, -1.70434100371686e-05, 0.0003941869371770338],
]
_GOLDEN_STEPS = (99, 199, 299, 359)
_GOLDEN_ESTIMATES = [
    [4.582700528868217, 0.5208593061597709, 1.0713003406035746],
    [4.805187901807433, 3.3417627008388555, 1.4991137756930595],
    [3.2498531134903574, 4.8009069077826725, 3.0879113514201366],
    [1.9955142358659623, 4.085978054061772, -2.113035571047867],
]
_GOLDEN_COV_359 = [
    [8.272019269371619e-06, 8.158747296285666e-06, -7.114390770262113e-06],
    [8.158747296285666e-06, 2.435695935772983e-05, -8.527786577868077e-06],
    [-7.114390770262113e-06, -8.527786577868077e-06, 0.00011049718745183615],
]


def _seed131_map():
    """Seed 131's layout, drawn as run_trial draws it (the first of its seven streams)."""
    layout_rng = np.random.default_rng(np.random.SeedSequence(131).spawn(7)[0])
    return SensorMap(generate_layout(layout_rng, LayoutParams()).arena, hc_sr04())


def _replay_seed131(**kwargs):
    """Steps 363 to 377 from the state after 362: each result, heading error and position error."""
    ekf = EkfLocalizer(
        _seed131_map(), WheelOdometry(), _SEED131_POSE_362, _SEED131_COV_362, **kwargs
    )
    steps = []
    for travel, scan, truth in zip(_SEED131_TRAVEL, _SEED131_SCANS, _SEED131_TRUTH, strict=True):
        ekf.predict(travel)
        result = ekf.update(scan)
        error = ekf.pose - np.asarray(truth)
        steps.append((result, abs(float(wrap_angle(error[2]))), float(np.hypot(*error[:2]))))
    return steps


def test_an_inconsistent_scan_after_a_wrong_association_is_held_and_the_reset_recovers():
    """From seed 131's state after step 362, the scan of step 370 is held (P kept), which makes
    it the third counting scan in a row: the reset fires at step 370, 0.15 rad off, instead of at
    376, 0.21 rad off, and the heading is back within 0.06 rad from step 373 (the old filter was
    0.19 to 0.45 rad off there and lost). The hold cannot undo step 363, whose update is
    consistent with its prior."""
    steps = _replay_seed131()
    headings = [heading for _, heading, _ in steps]
    assert max(headings[10:]) <= 0.06, (
        f"Steps 373-377 should be within 0.06 rad of the true heading, were {headings[10:]}"
    )
    assert steps[-1][2] < 0.03, f"Step 377 should be within 3 cm, was {steps[-1][2]:.3f} m"
    assert steps[7][0].held, f"Step 370's update (NIS {steps[7][0].nis:.1f}) should be held"
    assert steps[7][0].reset, "Step 370 should complete a reset (the third counting scan)"


def test_with_the_hold_off_seed131s_scans_lose_the_heading():
    """``nis_quantile=None`` is the filter before the hold: the same scans leave it 0.45 rad and
    0.23 m off at step 377, as in EST-3's lost run."""
    steps = _replay_seed131(nis_quantile=None)
    assert not any(result.held for result, _, _ in steps), "With the hold off nothing is held."
    assert steps[-1][1] > 0.4 and steps[-1][2] > 0.2, f"Step 377: {steps[-1][1:]}"


def test_a_scan_whose_nis_exceeds_the_quantile_keeps_the_covariance():
    """Seed 131's step 370: three sonars each pass their 3-sigma gate, but together give NIS
    23.4 > chi2_0.999(3) = 16.3. The update is held: the mean moves, P stays the prior. The old
    filter (``nis_quantile=None``) shrank sigma_theta from 19.9 to 4.0 mrad on this scan."""
    held = EkfLocalizer(_seed131_map(), WheelOdometry(), _SEED131_PRIOR_370, _SEED131_PRIOR_COV_370)
    result = held.update(_SEED131_SCANS[7])
    assert result.used.sum() == 3, f"Three sonars should pass the gate: {result}"
    assert result.nis == pytest.approx(23.4, abs=0.1), f"NIS {result.nis}"
    np.testing.assert_array_equal(
        held.covariance, _SEED131_PRIOR_COV_370, err_msg="A held update keeps the prior P."
    )
    assert result.held, "NIS above the 0.999 quantile should hold the update."
    assert not np.array_equal(held.pose, _SEED131_PRIOR_370), "A held update still moves the mean."
    old = EkfLocalizer(
        _seed131_map(),
        WheelOdometry(),
        _SEED131_PRIOR_370,
        _SEED131_PRIOR_COV_370,
        nis_quantile=None,
    )
    assert not old.update(_SEED131_SCANS[7]).held
    assert math.sqrt(old.covariance[2, 2]) < 0.005, "With the hold off, P shrinks as before."


class _KinkedMap(_LinearMap):
    """One beam reading ``2 - x`` below x = 0.25 and ``1.75 - 2 (x - 0.25)`` above: continuous,
    linear within 0.03 of any x but 0.25, with the slope doubling at the kink."""

    def __init__(self):
        super().__init__(2.0, [-1.0, 0.0, 0.0])

    def expected_ranges(self, poses):
        x = np.asarray(poses, dtype=np.float64).reshape(-1, 3)[:, 0]
        return np.where(x < 0.25, 2.0 - x, 1.75 - 2.0 * (x - 0.25))[:, None]


def test_an_update_whose_iteration_refuses_a_step_keeps_the_covariance():
    """Reading 1.25 (x = 0.5) from x = 0 with P = I: the first step, on slope -1, goes to
    x ~ 0.75; relinearized there on slope -2 the next step goes back 0.25 to x ~ 0.5, beyond the
    0.03 window, and is refused. The update did not converge: the first iterate's mean stands,
    P stays the prior, and the result says ``held``. Its NIS, 0.5625, is far under the quantile."""
    ekf = _ekf(_KinkedMap(), [0.0, 0.0, 0.0], np.eye(3))
    result = ekf.update([1.25])
    assert result.iterations == 1, f"The second iterate should be refused: {result}"
    np.testing.assert_array_equal(ekf.covariance, np.eye(3), err_msg="P should stay the prior.")
    assert result.held, "A refused iteration should hold the update."
    assert ekf.pose[0] == pytest.approx(0.75, abs=1e-3), f"The first iterate stands: {ekf.pose}"
    unheld = _ekf(_KinkedMap(), [0.0, 0.0, 0.0], np.eye(3), nis_quantile=None)
    assert not unheld.update([1.25]).held and unheld.covariance[0, 0] < 1e-3, (
        "With the hold off the first iterate's covariance stands, as before."
    )


def _inconsistent_pair(estimate_x=0.0):
    """Two linear beams (2 - x, 3 - x) and P = 1e-8 I, so S ~ sigma_r^2 = 1e-4 per beam: readings
    0.029 above the expected ranges pass each 3-sigma gate (8.4 < 9) but give NIS ~ 16.8, above
    chi2_0.999(2) = 14.1; readings equal to them give NIS 0."""
    ekf_map = _TwoBeamMap(jump_at=np.inf)
    expected = ekf_map.expected_ranges([estimate_x, 0.0, 0.0])[0]
    return ekf_map, expected + 0.029, expected


@pytest.mark.parametrize(
    ("sequence", "fired"),
    [
        (["off", "off", "off"], [False, False, True]),
        (["off", "clean", "off", "clean", "off"], [False] * 5),
    ],
)
def test_held_scans_count_toward_the_reset_only_in_a_row(sequence, fired):
    """A held scan with at least ``min_candidates`` candidates counts toward the reset like a
    mostly gated one: three in a row reset; held scans between consistent ones never do."""
    range_map, off, clean = _inconsistent_pair()
    ekf = _ekf(range_map, [0.0, 0.0, 0.0], 1e-8 * np.eye(3))
    results = [ekf.update(off if name == "off" else clean) for name in sequence]
    assert [r.reset for r in results] == fired, [r.reset for r in results]
    assert [r.held for r in results] == [name == "off" for name in sequence], results


@pytest.mark.parametrize(("min_candidates", "fires"), [(1, True), (2, False)])
def test_a_held_scan_needs_min_candidates_to_count(min_candidates, fires):
    """The refused one-beam update of ``_KinkedMap`` is held; with ``reset_after=1`` it resets only
    when one candidate is enough to count."""
    ekf = _ekf(
        _KinkedMap(), [0.0, 0.0, 0.0], np.eye(3), reset_after=1, min_candidates=min_candidates
    )
    result = ekf.update([1.25])
    assert result.reset is fires, result
    assert result.held, result


# HC-SR04 seed 94 (own dropout), run_trial(94, hc_sr04(), steps=330, kidnap_at=300), recorded at
# #203's head (c85e3d7), before the off-map count existed.
_SEED94_STEPS = (100, 299, 305, 312, 313, 329)
_SEED94_ESTIMATES = np.array(
    [
        [0.2329974513688819, 0.8614149992349731, -1.6905714911986625],
        [4.750037206591102, 1.6825319720818241, 1.543466446303519],
        [5.586591518567488, 1.2444908521897686, 1.3586444454368554],
        [6.732872757871282, 1.6800583452649818, 1.1722000498103604],
        [6.744119115629759, 1.7083568058243463, 1.2128330822545994],
        [6.898107464356389, 2.1733488498611746, 1.1986338593519141],
    ]
)
_SEED94_COV_312 = np.array(
    [
        [0.01021132491825088, 0.01253644554677795, 0.0064148167761099465],
        [0.01253644554677795, 0.015538543454813513, 0.008066866893636782],
        [0.0064148167761099465, 0.008066866893636782, 0.0062702309703077],
    ]
)
_SEED94_COV_329 = np.array(
    [
        [0.005939419860604414, 0.008945279934051795, 0.0017206244105699992],
        [0.008945279934051795, 0.018495765108921425, 0.009651712510121122],
        [0.0017206244105699992, 0.009651712510121122, 0.012779112568394117],
    ]
)


@pytest.mark.parametrize(("on_map", "fired"), [(False, [False, False, True]), (True, [False] * 3)])
def test_blind_scans_off_the_map_count_toward_the_reset(on_map, fired):
    """A dropout (NaN) leaves a scan with no candidate. With the estimate off the map (inside an
    obstacle or beyond a wall, where no robot can stand) such a scan counts, so the third resets
    and the filter reports itself lost; on the map it is a plain dropout and never counts."""
    range_map = _LinearMap(2.0, [-1.0, 0.0, 0.0], on_map=on_map)
    ekf = _ekf(range_map, [0.0, 0.0, 0.0], 1e-4 * np.eye(3))
    results = [ekf.update([np.nan]) for _ in range(3)]
    assert [r.reset for r in results] == fired, [r.reset for r in results]
    assert ekf.recent_resets == (0 if on_map else 1), ekf.recent_resets


def test_a_blind_scan_beside_a_wall_is_on_the_map():
    """The off-map test has no margin: an estimate whose centre is only 0.05 m from the west wall,
    closer than a robot's 0.1 m radius, is still inside the arena and so on the map; with a margin
    of the robot's radius it would count as lost. A blind scan (every reading a dropout) from there
    does not count, so with ``reset_after=1`` no reset."""
    range_map = SensorMap(ROOM, hc_sr04())
    ekf = _ekf(range_map, [0.05, 2.0, 0.0], 1e-4 * np.eye(3), reset_after=1)
    result = ekf.update(np.full(range_map.num_readings, np.nan))
    assert not (result.used.any() or result.gated.any()), f"The scan should be blind: {result}"
    assert not result.reset, (
        "An estimate beside a wall is on the map; its blind scan must not count."
    )


def test_a_scan_with_candidates_off_the_map_counts_as_before():
    """Off the map, a scan whose beam is used and agrees counts as it always did: not at all.
    The filter does not even ask whether its estimate is on the map; only a blind scan does."""
    range_map = _LinearMap(2.0, [-1.0, 0.0, 0.0], on_map=False)
    ekf = _ekf(range_map, [0.0, 0.0, 0.0], 1e-4 * np.eye(3), reset_after=1)
    result = ekf.update([2.0])
    assert result.used[0] and not result.reset, f"A consistent used beam does not count: {result}"
    assert range_map.free_calls == 0, (
        "A scan with candidates must not ask whether it is on the map."
    )


def test_count_off_map_false_is_the_filter_before():
    """``count_off_map=False`` is the filter before the off-map count: HC-SR04 seed 94 (own
    dropout), kidnapped at step 300, gives the estimates and covariances recorded at #203's head
    (c85e3d7). The default first counts a blind off-map scan at step 311 and resets at 313: equal
    to the record before, different after."""
    steps = list(_SEED94_STEPS)
    old = run_trial(
        94, hc_sr04(), steps=330, kidnap_at=300, make_localizer=ekf_factory(count_off_map=False)
    )
    np.testing.assert_allclose(old.estimates[steps], _SEED94_ESTIMATES, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(old.covariances[312], _SEED94_COV_312, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(old.covariances[329], _SEED94_COV_329, rtol=1e-9, atol=1e-15)
    new = run_trial(94, hc_sr04(), steps=330, kidnap_at=300)
    np.testing.assert_allclose(new.estimates[steps[:4]], _SEED94_ESTIMATES[:4], rtol=1e-9)
    np.testing.assert_allclose(new.covariances[312], _SEED94_COV_312, rtol=1e-9, atol=1e-15)
    assert not np.allclose(new.covariances[329], _SEED94_COV_329, rtol=1e-6), (
        "The default should reset at step 313 and so differ from the filter before."
    )


def test_an_estimate_outside_the_walls_reports_itself_lost():
    """An HC-SR04 filter whose estimate sits beyond the east wall (5.6 m in a 5 m room) while the
    robot stands inside: every sonar is nonlinear or reads no echo there, so no scan has a
    candidate. Before the off-map count none of them counted and the filter stayed quiet."""
    sensor = hc_sr04()
    range_map = SensorMap(ROOM, sensor)
    estimate = np.array([5.6, 2.0, 0.3])
    assert not range_map.free(estimate[None, :2], 0.0)[0], "The estimate must be off the map."
    ekf = _ekf(range_map, estimate, np.diag([0.04, 0.04, 0.01]))
    for step in range(30):
        result = ekf.update(sensor.read(ROOM, TRUTH, np.random.default_rng(step))[0])
        assert not (result.used.any() or result.gated.any()), f"Scan {step} has candidates."
    assert ekf.recent_resets >= 2, f"Off the map the filter should report itself lost: {ekf}"


def test_tracking_never_asks_whether_the_estimate_is_on_the_map(monkeypatch):
    """Tracking (EST-1 seed 1, HC-SR04 at 30 % dropout): the filter asks ``free`` only on a scan
    with no candidate, and every answer is yes, so the off-map count never fires."""
    scans, asked, answers = [], [], []
    update, free = EkfLocalizer.update, SensorMap.free

    def recorded_update(self, readings):
        result = update(self, readings)
        scans.append(not (result.used.any() or result.gated.any()))
        return result

    def recorded_free(self, points, margin):
        answer = free(self, points, margin)
        asked.append(len(scans))  # the scan being updated is the next one appended
        answers.extend(answer.tolist())
        return answer

    monkeypatch.setattr(EkfLocalizer, "update", recorded_update)
    monkeypatch.setattr(SensorMap, "free", recorded_free)
    run_trial(1, hc_sr04(), dropout_prob=0.3)
    assert all(answers), "A tracked estimate should always be on the map."
    assert all(scans[index] for index in asked), "free was asked on a scan with candidates."


def test_the_hold_off_is_the_filter_before_the_hold():
    """``nis_quantile=None`` turns off the whole hold: seed 131 (HC-SR04, 30 % dropout) for 360
    steps gives the estimates and covariance recorded with the filter before the hold. The default
    differs: it holds a few routine scans in tracking (at step 180 on this seed)."""
    old = run_trial(
        131, hc_sr04(), dropout_prob=0.3, steps=360, make_localizer=ekf_factory(nis_quantile=None)
    )
    steps = list(_GOLDEN_STEPS)
    np.testing.assert_allclose(old.estimates[steps], _GOLDEN_ESTIMATES, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(old.covariances[-1], _GOLDEN_COV_359, rtol=1e-9, atol=1e-15)
    new = run_trial(131, hc_sr04(), dropout_prob=0.3, steps=360)
    assert not np.allclose(new.covariances[-1], _GOLDEN_COV_359, rtol=1e-6), (
        "The default should hold some scans and so differ from the filter before the hold."
    )


def test_seed131_stays_tracked_canary():
    """Canary, not proof: seed 131's trajectory is chaotic (with the hold it departs from the old
    one at step 180), so this guards the whole 400-step run only coarsely; the replay test above is
    the causal check. The old filter was 0.23 m off at step 377 and 6.45 m by step 586.

    To re-find a lost seed after a change: run EST-3's HC-SR04 job with ``--records`` (TESTING.md),
    load the ``hc_sr04_0.3_ekf_tracking_seed*.npz`` records with ``TrialRecord.load`` and take the
    one with the largest position error."""
    trial = run_trial(131, hc_sr04(), dropout_prob=0.3, steps=400)
    error = np.hypot(*(trial.estimates - trial.true_poses)[:, :2].T)
    assert error.max() < 0.10, f"Seed 131 should stay within 10 cm; peaked at {error.max():.3f} m"


def test_after_a_large_bump_the_covariance_does_not_collapse_onto_a_wrong_pose():
    """A still lidar robot bumped 1.5 m cannot be recovered. The old filter's update after each
    reset took one unconverged step and shrank the position sigma to millimetres round a pose
    1.5 m off (75 of 100 scans); a held update keeps P, and ``recent_resets`` says lost."""
    sensor = lidar()
    ekf = _ekf(SensorMap(ROOM, sensor), TRUTH, STILL_P0)
    bumped = TRUTH + np.array([1.5, 0.0, 0.0])
    collapsed, recent = 0, []
    for step in range(SETTLE_SCANS + 100):
        ekf.predict([0.0, 0.0])
        ekf.update(_scan(sensor, TRUTH if step < SETTLE_SCANS else bumped, seed=step))
        if step >= SETTLE_SCANS:
            position, _ = _pose_error(ekf.pose, bumped)
            sigma = math.sqrt(max(ekf.covariance[0, 0], ekf.covariance[1, 1]))
            collapsed += position > 0.5 and sigma < 0.01
            recent.append(ekf.recent_resets)
    assert collapsed <= 5, f"{collapsed} of 100 scans collapsed below 1 cm while 1.5 m off"
    assert min(recent[20:]) >= 2, f"recent_resets should keep saying lost: {recent}"


def test_after_a_kidnapping_the_covariance_does_not_collapse_onto_a_wrong_pose():
    """EST-4's trial (seed 101, lidar, kidnapped at step 300): the old filter was within 1 cm of
    certainty while over 0.5 m off on 92 of the 300 steps after the kidnapping."""
    trial = run_trial(101, lidar(), kidnap_at=300)
    error = np.hypot(*(trial.estimates - trial.true_poses)[300:, :2].T)
    sigma = np.sqrt(np.maximum(trial.covariances[300:, 0, 0], trial.covariances[300:, 1, 1]))
    collapsed = int(np.count_nonzero((error > 0.5) & (sigma < 0.01)))
    assert collapsed <= 10, f"{collapsed} of 300 post-kidnap steps collapsed onto a wrong pose"


def test_a_heading_bump_the_sonars_barely_see_fires_no_reset():
    """A documented limit, not a goal: bumped 0.3 rad, the HC-SR04's sonars 0, 1 and 3 change by
    under 3 cm and are mostly used, sonar 4 turns nonlinear and sonar 2 is gated, so a scan rarely
    has half its candidates gated and never three in a row: no reset, and the heading stays about
    0.3 rad off. (It was once put down to beams turning nonlinear; only one does.) A fix must change
    this test on purpose."""
    ekf, bumped, after = _bumped_still_robot(hc_sr04(), [0.0, 0.0, 0.3], scans_after=30)
    for result, _ in after:
        assert result.nonlinear[4] and not result.used[2] and not result.reset, result
    _, heading = _pose_error(ekf.pose, bumped)
    assert heading > 0.25, f"The 0.3 rad error should remain, it is {heading:.3f} rad"


@pytest.mark.parametrize(
    ("dof", "exact"),
    [(1, 10.828), (2, 13.816), (3, 16.266), (5, 20.515), (12, 32.909), (40, 73.402)],
)
def test_chi2_quantile_is_within_3_5_percent_of_the_table(dof, exact):
    """Wilson-Hilferty against the exact 0.999 quantiles (worst at 1 degree of freedom, +3.0 %)."""
    assert chi2_quantile(0.999, dof) == pytest.approx(exact, rel=0.035)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_chi2_quantile_refuses_a_probability_outside_0_1(q):
    with pytest.raises(ValueError, match="quantile"):
        chi2_quantile(q, 3)
