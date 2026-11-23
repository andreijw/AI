"""Scan-to-map matching on a distance field and the scan-matching SLAM front end: the residual
Jacobian against finite differences, recovery of known poses, the prior's role, the acceptance
rules, and the filter's predict, mapping and never-raise contract."""

import math

import numpy as np
import pytest

from rl_navigation.estimation import predict_pose
from rl_navigation.estimation.ekf import chi2_quantile
from rl_navigation.estimation.kalman import symmetrized
from rl_navigation.estimation.poses import pose_difference
from rl_navigation.mapping import LIDAR_LOG_ODDS, OccupancyGrid
from rl_navigation.mapping.fields import DistanceField, distance_field
from rl_navigation.sim import Arena, WheelOdometry, hc_sr04, lidar, rotation
from rl_navigation.slam.scan_matching import (
    ScanMatchingSlam,
    ScanMatchSettings,
    match_points,
    match_residuals,
    match_scan,
    scan_endpoints,
)
from tests.helpers import assert_gradients_match

SETTINGS = ScanMatchSettings()
# The front-end tests fix their own floor, so they do not depend on the tuned default.
SLAM_SETTINGS = ScanMatchSettings(pose_noise_floor=(0.01, 0.01, 0.01))
# The room's walls and box edges lie on cell borders, so every surface is mapped half a cell
# (2.5 cm) behind itself and the nearer surfaces pull the match by up to that much: the
# correlated centre bias of a 5 cm grid. A match is right within half a cell here.
HALF_CELL = 0.025
SENSOR = lidar(num_beams=360, noise_std=0.0)
ODOMETRY = WheelOdometry()
ROOM = Arena(
    5.0,
    5.0,
    circle_centers=[[3.5, 3.5]],
    circle_radii=[0.4],
    boxes=[[1.0, 3.2, 1.8, 4.0]],
)
ORIGIN = (-1.0, -1.0)
SHAPE = (140, 140)
MAPPED_FROM = np.array([[2.5, 2.0, 0.3], [1.5, 1.5, 1.0], [3.8, 2.2, -2.0], [2.6, 4.4, 2.5]])
TRUE_POSE = np.array([2.4, 2.1, 0.4])
WIDE = 100.0 * np.eye(3)


def _room_grid(poses=MAPPED_FROM, sensor=SENSOR):
    grid = OccupancyGrid(sensor, SHAPE, SETTINGS.resolution, LIDAR_LOG_ODDS, ORIGIN)
    for pose in poses:
        grid.update(pose, sensor.true_ranges(ROOM, pose[None])[0])
    return grid


@pytest.fixture(scope="module")
def room_field():
    occupied = _room_grid().occupied()
    return DistanceField.from_occupied(occupied, SETTINGS.resolution, SETTINGS.field_reach, ORIGIN)


def _scan(pose=TRUE_POSE, sensor=SENSOR):
    return sensor.true_ranges(ROOM, np.asarray(pose)[None])[0]


def _points(pose=TRUE_POSE):
    points, _ = scan_endpoints(SENSOR, np.minimum(_scan(pose), SENSOR.max_range))
    return points


# The end points and the residuals ----------------------------------------------------------------


def test_scan_endpoints_are_the_usable_beams_in_the_robots_frame():
    sensor = lidar(num_beams=4, noise_std=0.0)
    readings = np.array([1.0, np.nan, sensor.max_range, 0.5])
    points, beams = scan_endpoints(sensor, readings)
    np.testing.assert_array_equal(beams, [0, 3])
    np.testing.assert_allclose(points, [[1.0, 0.0], [0.0, -0.5]], atol=1e-12)


def test_scan_endpoints_refuse_a_sensor_with_cones():
    with pytest.raises(ValueError, match="one ray per"):
        scan_endpoints(hc_sr04(), np.ones(5))


def test_the_residual_jacobian_matches_finite_differences(room_field):
    points = _points()
    pose = TRUE_POSE + np.array([0.013, -0.021, 0.017])
    terms = match_residuals(room_field, points, pose, SETTINGS.max_distance)
    keep = terms.inliers
    assert keep.sum() > 200

    def residuals(at):
        return match_residuals(room_field, points[keep], at, SETTINGS.max_distance).residuals

    param = pose.copy()

    def loss():
        return 0.5 * float(np.sum(residuals(param) ** 2))

    analytic = terms.jacobian[keep].T @ terms.residuals[keep]
    assert_gradients_match(loss, {"pose": param}, {"pose": analytic}, rtol=2e-3, atol=1e-6)


def test_the_residual_jacobian_matches_finite_differences_under_random_weights(room_field):
    """Each row on its own: a random weighting of the residuals, ``w @ r``, has gradient
    ``J^T w``, so one wrong row cannot hide behind the others."""
    points = _points()
    pose = TRUE_POSE + np.array([-0.017, 0.011, -0.013])
    terms = match_residuals(room_field, points, pose, SETTINGS.max_distance)
    keep = terms.inliers
    weights = np.random.default_rng(7).uniform(-1.0, 1.0, int(keep.sum()))
    param = pose.copy()

    def loss():
        return float(
            weights
            @ match_residuals(room_field, points[keep], param, SETTINGS.max_distance).residuals
        )

    analytic = terms.jacobian[keep].T @ weights
    assert_gradients_match(loss, {"pose": param}, {"pose": analytic}, rtol=2e-3, atol=1e-6)


def test_a_point_beyond_the_truncation_is_no_inlier_and_has_no_gradient():
    """Distances between ``max_distance`` and the field's reach exist in the field, but such a
    point is an outlier: residual ``max_distance``, zero Jacobian row."""
    occupied = np.zeros((30, 30), dtype=bool)
    occupied[:, 10] = True
    field = DistanceField.from_occupied(occupied, 0.05, 0.3, (0.0, 0.0))
    wall_x = 10.5 * 0.05
    points = np.array([[wall_x + 0.12, 0.73], [wall_x + 0.17, 0.73]])
    terms = match_residuals(field, points, np.zeros(3), 0.15)
    np.testing.assert_array_equal(terms.inliers, [True, False])
    assert terms.residuals[0] == pytest.approx(0.12, abs=1e-12)
    assert terms.residuals[1] == 0.15
    np.testing.assert_array_equal(terms.jacobian[1], np.zeros(3))


def test_an_inlier_near_the_truncation_keeps_its_exact_gradient():
    """The field reaches ``max_distance + resolution sqrt 2``, so an inlier's four cell centres
    are never clamped: 0.12 m from a straight wall with the truncation at 0.13 m, the gradient
    across the wall is exactly 1 (a field clamped at 0.13 would give 0.6)."""
    settings = ScanMatchSettings(max_distance=0.13)
    occupied = np.zeros((30, 30), dtype=bool)
    occupied[:, 10] = True
    field = DistanceField.from_occupied(occupied, 0.05, settings.field_reach, (0.0, 0.0))
    point = np.array([[10.5 * 0.05 + 0.12, 0.73]])
    terms = match_residuals(field, point, np.zeros(3), settings.max_distance)
    assert terms.inliers[0]
    np.testing.assert_allclose(terms.jacobian[0, :2], [1.0, 0.0], atol=1e-12)


# Matching -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset", [[0.04, -0.03, 0.03], [-0.05, 0.02, -0.04], [0.0, 0.06, 0.05], [0.03, 0.03, -0.06]]
)
def test_a_match_recovers_a_known_pose_from_an_offset_prior(room_field, offset):
    match = match_points(room_field, _points(), TRUE_POSE + np.array(offset), WIDE, SETTINGS)
    error = pose_difference(match.pose, TRUE_POSE)
    assert match.accepted, match.reason
    assert math.hypot(error[0], error[1]) < HALF_CELL
    assert abs(error[2]) < 0.01
    assert match.termination == "converged"


def test_matches_from_different_priors_land_on_one_pose(room_field):
    """The basin: from offsets up to 6 cm and 0.06 rad the match reaches the same minimum."""
    offsets = [[0.0, 0.0, 0.0], [0.04, -0.03, 0.03], [-0.05, 0.02, -0.04], [0.03, 0.06, -0.06]]
    poses = [
        match_points(room_field, _points(), TRUE_POSE + np.array(o), WIDE, SETTINGS).pose
        for o in offsets
    ]
    for pose in poses[1:]:
        error = pose_difference(pose, poses[0])
        assert math.hypot(error[0], error[1]) < 0.002 and abs(error[2]) < 0.002


def test_a_match_across_pi_wraps_the_heading(room_field):
    pose = np.array([2.4, 2.1, math.pi - 0.01])
    prior = np.array([2.41, 2.09, -math.pi + 0.01])
    covariance = np.diag([0.03, 0.03, 0.03]) ** 2
    match = match_points(room_field, _points(pose), prior, covariance, SETTINGS)
    assert match.accepted, match.reason
    assert abs(pose_difference(match.pose, pose)[2]) < 0.01


def _phantoms(count=24, radius=0.5):
    """End points in open floor, more than ``d_max`` from every surface: outliers in the grid."""
    angles = np.linspace(-math.pi, math.pi, count, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def test_the_last_cost_is_the_truncated_quadratic_over_every_beam_in_the_grid(room_field):
    """Outliers in the grid count at ``d_max`` each: the cost is over the fixed set ``B``."""
    points = np.concatenate([_points(), _phantoms()])
    prior = TRUE_POSE + np.array([0.03, -0.02, 0.02])
    covariance = np.diag([0.05, 0.05, 0.05]) ** 2
    match = match_points(room_field, points, prior, covariance, SETTINGS)
    c, s = math.cos(match.pose[2]), math.sin(match.pose[2])
    world = points @ np.array([[c, s], [-s, c]]) + match.pose[:2]
    sample = room_field.sample(world)
    beams = room_field.sample(points @ rotation(prior[2]).T + prior[:2]).inside
    distances = np.where(sample.inside, np.minimum(sample.values, SETTINGS.max_distance), 0.15)
    e = pose_difference(match.pose, prior)
    expected = 0.5 * np.sum(
        distances[beams] ** 2
    ) / SETTINGS.point_std**2 + 0.5 * e @ np.linalg.solve(covariance, e)
    assert match.costs[-1] == pytest.approx(expected, rel=1e-12)
    assert match.beams == beams.sum()
    assert match.beams - np.count_nonzero(match.inliers) >= 24


def test_a_step_that_raises_the_cost_is_halved():
    """Scattered occupied cells and points (a seeded case where full Gauss-Newton steps
    overshoot): steps are halved, and the cost never rises."""
    rng = np.random.default_rng(0)
    occupied = rng.random((40, 40)) < 0.03
    field = DistanceField.from_occupied(occupied, 0.05, SETTINGS.field_reach, (0.0, 0.0))
    points = rng.uniform(-0.6, 0.6, (40, 2))
    match = match_points(field, points, np.array([1.0, 1.0, 0.0]), WIDE, SETTINGS)
    assert match.halvings > 0
    assert all(b <= a for a, b in zip(match.costs, match.costs[1:], strict=False))


def test_two_gauss_newton_steps_by_hand(room_field):
    """Two full steps under a tight prior, worked in the test from ``match_residuals``: the
    second step's gradient carries the prior's pull ``Pbar^-1 (xi - xibar)``, which the first,
    taken at the prior itself, does not."""
    settings = ScanMatchSettings(max_iterations=2, max_halvings=0)
    points = _points()
    prior = TRUE_POSE + np.array([0.012, -0.009, 0.008])
    covariance = np.diag([0.004, 0.004, 0.004]) ** 2
    information = symmetrized(np.linalg.inv(covariance))
    weight = settings.point_std**-2
    inside = room_field.sample(points @ rotation(prior[2]).T + prior[:2]).inside
    xi = prior.copy()
    for _ in range(2):
        terms = match_residuals(room_field, points[inside], xi, settings.max_distance)
        H = weight * terms.jacobian.T @ terms.jacobian + information
        b = weight * terms.jacobian.T @ terms.residuals + information @ pose_difference(xi, prior)
        xi = xi - np.linalg.solve(H, b)
    match = match_points(room_field, points, prior, covariance, settings)
    assert match.halvings == 0 and match.iterations == 2 and len(match.costs) == 3
    np.testing.assert_allclose(match.pose, xi, rtol=0, atol=1e-12)


def test_a_match_with_too_few_inliers_is_rejected(room_field):
    """Twenty end points that all fit the map, against a floor of 30."""
    points = _points()[::18][:20]
    match = match_points(room_field, points, TRUE_POSE, WIDE, SETTINGS)
    assert np.count_nonzero(match.inliers) == 20
    assert not match.accepted
    assert match.reason == "few_inliers"


def test_a_step_that_raises_the_cost_is_taken_at_half_length():
    """The seeded scattered case's fourth iteration, by hand: its full Gauss-Newton step raises
    the cost, and the step taken is exactly half of it."""
    rng = np.random.default_rng(0)
    occupied = rng.random((40, 40)) < 0.03
    field = DistanceField.from_occupied(occupied, 0.05, SETTINGS.field_reach, (0.0, 0.0))
    points = rng.uniform(-0.6, 0.6, (40, 2))
    prior = np.array([1.0, 1.0, 0.0])
    before = match_points(field, points, prior, WIDE, ScanMatchSettings(max_iterations=3))
    after = match_points(field, points, prior, WIDE, ScanMatchSettings(max_iterations=4))
    assert (before.halvings, after.halvings) == (0, 1)
    inside = field.sample(points @ rotation(prior[2]).T + prior[:2]).inside
    information = np.linalg.inv(WIDE)
    weight = SETTINGS.point_std**-2

    def cost(xi):
        r = match_residuals(field, points[inside], xi, SETTINGS.max_distance).residuals
        e = pose_difference(xi, prior)
        return 0.5 * weight * r @ r + 0.5 * e @ information @ e

    terms = match_residuals(field, points[inside], before.pose, SETTINGS.max_distance)
    H = weight * terms.jacobian.T @ terms.jacobian + information
    b = weight * terms.jacobian.T @ terms.residuals + information @ pose_difference(
        before.pose, prior
    )
    step = -np.linalg.solve(H, b)
    assert cost(before.pose + step) > cost(before.pose)
    np.testing.assert_allclose(after.pose, before.pose + 0.5 * step, rtol=0, atol=1e-12)


def test_a_scan_mostly_off_the_map_fails_the_inlier_share(room_field):
    settings = ScanMatchSettings(min_inlier_share=0.95)
    points = np.concatenate([_points(), _phantoms(count=60)])
    match = match_points(room_field, points, TRUE_POSE, WIDE, settings)
    assert not match.accepted
    assert match.reason == "inlier_share"
    assert match_points(room_field, _points(), TRUE_POSE, WIDE, settings).accepted


def test_inliers_that_fit_loosely_fail_the_residual_rms(room_field):
    """The noise-free scan's inliers sit about 2 cm from the cell centres (the centre bias):
    under a 5 mm limit the match is rejected for its RMS, and passes under the default."""
    tight = ScanMatchSettings(max_residual_rms=0.005)
    match = match_points(room_field, _points(), TRUE_POSE, WIDE, tight)
    assert not match.accepted
    assert match.reason == "residual_rms"
    assert match.residual_rms > 0.005
    assert match_points(room_field, _points(), TRUE_POSE, WIDE, SETTINGS).accepted


def test_the_cost_never_rises_between_iterates(room_field):
    prior = TRUE_POSE + np.array([0.08, -0.07, 0.09])
    match = match_points(room_field, _points(), prior, WIDE, SETTINGS)
    assert len(match.costs) >= 3
    assert all(b <= a for a, b in zip(match.costs, match.costs[1:], strict=False))


def test_the_covariance_is_the_inverse_hessian_at_the_returned_pose(room_field):
    points = _points()
    prior = TRUE_POSE + np.array([0.02, 0.01, -0.02])
    covariance = np.diag([0.04, 0.04, 0.04]) ** 2
    match = match_points(room_field, points, prior, covariance, SETTINGS)
    terms = match_residuals(room_field, points, match.pose, SETTINGS.max_distance)
    J = terms.jacobian[terms.inliers]
    H = J.T @ J / SETTINGS.point_std**2 + np.linalg.inv(covariance)
    np.testing.assert_allclose(match.covariance, np.linalg.inv(H), rtol=1e-9, atol=1e-15)
    np.testing.assert_array_equal(match.covariance, match.covariance.T)


def test_one_wall_leaves_the_along_wall_variance_to_the_prior():
    occupied = np.zeros((60, 60), dtype=bool)
    occupied[:, 40] = True
    field = DistanceField.from_occupied(occupied, 0.05, SETTINGS.field_reach, (0.0, 0.0))
    wall_x = 40.5 * 0.05
    pose = np.array([1.0, 1.5, 0.0])
    ys = np.linspace(-0.6, 0.6, 61)
    points = np.column_stack([np.full_like(ys, wall_x - pose[0]), ys])
    covariance = np.diag([0.02, 0.02, 0.02]) ** 2
    match = match_points(field, points, pose, covariance, SETTINGS)
    assert match.accepted, match.reason
    assert match.covariance[1, 1] >= 0.9 * covariance[1, 1]
    assert match.covariance[0, 0] < 0.1 * covariance[0, 0]


def test_with_no_inliers_the_prior_stands(room_field):
    far = np.full((50, 2), 30.0)
    prior = TRUE_POSE
    match = match_points(room_field, far, prior, WIDE, SETTINGS)
    assert not match.accepted
    assert match.reason == "no_beams"
    np.testing.assert_array_equal(match.pose, prior)


def test_a_match_far_from_a_tight_prior_fails_the_gate(room_field):
    """The scan pulls 5 cm from a prior that claims 5 mm: the move is far outside chi-square."""
    prior = TRUE_POSE + np.array([0.05, 0.0, 0.0])
    covariance = np.diag([0.005, 0.005, 0.005]) ** 2
    match = match_points(room_field, _points(), prior, covariance, SETTINGS)
    assert not match.accepted
    assert match.reason == "gate"
    assert match.gate_distance > chi2_quantile(SETTINGS.gate_quantile, 3)


def test_a_scan_that_does_not_fit_the_map_is_rejected(room_field):
    rng = np.random.default_rng(2)
    angles = rng.uniform(-math.pi, math.pi, 300)
    ranges = rng.uniform(0.2, 2.0, 300)
    points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
    match = match_points(room_field, points, TRUE_POSE, WIDE, SETTINGS)
    assert not match.accepted
    assert match.reason in {"few_inliers", "inlier_share", "residual_rms"}


def test_a_match_that_runs_out_of_iterations_says_so(room_field):
    settings = ScanMatchSettings(max_iterations=1)
    prior = TRUE_POSE + np.array([0.06, -0.05, 0.05])
    match = match_points(room_field, _points(), prior, WIDE, settings)
    assert match.termination == "capped"
    assert match.iterations == 1


def test_match_scan_is_match_points_on_the_scans_end_points(room_field):
    readings = np.minimum(_scan(), SENSOR.max_range)
    readings[::7] = np.nan
    prior = TRUE_POSE + np.array([0.02, 0.01, 0.01])
    via_scan = match_scan(room_field, SENSOR, readings, prior, WIDE, SETTINGS)
    points, _ = scan_endpoints(SENSOR, readings)
    via_points = match_points(room_field, points, prior, WIDE, SETTINGS)
    np.testing.assert_array_equal(via_scan.pose, via_points.pose)


@pytest.mark.parametrize(
    "change",
    [
        {"point_std": 0.0},
        {"max_distance": 0.0},
        {"max_iterations": 0},
        {"min_inlier_share": 1.5},
        {"gate_quantile": 1.0},
        {"pose_noise_floor": (0.0, -0.01, 0.0)},
        {"pose_noise_floor": (0.0, 0.0)},
    ],
)
def test_bad_settings_are_refused(change):
    with pytest.raises(ValueError):
        ScanMatchSettings(**change)


# The front end ----------------------------------------------------------------------------------


def _slam(settings=SLAM_SETTINGS, start=TRUE_POSE, covariance=None, sensor=SENSOR):
    covariance = np.diag([0.02, 0.02, 0.02]) ** 2 if covariance is None else covariance
    return ScanMatchingSlam(sensor, ODOMETRY, start, covariance, SHAPE, ORIGIN, settings)


def test_predict_is_the_shared_pose_predict_plus_the_noise_floor():
    settings = ScanMatchSettings(pose_noise_floor=(0.001, 0.002, 0.003))
    slam = _slam(settings)
    covariance = slam.covariance
    travel = np.array([0.031, 0.027])
    slam.predict(travel)
    prediction = predict_pose(TRUE_POSE, travel, ODOMETRY)
    expected = prediction.F @ covariance @ prediction.F.T + prediction.noise
    expected = 0.5 * (expected + expected.T) + np.diag([0.001, 0.002, 0.003]) ** 2
    np.testing.assert_array_equal(slam.pose, prediction.pose)
    np.testing.assert_allclose(slam.covariance, expected, rtol=1e-12, atol=1e-18)


def test_the_first_usable_scan_is_mapped_at_the_start():
    slam = _slam()
    result = slam.update(_scan())
    assert result.mapped and result.match is None
    expected = OccupancyGrid(SENSOR, SHAPE, SETTINGS.resolution, LIDAR_LOG_ODDS, ORIGIN)
    expected.update(TRUE_POSE, _scan())
    np.testing.assert_array_equal(slam.grid.log_odds, expected.log_odds)
    np.testing.assert_array_equal(slam.pose, TRUE_POSE)


@pytest.mark.parametrize(
    "readings",
    [
        np.full(360, np.nan),
        np.full(360, SENSOR.max_range),
        np.where(np.arange(360) % 2 == 0, np.nan, SENSOR.max_range),
    ],
)
def test_a_scan_without_ranges_only_predicts(readings):
    slam = _slam()
    slam.update(_scan())
    before = slam.grid.log_odds.copy()
    pose, covariance = slam.pose, slam.covariance
    result = slam.update(readings)
    assert not result.mapped and not result.used.any()
    np.testing.assert_array_equal(slam.grid.log_odds, before)
    np.testing.assert_array_equal(slam.pose, pose)
    np.testing.assert_array_equal(slam.covariance, covariance)


def test_a_moved_robot_is_matched_and_its_scan_mapped():
    slam = _slam()
    slam.update(_scan())
    travel = ODOMETRY.wheel_travel(0.3, 0.0, 0.1)
    moved = predict_pose(TRUE_POSE, travel, ODOMETRY).pose
    slam.predict(travel + np.array([0.004, -0.004]))  # the encoders slip a little
    readings = _scan(moved)
    readings[::5] = np.nan  # dropouts: beams that cannot be used
    result = slam.update(readings)
    _, beams = scan_endpoints(SENSOR, readings)
    expected = np.zeros(SENSOR.num_readings, dtype=bool)
    assert result.match is not None
    expected[beams[result.match.inliers]] = True
    np.testing.assert_array_equal(result.used, expected)
    assert not result.used[::5].any()
    assert result.mapped and result.match is not None and result.match.accepted
    error = pose_difference(slam.pose, moved)
    assert math.hypot(error[0], error[1]) < HALF_CELL and abs(error[2]) < 0.01
    assert result.used.sum() == np.count_nonzero(result.match.inliers)


def test_each_match_reads_the_field_of_the_map_as_it_stands():
    """The field is rebuilt whenever the occupied cells changed since the last match."""
    slam = _slam()
    slam.update(_scan())
    for step in range(3):
        slam.predict(ODOMETRY.wheel_travel(0.3, 0.5, 0.1))
        before = slam.grid.occupied().copy()
        result = slam.update(_scan(slam.pose + np.array([0.002 * step, 0.0, 0.0])))
        assert result.mapped
        expected = distance_field(before, SETTINGS.resolution, SETTINGS.field_reach)
        assert slam.field is not None
        np.testing.assert_array_equal(slam.field.values, expected)
    assert not np.array_equal(slam.grid.occupied(), _first_scan_occupied())


def _first_scan_occupied():
    grid = OccupancyGrid(SENSOR, SHAPE, SETTINGS.resolution, LIDAR_LOG_ODDS, ORIGIN)
    grid.update(TRUE_POSE, _scan())
    return grid.occupied()


def test_a_rejected_match_leaves_the_map_and_keeps_the_prediction():
    slam = _slam(covariance=np.diag([0.002, 0.002, 0.002]) ** 2)
    slam.update(_scan())
    before = slam.grid.log_odds.copy()
    slam.predict(np.zeros(2))
    pose, covariance = slam.pose, slam.covariance
    result = slam.update(_scan(TRUE_POSE + np.array([0.06, 0.0, 0.0])))
    assert result.match is not None and not result.match.accepted
    assert not result.mapped
    np.testing.assert_array_equal(slam.grid.log_odds, before)
    np.testing.assert_array_equal(slam.pose, pose)
    np.testing.assert_array_equal(slam.covariance, covariance)


def test_a_match_off_the_bounds_is_rejected():
    slam = ScanMatchingSlam(
        SENSOR,
        ODOMETRY,
        TRUE_POSE,
        0.02**2 * np.eye(3),
        SHAPE,
        ORIGIN,
        SLAM_SETTINGS,
        bounds=(0.0, 0.0, 2.401, 5.0),
    )
    slam.update(_scan())
    result = slam.update(_scan(TRUE_POSE + np.array([0.04, 0.0, 0.0])))
    assert result.match is not None and result.match.reason == "off_bounds"
    assert slam.pose[0] == TRUE_POSE[0]


def test_the_front_end_is_deterministic():
    runs = []
    for _ in range(2):
        slam = _slam()
        slam.update(_scan())
        for k in range(4):
            slam.predict(ODOMETRY.wheel_travel(0.3, 0.4, 0.1))
            slam.update(_scan(slam.pose + np.array([0.003, -0.002, 0.004 * k])))
        runs.append((slam.pose, slam.covariance, slam.grid.log_odds.copy()))
    for a, b in zip(runs[0], runs[1], strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_front_end_refuses_a_sensor_with_cones():
    with pytest.raises(ValueError, match="one ray per"):
        _slam(sensor=hc_sr04())


def test_the_front_end_refuses_a_scan_of_the_wrong_length():
    with pytest.raises(ValueError, match="readings"):
        _slam().update(np.ones(10))
