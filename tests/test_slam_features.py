"""rl_navigation.slam.features: circles fitted to the clusters of a lidar scan.

Arcs are built as a lidar sees them: beams from the sensor at the origin, each reading the
distance along its beam to the circle, plus Gaussian range noise. Every random input is seeded.
"""

import math

import numpy as np
import pytest

from rl_navigation.sim import Arena, generate_layout, hc_sr04, lidar
from rl_navigation.slam.features import (
    FeatureSettings,
    circle_residuals,
    extract_circles,
    fit_circle,
    incidence_weights,
    scan_points,
    segment_scan,
)

SIGMA = 0.01


def ray_circle_ranges(angles, centre, radius):
    """Distance along each beam from the origin to the near side of the circle (NaN on a miss)."""
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    b = directions @ np.asarray(centre, dtype=float)
    disc = b * b - (float(np.dot(centre, centre)) - radius * radius)
    return np.where(disc >= 0.0, b - np.sqrt(np.maximum(disc, 0.0)), np.nan)


def arc(centre, radius, count, span, rng=None, noise=0.0):
    """``count`` beams spread over the middle ``span`` of the angle the circle subtends, as
    ``(points, directions)``, with ``noise`` metres of range noise along each beam."""
    centre = np.asarray(centre, dtype=float)
    half = math.asin(radius / float(np.hypot(*centre))) * span
    angles = math.atan2(centre[1], centre[0]) + np.linspace(-half, half, count)
    ranges = ray_circle_ranges(angles, centre, radius)
    if noise:
        ranges = ranges + noise * rng.standard_normal(count)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return ranges[:, None] * directions, directions


def scan_of(arena, sensor, pose, rng=None, noise=0.0):
    """A noise-free (or seeded noisy) reading of ``sensor`` at ``pose``, clipped as a sensor does."""
    true = sensor.true_ranges(arena, np.asarray(pose, dtype=float))[0]
    if noise:
        true = true + noise * rng.standard_normal(len(true))
    return np.clip(true, sensor.min_range, sensor.max_range)


def fit_nees(fit, centre, radius):
    error = fit.z - np.array([centre[0], centre[1], radius])
    return float(error @ np.linalg.solve(fit.covariance, error))


# Fit ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("centre", "radius", "count", "span"),
    [
        ((1.5, 0.4), 0.2, 8, 0.9),
        ((-0.8, 1.9), 0.5, 40, 0.98),
        ((0.0, -1.2), 0.35, 12, 0.6),
    ],
)
def test_a_noise_free_arc_gives_the_circle_exactly(centre, radius, count, span):
    points, directions = arc(centre, radius, count, span)
    fit = fit_circle(points, directions, SIGMA, FeatureSettings())
    assert fit is not None, "a clean arc of at least min_points must be fitted"
    np.testing.assert_allclose(
        fit.z, [*centre, radius], atol=1e-9, err_msg="the fit should recover centre and radius"
    )


def test_the_fit_covariance_is_consistent_over_many_noisy_arcs():
    """Mean NEES of the (centre, radius) fit over 2000 seeded arcs lies near 3 (3 degrees of
    freedom): the per-beam weights sigma cos(incidence) make the covariance neither too tight nor
    too loose at 12 points over the middle of the visible half."""
    rng = np.random.default_rng(7)
    fits = [
        fit_circle(*arc((1.5, 0.3), 0.35, 12, 0.8, rng, SIGMA), SIGMA, FeatureSettings())
        for _ in range(2000)
    ]
    kept = [fit for fit in fits if fit is not None]
    assert len(kept) >= 0.95 * len(fits), (
        "the 0.99 residual test refuses about 1 % of true arcs and the two-line test about 2 %"
    )
    nees = [fit_nees(fit, (1.5, 0.3), 0.35) for fit in kept]
    assert 2.6 <= np.mean(nees) <= 3.4, f"mean fit NEES {np.mean(nees):.2f}, expected about 3"


def test_the_residual_jacobian_matches_central_differences():
    rng = np.random.default_rng(3)
    points, directions = arc((1.2, -0.5), 0.3, 10, 0.8, rng, SIGMA)
    params = np.array([1.21, -0.49, 0.29])
    weights = incidence_weights(params, points, directions, SIGMA, FeatureSettings().cos_floor)
    _, jacobian = circle_residuals(params, points, weights)
    eps = 1e-6
    numeric = np.stack(
        [
            (
                circle_residuals(params + eps * unit, points, weights)[0]
                - circle_residuals(params - eps * unit, points, weights)[0]
            )
            / (2 * eps)
            for unit in np.eye(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-6, atol=1e-8)


def test_the_weights_follow_the_incidence_with_a_floor():
    """A beam hitting the surface square on is weighted sigma; one at 60 degrees, sigma cos 60;
    a grazing one, sigma times the floor."""
    params = np.array([2.0, 0.0, 1.0])
    points = np.array([[1.0, 0.0], [2.0 - math.cos(math.pi / 3), math.sin(math.pi / 3)]])
    directions = np.array([[1.0, 0.0], [1.0, 0.0]])
    weights = incidence_weights(params, points, directions, SIGMA, 0.3)
    np.testing.assert_allclose(weights, [SIGMA, SIGMA * 0.5])
    grazing = incidence_weights(params, np.array([[2.0, 1.0]]), np.array([[1.0, 0.0]]), SIGMA, 0.3)
    np.testing.assert_allclose(grazing, [SIGMA * 0.3])


def test_the_covariance_is_the_inverse_of_the_weighted_normal_matrix():
    rng = np.random.default_rng(11)
    points, directions = arc((0.9, 1.1), 0.4, 15, 0.8, rng, SIGMA)
    fit = fit_circle(points, directions, SIGMA, FeatureSettings())
    weights = incidence_weights(fit.z, points, directions, SIGMA, FeatureSettings().cos_floor)
    residuals, jacobian = circle_residuals(fit.z, points, weights)
    np.testing.assert_allclose(fit.covariance, np.linalg.inv(jacobian.T @ jacobian), rtol=1e-9)
    assert fit.ssr == pytest.approx(float(residuals @ residuals), rel=1e-9)


@pytest.mark.parametrize(
    "points",
    [
        np.array([[1.0, y] for y in np.linspace(-0.3, 0.3, 10)]),  # collinear
        np.full((10, 2), 1.0),  # one point repeated
        np.array([[1.0, 0.0]] * 4 + [[np.nan, 0.0]] + [[1.0, 0.1]] * 5),  # not finite
    ],
    ids=["collinear", "repeated", "nan"],
)
def test_a_degenerate_segment_is_refused_without_raising(points):
    directions = points / np.linalg.norm(np.nan_to_num(points, nan=1.0), axis=1, keepdims=True)
    assert fit_circle(points, directions, SIGMA, FeatureSettings()) is None


def test_a_fit_less_certain_than_max_fit_std_is_refused():
    rng = np.random.default_rng(5)
    points, directions = arc((2.0, 0.0), 0.3, 10, 0.8, rng, SIGMA)
    assert fit_circle(points, directions, SIGMA, FeatureSettings()) is not None
    strict = FeatureSettings(max_fit_std=1e-4)
    assert fit_circle(points, directions, SIGMA, strict) is None, "sigma above the bound kept"


# Segmentation ------------------------------------------------------------------------------------


def test_segments_break_where_neighbouring_points_jump():
    ranges = np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    angles = np.radians(np.arange(6.0))
    points = ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    segments = segment_scan(points, ranges, np.ones(6, bool), math.radians(1.0), FeatureSettings())
    assert [s.tolist() for s in segments] == [[0, 1, 2], [3, 4, 5]]


def test_an_unusable_beam_splits_a_segment():
    ranges = np.ones(6)
    usable = np.array([True, True, False, True, True, True])
    angles = np.radians(np.arange(6.0))
    points = ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    segments = segment_scan(points, ranges, usable, math.radians(1.0), FeatureSettings())
    assert [s.tolist() for s in segments] == [[0, 1], [3, 4, 5]]


def test_a_full_circle_scan_joins_its_last_segment_to_its_first():
    count = 360
    angles = 2 * math.pi * np.arange(count) / count
    ranges = np.full(count, 3.0)
    ranges[5:300] = 1.0  # a near patch in the middle; the far wall wraps round index 0
    points = ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    segments = segment_scan(
        points, ranges, np.ones(count, bool), 2 * math.pi / count, FeatureSettings(), wrap=True
    )
    assert len(segments) == 2
    joined = next(s for s in segments if 0 in s)
    assert joined.tolist() == [*range(300, 360), *range(0, 5)], "wrapped segment in beam order"


def test_a_scan_with_nothing_usable_has_no_segment():
    segments = segment_scan(
        np.zeros((4, 2)), np.zeros(4), np.zeros(4, bool), 0.1, FeatureSettings(), wrap=True
    )
    assert segments == []


def test_scan_points_place_readings_along_their_beams():
    sensor = lidar(num_beams=4)
    points, usable, directions = scan_points(sensor, np.array([1.0, 2.0, sensor.max_range, np.nan]))
    np.testing.assert_allclose(points[:2], [[1.0, 0.0], [0.0, 2.0]], atol=1e-12)
    assert usable.tolist() == [True, True, False, False]
    np.testing.assert_allclose(directions[1], [0.0, 1.0], atol=1e-12)


# Whole scans -------------------------------------------------------------------------------------


def test_two_circles_are_found_where_they_are():
    arena = Arena(6.0, 6.0, circle_centers=[[4.0, 3.0], [2.0, 4.5]], circle_radii=[0.4, 0.3])
    sensor = lidar(num_beams=360)
    pose = np.array([2.0, 2.5, 0.4])
    readings = scan_of(arena, sensor, pose, np.random.default_rng(1), SIGMA)
    features = extract_circles(sensor, readings)
    assert len(features) == 2, f"expected the two circles, got {len(features)} features"
    c, s = math.cos(pose[2]), math.sin(pose[2])
    to_body = np.array([[c, s], [-s, c]])
    for centre, radius in zip(arena.circle_centers, arena.circle_radii, strict=True):
        truth = np.array([*(to_body @ (centre - pose[:2])), radius])
        nees = [float((f.z - truth) @ np.linalg.solve(f.covariance, f.z - truth)) for f in features]
        assert min(nees) < 14.2, "each circle within chi-square(3) 0.997 of a feature"
    assert all(len(f.beams) >= FeatureSettings().min_points for f in features)


def test_a_circle_across_the_scan_start_is_one_feature():
    arena = Arena(6.0, 6.0, circle_centers=[[4.0, 3.0]], circle_radii=[0.4])
    sensor = lidar(num_beams=360)
    readings = scan_of(arena, sensor, [2.0, 3.0, 0.0])  # dead ahead: beams 355..359 and 0..5
    features = extract_circles(sensor, readings)
    assert len(features) == 1
    beams = set(features[0].beams.tolist())
    assert 0 in beams and 359 in beams
    np.testing.assert_allclose(features[0].z, [2.0, 0.0, 0.4], atol=1e-6)


def test_walls_alone_give_no_feature():
    sensor = lidar(num_beams=360)
    readings = scan_of(Arena(3.0, 3.0), sensor, [1.2, 1.4, 0.3], np.random.default_rng(2), SIGMA)
    assert extract_circles(sensor, readings) == []


def corner_features(settings, scans=20):
    """Features near a small box's corner over seeded scans: a 0.2 m box seen across its corner
    from 2 m, about four beams on each face, the shape a short arc can mimic."""
    arena = Arena(6.0, 6.0, boxes=[[3.0, 3.0, 3.2, 3.2]])
    sensor = lidar(num_beams=360)
    rng = np.random.default_rng(4)
    found = 0
    for _ in range(scans):
        readings = scan_of(arena, sensor, [1.6, 1.6, 0.0], rng, SIGMA)
        found += sum(np.hypot(*f.z[:2]) < 2.6 for f in extract_circles(sensor, readings, settings))
    return found


def test_a_box_corner_seen_diagonally_is_not_a_circle():
    assert corner_features(FeatureSettings()) == 0, "a box corner was taken for a circle"


def test_two_faces_meeting_at_a_corner_are_not_a_circle():
    """Two straight faces meeting 2 m ahead, each 0.2 m long and turned 60 degrees from the line
    of sight, sampled noise-free at 9 points: a 0.22 m circle passes the residual test, the
    radius range and the sigma bound, so only the two-line test can refuse it."""
    turn = math.radians(60.0)
    steps = np.linspace(0.0, 0.2, 5)
    corner = np.array([2.0, 0.0])
    points = np.vstack(
        [
            corner + steps[::-1, None] * [math.cos(turn), -math.sin(turn)],
            corner + steps[1:, None] * [math.cos(turn), math.sin(turn)],
        ]
    )
    directions = points / np.linalg.norm(points, axis=1, keepdims=True)
    without_test = fit_circle(points, directions, SIGMA, FeatureSettings(corner_penalty=1e9))
    assert without_test is not None, "the case must otherwise pass as a circle"
    assert fit_circle(points, directions, SIGMA, FeatureSettings()) is None


def test_a_circle_with_too_few_beams_is_dropped():
    settings = FeatureSettings()
    rng = np.random.default_rng(8)
    points, directions = arc((2.0, 0.0), 0.3, settings.min_points - 1, 0.8, rng, SIGMA)
    assert fit_circle(points, directions, SIGMA, settings) is None


def test_hc_sr04_scans_never_give_a_feature():
    """Five single readings 30 degrees apart cannot carry a circle: no layout or pose does."""
    sensor = hc_sr04()
    layout = generate_layout(np.random.default_rng(3))
    rng = np.random.default_rng(9)
    for pose in [layout.start + [dx, dy, dt] for dx, dy, dt in rng.uniform(-0.05, 0.05, (50, 3))]:
        assert extract_circles(sensor, scan_of(layout.arena, sensor, pose, rng, SIGMA)) == []


def test_a_concave_arc_from_inside_a_circle_is_refused():
    """From inside a circle the sensor sees its far side, curving away: not an obstacle's face."""
    arena = Arena(6.0, 6.0, circle_centers=[[3.1, 3.0]], circle_radii=[0.6])
    sensor = lidar(num_beams=360)
    assert extract_circles(sensor, scan_of(arena, sensor, [3.0, 3.0, 0.0])) == []


def test_a_radius_outside_the_range_is_refused():
    arena = Arena(8.0, 8.0, circle_centers=[[5.5, 4.0]], circle_radii=[1.0])
    sensor = lidar(num_beams=360)
    readings = scan_of(arena, sensor, [3.0, 4.0, 0.0])
    assert extract_circles(sensor, readings) == []
    wide = FeatureSettings(radius_range=(0.1, 1.2))
    assert len(extract_circles(sensor, readings, wide)) == 1


def test_an_arc_with_two_outliers_fails_the_residual_test():
    rng = np.random.default_rng(12)
    settings = FeatureSettings(corner_penalty=1e9)  # only the residual test may refuse it
    points, directions = arc((1.5, 0.0), 0.35, 14, 0.8, rng, SIGMA)
    assert fit_circle(points, directions, SIGMA, settings) is not None
    bumped = points.copy()
    bumped[[4, 9]] += 6 * SIGMA * directions[[4, 9]]  # two readings 6 sigma long
    assert fit_circle(bumped, directions, SIGMA, settings) is None, "outliers passed the test"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_points": 3},
        {"radius_range": (0.5, 0.2)},
        {"radius_range": (0.0, 0.5)},
        {"fit_quantile": 1.0},
        {"cos_floor": 0.0},
        {"cos_floor": 1.5},
        {"breakpoint_gap": -0.01},
        {"breakpoint_factor": -1.0},
        {"corner_penalty": -1.0},
        {"max_iterations": 0},
        {"max_fit_std": 0.0},
    ],
)
def test_feature_settings_refuse_values_out_of_range(kwargs):
    with pytest.raises(ValueError):
        FeatureSettings(**kwargs)


def test_extract_circles_checks_the_scan_length():
    with pytest.raises(ValueError):
        extract_circles(lidar(num_beams=8), np.ones(7))


def test_the_residual_test_has_n_minus_3_degrees_of_freedom():
    """Ten readings pushed 1.4 cm along their beams in a fixed +-pattern leave a weighted sum of
    squares (19.4) between the 0.99 quantiles of chi-square with 7 and 8 degrees of freedom
    (18.5 and 20.1): three parameters were fitted, so the segment is refused; one degree of freedom
    more would keep it. The two-line test is switched off so only the residual test decides."""
    from rl_navigation.estimation.ekf import chi2_quantile

    count = 10
    points, directions = arc((1.5, 0.2), 0.35, count, 0.8)
    pattern = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    pushed = points + 0.014 * pattern[:, None] * directions
    loose = FeatureSettings(corner_penalty=1e9, fit_quantile=0.999999)
    ssr = fit_circle(pushed, directions, SIGMA, loose).ssr
    assert chi2_quantile(0.99, count - 3) < ssr <= chi2_quantile(0.99, count - 2), "case moved"
    assert fit_circle(pushed, directions, SIGMA, FeatureSettings(corner_penalty=1e9)) is None


def test_extract_circles_refuses_a_noise_free_sensor():
    """The weights are noise_std times a cosine, so zero noise has no fit covariance."""
    with pytest.raises(ValueError):
        extract_circles(lidar(num_beams=360, noise_std=0.0), np.full(360, 2.0))


def test_the_breakpoint_scales_with_the_nearer_reading():
    """Beams 1 degree apart reading 1.0 m and 1.24 m are 0.241 m apart. With a factor of 10 the
    limit at the nearer range is 0.05 + 10 (pi/180) 1.0 = 0.225 m, so they split; at the farther
    range it would be 0.266 m, which would join them."""
    spacing = math.radians(1.0)
    ranges = np.array([1.0, 1.24])
    angles = np.array([0.0, spacing])
    points = ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    settings = FeatureSettings(breakpoint_factor=10.0)
    gap = float(np.hypot(*(points[1] - points[0])))
    near = settings.breakpoint_gap + settings.breakpoint_factor * spacing * ranges.min()
    far = settings.breakpoint_gap + settings.breakpoint_factor * spacing * ranges.max()
    assert near < gap <= far, "case moved"
    segments = segment_scan(points, ranges, np.ones(2, bool), spacing, settings)
    assert [s.tolist() for s in segments] == [[0], [1]]
