"""rl_navigation.slam.feature_evaluation: circle features scored against the simulator's truth."""

import math

import numpy as np
import pytest

from rl_navigation.sim import Arena, lidar
from rl_navigation.slam.feature_evaluation import (
    FeatureRun,
    feature_trial,
    match_features,
    radius_bias_t2,
    summarize_features,
    visible_circles,
)
from rl_navigation.slam.features import CircleFeature


def feature_at(z):
    return CircleFeature(np.asarray(z, dtype=float), np.eye(3) * 1e-4, np.arange(8), 1.0)


def test_visible_circles_counts_the_beams_that_reach_each_circle():
    """From (1, 3) facing +x: a 0.4 m circle 2 m ahead spans about 23 beams of a 1 degree lidar; a
    0.4 m circle straight behind the first is hidden; a small one 1 m to the left spans about 12."""
    arena = Arena(
        8.0,
        6.0,
        circle_centers=[[3.0, 3.0], [5.0, 3.0], [1.0, 4.0]],
        circle_radii=[0.4, 0.4, 0.1],
    )
    sensor = lidar(num_beams=360)
    pose = np.array([1.0, 3.0, 0.0])
    true = sensor.true_ranges(arena, pose)[0]
    assert visible_circles(arena, sensor, pose, true, 8).tolist() == [0, 2]
    assert visible_circles(arena, sensor, pose, true, 13).tolist() == [0]


def test_match_features_names_the_true_circle_or_minus_one():
    arena = Arena(6.0, 6.0, circle_centers=[[4.0, 3.0], [2.0, 5.0]], circle_radii=[0.3, 0.3])
    pose = np.array([2.0, 3.0, math.pi / 2])  # facing +y: circle 0 is 2 m to the right
    features = [
        feature_at([0.0, -2.0, 0.3]),
        feature_at([2.0, 0.01, 0.3]),
        feature_at([1.0, 1.0, 0.3]),
    ]
    assert match_features(arena, pose, features).tolist() == [0, 1, -1]


def test_radius_bias_t2_is_the_mean_squared_t_over_circles_with_enough_fits():
    """Circle 0: 30 errors of 2 mm +- 1 mm, mean 2 mm, sample sd 1 mm sqrt(30/29); circle 1:
    errors +-1 mm, mean 0; circle 2: 10 fits, too few to count."""
    errors = np.concatenate(
        [0.002 + 0.001 * np.tile([1.0, -1.0], 15), 0.001 * np.tile([1.0, -1.0], 15)]
    )
    errors = np.concatenate([errors, np.full(10, 0.05)])
    circles = np.repeat([0, 1, 2], [30, 30, 10])
    t0 = 0.002 / (0.001 * math.sqrt(30 / 29) / math.sqrt(30))
    assert radius_bias_t2(circles, errors, min_fits=30) == pytest.approx((t0**2 + 0.0) / 2)
    assert math.isnan(radius_bias_t2(circles[:10], errors[:10], min_fits=30))


def run(**overrides):
    fields = {
        "seed": 1,
        "scans": 10,
        "fits": 4,
        "false_fits": 1,
        "visible": 3,
        "detected": 2,
        "points": np.array([8, 12, 25]),
        "centre_errors": np.array([0.01, 0.02, 0.03]),
        "radius_errors": np.array([-0.002, 0.001, 0.003]),
        "nees": np.array([1.0, 2.0, 6.0]),
        "circles": np.array([0, 0, 1]),
        "extract_seconds": np.full(10, 0.002),
        "collided": False,
    }
    fields.update(overrides)
    return FeatureRun(**fields)


def test_summarize_features_counts_detections_and_false_fits():
    summary = summarize_features([run(), run(seed=2, fits=6, false_fits=0, visible=1, detected=1)])
    assert summary["detection_rate"] == pytest.approx(3 / 4)
    assert summary["false_fit_rate"] == pytest.approx(1 / 10)
    assert summary["fits_per_scan"] == pytest.approx(10 / 20)
    assert summary["centre_error_median"] == pytest.approx(0.02)
    assert summary["radius_error_mean"] == pytest.approx(2 * (0.002 / 3) / 2)
    assert summary["fit_nees_mean"] == pytest.approx(3.0)
    assert summary["fit_nees_mean_8_to_11"] == pytest.approx(1.0)
    assert summary["fit_nees_mean_20_up"] == pytest.approx(6.0)
    assert summary["extract_ms_median"] == pytest.approx(2.0)
    assert summary["runs"] == 2.0 and summary["collisions"] == 0.0


def test_summarize_features_refuses_no_runs():
    with pytest.raises(ValueError):
        summarize_features([])


def test_a_short_seeded_trial_scores_finite_figures():
    """Smoke: 40 steps of seed 1 with the 1 degree lidar join the pieces; not a measurement."""
    trial = feature_trial(1, lidar(num_beams=360), steps=40)
    assert trial.scans == 40
    assert trial.fits >= len(trial.nees) and trial.detected <= trial.visible
    summary = summarize_features([trial])
    assert all(np.isfinite(v) or math.isnan(v) for v in summary.values())
    assert np.isfinite(summary["false_fit_rate"])


def test_a_circle_beyond_the_maximum_range_is_not_visible():
    """Beams that end past max_range read as no echo, so a circle there cannot be detected and
    is not counted as missed."""
    arena = Arena(12.0, 6.0, circle_centers=[[6.0, 3.0]], circle_radii=[0.5])
    sensor = lidar(num_beams=360)  # max_range 4 m; the circle's face is 4.5 m away
    pose = np.array([1.0, 3.0, 0.0])
    assert visible_circles(arena, sensor, pose, sensor.true_ranges(arena, pose)[0], 8).size == 0
