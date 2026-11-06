"""Seeded localization trials against the simulator's ground truth, and their summary."""

import math

import numpy as np
import pytest

from rl_navigation.estimation.evaluation import Trial, average_nees, run_trial, summarize
from rl_navigation.sim import LayoutParams, WheelOdometry, hc_sr04, lidar

# Two-sided 95 % bounds of chi-square(3 R) / R for R = 30 runs (Wilson-Hilferty): the per-step
# band of the average NEES over 30 independent runs. The tests apply it to that average taken
# over 100 steps as well, which varies far less than one step's, so the band is loose: it catches
# only a large miscalibration of the filter's noise (a factor of about 4), not a small one.
ANEES_BOUNDS_30_RUNS = (2.19, 3.94)
EMPTY_ARENA = LayoutParams(width=4.0, height=4.0, num_obstacles=0)


def _trial(estimates, truth, dead_reckoning, covariance=None):
    estimates, truth = np.asarray(estimates, float), np.asarray(truth, float)
    covariance = (
        np.broadcast_to(np.eye(3), (len(truth), 3, 3)) if covariance is None else covariance
    )
    return Trial(
        true_poses=truth,
        estimates=estimates,
        covariances=np.asarray(covariance, float),
        dead_reckoning=np.asarray(dead_reckoning, float),
        beams_used=np.array([5, 3]),
        collided=False,
    )


def test_summary_by_hand():
    # Errors (3, 4, 0) then (0, 0, 0.1) under P = diag(1, 4, 0.01): NEES 9 + 16/4 = 13, then 1.
    truth = np.zeros((2, 3))
    trial = _trial(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.1]],
        truth,
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        covariance=np.broadcast_to(np.diag([1.0, 4.0, 0.01]), (2, 3, 3)),
    )
    summary = summarize([trial])
    expected = {
        "position_rmse": math.sqrt(12.5),
        "heading_rmse": math.sqrt(0.005),
        "final_position_error": 0.0,
        "max_position_error": 5.0,
        "mean_nees": 7.0,
        "median_nees": 7.0,
        "nees_below_upper_95": 0.5,
        "beams_used": 4.0,
        "lost": 1.0,
        "steps": 2.0,
        "collisions": 0.0,
        "dead_reckoning_position_rmse": math.sqrt(0.5),
        "dead_reckoning_heading_rmse": 0.0,
        "dead_reckoning_final_position_error": 0.0,
        "dead_reckoning_max_position_error": 1.0,
        "dead_reckoning_lost": 1.0,
    }
    assert summary.keys() == expected.keys()
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_heading_errors_wrap():
    truth = [[0.0, 0.0, math.pi - 0.01]] * 2
    trial = _trial([[0.0, 0.0, -math.pi + 0.01]] * 2, truth, truth)
    assert summarize([trial])["heading_rmse"] == pytest.approx(0.02, rel=1e-9), (
        "An estimate 0.02 rad across +-pi is 0.02 rad off, not 2 pi - 0.02."
    )


def test_average_nees_is_per_step_over_runs_cut_to_the_shortest():
    short = _trial([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    long = _trial([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]], np.zeros((2, 3)), np.zeros((2, 3)))
    np.testing.assert_allclose(average_nees([short, long]), [5.0], rtol=1e-12)


def test_a_trial_is_reproducible_from_its_seed():
    first = run_trial(4, hc_sr04(), steps=30)
    again = run_trial(4, hc_sr04(), steps=30)
    other = run_trial(5, hc_sr04(), steps=30)
    np.testing.assert_array_equal(first.true_poses, again.true_poses)
    np.testing.assert_array_equal(first.estimates, again.estimates)
    np.testing.assert_array_equal(first.dead_reckoning, again.dead_reckoning)
    assert not np.array_equal(first.true_poses, other.true_poses), "Seeds should differ."


@pytest.mark.parametrize("make_sensor", [hc_sr04, lidar])
@pytest.mark.parametrize("dropout_prob", [None, 0.3])
def test_the_filter_beats_dead_reckoning(make_sensor, dropout_prob):
    """Each preset, with its own dropout rate and with 30 %: the pose error is a fraction of
    dead reckoning's on the same encoder readings."""
    summary = summarize(
        [run_trial(seed, make_sensor(), steps=150, dropout_prob=dropout_prob) for seed in (1, 2)]
    )
    assert summary["position_rmse"] < 0.5 * summary["dead_reckoning_position_rmse"], summary
    assert summary["steps"] == 150 and summary["collisions"] == 0, summary


def _empty_arena_anees(filter_odometry=None):
    trials = [
        run_trial(
            seed,
            lidar(),
            steps=100,
            layout_params=EMPTY_ARENA,
            filter_odometry=filter_odometry,
        )
        for seed in range(1, 31)
    ]
    return float(average_nees(trials).mean())


def test_the_filter_is_consistent_in_an_empty_arena():
    """The ANEES over 30 seeded runs, averaged over 100 steps, lies in the chi-square band."""
    anees = _empty_arena_anees()
    low, high = ANEES_BOUNDS_30_RUNS
    assert low <= anees <= high, f"ANEES {anees:.2f} should lie in [{low}, {high}]."


def test_a_filter_told_the_encoders_are_four_times_quieter_is_overconfident():
    """The consistency test can fail. Measured on these seeds (2026-11-06): the matched filter
    3.05; told k / 2, 3.84, still inside the band; told k / 4, 6.66, above it."""
    anees = _empty_arena_anees(WheelOdometry(noise_per_metre=2.5e-4 / 4))
    assert anees > ANEES_BOUNDS_30_RUNS[1], f"ANEES {anees:.2f} should exceed the band."


def test_a_trial_without_steps_is_refused_by_the_summaries():
    """A run that collided on its first step has no rows: averaging over it would be NaN or an
    error deep in NumPy, so the summaries name the problem instead."""
    empty = Trial(
        true_poses=np.zeros((0, 3)),
        estimates=np.zeros((0, 3)),
        covariances=np.zeros((0, 3, 3)),
        dead_reckoning=np.zeros((0, 3)),
        beams_used=np.zeros(0, dtype=np.int_),
        collided=True,
    )
    full = _trial([[0.0, 0.0, 0.0]] * 2, np.zeros((2, 3)), np.zeros((2, 3)))
    for summary in (summarize, average_nees):
        with pytest.raises(ValueError, match="no steps"):
            summary([full, empty])
        with pytest.raises(ValueError, match="no trials"):
            summary([])
