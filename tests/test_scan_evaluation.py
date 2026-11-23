"""The scan-matching desk tool (``SCAN-1``, ``SLAM-4``, ``SLAM-5``): the frozen definitions, the
named perturbation stream, the denominators, and the map scored from aligned poses."""

import dataclasses
import math

import numpy as np
import pytest

from rl_navigation.estimation.evaluation import Trial
from rl_navigation.estimation.runs import record_run
from rl_navigation.mapping import LIDAR_LOG_ODDS
from rl_navigation.mapping.evaluation import build_map, score_map
from rl_navigation.sim import Arena, lidar
from rl_navigation.slam import scan_evaluation
from rl_navigation.slam.scan_evaluation import (
    SCAN1_CORRECT,
    SCAN1_EVERY,
    SCAN1_PERTURBATION,
    SCAN1_PRIOR_VARIANCE,
    SCAN1_START,
    SCAN1_STREAM,
    Scan1Trials,
    ScanSlamRun,
    arena_frame,
    scan1_trials,
    scan_slam_trial,
    score_scan_slam,
    summarize_scan1,
    summarize_scan_slam,
)

SENSOR = lidar(num_beams=360)


def test_the_scan1_definitions_are_frozen():
    """Registered gates read these; changing one after registration needs the maintainer."""
    assert (SCAN1_START, SCAN1_EVERY) == (100, 25)
    assert SCAN1_PERTURBATION == (0.03, 0.03, 0.03)
    assert SCAN1_PRIOR_VARIANCE == 100.0
    assert SCAN1_CORRECT == 0.05
    assert SCAN1_STREAM == 0x5CA1
    assert set(scan_evaluation.SCAN1_ROWS) == {"a", "basin", "64"}
    assert set(scan_evaluation.SLAM4_ROWS) == {"a", "b", "c", "d", "floor_5mm"}


def _sensor_figures(sensor):
    return sensor.num_readings, sensor.noise_std, sensor.dropout_prob


@pytest.mark.parametrize(
    ("table", "name", "beams", "noise", "dropout", "perturbation", "floor"),
    [
        (scan_evaluation.SCAN1_ROWS, "a", 360, 0.01, None, (0.03, 0.03, 0.03), None),
        (scan_evaluation.SCAN1_ROWS, "basin", 360, 0.01, None, (0.1, 0.1, 0.1), None),
        (scan_evaluation.SCAN1_ROWS, "64", 64, 0.01, None, (0.03, 0.03, 0.03), None),
        (scan_evaluation.SLAM4_ROWS, "a", 360, 0.01, None, (0.03, 0.03, 0.03), None),
        (scan_evaluation.SLAM4_ROWS, "b", 360, 0.01, 0.3, (0.03, 0.03, 0.03), None),
        (scan_evaluation.SLAM4_ROWS, "c", 64, 0.01, None, (0.03, 0.03, 0.03), None),
        (scan_evaluation.SLAM4_ROWS, "d", 360, 0.02, None, (0.03, 0.03, 0.03), None),
        (
            scan_evaluation.SLAM4_ROWS,
            "floor_5mm",
            360,
            0.01,
            None,
            (0.03, 0.03, 0.03),
            (0.005, 0.005, 0.005),
        ),
    ],
)
def test_each_frozen_row_is_its_registered_sensor_dropout_and_floor(
    table, name, beams, noise, dropout, perturbation, floor
):
    row = table[name]
    assert _sensor_figures(row.sensor()) == (beams, noise, 0.0)
    assert row.dropout_prob == dropout
    assert row.perturbation == perturbation
    assert row.pose_noise_floor == floor


def test_arena_frame_grows_the_arena_by_the_margin():
    shape, origin = arena_frame(Arena(5.0, 5.0), 0.05, 1.0)
    assert shape == (140, 140)
    assert origin == (-1.0, -1.0)
    shape, origin = arena_frame(Arena(4.0, 3.0), 0.05, 0.5)
    assert shape == (80, 100)
    assert origin == (-0.5, -0.5)


@pytest.fixture(scope="module")
def scan1():
    return scan1_trials(1, SENSOR, steps=160)


def test_scan1_matches_every_25th_scan_from_step_100(scan1):
    np.testing.assert_array_equal(scan1.steps, [100, 125, 150])


def test_scan1_perturbations_come_from_the_named_stream(scan1):
    rng = np.random.default_rng(np.random.SeedSequence([1, SCAN1_STREAM]))
    draws = rng.standard_normal((3, 3)) * np.array(SCAN1_PERTURBATION)
    run = record_run(1, SENSOR, steps=160)
    offsets = scan1.priors - run.true_poses[scan1.steps]
    offsets[:, 2] = (offsets[:, 2] + math.pi) % (2 * math.pi) - math.pi
    np.testing.assert_allclose(offsets, draws, atol=1e-12)
    again = scan1_trials(1, SENSOR, steps=160)
    np.testing.assert_array_equal(again.errors, scan1.errors)


def test_scan1_on_a_true_pose_map_is_centimetre_accurate(scan1):
    position = np.hypot(scan1.errors[:, 0], scan1.errors[:, 1])
    assert (position < 0.03).all()
    assert scan1.accepted.all()


def _scan1(nees, errors, terminations, accepted, seed=1):
    count = len(nees)
    return Scan1Trials(
        seed=seed,
        steps=np.arange(count),
        priors=np.zeros((count, 3)),
        errors=np.asarray(errors, dtype=np.float64),
        nees=np.asarray(nees, dtype=np.float64),
        terminations=list(terminations),
        accepted=np.asarray(accepted, dtype=bool),
        inlier_share=np.full(count, 0.9),
        residual_rms=np.full(count, 0.01),
        seconds=np.full(count, 0.002),
    )


def test_summarize_scan1_counts_every_trial_in_the_nees():
    """A rejected or non-converged trial is not dropped: NEES is over every trial; converged and
    correct are separate shares."""
    trials = [
        _scan1(
            [2.0, 4.0, 30.0],
            [[0.001, 0.0, 0.0], [0.0, 0.004, 0.001], [0.06, 0.0, 0.0]],
            ["converged", "converged", "capped"],
            [True, True, False],
        ),
        _scan1([3.0], [[0.0, 0.002, 0.0]], ["stalled"], [True], seed=2),
    ]
    summary = summarize_scan1(trials)
    assert summary["trials"] == 4.0
    assert summary["nees_mean"] == pytest.approx((2.0 + 4.0 + 30.0 + 3.0) / 4)
    assert summary["correct_share"] == pytest.approx(0.75)
    assert summary["converged_share"] == pytest.approx(0.5)
    assert summary["capped_share"] == pytest.approx(0.25)
    assert summary["accepted_share"] == pytest.approx(0.75)
    assert summary["position_error_median"] == pytest.approx(0.003)


def test_the_residual_medians_skip_matches_without_inliers():
    """A match with no inliers has no RMS (NaN); the medians are over the matches that have one."""
    trial = _scan1([2.0, 3.0, 4.0], np.zeros((3, 3)), ["converged"] * 3, [True, True, False])
    trial = dataclasses.replace(trial, residual_rms=np.array([0.01, math.nan, 0.03]))
    assert summarize_scan1([trial])["residual_rms_median"] == pytest.approx(0.02)


def test_summarize_scan1_refuses_no_trials():
    with pytest.raises(ValueError, match="trials"):
        summarize_scan1([])


@pytest.fixture(scope="module")
def short_run():
    return scan_slam_trial(1, SENSOR, steps=60)


def test_a_short_scan_matching_run_tracks_the_robot(short_run):
    score = score_scan_slam(short_run)
    assert score["ate"] < 0.02
    assert score["accepted_share"] > 0.9
    assert short_run.readings.shape == (60, 360)
    assert len(short_run.updates) == 60 == len(short_run.update_seconds)


def test_a_runs_residual_median_skips_matches_without_inliers(short_run):
    updates = list(short_run.updates)
    first = next(k for k, u in enumerate(updates) if u.match is not None)
    updates[first] = dataclasses.replace(
        updates[first], match=dataclasses.replace(updates[first].match, residual_rms=math.nan)
    )
    run = dataclasses.replace(short_run, updates=updates)
    finite = [u.match.residual_rms for u in updates if u.match is not None][1:]
    score = score_scan_slam(run)
    assert score["residual_rms_median"] == pytest.approx(float(np.median(finite)))
    assert math.isfinite(summarize_scan_slam([run])["residual_rms_median"])


def test_the_map_is_scored_from_the_aligned_estimates():
    """A trajectory moved rigidly (rotated 0.2 rad about the origin and shifted) aligns back
    exactly, so its map scores as the true poses' map does."""
    recorded = record_run(2, SENSOR, steps=80)
    angle = 0.2
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    estimates = recorded.true_poses.copy()
    estimates[:, :2] = recorded.true_poses[:, :2] @ rotation.T + np.array([0.3, -0.2])
    estimates[:, 2] = (estimates[:, 2] + angle + math.pi) % (2 * math.pi) - math.pi
    trial = Trial(
        true_poses=recorded.true_poses,
        estimates=estimates,
        covariances=np.tile(1e-4 * np.eye(3), (80, 1, 1)),
        dead_reckoning=recorded.dead_reckoning,
        beams_used=np.zeros(80, dtype=np.int_),
        collided=False,
    )
    run = ScanSlamRun(
        seed=2,
        trial=trial,
        arena=recorded.arena,
        sensor=SENSOR,
        readings=recorded.readings,
        updates=[],
        update_seconds=np.zeros(80),
    )
    score = score_scan_slam(run)
    expected = score_map(build_map(recorded, LIDAR_LOG_ODDS), recorded.arena, recorded.true_poses)
    assert score["ate"] == pytest.approx(0.0, abs=1e-9)
    for key in ("occupied_precision", "cell_accuracy", "visible_surface_recall"):
        assert score[f"map_{key}"] == pytest.approx(expected[key], abs=1e-12)


def test_summarize_scan_slam_reports_the_gated_figures(short_run):
    summary = summarize_scan_slam([short_run])
    for key in (
        "ate_mean",
        "ate_median",
        "aligned_failures",
        "ate_ratio_to_dead_reckoning",
        "map_occupied_precision",
        "map_cell_accuracy",
        "map_visible_surface_recall",
        "accepted_share",
        "capped_share",
        "aligned_nees_mean",
        "update_ms_median",
        "update_ms_p99",
        "runs",
    ):
        assert key in summary, key
    assert summary["runs"] == 1.0
    assert summary["ate_ratio_to_dead_reckoning"] == pytest.approx(
        summary["ate_mean"] / summary["dead_reckoning_ate_mean"]
    )


def test_summarize_scan_slam_refuses_no_runs():
    with pytest.raises(ValueError, match="runs"):
        summarize_scan_slam([])
