"""Seeded localization trials against the simulator's ground truth, and their summary."""

import math

import numpy as np
import pytest

from rl_navigation.estimation import EkfLocalizer, MclSettings, SensorMap
from rl_navigation.estimation.evaluation import (
    Trial,
    average_nees,
    ekf,
    localization_steps,
    localization_success,
    mcl,
    run_trial,
    summarize,
    summarize_localization,
)
from rl_navigation.sim import Arena, LayoutParams, WheelOdometry, hc_sr04, lidar

# Two-sided 95 % bounds of chi-square(3 R) / R for R = 30 runs (Wilson-Hilferty): the per-step
# band of the average NEES over 30 independent runs. The tests apply it to that average taken
# over 100 steps as well, which varies far less than one step's, so the band is loose: it catches
# only a large miscalibration of the filter's noise (a factor of about 4), not a small one.
ANEES_BOUNDS_30_RUNS = (2.19, 3.94)
EMPTY_ARENA = LayoutParams(width=4.0, height=4.0, num_obstacles=0)
# A small room with three small obstacles, where 300 particles are enough to localize globally
# within 8 s.
SMALL_ROOM = LayoutParams(
    width=2.5,
    height=2.0,
    num_obstacles=3,
    box_size=(0.3, 0.6),
    circle_radius=(0.2, 0.3),
    min_start_goal_distance=1.0,
)
FEW_PARTICLES = MclSettings(num_particles=300)


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


def test_an_ekf_trial_is_pinned():
    """Recorded on develop 2f4a780 (2026-11-06), before MCL split the seed into more streams: a
    change to the streams, the controller or the EKF that moves EST-1 shows here first."""
    np.testing.assert_allclose(
        run_trial(4, hc_sr04(), steps=30).estimates[-1],
        [4.134178821496811, 2.3493711089355878, -0.36424679143710614],
        rtol=1e-12,
    )


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


# MCL in the trials ------------------------------------------------------------------------------


def test_the_default_localizer_is_the_ekf():
    default = run_trial(4, hc_sr04(), steps=30)
    explicit = run_trial(4, hc_sr04(), steps=30, make_localizer=ekf())
    np.testing.assert_array_equal(default.estimates, explicit.estimates)
    np.testing.assert_array_equal(default.covariances, explicit.covariances)


def test_the_ekf_factory_passes_the_reset_settings_on():
    make = ekf(reset_after=None, min_candidates=4, reset_std=(0.3, 0.3, 0.2), reset_window=10)
    localizer = make(
        range_map=SensorMap(Arena(4.0, 4.0), lidar()),
        odometry=WheelOdometry(),
        start=np.array([2.0, 2.0, 0.0]),
        covariance=0.01 * np.eye(3),
        rng=np.random.default_rng(0),
        robot_radius=0.1,
    )
    assert isinstance(localizer, EkfLocalizer)
    assert localizer.reset_after is None, "The factory should pass reset_after on."
    assert localizer.min_candidates == 4, "The factory should pass min_candidates on."
    np.testing.assert_array_equal(localizer.reset_std, [0.3, 0.3, 0.2])
    assert localizer.reset_window == 10, "The factory should pass reset_window on."


@pytest.mark.parametrize("ekf_argument", [{"iterations": 2}, {"linearity_tol": 0.01}])
def test_ekf_arguments_beside_a_factory_are_refused(ekf_argument):
    """They would be silently ignored."""
    with pytest.raises(ValueError, match="make_localizer"):
        run_trial(1, lidar(), steps=1, make_localizer=mcl(), **ekf_argument)


@pytest.mark.parametrize("kidnap_at", [0, 30, -1])
def test_a_kidnapping_outside_the_run_is_refused(kidnap_at):
    with pytest.raises(ValueError, match="kidnap_at"):
        run_trial(1, lidar(), steps=30, kidnap_at=kidnap_at)


def test_a_kidnapping_moves_the_robot_at_least_a_metre():
    """In an empty 2 x 2 m room most free poses lie within 1 m of each other, so a draw that
    ignored the distance would often land close."""
    room = LayoutParams(width=2.0, height=2.0, num_obstacles=0, min_start_goal_distance=0.5)
    for seed in range(1, 9):
        trial = run_trial(seed, lidar(), steps=3, layout_params=room, kidnap_at=1)
        assert trial.kidnap_at == 1
        jump = np.hypot(*(trial.true_poses[1, :2] - trial.true_poses[0, :2]))
        assert jump > 1.0 - 0.07, (  # less the two steps' travel, 0.03 m each
            f"Seed {seed}: the kidnapping moved the robot {jump:.2f} m; it should move at least 1 m."
        )


def _localization_trial(position_errors, heading_errors=None, kidnap_at=None):
    position_errors = np.asarray(position_errors, float)
    estimates = np.zeros((len(position_errors), 3))
    estimates[:, 0] = position_errors
    if heading_errors is not None:
        estimates[:, 2] = heading_errors
    return Trial(
        true_poses=np.zeros_like(estimates),
        estimates=estimates,
        covariances=np.broadcast_to(np.eye(3), (len(estimates), 3, 3)),
        dead_reckoning=np.zeros_like(estimates),
        beams_used=np.zeros(len(estimates), dtype=np.int_),
        collided=False,
        kidnap_at=kidnap_at,
    )


def test_localization_needs_the_whole_hold_window_within_tolerance():
    """deadline 4, hold 3: rows 4, 5 and 6 must all be within 0.2 m and 0.2 rad."""
    settled = [1.0] * 4 + [0.0] * 6
    assert localization_success(_localization_trial(settled), deadline=4, hold=3) == (True, None)
    slipped = [1.0] * 4 + [0.0, 0.25, 0.0] + [0.0] * 3
    assert localization_success(_localization_trial(slipped), deadline=4, hold=3)[0] is False
    turned = _localization_trial(settled, heading_errors=[0.0] * 5 + [0.25] + [0.0] * 4)
    assert localization_success(turned, deadline=4, hold=3)[0] is False
    late = [1.0] * 5 + [0.0] * 5  # row 4 still off
    assert localization_success(_localization_trial(late), deadline=4, hold=3)[0] is False


def test_a_run_that_ends_before_its_window_is_not_localized():
    assert localization_success(_localization_trial([0.0] * 6), deadline=4, hold=3) == (False, None)


def test_recovery_is_measured_from_the_kidnapping():
    """Kidnapped at 7, deadline 4, hold 3: rows 11 to 13."""
    recovered = [0.0] * 7 + [2.0] * 4 + [0.0] * 3
    trial = _localization_trial(recovered, kidnap_at=7)
    assert localization_success(trial, deadline=4, hold=3) == (True, True)
    lost_again = [0.0] * 7 + [2.0] * 4 + [0.0, 2.0, 0.0]
    trial = _localization_trial(lost_again, kidnap_at=7)
    assert localization_success(trial, deadline=4, hold=3) == (True, False)


def test_a_window_reaching_past_the_kidnapping_is_refused():
    trial = _localization_trial([0.0] * 14, kidnap_at=6)
    with pytest.raises(ValueError, match="kidnap_at"):
        localization_success(trial, deadline=4, hold=3)


def test_localization_summary_by_hand():
    """Deadline 4, hold 3, kidnapped at 7. The first trial is within tolerance from row 4 to the
    window's end (4 steps) and from row 11 (4 steps after the kidnapping); the third from row 2
    (2 steps) and row 9 (2 steps); the second never; the fourth repeats the first. Medians over
    the three that localized, 4, 2 and 4: 4 (their mean would be 3.33)."""
    first = _localization_trial([1.0] * 4 + [0.0] * 3 + [2.0] * 4 + [0.0] * 3, kidnap_at=7)
    never = _localization_trial([1.0] * 14, kidnap_at=7)
    early = _localization_trial([1.0] * 2 + [0.0] * 5 + [2.0] * 2 + [0.0] * 5, kidnap_at=7)
    summary = summarize_localization([first, never, early, first], deadline=4, hold=3)
    assert summary == {
        "global_share": pytest.approx(3 / 4),
        "recovered_share": pytest.approx(3 / 4),
        "median_steps_to_localize": pytest.approx(4.0),
        "median_steps_to_recover": pytest.approx(4.0),
        "trials": 4.0,
        "collisions": 0.0,
    }


def test_the_median_steps_skip_the_runs_that_never_localized():
    """Localized in 4 and 2 steps, recovered in 4 and 2, and one run never: the medians are over
    the two that did, 3.0 each, as Python floats."""
    first = _localization_trial([1.0] * 4 + [0.0] * 3 + [2.0] * 4 + [0.0] * 3, kidnap_at=7)
    never = _localization_trial([1.0] * 14, kidnap_at=7)
    early = _localization_trial([1.0] * 2 + [0.0] * 5 + [2.0] * 2 + [0.0] * 5, kidnap_at=7)
    summary = summarize_localization([first, never, early], deadline=4, hold=3)
    for key in ("median_steps_to_localize", "median_steps_to_recover"):
        assert summary[key] == 3.0, key
        assert type(summary[key]) is float, key


def test_steps_to_localize_count_from_the_last_step_out_of_tolerance():
    """Off at rows 0 and 2, within from row 3 to the window's end at row 6: 3 steps. A slip after
    the window does not count; one inside it means not localized."""
    assert localization_steps(
        _localization_trial([1.0, 0.0, 1.0] + [0.0] * 4 + [1.0]), deadline=4, hold=3
    ) == (3, None)
    assert localization_steps(
        _localization_trial([0.0] * 5 + [1.0] + [0.0] * 4), deadline=4, hold=3
    ) == (None, None)


def test_a_summary_where_nothing_localized_has_no_median():
    summary = summarize_localization(
        [_localization_trial([1.0] * 14, kidnap_at=7)], deadline=4, hold=3
    )
    assert summary["global_share"] == 0.0
    assert math.isnan(summary["median_steps_to_localize"])
    assert math.isnan(summary["median_steps_to_recover"])


def test_a_summary_without_kidnappings_reports_no_recovery():
    summary = summarize_localization([_localization_trial([0.0] * 10)], deadline=4, hold=3)
    assert summary == {
        "global_share": 1.0,
        "median_steps_to_localize": 0.0,
        "trials": 1.0,
        "collisions": 0.0,
    }


@pytest.mark.parametrize("make_sensor", [hc_sr04, lidar])
@pytest.mark.parametrize("dropout_prob", [None, 0.3])
def test_mcl_beats_dead_reckoning(make_sensor, dropout_prob):
    """Started like the EKF, around the true pose: tracking, not global localization."""
    summary = summarize(
        [
            run_trial(
                seed,
                make_sensor(),
                steps=150,
                dropout_prob=dropout_prob,
                make_localizer=mcl(FEW_PARTICLES),
            )
            for seed in (1, 2)
        ]
    )
    assert summary["position_rmse"] < 0.5 * summary["dead_reckoning_position_rmse"], summary
    assert summary["steps"] == 150 and summary["collisions"] == 0, summary


def test_mcl_localizes_globally_and_recovers_from_kidnappings():
    """Six seeded runs of a small room with 300 particles: a global start, a kidnapping at step
    100, and 8 s to localize each time. Measured 2026-11-06: localized 6 of 6 and recovered 4 of
    6 (12 and 6 of 12 over seeds 1 to 12). The rate on full layouts is TESTING.md's EST-2."""
    outcomes = [
        localization_success(
            run_trial(
                seed,
                lidar(),
                steps=200,
                layout_params=SMALL_ROOM,
                make_localizer=mcl(FEW_PARTICLES, global_start=True),
                kidnap_at=100,
            ),
            deadline=80,
            hold=20,
        )
        for seed in range(1, 7)
    ]
    localized = sum(found for found, _ in outcomes)
    recovered = sum(bool(again) for _, again in outcomes)
    assert localized >= 5, f"At least 5 of 6 runs should localize from anywhere: {outcomes}."
    assert recovered >= 3, f"At least 3 of 6 runs should recover from the kidnapping: {outcomes}."


@pytest.mark.parametrize(("deadline", "hold"), [(-1, 3), (4, 0)])
def test_a_window_out_of_range_is_refused(deadline, hold):
    with pytest.raises(ValueError, match="deadline"):
        localization_success(_localization_trial([0.0] * 10), deadline=deadline, hold=hold)


def test_a_localization_summary_needs_trials():
    with pytest.raises(ValueError, match="no trials"):
        summarize_localization([])


def test_a_room_too_small_to_kidnap_in_is_named():
    """A 1 x 1 m room keeps the robot 0.2 m from the walls: no two poses are 1 m apart."""
    tiny = LayoutParams(width=1.0, height=1.0, num_obstacles=0, min_start_goal_distance=0.1)
    with pytest.raises(ValueError, match="kidnap"):
        run_trial(1, lidar(), steps=5, layout_params=tiny, kidnap_at=2)
