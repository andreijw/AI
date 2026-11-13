"""rl_navigation.slam.evaluation: landmark SLAM runs scored against the simulator's truth."""

import math

import numpy as np
import pytest

import rl_navigation.estimation.evaluation as estimation_evaluation
import rl_navigation.slam.evaluation as slam_evaluation
from rl_navigation.estimation.evaluation import Trial, run_trial
from rl_navigation.mapping.evaluation import record_run
from rl_navigation.sim import Arena, lidar
from rl_navigation.sim.layouts import LayoutParams
from rl_navigation.slam.ekf_slam import SlamUpdate
from rl_navigation.slam.evaluation import (
    SlamRun,
    align_se2,
    aligned_trajectory_error,
    collapsed,
    loop_closures,
    oracle_labels,
    score_landmarks,
    score_run,
    slam_trial,
    summarize_slam,
)
from rl_navigation.slam.features import extract_circles

SENSOR = lidar(num_beams=360)
SMALL_EMPTY_ROOM = LayoutParams(width=2.0, height=2.0, num_obstacles=0, min_start_goal_distance=0.5)
ARENA = Arena(
    5.0,
    5.0,
    circle_centers=[[1.0, 1.0], [4.0, 1.0], [2.5, 4.0]],
    circle_radii=[0.3, 0.4, 0.25],
)


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# The trial ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", [False, True])
def test_a_slam_trial_drives_the_same_robot_as_run_trial(oracle):
    run = slam_trial(4, SENSOR, steps=40, oracle=oracle)
    reference = run_trial(4, SENSOR, steps=40)
    np.testing.assert_array_equal(run.trial.true_poses, reference.true_poses)
    np.testing.assert_array_equal(run.trial.dead_reckoning, reference.dead_reckoning)
    assert len(run.updates) == len(run.update_seconds) == 40
    assert run.landmarks.shape == (len(run.labels), 3)
    assert run.seen.shape == (len(run.arena.circle_radii),)


def test_the_oracle_labels_each_feature_with_the_circle_its_beams_hit():
    """A check of the oracle itself: the noise-free end points of a labelled feature's beams,
    seen from the true pose, lie on the circle it names."""
    recorded = record_run(2, SENSOR, steps=60)
    true_ranges = SENSOR.true_ranges(recorded.arena, recorded.true_poses)
    checked = 0
    for pose, scan, true in zip(recorded.true_poses, recorded.readings, true_ranges, strict=True):
        features = extract_circles(SENSOR, scan)
        for feature, label in zip(
            features, oracle_labels(recorded.arena, pose, features), strict=True
        ):
            if label < 0:
                continue
            angles = pose[2] + SENSOR.mount_angles[feature.beams]
            ends = pose[:2] + true[feature.beams, None] * np.column_stack(
                [np.cos(angles), np.sin(angles)]
            )
            distance = np.hypot(*(ends - recorded.arena.circle_centers[label]).T)
            on_circle = np.abs(distance - recorded.arena.circle_radii[label]) < 1e-6
            assert on_circle.mean() > 0.9
            checked += 1
    assert checked > 20


def test_the_oracle_run_labels_each_landmark_with_the_circle_it_lies_on():
    """The oracle reads the true pose of the scan being processed: each landmark of an oracle
    run lies on the circle its label names (the start is within about 5 cm)."""
    run = slam_trial(2, SENSOR, steps=60, oracle=True)
    assert len(run.labels) >= 1 and (run.labels >= 0).all()
    offsets = np.hypot(*(run.landmarks[:, :2] - run.arena.circle_centers[run.labels]).T)
    assert offsets.max() < 0.15


def test_the_oracle_labels_each_scan_from_that_scans_true_pose(monkeypatch):
    seen_poses = []

    def recording(arena, pose, features):
        seen_poses.append(np.array(pose))
        return oracle_labels(arena, pose, features)

    monkeypatch.setattr(slam_evaluation, "oracle_labels", recording)
    run = slam_trial(2, SENSOR, steps=20, oracle=True)
    np.testing.assert_array_equal(np.array(seen_poses), run.trial.true_poses)


def test_a_run_ending_at_a_collision_keeps_the_oracle_in_step(monkeypatch):
    """The wander never collides on its own, so its look-ahead is blinded: the robot drives into
    a wall, the run ends early, and the oracle's true poses still match the trial's rows."""
    monkeypatch.setattr(estimation_evaluation, "segment_hits", lambda *args: False)
    run = slam_trial(1, SENSOR, steps=300, oracle=True, layout_params=SMALL_EMPTY_ROOM)
    recorded = record_run(1, SENSOR, steps=300, layout_params=SMALL_EMPTY_ROOM)
    assert run.trial.collided and 0 < len(run.trial.true_poses) < 300
    np.testing.assert_array_equal(run.trial.true_poses, recorded.true_poses)
    assert len(run.updates) == len(run.trial.true_poses)


@pytest.mark.parametrize("oracle", [False, True])
def test_a_run_without_steps_is_refused_as_record_run_refuses_it(oracle):
    with pytest.raises(ValueError, match="no steps") as recorded:
        record_run(3, SENSOR, steps=0)
    with pytest.raises(ValueError, match="no steps") as slam:
        slam_trial(3, SENSOR, steps=0, oracle=oracle)
    assert type(slam.value) is type(recorded.value)


# Alignment ---------------------------------------------------------------------------------------


def test_a_rotated_and_shifted_trajectory_aligns_exactly():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.0, 5.0, (50, 2))
    moved = truth @ rotation(0.3).T + np.array([1.0, -2.0])
    R, t = align_se2(moved, truth)
    np.testing.assert_allclose(moved @ R.T + t, truth, atol=1e-12)
    assert aligned_trajectory_error(moved, truth) < 1e-12
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_a_reflected_trajectory_is_not_aligned_by_a_reflection():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.0, 5.0, (50, 2))
    mirrored = truth * np.array([-1.0, 1.0])
    R, _ = align_se2(mirrored, truth)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    assert aligned_trajectory_error(mirrored, truth) > 0.5


# Scoring a map -----------------------------------------------------------------------------------


def test_score_landmarks_counts_duplicates_spurious_and_missed_circles():
    """Two landmarks on circle 0 (one 2 cm off, one 4 cm off: a duplicate), one 30 cm from every
    circle (spurious), none near circle 1 (missed), and circle 2 never seen: its landmark is
    assigned, but a circle never seen counts neither as mapped nor as missed."""
    landmarks = np.array([[1.02, 1.0, 0.3], [1.0, 1.04, 0.31], [2.5, 2.5, 0.3], [2.5, 4.0, 0.25]])
    score = score_landmarks(landmarks, ARENA, np.array([True, True, False]), np.eye(2), np.zeros(2))
    assert score.assigned.tolist() == [0, 0, -1, 2]
    assert (score.spurious, score.duplicates, score.mapped, score.seen) == (1, 1, 1, 2)
    np.testing.assert_allclose(score.errors[:2], [0.02, 0.04], atol=1e-12)


def test_a_landmark_beyond_the_mapped_radius_counts_as_assigned_but_not_mapped():
    landmarks = np.array([[1.08, 1.0, 0.3]])
    score = score_landmarks(
        landmarks, ARENA, np.array([True, False, False]), np.eye(2), np.zeros(2)
    )
    assert score.assigned.tolist() == [0] and score.spurious == 0 and score.mapped == 0


def test_score_landmarks_uses_the_aligned_positions():
    """A map rotated with its trajectory scores like the true map once aligned: no spurious."""
    R = rotation(0.4)
    t = np.array([0.7, -0.3])
    true = np.column_stack([ARENA.circle_centers, ARENA.circle_radii])
    rotated = true.copy()
    rotated[:, :2] = (true[:, :2] - t) @ R  # the inverse of x -> R x + t
    seen = np.ones(3, dtype=bool)
    unaligned = score_landmarks(rotated, ARENA, seen, np.eye(2), np.zeros(2))
    aligned = score_landmarks(rotated, ARENA, seen, R, t)
    assert unaligned.spurious > 0
    assert (aligned.spurious, aligned.duplicates, aligned.mapped) == (0, 0, 3)


# Loop closures -----------------------------------------------------------------------------------


def test_a_landmark_seen_again_after_150_scans_is_a_loop_closure():
    errors = np.linspace(0.0, 1.0, 400)
    stds = np.linspace(0.1, 0.5, 400)
    events = loop_closures([(10, 0), (160, 0), (170, 1), (318, 1)], errors, stds)
    assert events.shape == (1, 4)
    np.testing.assert_allclose(events[0], [errors[159], errors[160], stds[159], stds[160]])
    assert loop_closures([(10, 0), (159, 0)], errors, stds).shape == (0, 4)


# Smoke ---------------------------------------------------------------------------------------------


def test_slam_runs_and_scores_on_two_short_seeds():
    """Not evidence of mapping well: only that the pieces join and the figures are finite."""
    runs = [slam_trial(seed, SENSOR, steps=100) for seed in (1, 2)]
    summary = summarize_slam(runs)
    for key in (
        "position_rmse",
        "ate_mean",
        "ate_median",
        "nees_below_upper_95",
        "spurious_per_run",
        "duplicates_per_run",
        "update_ms_median",
        "update_ms_p99",
    ):
        assert math.isfinite(summary[key]), key
    assert summary["runs"] == 2.0


def test_slam_beats_dead_reckoning_on_a_seed_with_circles():
    run = slam_trial(3, SENSOR, steps=300)
    summary = summarize_slam([run])
    assert len(run.landmarks) >= 1
    assert summary["position_rmse"] < summary["dead_reckoning_position_rmse"]


def test_aligned_failures_counts_the_runs_whose_ate_exceeds_the_threshold(monkeypatch):
    """Runs with ATE 0.02, 0.10 (not above) and 0.25 m: one aligned failure at the default
    0.10 m, two at 0.05 m."""
    runs = [slam_trial(seed, SENSOR, steps=20) for seed in (1, 2, 3)]
    ates = iter([0.02, 0.10, 0.25] * 2)
    monkeypatch.setattr(slam_evaluation, "aligned_trajectory_error", lambda *args: next(ates))
    assert slam_evaluation.ALIGNED_FAILURE_ATE == 0.10
    assert summarize_slam(runs)["aligned_failures"] == 1.0
    assert summarize_slam(runs, failure_ate=0.05)["aligned_failures"] == 2.0


# Hand-built runs: the figures the gates read, by hand -----------------------------------------------

TWO_CIRCLES = Arena(5.0, 3.0, circle_centers=[[1.0, 1.0], [4.0, 1.0]], circle_radii=[0.3, 0.4])
SQUARE = np.array([[0.5, 0.5, 0.0], [2.5, 0.5, 1.6], [2.5, 2.5, 3.1], [0.5, 2.5, -1.6]])


def hand_run(landmarks, blocks, first, seen, cross=None):
    """A four-step run whose estimates are the truth (alignment = identity), with two landmarks
    whose covariance blocks are ``blocks`` (and ``cross`` between them)."""
    landmarks = np.asarray(landmarks, dtype=float)
    P = np.zeros((9, 9))
    P[:3, :3] = 0.01 * np.eye(3)
    P[3:6, 3:6], P[6:9, 6:9] = blocks
    if cross is not None:
        P[3:6, 6:9] = cross
        P[6:9, 3:6] = np.asarray(cross).T
    steps = len(SQUARE)
    trial = Trial(
        true_poses=SQUARE.copy(),
        estimates=SQUARE.copy(),
        covariances=np.tile(0.01 * np.eye(3), (steps, 1, 1)),
        dead_reckoning=SQUARE.copy(),
        beams_used=np.zeros(steps, dtype=np.int_),
        collided=False,
    )
    return SlamRun(
        seed=0,
        trial=trial,
        arena=TWO_CIRCLES,
        sensor=SENSOR,
        landmarks=landmarks,
        labels=np.full(2, -1, dtype=np.intp),
        state_covariance=P,
        first_estimates=np.asarray(first, dtype=float),
        updates=[SlamUpdate(used=np.zeros(360, dtype=bool)) for _ in range(steps)],
        update_seconds=np.zeros(steps),
        seen=np.asarray(seen, dtype=bool),
        min_points=8,
    )


BLOCKS = (np.diag([4e-4, 1e-4, 2.5e-5]), np.diag([1e-4, 4e-4, 1e-4]))
LANDMARKS = [[1.03, 1.0, 0.32], [4.0, 1.04, 0.38]]
FIRST = [[1.0, 1.0, 0.3], [4.0, 1.1, 0.4]]


def test_score_run_gives_the_hand_worked_nees_and_drift():
    """Radius NEES 0.02^2 / 2.5e-5 = 16 and 0.02^2 / 1e-4 = 4 (variances unlike the y ones); absolute NEES 0.03^2 / 4e-4 = 2.25
    and 0.04^2 / 4e-4 = 4; the pair's distance |(-2.97, -0.04)| against 3 m, with variance
    u^T (P_00 + P_11) u = 5e-4 for any unit u since P_00 + P_11 = 5e-4 I; drift 0.03 and 0.06."""
    score = score_run(hand_run(LANDMARKS, BLOCKS, FIRST, [True, True]))
    assert score.ate == pytest.approx(0.0, abs=1e-12)
    assert score.landmarks.assigned.tolist() == [0, 1]
    np.testing.assert_allclose(score.radius_nees, [16.0, 4.0], rtol=1e-9)
    np.testing.assert_allclose(score.absolute_nees, [2.25, 4.0], rtol=1e-9)
    np.testing.assert_allclose(
        score.pair_nees, [(math.hypot(2.97, 0.04) - 3.0) ** 2 / 5e-4], rtol=1e-9
    )
    np.testing.assert_allclose(score.radius_errors, [0.02, -0.02], atol=1e-12)
    assert score.first_estimate_drift == pytest.approx(0.06, abs=1e-12)


def test_a_pair_without_distance_variance_is_left_out():
    """Two landmarks perfectly correlated (cross covariance equal to their blocks): their
    distance has no variance, so the pair has no NEES."""
    block = np.diag([1e-4, 1e-4, 1e-4])
    run = hand_run(LANDMARKS, (block, block), FIRST, [True, True], cross=block)
    score = score_run(run)
    assert score.pair_nees.shape == (0,)
    np.testing.assert_allclose(score.radius_nees, [4.0, 4.0], rtol=1e-9)


def test_circles_mapped_pools_mapped_over_seen_circles_across_runs():
    """Run 1 sees circle 0 only (mapped); its landmark on unseen circle 1 does not count. Run 2
    sees both and maps circle 0 only (its other landmark is 8 cm off). Pooled: 2 of 3 = 0.667,
    not the mean of the runs' shares (0.75)."""
    first = hand_run(LANDMARKS, BLOCKS, FIRST, [True, False])
    second = hand_run([[1.03, 1.0, 0.32], [4.0, 1.08, 0.38]], BLOCKS, FIRST, [True, True])
    summary = summarize_slam([first, second])
    assert summary["circles_mapped"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert summary["spurious_per_run"] == 0.0 and summary["duplicates_per_run"] == 0.0
    assert summary["radius_nees_mean"] == pytest.approx((16.0 + 4.0 + 16.0 + 4.0) / 4, rel=1e-9)
    assert summary["drift_max"] == pytest.approx(0.06, abs=1e-12)


def test_summarize_slam_refuses_no_runs():
    with pytest.raises(ValueError, match="needs runs"):
        summarize_slam([])


# Consistency over runs: collapses and the per-run NEES bounds -----------------------------------


@pytest.mark.parametrize(
    "series, expected",
    [
        ([1.0] * 10 + [30.0] * 25, True),  # rises through 20 and stays there
        ([1.0] * 10 + [30.0] + [1.0] * 30, False),  # one spike: the next 20 scans' median is low
        ([30.0] * 40, False),  # above 20 from the start: no rise
        ([1.0] * 10 + [19.9] * 30, False),  # never reaches 20
        ([5.0, 25.0] + [21.0] * 19, True),  # held for exactly the 20 scans from the rise
    ],
)
def test_a_collapse_is_a_rise_through_nees_20_that_holds_for_20_scans(series, expected):
    assert slam_evaluation.COLLAPSE_NEES == 20.0 and slam_evaluation.COLLAPSE_HOLD == 20
    assert collapsed(np.array(series)) is expected


def nees_run(nees):
    """A run whose pose NEES at each step is ``nees`` (an x error against a 0.01 I covariance),
    with no landmark."""
    nees = np.asarray(nees, dtype=float)
    steps = len(nees)
    truth = np.column_stack([np.linspace(0.5, 4.5, steps), np.full(steps, 1.5), np.zeros(steps)])
    estimates = truth.copy()
    estimates[:, 0] += np.sqrt(0.01 * nees)
    trial = Trial(
        true_poses=truth,
        estimates=estimates,
        covariances=np.tile(0.01 * np.eye(3), (steps, 1, 1)),
        dead_reckoning=truth.copy(),
        beams_used=np.zeros(steps, dtype=np.int_),
        collided=False,
    )
    return SlamRun(
        seed=0,
        trial=trial,
        arena=TWO_CIRCLES,
        sensor=SENSOR,
        landmarks=np.zeros((0, 3)),
        labels=np.zeros(0, dtype=np.intp),
        state_covariance=0.01 * np.eye(3),
        first_estimates=np.zeros((0, 3)),
        updates=[SlamUpdate(used=np.zeros(360, dtype=bool)) for _ in range(steps)],
        update_seconds=np.zeros(steps),
        seen=np.zeros(2, dtype=bool),
        min_points=8,
    )


def test_summarize_slam_counts_collapses_and_bounds_the_per_run_nees():
    """Four runs of 40 steps with mean NEES 0.25, 1.0, 3.0 and 15.625 (the last collapses at step
    10 from 1 to 20.5): one collapse; the median of the per-run means is (1 + 3) / 2 = 2; one of
    four runs below 0.5 (low_nees_runs 0.25) and one above 6 (high_nees_runs 0.25)."""
    runs = [
        nees_run([0.25] * 40),
        nees_run([1.0] * 40),
        nees_run([3.0] * 40),
        nees_run([1.0] * 10 + [20.5] * 30),
    ]
    summary = summarize_slam(runs)
    assert summary["collapses"] == 1.0
    assert summary["run_nees_mean_median"] == pytest.approx(2.0, rel=1e-9)
    assert summary["low_nees_runs"] == pytest.approx(0.25, abs=1e-12)
    assert summary["high_nees_runs"] == pytest.approx(0.25, abs=1e-12)
    assert slam_evaluation.LOW_MEAN_NEES == 0.5
