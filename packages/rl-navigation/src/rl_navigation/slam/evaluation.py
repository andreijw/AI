"""Seeded landmark-SLAM runs scored against the simulator's truth (``SLAM-1``).

A desk tool: :mod:`rl_navigation.slam` does not import it. Each run is ``run_trial``'s wandering
robot with :class:`~rl_navigation.slam.ekf_slam.EkfSlam` as its filter, built by
:class:`SlamFactory`, which keeps the filter and each update's report. The known-correspondence
row hands the filter an *oracle*: each feature, placed with the *true* pose from
:func:`~rl_navigation.mapping.evaluation.record_run` (whose poses are ``run_trial``'s, bit for
bit), is labelled with the true circle within 0.15 m of it; the filter still gates those labels.

**Every map figure is computed after alignment.** The estimated trajectory is aligned to the
true one by the best rotation and translation (Umeyama 1991 without scale), and the same
transform is applied to the landmarks: SLAM cannot recover the start's error nor the heading
drift before its first sighting, so absolute error against dead reckoning is a low bar and the
aligned trajectory error (ATE) and the map figures are the SLAM figures.

- A landmark is *assigned* to its nearest true circle within :data:`MATCH_RADIUS`, otherwise it
  is *spurious*. A circle's *duplicates* are its assigned landmarks beyond the first.
- A circle is *seen* when, at some step, at least ``min_points`` noise-free beams the sensor can
  report end on it; it is *mapped* when an assigned landmark lies within :data:`MAPPED_RADIUS`.
- Consistency: the radius NEES (1 degree of freedom) and the pair-distance NEES, the distance
  between two landmarks assigned to different circles against its linearized variance (1
  degree, free of the unobservable frame); the absolute landmark position NEES includes the
  frame's uncertainty and is reported only.
- A *loop closure* is a landmark associated again after :data:`LOOP_GAP` scans unseen.

Imports NumPy, ``rl_navigation.sim``, ``rl_navigation.estimation`` (``run_trial``,
``summarize``), ``rl_navigation.mapping.evaluation`` (``record_run``) and the slam package. The
mapping desk tool now serves mapping, the features' scoring and this oracle; a fourth user should
promote ``record_run`` to a neutral module. ``SlamFactory`` is the fourth implementation of the
six-keyword factory seam; a fifth user, or a change to the keywords, should refactor the seam to
one builder.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.averages import mean_or_nan, median_or_nan
from rl_navigation.estimation.evaluation import Localizer, Trial, run_trial, summarize
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.estimation.poses import pose_difference
from rl_navigation.mapping.evaluation import record_run
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.layouts import LayoutParams
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor
from rl_navigation.slam.ekf_slam import EkfSlam, SlamSettings, SlamUpdate
from rl_navigation.slam.feature_evaluation import MATCH_RADIUS, match_features, visible_circles
from rl_navigation.slam.features import CircleFeature

# The filter's pose must stay within the arena grown by this many metres.
BOUNDS_MARGIN = 1.0

# A circle is mapped when an assigned landmark's centre lies within this many metres of it.
MAPPED_RADIUS = 0.05

# A landmark associated again after at least this many scans unseen closes a loop.
LOOP_GAP = 150

# A run whose aligned trajectory error exceeds this many metres has failed (aligned_failures): unlike
# lost, it does not count a right map in a shifted frame.
ALIGNED_FAILURE_ATE = 0.10

# A run whose mean pose NEES exceeds this is in the overconfident tail (3 is consistent).
HIGH_MEAN_NEES = 6.0


def oracle_labels(
    arena: Arena, true_pose: ArrayLike, features: Sequence[CircleFeature]
) -> NDArray[np.intp]:
    """``(F,)`` the true circle each feature, placed with the true pose, lies within
    :data:`MATCH_RADIUS` of, or -1: the known-correspondence labels."""
    return match_features(arena, true_pose, features)


class _RecordedSlam:
    """An :class:`EkfSlam` as ``run_trial`` drives it, recording each update (and its time), and
    labelling the features from the true poses when it is given them."""

    def __init__(self, slam: EkfSlam, arena: Arena, true_poses: FloatArray | None) -> None:
        self.slam = slam
        self.arena = arena
        self.true_poses = true_poses
        self.updates: list[SlamUpdate] = []
        self.seconds: list[float] = []

    @property
    def pose(self) -> FloatArray:
        return self.slam.pose

    @property
    def covariance(self) -> FloatArray:
        return self.slam.covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        self.slam.predict(wheel_travel)

    def update(self, readings: ArrayLike) -> SlamUpdate:
        associate = None
        if self.true_poses is not None:
            pose = self.true_poses[len(self.updates)]

            def associate(features: Sequence[CircleFeature]) -> Sequence[int]:
                return oracle_labels(self.arena, pose, features).tolist()

        start = time.perf_counter()
        result = self.slam.update(readings, associate)
        self.seconds.append(time.perf_counter() - start)
        self.updates.append(result)
        return result


class SlamFactory:
    """A ``run_trial`` factory of :class:`EkfSlam` at the trial's noisy start and covariance,
    bounded to the arena grown by :data:`BOUNDS_MARGIN`. It keeps the filter it built.

    Args:
        settings: The filter's settings; None uses the defaults.
        oracle_poses: ``(T, 3)`` the run's true poses, for the oracle's labels; None uses the
            maximum-likelihood association.
    """

    def __init__(
        self, settings: SlamSettings | None = None, oracle_poses: FloatArray | None = None
    ) -> None:
        self.settings = settings
        self.oracle_poses = oracle_poses
        self.made: _RecordedSlam | None = None

    def __call__(
        self,
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer:
        arena = range_map.arena
        bounds = (
            -BOUNDS_MARGIN,
            -BOUNDS_MARGIN,
            arena.width + BOUNDS_MARGIN,
            arena.height + BOUNDS_MARGIN,
        )
        slam = EkfSlam(range_map.sensor, odometry, start, covariance, self.settings, bounds=bounds)
        self.made = _RecordedSlam(slam, arena, self.oracle_poses)
        return self.made


@dataclass(frozen=True, eq=False)
class SlamRun:
    """One seeded SLAM run.

    Attributes:
        seed: The run's seed.
        trial: ``run_trial``'s record (true poses, estimates, pose covariances, dead reckoning).
        arena: The true arena.
        sensor: The lidar.
        landmarks: ``(L, 3)`` the final map.
        labels: ``(L,)`` the oracle's labels of the landmarks (-1 on the ML path).
        state_covariance: ``(3 + 3L, 3 + 3L)`` the final covariance.
        first_estimates: ``(L, 3)`` FEJ's first estimates.
        updates: Each step's :class:`SlamUpdate`.
        update_seconds: ``(T,)`` the wall time of each update, extraction included.
        seen: ``(C,)`` whether each circle was seen (see the module docstring).
        min_points: The extraction's ``min_points``, which ``seen`` used.
    """

    seed: int
    trial: Trial
    arena: Arena
    sensor: RangeSensor
    landmarks: FloatArray
    labels: NDArray[np.intp]
    state_covariance: FloatArray
    first_estimates: FloatArray
    updates: list[SlamUpdate]
    update_seconds: FloatArray
    seen: NDArray[np.bool_]
    min_points: int


def slam_trial(
    seed: int,
    sensor: RangeSensor,
    *,
    settings: SlamSettings | None = None,
    oracle: bool = False,
    steps: int = 600,
    dropout_prob: float | None = None,
    layout_params: LayoutParams | None = None,
) -> SlamRun:
    """Run the landmark SLAM on a seeded wandering run (``run_trial``'s).

    Args:
        seed: The run's seed.
        sensor: The lidar.
        settings: The filter's settings; None uses the defaults.
        oracle: Label the features from the true poses (known correspondence).
        steps: Steps to run; a collision ends the run early.
        dropout_prob: Dropout probability; None uses the sensor's own.
        layout_params: How the layout is drawn; None uses the defaults.

    Raises:
        ValueError: The run has no steps (``steps`` 0, or a collision on the first step).
    """
    settings = SlamSettings() if settings is None else settings
    poses = None
    if oracle:
        poses = record_run(
            seed, sensor, steps=steps, dropout_prob=dropout_prob, layout_params=layout_params
        ).true_poses
    factory = SlamFactory(settings, poses)
    trial = run_trial(
        seed,
        sensor,
        steps=steps,
        dropout_prob=dropout_prob,
        layout_params=layout_params,
        make_localizer=factory,
    )
    made = factory.made
    if made is None or len(trial.true_poses) == 0:
        raise ValueError(
            f"slam_trial: seed {seed} has no steps (steps={steps}, or a collision on the first "
            "step); nothing to map."
        )
    arena = made.arena
    min_points = settings.features.min_points
    seen = np.zeros(len(arena.circle_radii), dtype=bool)
    for pose, true in zip(
        trial.true_poses, sensor.true_ranges(arena, trial.true_poses), strict=True
    ):
        seen[visible_circles(arena, sensor, pose, true, min_points)] = True
    slam = made.slam
    return SlamRun(
        seed=seed,
        trial=trial,
        arena=arena,
        sensor=sensor,
        landmarks=slam.landmarks,
        labels=slam.labels,
        state_covariance=slam.state_covariance,
        first_estimates=slam.first_estimates,
        updates=made.updates,
        update_seconds=np.array(made.seconds),
        seen=seen,
        min_points=min_points,
    )


def align_se2(estimated: ArrayLike, truth: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """The rotation ``R`` and translation ``t`` minimizing ``sum |R e_i + t - g_i|^2`` (Umeyama
    1991 without scale; never a reflection).

    Args:
        estimated: ``(T, 2)`` estimated positions.
        truth: ``(T, 2)`` true positions.

    Returns:
        ``(R, t)``, ``(2, 2)`` and ``(2,)``: ``estimated @ R.T + t`` is aligned.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mean_e, mean_g = estimated.mean(axis=0), truth.mean(axis=0)
    U, _, Vt = np.linalg.svd((truth - mean_g).T @ (estimated - mean_e))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mean_g - R @ mean_e


def aligned_trajectory_error(estimated: ArrayLike, truth: ArrayLike) -> float:
    """The position RMSE after :func:`align_se2`: the ATE, in metres."""
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    R, t = align_se2(estimated, truth)
    return float(np.sqrt(np.mean(np.sum((estimated @ R.T + t - truth) ** 2, axis=1))))


class LandmarkScore(NamedTuple):
    """A map scored against the true circles, after alignment.

    Attributes:
        assigned: ``(L,)`` each landmark's circle, or -1 (spurious).
        errors: ``(L,)`` each landmark's aligned centre distance to its circle (NaN if spurious).
        spurious: Landmarks assigned to no circle.
        duplicates: Assigned landmarks beyond the first, summed over circles.
        mapped: Seen circles with an assigned landmark within :data:`MAPPED_RADIUS`.
        seen: Circles seen.
    """

    assigned: NDArray[np.intp]
    errors: FloatArray
    spurious: int
    duplicates: int
    mapped: int
    seen: int


def score_landmarks(
    landmarks: ArrayLike,
    arena: Arena,
    seen: ArrayLike,
    rotation: ArrayLike,
    translation: ArrayLike,
) -> LandmarkScore:
    """Assign each landmark, moved by ``x -> rotation x + translation``, to its nearest true
    circle within :data:`MATCH_RADIUS`, and count the map's errors (see the module docstring).

    Args:
        landmarks: ``(L, 3)`` the map.
        arena: The true arena.
        seen: ``(C,)`` whether each circle was seen.
        rotation: ``(2, 2)`` the alignment's rotation.
        translation: ``(2,)`` its translation.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).reshape(-1, 3)
    seen = np.asarray(seen, dtype=bool)
    count = len(arena.circle_radii)
    aligned = landmarks[:, :2] @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    )
    assigned = np.full(len(landmarks), -1, dtype=np.intp)
    errors = np.full(len(landmarks), np.nan)
    if count and len(landmarks):
        distances = np.hypot(*(aligned[:, None, :] - arena.circle_centers[None]).transpose(2, 0, 1))
        nearest = distances.argmin(axis=1)
        within = distances[np.arange(len(landmarks)), nearest] <= MATCH_RADIUS
        assigned = np.where(within, nearest, -1).astype(np.intp)
        errors = np.where(within, distances[np.arange(len(landmarks)), nearest], np.nan)
    per_circle = np.bincount(assigned[assigned >= 0], minlength=count)
    close = assigned[errors <= MAPPED_RADIUS]
    mapped = np.zeros(count, dtype=bool)
    mapped[close] = True
    return LandmarkScore(
        assigned=assigned,
        errors=errors,
        spurious=int((assigned < 0).sum()),
        duplicates=int(np.maximum(per_circle - 1, 0).sum()),
        mapped=int((mapped & seen).sum()),
        seen=int(seen.sum()),
    )


def loop_closures(
    events: Sequence[tuple[int, int]],
    position_errors: ArrayLike,
    position_stds: ArrayLike,
    gap: int = LOOP_GAP,
) -> FloatArray:
    """``(E, 4)`` for each association of a landmark unseen for at least ``gap`` scans: the
    position error and standard deviation on the scan before and on the scan of the event,
    ``(error_before, error_after, std_before, std_after)``.

    Args:
        events: ``(scan, landmark)`` associations in scan order.
        position_errors: ``(T,)`` the position error after each scan.
        position_stds: ``(T,)`` ``sqrt(trace P_pp)`` of the position after each scan.
        gap: Scans unseen that make a loop closure.
    """
    errors = np.asarray(position_errors, dtype=np.float64)
    stds = np.asarray(position_stds, dtype=np.float64)
    last: dict[int, int] = {}
    rows = []
    for scan, landmark in events:
        if landmark in last and scan - last[landmark] >= gap and scan >= 1:
            rows.append((errors[scan - 1], errors[scan], stds[scan - 1], stds[scan]))
        last[landmark] = scan
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


@dataclass(frozen=True, eq=False)
class RunScore:
    """One run's SLAM figures (see :func:`score_run`)."""

    ate: float
    circles: int
    landmarks: LandmarkScore
    mean_nees: float
    median_nees: float
    radius_nees: FloatArray
    pair_nees: FloatArray
    absolute_nees: FloatArray
    radius_errors: FloatArray
    first_estimate_drift: float
    loops: FloatArray


def score_run(run: SlamRun) -> RunScore:
    """A run's aligned trajectory error, its map's errors and consistency, and its loops."""
    trial = run.trial
    R, t = align_se2(trial.estimates[:, :2], trial.true_poses[:, :2])
    score = score_landmarks(run.landmarks, run.arena, run.seen, R, t)
    P = run.state_covariance
    radius_nees, radius_errors, absolute_nees = [], [], []
    for j in np.flatnonzero(score.assigned >= 0):
        circle = score.assigned[j]
        block = slice(3 + 3 * j, 6 + 3 * j)
        error = run.landmarks[j, 2] - run.arena.circle_radii[circle]
        radius_errors.append(error)
        radius_nees.append(error**2 / P[block, block][2, 2])
        offset = run.landmarks[j, :2] - run.arena.circle_centers[circle]
        absolute_nees.append(offset @ np.linalg.solve(P[block, block][:2, :2], offset))
    pair_nees = []
    assigned = np.flatnonzero(score.assigned >= 0)
    for a, i in enumerate(assigned):
        for j in assigned[a + 1 :]:
            if score.assigned[i] == score.assigned[j]:
                continue
            difference = run.landmarks[i, :2] - run.landmarks[j, :2]
            distance = float(np.hypot(*difference))
            if not distance > 0.0:
                continue  # no direction to linearize along
            gradient = np.zeros(len(P))
            gradient[3 + 3 * i : 5 + 3 * i] = difference / distance
            gradient[3 + 3 * j : 5 + 3 * j] = -difference / distance
            true_distance = np.hypot(
                *(
                    run.arena.circle_centers[score.assigned[i]]
                    - run.arena.circle_centers[score.assigned[j]]
                )
            )
            variance = float(gradient @ P @ gradient)
            if not variance > 0.0:
                continue  # a distance the covariance claims to know exactly has no NEES
            pair_nees.append((distance - true_distance) ** 2 / variance)
    error = pose_difference(trial.estimates, trial.true_poses)
    position_errors = np.hypot(error[:, 0], error[:, 1])
    stds = np.sqrt(trial.covariances[:, 0, 0] + trial.covariances[:, 1, 1])
    events = [(scan, int(j)) for scan, update in enumerate(run.updates) for j in update.landmarks]
    drift = np.hypot(*(run.landmarks[:, :2] - run.first_estimates[:, :2]).T)
    pose = summarize([trial])
    return RunScore(
        ate=aligned_trajectory_error(trial.estimates[:, :2], trial.true_poses[:, :2]),
        circles=len(run.arena.circle_radii),
        landmarks=score,
        mean_nees=pose["mean_nees"],
        median_nees=pose["median_nees"],
        radius_nees=np.array(radius_nees),
        pair_nees=np.array(pair_nees),
        absolute_nees=np.array(absolute_nees),
        radius_errors=np.array(radius_errors),
        first_estimate_drift=float(drift.max(initial=0.0)),
        loops=loop_closures(events, position_errors, stds),
    )


def summarize_slam(
    runs: Sequence[SlamRun], failure_ate: float = ALIGNED_FAILURE_ATE
) -> dict[str, float]:
    """The SLAM figures over runs.

    Returns:
        ``summarize``'s pose figures over the trials (absolute: ``position_rmse``, ``lost``,
        ``mean_nees``, ``nees_below_upper_95``, the ``dead_reckoning_`` figures, ...) and:
        ``ate_mean``, ``ate_median``, ``ate_max``, ``aligned_failures`` (runs whose ATE exceeds
        ``failure_ate``, default :data:`ALIGNED_FAILURE_ATE`), and ``ate_mean`` split into
        ``ate_mean_two_or_more_circles`` and ``ate_mean_at_most_one_circle`` (with their
        ``runs_``); ``run_nees_median_q25``, ``_q50`` and ``_q75`` (quantiles of the per-run
        median NEES) and ``high_nees_runs`` (the share of runs whose mean NEES exceeds
        :data:`HIGH_MEAN_NEES`); per run ``landmarks_per_run``, ``spurious_per_run`` and
        ``duplicates_per_run``; ``circles_mapped`` (mapped over seen circles); over assigned
        landmarks ``radius_nees_mean`` and ``_median``, ``pair_nees_mean`` and ``_median``,
        ``absolute_nees_mean`` (2 degrees of freedom, gauge included) and ``radius_error_mean``;
        ``drift_median`` and ``drift_max`` (per run, the largest distance of a landmark from its
        first estimate); ``loop_closures`` and their median ``loop_error_before``,
        ``loop_error_after``, ``loop_std_before`` and ``loop_std_after``; ``update_ms_median``
        and ``update_ms_p99`` per scan; the totals ``rejected``, ``skipped``, ``ambiguous`` and
        ``rejected_updates``; ``runs``. Means and medians of nothing are NaN.

    Raises:
        ValueError: No runs.
    """
    if not runs:
        raise ValueError("summarize_slam needs runs; got none.")
    scores = [score_run(run) for run in runs]
    ates = np.array([score.ate for score in scores])
    many = np.array([score.circles >= 2 for score in scores])
    loops = np.concatenate([score.loops for score in scores])
    seconds = np.concatenate([run.update_seconds for run in runs])
    updates = [update for run in runs for update in run.updates]
    run_medians = np.array([score.median_nees for score in scores])
    seen = sum(score.landmarks.seen for score in scores)

    def pooled(name: str) -> FloatArray:
        return np.concatenate([getattr(score, name) for score in scores])

    return {
        **summarize([run.trial for run in runs]),
        "ate_mean": float(ates.mean()),
        "ate_median": float(np.median(ates)),
        "ate_max": float(ates.max()),
        "aligned_failures": float((ates > failure_ate).sum()),
        "ate_mean_two_or_more_circles": mean_or_nan(ates[many]),
        "ate_mean_at_most_one_circle": mean_or_nan(ates[~many]),
        "runs_two_or_more_circles": float(many.sum()),
        "runs_at_most_one_circle": float((~many).sum()),
        "run_nees_median_q25": float(np.quantile(run_medians, 0.25)),
        "run_nees_median_q50": float(np.quantile(run_medians, 0.5)),
        "run_nees_median_q75": float(np.quantile(run_medians, 0.75)),
        "high_nees_runs": float(np.mean([score.mean_nees > HIGH_MEAN_NEES for score in scores])),
        "landmarks_per_run": float(np.mean([len(run.landmarks) for run in runs])),
        "spurious_per_run": float(np.mean([score.landmarks.spurious for score in scores])),
        "duplicates_per_run": float(np.mean([score.landmarks.duplicates for score in scores])),
        "circles_mapped": (
            sum(score.landmarks.mapped for score in scores) / seen if seen else float("nan")
        ),
        "radius_nees_mean": mean_or_nan(pooled("radius_nees")),
        "radius_nees_median": median_or_nan(pooled("radius_nees")),
        "pair_nees_mean": mean_or_nan(pooled("pair_nees")),
        "pair_nees_median": median_or_nan(pooled("pair_nees")),
        "absolute_nees_mean": mean_or_nan(pooled("absolute_nees")),
        "radius_error_mean": mean_or_nan(pooled("radius_errors")),
        "drift_median": median_or_nan([score.first_estimate_drift for score in scores]),
        "drift_max": float(max(score.first_estimate_drift for score in scores)),
        "loop_closures": float(len(loops)),
        "loop_error_before": median_or_nan(loops[:, 0]),
        "loop_error_after": median_or_nan(loops[:, 1]),
        "loop_std_before": median_or_nan(loops[:, 2]),
        "loop_std_after": median_or_nan(loops[:, 3]),
        "update_ms_median": float(np.median(seconds) * 1e3),
        "update_ms_p99": float(np.quantile(seconds, 0.99) * 1e3),
        "rejected": float(sum(update.rejected for update in updates)),
        "skipped": float(sum(update.skipped for update in updates)),
        "ambiguous": float(sum(update.ambiguous for update in updates)),
        "rejected_updates": float(sum(update.rejected_update for update in updates)),
        "runs": float(len(runs)),
    }
