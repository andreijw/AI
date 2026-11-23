"""Scan matching scored against the simulator's truth (``SCAN-1``, ``SLAM-4``, ``SLAM-5``).

A desk tool: :mod:`rl_navigation.slam` does not import it. The definitions here are frozen
before any gate is registered; a change after registration is the maintainer's decision.

- **``SCAN-1``, the matcher alone** (:func:`scan1_trials`). A seeded run
  (:func:`~rl_navigation.estimation.runs.record_run`) is mapped from its *true* poses. At every
  :data:`SCAN1_EVERY`-th step from :data:`SCAN1_START`, before that step's scan is mapped, the
  scan is matched from the true pose plus ``N(0, diag(SCAN1_PERTURBATION)^2)``, drawn from the
  run's own stream ``SeedSequence([seed, SCAN1_STREAM])``, with the wide prior covariance
  ``SCAN1_PRIOR_VARIANCE I``. Every trial counts in the NEES (3 degrees of freedom; a rejected or
  non-converged one is not dropped); *correct* (position error below :data:`SCAN1_CORRECT`),
  *converged* (the step fell under the tolerance) and *accepted* are separate shares.
- **``SLAM-4``, the front end** (:func:`scan_slam_trial`): ``run_trial``'s wandering robot with
  :class:`~rl_navigation.slam.scan_matching.ScanMatchingSlam` as its filter, its grid on the
  arena grown by :data:`~rl_navigation.slam.evaluation.BOUNDS_MARGIN`. The trajectory is aligned
  to the truth (Umeyama without scale, as ``SLAM-1``), and the map is scored by rebuilding it
  from the aligned estimates (:func:`~rl_navigation.mapping.evaluation.map_from_poses`) and
  scoring it as ``MAP-1`` does. The pose NEES is reported, aligned and not; it is not a gate:
  the covariance is the pose's relative to the map, whose drift is not in the state.
- **``SLAM-5``, the speed** (:func:`slam5_timing`): each update's wall time (field, match and
  mapping) over the runs of :data:`SLAM5_SEEDS`, one process, alone on the machine.

:data:`SCAN1_ROWS` and :data:`SLAM4_ROWS` name the rows; :func:`scan1_row` and
:func:`slam4_row` run one over seeds. Imports NumPy, ``rl_navigation.sim``,
``rl_navigation.estimation`` (``run_trial``, ``summarize``, ``pose_nees``, ``runs``),
``rl_navigation.mapping`` (the core and ``evaluation``) and the slam package.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.averages import mean_or_nan, median_or_nan
from rl_navigation.estimation.evaluation import (
    NEES_UPPER_95,
    FactoryContext,
    Localizer,
    Trial,
    pose_nees,
    run_trial,
    summarize,
)
from rl_navigation.estimation.poses import pose_difference
from rl_navigation.estimation.runs import record_run
from rl_navigation.mapping.evaluation import map_from_poses, score_map
from rl_navigation.mapping.fields import DistanceField
from rl_navigation.mapping.grid import OccupancyGrid
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.kinematics import wrap_angle
from rl_navigation.sim.layouts import LayoutParams
from rl_navigation.sim.sensors import RangeSensor, lidar
from rl_navigation.slam.evaluation import (
    ALIGNED_FAILURE_ATE,
    BOUNDS_MARGIN,
    align_se2,
    aligned_trajectory_error,
    arena_bounds,
)
from rl_navigation.slam.scan_matching import (
    ScanMatchingSlam,
    ScanMatchSettings,
    ScanMatchUpdate,
    match_scan,
)

# SCAN-1: the first matched step, the spacing, the prior's perturbation (m, m, rad), the wide
# prior covariance's variance, the "correct" position error (m) and the perturbations' stream.
SCAN1_START = 100
SCAN1_EVERY = 25
SCAN1_PERTURBATION = (0.03, 0.03, 0.03)
SCAN1_PRIOR_VARIANCE = 100.0
SCAN1_CORRECT = 0.05
SCAN1_STREAM = 0x5CA1

# SLAM-5: the runs timed (tuning seeds: timing reads no learning result).
SLAM5_SEEDS = (1, 2, 3)

# The map figures SLAM-4 averages over runs.
MAP_FIGURES = (
    "occupied_precision",
    "cell_accuracy",
    "visible_surface_recall",
    "observed_free_share",
    "false_free_cells",
)


class Row(NamedTuple):
    """One row of ``SCAN-1`` or ``SLAM-4``: the sensor, the dropout (None: the sensor's own), the
    prior's perturbation (``SCAN-1``) and a pose-noise floor replacing the settings' (``SLAM-4``;
    None keeps it)."""

    sensor: Callable[[], RangeSensor]
    dropout_prob: float | None = None
    perturbation: tuple[float, float, float] = SCAN1_PERTURBATION
    pose_noise_floor: tuple[float, float, float] | None = None


_LIDAR_360 = partial(lidar, num_beams=360)

SCAN1_ROWS: dict[str, Row] = {
    "a": Row(_LIDAR_360),
    "basin": Row(_LIDAR_360, perturbation=(0.1, 0.1, 0.1)),
    "64": Row(lidar),
}
SLAM4_ROWS: dict[str, Row] = {
    "a": Row(_LIDAR_360),
    "b": Row(_LIDAR_360, dropout_prob=0.3),
    "c": Row(lidar),
    "d": Row(partial(lidar, num_beams=360, noise_std=0.02)),
    "floor_5mm": Row(_LIDAR_360, pose_noise_floor=(0.005, 0.005, 0.005)),
}


def arena_frame(
    arena: Arena, resolution: float, margin: float = BOUNDS_MARGIN
) -> tuple[tuple[int, int], tuple[float, float]]:
    """``(shape, origin)`` of a grid covering the arena grown by ``margin`` on every side."""
    rows = math.ceil((arena.height + 2.0 * margin) / resolution - 1e-9)
    cols = math.ceil((arena.width + 2.0 * margin) / resolution - 1e-9)
    return (rows, cols), (-margin, -margin)


# SCAN-1 -------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Scan1Trials:
    """One run's ``SCAN-1`` trials, ``M`` of them.

    Attributes:
        seed: The run's seed.
        steps: ``(M,)`` the matched steps.
        priors: ``(M, 3)`` the perturbed priors.
        errors: ``(M, 3)`` matched pose minus true pose (heading wrapped).
        nees: ``(M,)`` ``e^T P^-1 e`` against the match's covariance (inf when it is not finite).
        terminations: ``(M,)`` ``"converged"``, ``"capped"`` or ``"stalled"``.
        accepted: ``(M,)`` whether each match passed its checks.
        inlier_share: ``(M,)`` inliers over beams in the grid (NaN without beams).
        residual_rms: ``(M,)`` the inliers' RMS distance in metres.
        seconds: ``(M,)`` each match's wall time, the field included.
    """

    seed: int
    steps: NDArray[np.intp]
    priors: FloatArray
    errors: FloatArray
    nees: FloatArray
    terminations: list[str]
    accepted: NDArray[np.bool_]
    inlier_share: FloatArray
    residual_rms: FloatArray
    seconds: FloatArray


def _nees_of(error: FloatArray, covariance: FloatArray) -> float:
    if not np.isfinite(covariance).all():
        return math.inf
    try:
        return float(error @ np.linalg.solve(covariance, error))
    except np.linalg.LinAlgError:
        return math.inf


def _finite_median(values: ArrayLike) -> float:
    """The median of the finite values (NaN when there are none): a match without inliers has no
    RMS and would otherwise make every median NaN."""
    array = np.asarray(values, dtype=np.float64)
    return median_or_nan(array[np.isfinite(array)])


def scan1_trials(
    seed: int,
    sensor: RangeSensor,
    *,
    settings: ScanMatchSettings | None = None,
    steps: int = 600,
    dropout_prob: float | None = None,
    perturbation: tuple[float, float, float] = SCAN1_PERTURBATION,
    layout_params: LayoutParams | None = None,
) -> Scan1Trials:
    """The matcher on one seeded run's map built from true poses (see the module docstring).

    Raises:
        ValueError: The run has no steps.
    """
    settings = ScanMatchSettings() if settings is None else settings
    run = record_run(
        seed, sensor, steps=steps, dropout_prob=dropout_prob, layout_params=layout_params
    )
    shape, origin = arena_frame(run.arena, settings.resolution)
    grid = OccupancyGrid(sensor, shape, settings.resolution, settings.model, origin)
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCAN1_STREAM]))
    spread = np.asarray(perturbation, dtype=np.float64)
    wide = SCAN1_PRIOR_VARIANCE * np.eye(3)
    rows: list[tuple] = []
    for step, (pose, scan) in enumerate(zip(run.true_poses, run.readings, strict=True)):
        if step >= SCAN1_START and (step - SCAN1_START) % SCAN1_EVERY == 0:
            prior = pose + spread * rng.standard_normal(3)
            prior[2] = wrap_angle(prior[2])
            start = time.perf_counter()
            field = DistanceField.from_occupied(
                grid.occupied(), settings.resolution, settings.field_reach, origin
            )
            match = match_scan(field, sensor, scan, prior, wide, settings)
            seconds = time.perf_counter() - start
            error = pose_difference(match.pose, pose)
            share = int(match.inliers.sum()) / match.beams if match.beams else math.nan
            rows.append(
                (
                    step,
                    prior,
                    error,
                    _nees_of(error, match.covariance),
                    match.termination,
                    match.accepted,
                    share,
                    match.residual_rms,
                    seconds,
                )
            )
        grid.update(pose, scan)
    columns = list(zip(*rows, strict=True)) if rows else [()] * 9
    return Scan1Trials(
        seed=seed,
        steps=np.array(columns[0], dtype=np.intp),
        priors=np.array(columns[1], dtype=np.float64).reshape(-1, 3),
        errors=np.array(columns[2], dtype=np.float64).reshape(-1, 3),
        nees=np.array(columns[3], dtype=np.float64),
        terminations=list(columns[4]),
        accepted=np.array(columns[5], dtype=bool),
        inlier_share=np.array(columns[6], dtype=np.float64),
        residual_rms=np.array(columns[7], dtype=np.float64),
        seconds=np.array(columns[8], dtype=np.float64),
    )


def summarize_scan1(runs: Sequence[Scan1Trials]) -> dict[str, float]:
    """``SCAN-1``'s figures pooled over every trial of every run.

    Returns:
        ``trials``; ``nees_mean`` and ``nees_median`` over every trial; the shares
        ``correct_share`` (position error below :data:`SCAN1_CORRECT`), ``converged_share``,
        ``capped_share``, ``stalled_share`` and ``accepted_share``; ``position_error_median``
        and ``_p95`` (m), ``heading_error_median`` and ``_p95`` (rad); ``inlier_share_median``
        and ``residual_rms_median`` (m), over the trials that have one (a match without inliers
        has no RMS); ``match_ms_median`` and ``_p99``.

    Raises:
        ValueError: No trials.
    """
    if not runs or not sum(len(run.nees) for run in runs):
        raise ValueError("summarize_scan1 needs trials; got none.")
    errors = np.concatenate([run.errors for run in runs])
    nees = np.concatenate([run.nees for run in runs])
    terminations = np.array([t for run in runs for t in run.terminations])
    position = np.hypot(errors[:, 0], errors[:, 1])
    heading = np.abs(errors[:, 2])
    seconds = np.concatenate([run.seconds for run in runs])
    return {
        "trials": float(len(nees)),
        "nees_mean": float(nees.mean()),
        "nees_median": float(np.median(nees)),
        "correct_share": float(np.mean(position < SCAN1_CORRECT)),
        "converged_share": float(np.mean(terminations == "converged")),
        "capped_share": float(np.mean(terminations == "capped")),
        "stalled_share": float(np.mean(terminations == "stalled")),
        "accepted_share": float(np.concatenate([run.accepted for run in runs]).mean()),
        "position_error_median": float(np.median(position)),
        "position_error_p95": float(np.quantile(position, 0.95)),
        "heading_error_median": float(np.median(heading)),
        "heading_error_p95": float(np.quantile(heading, 0.95)),
        "inlier_share_median": _finite_median(np.concatenate([r.inlier_share for r in runs])),
        "residual_rms_median": _finite_median(np.concatenate([r.residual_rms for r in runs])),
        "match_ms_median": float(np.median(seconds) * 1e3),
        "match_ms_p99": float(np.quantile(seconds, 0.99) * 1e3),
    }


# SLAM-4 -------------------------------------------------------------------------------------------


class _RecordedScanSlam:
    """A :class:`ScanMatchingSlam` as ``run_trial`` drives it, recording each scan, update and
    update time."""

    def __init__(self, slam: ScanMatchingSlam) -> None:
        self.slam = slam
        self.scans: list[FloatArray] = []
        self.updates: list[ScanMatchUpdate] = []
        self.seconds: list[float] = []

    @property
    def pose(self) -> FloatArray:
        return self.slam.pose

    @property
    def covariance(self) -> FloatArray:
        return self.slam.covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        self.slam.predict(wheel_travel)

    def update(self, readings: ArrayLike) -> ScanMatchUpdate:
        self.scans.append(np.array(readings, dtype=np.float64))
        start = time.perf_counter()
        result = self.slam.update(readings)
        self.seconds.append(time.perf_counter() - start)
        self.updates.append(result)
        return result


class ScanSlamFactory:
    """A ``run_trial`` factory of :class:`ScanMatchingSlam` at the trial's noisy start, its grid
    and bounds on the arena grown by :data:`BOUNDS_MARGIN`. It keeps what it built."""

    def __init__(self, settings: ScanMatchSettings | None = None) -> None:
        self.settings = ScanMatchSettings() if settings is None else settings
        self.made: _RecordedScanSlam | None = None
        self.arena: Arena | None = None

    def __call__(self, context: FactoryContext) -> Localizer:
        arena = context.range_map.arena
        shape, origin = arena_frame(arena, self.settings.resolution)
        slam = ScanMatchingSlam(
            context.range_map.sensor,
            context.odometry,
            context.start,
            context.covariance,
            shape,
            origin,
            self.settings,
            bounds=arena_bounds(arena),
        )
        self.arena, self.made = arena, _RecordedScanSlam(slam)
        return self.made


@dataclass(frozen=True, eq=False)
class ScanSlamRun:
    """One seeded scan-matching SLAM run.

    Attributes:
        seed: The run's seed.
        trial: ``run_trial``'s record (true poses, estimates, covariances, dead reckoning).
        arena: The true arena.
        sensor: The lidar.
        readings: ``(T, N)`` each step's scan.
        updates: Each step's :class:`ScanMatchUpdate`.
        update_seconds: ``(T,)`` each update's wall time (field, match and mapping).
    """

    seed: int
    trial: Trial
    arena: Arena
    sensor: RangeSensor
    readings: FloatArray
    updates: list[ScanMatchUpdate]
    update_seconds: FloatArray


def scan_slam_trial(
    seed: int,
    sensor: RangeSensor,
    *,
    settings: ScanMatchSettings | None = None,
    steps: int = 600,
    dropout_prob: float | None = None,
    layout_params: LayoutParams | None = None,
) -> ScanSlamRun:
    """Run the scan-matching front end on a seeded wandering run (``run_trial``'s).

    Raises:
        ValueError: The run has no steps.
    """
    factory = ScanSlamFactory(settings)
    trial = run_trial(
        seed,
        sensor,
        steps=steps,
        dropout_prob=dropout_prob,
        layout_params=layout_params,
        make_localizer=factory,
    )
    made, arena = factory.made, factory.arena
    if made is None or arena is None or len(trial.true_poses) == 0:
        raise ValueError(
            f"scan_slam_trial: seed {seed} has no steps (steps={steps}, or a collision on the "
            "first step)."
        )
    return ScanSlamRun(
        seed=seed,
        trial=trial,
        arena=arena,
        sensor=sensor,
        readings=np.array(made.scans, dtype=np.float64),
        updates=made.updates,
        update_seconds=np.array(made.seconds, dtype=np.float64),
    )


def _aligned(trial: Trial) -> tuple[FloatArray, FloatArray]:
    """The estimates and covariances moved by the trajectory's alignment to the truth."""
    R, t = align_se2(trial.estimates[:, :2], trial.true_poses[:, :2])
    poses = trial.estimates.copy()
    poses[:, :2] = trial.estimates[:, :2] @ R.T + t
    poses[:, 2] = wrap_angle(trial.estimates[:, 2] + math.atan2(R[1, 0], R[0, 0]))
    frame = np.eye(3)
    frame[:2, :2] = R
    return poses, frame @ trial.covariances @ frame.T


def score_scan_slam(
    run: ScanSlamRun, settings: ScanMatchSettings | None = None
) -> dict[str, float]:
    """One run's figures.

    Returns:
        ``ate`` and ``dead_reckoning_ate`` (aligned, m); ``map_<figure>`` for each of
        :func:`~rl_navigation.mapping.evaluation.score_map`'s figures, the map rebuilt from the
        aligned estimates; over the matches, ``accepted_share``, ``capped_share``,
        ``stalled_share``, ``inlier_share_median`` and ``residual_rms_median`` (over the matches
        that have one: a match without inliers has no RMS); over the steps,
        ``mapped_share``; ``nees_mean`` (absolute), ``aligned_nees_mean`` and
        ``aligned_nees_below_upper_95``.
    """
    settings = ScanMatchSettings() if settings is None else settings
    trial = run.trial
    true = trial.true_poses
    aligned, covariances = _aligned(trial)
    grid = map_from_poses(
        run.arena, run.sensor, aligned, run.readings, settings.model, resolution=settings.resolution
    )
    aligned_nees = pose_nees(replace(trial, estimates=aligned, covariances=covariances))
    matches = [update.match for update in run.updates if update.match is not None]
    terminations = np.array([match.termination for match in matches])
    return {
        "ate": aligned_trajectory_error(trial.estimates[:, :2], true[:, :2]),
        "dead_reckoning_ate": aligned_trajectory_error(trial.dead_reckoning[:, :2], true[:, :2]),
        **{f"map_{key}": value for key, value in score_map(grid, run.arena, true).items()},
        "accepted_share": mean_or_nan([match.accepted for match in matches]),
        "capped_share": mean_or_nan(terminations == "capped"),
        "stalled_share": mean_or_nan(terminations == "stalled"),
        "inlier_share_median": _finite_median(
            [match.inliers.sum() / match.beams for match in matches if match.beams]
        ),
        "residual_rms_median": _finite_median([match.residual_rms for match in matches]),
        "mapped_share": mean_or_nan([update.mapped for update in run.updates]),
        "nees_mean": float(pose_nees(trial).mean()),
        "aligned_nees_mean": float(aligned_nees.mean()),
        "aligned_nees_below_upper_95": float(np.mean(aligned_nees <= NEES_UPPER_95)),
    }


def summarize_scan_slam(
    runs: Sequence[ScanSlamRun],
    settings: ScanMatchSettings | None = None,
    failure_ate: float = ALIGNED_FAILURE_ATE,
) -> dict[str, float]:
    """``SLAM-4``'s figures over runs.

    Returns:
        ``summarize``'s pose figures over the trials (absolute) and: ``ate_mean``,
        ``ate_median``, ``ate_max``, ``aligned_failures`` (runs whose ATE exceeds
        ``failure_ate``), ``dead_reckoning_ate_mean`` and ``ate_ratio_to_dead_reckoning`` (the
        two means' ratio); the mean over runs of each ``map_`` figure of :data:`MAP_FIGURES`;
        the mean over runs of ``accepted_share``, ``capped_share``, ``stalled_share`` and
        ``mapped_share``, and ``accepted_share_min``; the medians over runs of
        ``inlier_share_median`` and ``residual_rms_median`` (runs with one); ``nees_mean_runs`` (absolute),
        ``aligned_nees_mean`` and ``aligned_nees_below_upper_95`` (means over runs);
        ``update_ms_median`` and ``_p99``; ``runs``.

    Raises:
        ValueError: No runs.
    """
    if not runs:
        raise ValueError("summarize_scan_slam needs runs; got none.")
    scores = [score_scan_slam(run, settings) for run in runs]

    def each(key: str) -> FloatArray:
        return np.array([score[key] for score in scores], dtype=np.float64)

    ates, dead = each("ate"), each("dead_reckoning_ate")
    seconds = np.concatenate([run.update_seconds for run in runs])
    return {
        **summarize([run.trial for run in runs]),
        "ate_mean": float(ates.mean()),
        "ate_median": float(np.median(ates)),
        "ate_max": float(ates.max()),
        "aligned_failures": float((ates > failure_ate).sum()),
        "dead_reckoning_ate_mean": float(dead.mean()),
        "ate_ratio_to_dead_reckoning": float(ates.mean() / dead.mean()),
        **{f"map_{key}": mean_or_nan(each(f"map_{key}")) for key in MAP_FIGURES},
        **{
            key: mean_or_nan(each(key))
            for key in ("accepted_share", "capped_share", "stalled_share", "mapped_share")
        },
        "accepted_share_min": float(np.nanmin(each("accepted_share")))
        if np.isfinite(each("accepted_share")).any()
        else math.nan,
        "inlier_share_median": _finite_median(each("inlier_share_median")),
        "residual_rms_median": _finite_median(each("residual_rms_median")),
        "nees_mean_runs": mean_or_nan(each("nees_mean")),
        "aligned_nees_mean": mean_or_nan(each("aligned_nees_mean")),
        "aligned_nees_below_upper_95": mean_or_nan(each("aligned_nees_below_upper_95")),
        "update_ms_median": float(np.median(seconds) * 1e3) if len(seconds) else math.nan,
        "update_ms_p99": float(np.quantile(seconds, 0.99) * 1e3) if len(seconds) else math.nan,
        "runs": float(len(runs)),
    }


# Rows and timing ----------------------------------------------------------------------------------


def _row_settings(row: Row, settings: ScanMatchSettings | None) -> ScanMatchSettings:
    base = ScanMatchSettings() if settings is None else settings
    if row.pose_noise_floor is None:
        return base
    return replace(base, pose_noise_floor=row.pose_noise_floor)


def scan1_row(
    name: str, seeds: Sequence[int], settings: ScanMatchSettings | None = None
) -> dict[str, float]:
    """``SCAN-1`` row ``name`` (:data:`SCAN1_ROWS`) over ``seeds``, summarised."""
    row = SCAN1_ROWS[name]
    sensor = row.sensor()
    return summarize_scan1(
        [
            scan1_trials(
                seed,
                sensor,
                settings=_row_settings(row, settings),
                dropout_prob=row.dropout_prob,
                perturbation=row.perturbation,
            )
            for seed in seeds
        ]
    )


def slam4_row(
    name: str, seeds: Sequence[int], settings: ScanMatchSettings | None = None
) -> dict[str, float]:
    """``SLAM-4`` row ``name`` (:data:`SLAM4_ROWS`) over ``seeds``, summarised."""
    row = SLAM4_ROWS[name]
    sensor = row.sensor()
    row_settings = _row_settings(row, settings)
    runs = [
        scan_slam_trial(seed, sensor, settings=row_settings, dropout_prob=row.dropout_prob)
        for seed in seeds
    ]
    return summarize_scan_slam(runs, row_settings)


def slam5_timing(
    seeds: Sequence[int] = SLAM5_SEEDS, settings: ScanMatchSettings | None = None
) -> dict[str, float]:
    """``SLAM-5``: the median and p99 update time in milliseconds of the 360-beam front end over
    the runs of ``seeds`` (600 steps each), and the number of updates timed."""
    seconds = np.concatenate(
        [scan_slam_trial(seed, _LIDAR_360(), settings=settings).update_seconds for seed in seeds]
    )
    return {
        "update_ms_median": float(np.median(seconds) * 1e3),
        "update_ms_p99": float(np.quantile(seconds, 0.99) * 1e3),
        "updates": float(len(seconds)),
    }
