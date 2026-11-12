"""Circle features scored against the simulator's truth, over seeded runs (``CIRC-1``).

A desk tool: :mod:`rl_navigation.slam` does not import it. Each run is
:func:`~rl_navigation.mapping.evaluation.record_run`'s wandering robot (the trajectories of
``run_trial``, bit for bit). Every scan's features are placed with the *true* pose and matched to
the true circle whose centre lies within :data:`MATCH_RADIUS`; a feature matching none is a false
fit. A circle is *visible* in a scan when at least ``min_points`` noise-free beams end on it, and
*detected* when a feature matched it. It is *reachable* when at least ``min_points`` of those
beams also carry a usable reading in the scan: the most a detector could find under dropout.
Matched features are scored by their centre and radius
error and by their NEES against the fit's own covariance (3 degrees of freedom).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.mapping.evaluation import record_run
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.sensors import RangeSensor, usable_readings
from rl_navigation.slam.features import CircleFeature, FeatureSettings, extract_circles

# A feature matches a true circle whose centre lies within this many metres of it.
MATCH_RADIUS = 0.15

# A beam ends on a circle when its noise-free end point lies within this many metres of it.
HIT_TOLERANCE = 1e-6

# A circle's radius bias is tested only with at least this many matched fits.
BIAS_MIN_FITS = 30


def _rotation(theta: float) -> FloatArray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def visible_circles(
    arena: Arena,
    sensor: RangeSensor,
    pose: ArrayLike,
    true_ranges: ArrayLike,
    min_points: int,
    readings: ArrayLike | None = None,
) -> NDArray[np.intp]:
    """Indices of the circles that at least ``min_points`` of the pose's noise-free beams end on,
    counting only beams whose noise-free range the sensor can report.

    Args:
        arena: The true arena.
        sensor: The sensor.
        pose: ``(3,)`` the true pose.
        true_ranges: ``(N,)`` the noise-free ranges at ``pose`` (``RangeSensor.true_ranges``).
        min_points: Beams a circle needs.
        readings: ``(N,)`` the scan actually read; when given, a beam counts only if its reading
            is also usable (the circle is *reachable*).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if not len(arena.circle_radii):
        return np.zeros(0, dtype=np.intp)
    angles = pose[2] + sensor.mount_angles
    ranges = np.asarray(true_ranges, dtype=np.float64)
    ends = sensor.mount_origins(pose[None])[0] + ranges[:, None] * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )
    offsets = ends[:, None, :] - arena.circle_centers[None]  # (N, C, 2)
    on = np.abs(np.hypot(offsets[..., 0], offsets[..., 1]) - arena.circle_radii) < HIT_TOLERANCE
    on &= usable_readings(ranges, sensor)[:, None]  # past max_range a beam reads no echo
    if readings is not None:
        on &= usable_readings(readings, sensor)[:, None]  # a dropout carries no point
    return np.flatnonzero(on.sum(axis=0) >= min_points).astype(np.intp)


def match_features(
    arena: Arena,
    pose: ArrayLike,
    features: Sequence[CircleFeature],
    match_radius: float = MATCH_RADIUS,
) -> NDArray[np.intp]:
    """``(F,)`` the true circle each feature, placed with the true ``pose``, lies within
    ``match_radius`` of (the nearest), or -1."""
    pose = np.asarray(pose, dtype=np.float64)
    if not features or not len(arena.circle_radii):
        return np.full(len(features), -1, dtype=np.intp)
    body = np.array([feature.z[:2] for feature in features])
    world = pose[:2] + body @ _rotation(float(pose[2])).T
    distances = np.hypot(*(world[:, None, :] - arena.circle_centers[None]).transpose(2, 0, 1))
    nearest = distances.argmin(axis=1)
    within = distances[np.arange(len(features)), nearest] <= match_radius
    return np.where(within, nearest, -1).astype(np.intp)


@dataclass(frozen=True, eq=False)
class FeatureRun:
    """One run's feature scores.

    Attributes:
        seed: The run's seed.
        scans: Scans read.
        fits: Features extracted, over every scan.
        false_fits: Features matching no true circle.
        visible: Circle-scans with a visible circle (see the module docstring).
        detected: Visible circle-scans with a feature matching that circle.
        reachable: Visible circle-scans whose circle keeps ``min_points`` usable readings.
        points: ``(M,)`` beams in each matched feature.
        centre_errors: ``(M,)`` distance from each matched feature's centre to the true one, m.
        radius_errors: ``(M,)`` fitted minus true radius, m.
        nees: ``(M,)`` each matched feature's NEES against its covariance.
        circles: ``(M,)`` the true circle each matched feature belongs to.
        extract_seconds: ``(scans,)`` wall time of each scan's :func:`extract_circles`.
        collided: Whether the run ended at a collision before its last step.
    """

    seed: int
    scans: int
    fits: int
    false_fits: int
    visible: int
    detected: int
    reachable: int
    points: NDArray[np.intp]
    centre_errors: FloatArray
    radius_errors: FloatArray
    nees: FloatArray
    circles: NDArray[np.intp]
    extract_seconds: FloatArray
    collided: bool


def feature_trial(
    seed: int,
    sensor: RangeSensor,
    *,
    steps: int = 600,
    dropout_prob: float | None = None,
    settings: FeatureSettings | None = None,
) -> FeatureRun:
    """Extract circles from every scan of a seeded run and score them against the truth.

    Args:
        seed: The run's seed (as ``run_trial``'s).
        sensor: The lidar.
        steps: Steps to run; a collision ends the run early.
        dropout_prob: Dropout probability; None uses the sensor's own.
        settings: The extraction settings; None uses the defaults.

    Returns:
        The run's scores.
    """
    settings = FeatureSettings() if settings is None else settings
    run = record_run(seed, sensor, steps=steps, dropout_prob=dropout_prob)
    arena = run.arena
    true_ranges = sensor.true_ranges(arena, run.true_poses)
    fits = false_fits = visible_count = detected = reachable = 0
    points: list[int] = []
    centre_errors: list[float] = []
    radius_errors: list[float] = []
    nees: list[float] = []
    circles: list[int] = []
    seconds = []
    for pose, readings, true in zip(run.true_poses, run.readings, true_ranges, strict=True):
        start = time.perf_counter()
        features = extract_circles(sensor, readings, settings)
        seconds.append(time.perf_counter() - start)
        matches = match_features(arena, pose, features)
        visible = visible_circles(arena, sensor, pose, true, settings.min_points)
        fits += len(features)
        false_fits += int((matches < 0).sum())
        visible_count += len(visible)
        reachable += len(visible_circles(arena, sensor, pose, true, settings.min_points, readings))
        detected += len(set(visible.tolist()) & set(matches.tolist()))
        to_body = _rotation(float(pose[2])).T
        for feature, circle in zip(features, matches, strict=True):
            if circle < 0:
                continue
            truth = np.array(
                [*(to_body @ (arena.circle_centers[circle] - pose[:2])), arena.circle_radii[circle]]
            )
            error = feature.z - truth
            points.append(len(feature.beams))
            centre_errors.append(float(np.hypot(error[0], error[1])))
            radius_errors.append(float(error[2]))
            nees.append(float(error @ np.linalg.solve(feature.covariance, error)))
            circles.append(int(circle))
    return FeatureRun(
        seed=seed,
        scans=len(run.true_poses),
        fits=fits,
        false_fits=false_fits,
        visible=visible_count,
        detected=detected,
        reachable=reachable,
        points=np.array(points, dtype=np.intp),
        centre_errors=np.array(centre_errors),
        radius_errors=np.array(radius_errors),
        nees=np.array(nees),
        circles=np.array(circles, dtype=np.intp),
        extract_seconds=np.array(seconds),
        collided=run.collided,
    )


def radius_bias_t2(
    circles: ArrayLike, radius_errors: ArrayLike, min_fits: int = BIAS_MIN_FITS
) -> float:
    """Mean over circles with at least ``min_fits`` fits of ``t^2``, ``t`` the circle's mean
    radius error over its standard error (sample sd / sqrt(n)): about 1 when the fit is unbiased,
    larger when each circle's fits share a bias. NaN when no circle has enough fits."""
    circles = np.asarray(circles)
    errors = np.asarray(radius_errors, dtype=np.float64)
    squares = []
    for circle in np.unique(circles):
        values = errors[circles == circle]
        if len(values) < min_fits:
            continue
        standard_error = values.std(ddof=1) / np.sqrt(len(values))
        squares.append((values.mean() / standard_error) ** 2)
    return float(np.mean(squares)) if squares else float("nan")


def _mean_or_nan(values: FloatArray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def summarize_features(runs: Sequence[FeatureRun]) -> dict[str, float]:
    """Pooled feature scores over runs.

    Returns:
        ``detection_rate`` and ``false_fit_rate`` (shares), ``reachable_rate`` (visible
        circle-scans that stay reachable, the detection rate's ceiling under dropout),
        ``fits_per_scan``; over matched fits
        ``centre_error_median`` and ``_p95``, ``radius_error_median`` and ``_p95`` (absolute) and
        ``radius_error_mean`` (signed), ``fit_nees_mean`` overall and by beams per fit (``_8_to_11``,
        ``_12_to_19``, ``_20_up``); ``radius_bias_t2`` over every run's circles (each run's
        circles counted apart); ``extract_ms_median`` and ``extract_ms_p99`` per scan; ``runs``
        and ``collisions``. Medians and means of no fits are NaN.

    Raises:
        ValueError: No runs.
    """
    if not runs:
        raise ValueError("summarize_features needs runs; got none.")
    points = np.concatenate([run.points for run in runs])
    centres = np.concatenate([run.centre_errors for run in runs])
    radii = np.concatenate([run.radius_errors for run in runs])
    nees = np.concatenate([run.nees for run in runs])
    keys = np.concatenate(
        [run.circles + 1_000_000 * index for index, run in enumerate(runs)]  # circles per run
    )
    milliseconds = 1e3 * np.concatenate([run.extract_seconds for run in runs])
    fits = sum(run.fits for run in runs)
    visible = sum(run.visible for run in runs)

    def quantile(values: FloatArray, q: float) -> float:
        return float(np.quantile(values, q)) if len(values) else float("nan")

    return {
        "detection_rate": sum(run.detected for run in runs) / visible if visible else float("nan"),
        "reachable_rate": sum(run.reachable for run in runs) / visible if visible else float("nan"),
        "false_fit_rate": sum(run.false_fits for run in runs) / fits if fits else float("nan"),
        "fits_per_scan": fits / sum(run.scans for run in runs),
        "centre_error_median": quantile(centres, 0.5),
        "centre_error_p95": quantile(centres, 0.95),
        "radius_error_median": quantile(np.abs(radii), 0.5),
        "radius_error_p95": quantile(np.abs(radii), 0.95),
        "radius_error_mean": _mean_or_nan(radii),
        "fit_nees_mean": _mean_or_nan(nees),
        "fit_nees_mean_8_to_11": _mean_or_nan(nees[points < 12]),
        "fit_nees_mean_12_to_19": _mean_or_nan(nees[(points >= 12) & (points < 20)]),
        "fit_nees_mean_20_up": _mean_or_nan(nees[points >= 20]),
        "radius_bias_t2": radius_bias_t2(keys, radii),
        "extract_ms_median": quantile(milliseconds, 0.5),
        "extract_ms_p99": quantile(milliseconds, 0.99),
        "runs": float(len(runs)),
        "collisions": float(sum(run.collided for run in runs)),
    }
