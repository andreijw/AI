"""Circle features from one range scan: the landmarks of the landmark SLAM.

A scan's usable readings become points in the robot's frame. They are split into segments where
neighbouring points jump apart (an adaptive breakpoint, Borges and Aldon 2004), bridging up to
``max_bridged_angle`` of unusable beams (dropouts) between two usable ones, and each segment
of at least ``min_points`` points is fitted with a circle:

- an algebraic fit (Kasa 1976) starts a weighted geometric Gauss-Newton fit of
  ``r_i = (|p_i - c| - rho) / w_i``;
- each residual's standard deviation ``w_i = sigma max(|u_i . n_i|, cos_floor)`` is the range
  noise projected on the surface normal ``n_i`` at the point (``u_i`` is the beam's direction): a
  reading along an oblique beam moves the point mostly along the surface, so an unweighted fit's
  covariance is too wide (mean NEES 1.75 against 3 on tuning seeds);
- the covariance of ``(c_x, c_y, rho)`` is ``(J^T J)^-1`` of the weighted residuals, weights held
  fixed at the solution.

A fit is kept only when it is plausible as the visible face of a cylinder: its weighted residuals
pass a chi-square test, its radius lies in ``radius_range``, its centre lies beyond the points (a
convex face, not the far side seen from inside), two straight lines (a box corner, or a wall) do
not explain the points better by ``corner_penalty`` sigma^2, and the result is sound: finite, a
positive-definite covariance, and no standard deviation above ``max_fit_std``. A segment whose
radius runs away (a straight surface) is refused as soon as it does. Nothing here raises on a
scan's content: a segment that cannot be fitted is dropped.

A circle needs many beams: the HC-SR04's five sonars, 30 degrees apart, never give a feature, so
landmark SLAM needs a lidar (the robot's has about 1 degree per beam).

Known limit: a circle touching a box, a wall or another circle shares one segment with it, and no
circle fits the joined points, so it is missed (most of the 20 % of visible circles missed on
tuning seeds; ``TESTING.md`` ``CIRC-1``).

Frames and units: metres and radians; a feature is in the robot's frame, ``x`` ahead.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.ekf import chi2_quantile
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.sensors import RangeSensor, usable_readings

# A segment's last beam joins its first across the scan's start when the mounts close a full turn
# within this many radians.
FULL_TURN_TOLERANCE = 1e-9

# Beams that fit in max_bridged_angle are counted with this much slack, so a cap of exactly n
# spacings bridges n beams despite rounding.
BRIDGE_TOLERANCE = 1e-9

# A fit whose radius exceeds this many times the largest radius allowed, at the algebraic start or
# during the iterations, is a straight surface whose radius runs away: it is refused at once.
DIVERGED_RADIUS_FACTOR = 3.0

# Gauss-Newton stops when no parameter moves by more than this many metres.
STEP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FeatureSettings:
    """How circles are cut out of a scan and fitted.

    Args:
        min_points: Fewest points a circle is fitted to (at least 4, one more than its
            parameters, so its residual test has a degree of freedom). 8 keeps the fit's NEES
            near 3 on tuning seeds; at 5 to 7 points it is overconfident.
        radius_range: ``(low, high)`` radii in metres a circle may have (``0 < low < high``).
        fit_quantile: Quantile of chi-square with ``n - 3`` degrees of freedom that the weighted
            residuals' sum of squares must not exceed, in ``(0, 1)``.
        cos_floor: Smallest incidence cosine in the residual weights, in ``(0, 1]``: a grazing
            beam is weighted as if at ``acos(cos_floor)``.
        breakpoint_gap: Fixed part of the breakpoint distance in metres (>= 0).
        breakpoint_factor: Neighbours split when farther apart than ``breakpoint_gap +
            breakpoint_factor * spacing * range`` (>= 0), ``spacing`` the beams' angle apart.
        corner_penalty: The circle is refused when one line, or two lines meeting at one of the
            segment's points, leave a smaller unweighted sum of squares by more than this many
            ``noise_std ** 2`` (>= 0).
        max_iterations: Gauss-Newton iterations at most (>= 1).
        max_fit_std: Largest standard deviation in metres of the centre's coordinates or the
            radius that a kept fit may have (> 0).
        max_bridged_angle: Largest angle in radians of unusable beams (a dropout, no echo)
            between two usable ones that a segment may span (finite, >= 0): ``floor(max_bridged_angle /
            spacing)`` beams; 0 splits a segment at every unusable beam. 3 degrees bridges 3 beams
            of the 1 degree lidar, which lifts detection at 30 % dropout from 10 % to 62 % of
            visible circles while false fits fall from 28 % to 0.4 % (tuning seeds; 2 beams leave
            2.3 % false fits). It bridges no beam of the 64-beam lidar (5.6 degrees), where one
            bridged beam leaves 11 degrees between two points and the fits grow overconfident
            (``TESTING.md`` ``CIRC-2``).

    Raises:
        ValueError: A value out of its range.
    """

    min_points: int = 8
    radius_range: tuple[float, float] = (0.1, 0.8)
    fit_quantile: float = 0.99
    cos_floor: float = 0.3
    breakpoint_gap: float = 0.05
    breakpoint_factor: float = 3.0
    corner_penalty: float = 3.0
    max_iterations: int = 10
    max_fit_std: float = 0.1
    max_bridged_angle: float = math.radians(3.0)

    def __post_init__(self) -> None:
        low, high = self.radius_range
        problems = [
            (self.min_points < 4, f"min_points must be at least 4, got {self.min_points}"),
            (not 0.0 < low < high, f"radius_range needs 0 < low < high, got {self.radius_range}"),
            (
                not 0.0 < self.fit_quantile < 1.0,
                f"fit_quantile must be in (0, 1), got {self.fit_quantile}",
            ),
            (not 0.0 < self.cos_floor <= 1.0, f"cos_floor must be in (0, 1], got {self.cos_floor}"),
            (self.breakpoint_gap < 0.0, f"breakpoint_gap must be >= 0, got {self.breakpoint_gap}"),
            (
                self.breakpoint_factor < 0.0,
                f"breakpoint_factor must be >= 0, got {self.breakpoint_factor}",
            ),
            (self.corner_penalty < 0.0, f"corner_penalty must be >= 0, got {self.corner_penalty}"),
            (self.max_iterations < 1, f"max_iterations must be >= 1, got {self.max_iterations}"),
            (self.max_fit_std <= 0.0, f"max_fit_std must be positive, got {self.max_fit_std}"),
            (
                not (math.isfinite(self.max_bridged_angle) and self.max_bridged_angle >= 0.0),
                f"max_bridged_angle must be finite and >= 0, got {self.max_bridged_angle}",
            ),
        ]
        wrong = [message for failed, message in problems if failed]
        if wrong:
            raise ValueError("FeatureSettings: " + "; ".join(wrong) + ".")


class CircleFeature(NamedTuple):
    """A circle fitted to one segment of a scan.

    Attributes:
        z: ``(3,)`` the centre ``(x, y)`` in the robot's frame and the radius, metres.
        covariance: ``(3, 3)`` covariance of ``z``.
        beams: ``(n,)`` indices of the scan's readings the circle was fitted to, in beam order.
        ssr: Weighted residuals' sum of squares, chi-square with ``n - 3`` degrees of freedom.
    """

    z: FloatArray
    covariance: FloatArray
    beams: NDArray[np.intp]
    ssr: float


def scan_points(
    sensor: RangeSensor, readings: ArrayLike
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """A scan's readings as points in the robot's frame.

    Args:
        sensor: The sensor that read the scan.
        readings: ``(N,)`` readings in metres, per :class:`RangeSensor`'s reading contract.

    Returns:
        ``(points, usable, directions)``: ``(N, 2)`` points (meaningless where not usable),
        ``(N,)`` :func:`usable_readings`, and ``(N, 2)`` unit beam directions.

    Raises:
        ValueError: ``readings`` is not ``(sensor.num_readings,)``.
    """
    ranges = np.asarray(readings, dtype=np.float64)
    if ranges.shape != (sensor.num_readings,):
        raise ValueError(
            f"scan_points needs {sensor.num_readings} readings, got shape {ranges.shape}."
        )
    usable = usable_readings(ranges, sensor)
    angles = sensor.mount_angles
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    origins = sensor.mount_origins(np.zeros((1, 3)))[0]
    points = origins + np.where(usable, ranges, 0.0)[:, None] * directions
    return points, usable, directions


def segment_scan(
    points: FloatArray,
    ranges: FloatArray,
    usable: NDArray[np.bool_],
    spacing: float,
    settings: FeatureSettings,
    wrap: bool = False,
) -> list[NDArray[np.intp]]:
    """Split a scan's usable points into runs of neighbours close enough to be one surface.

    The usable beams are taken in beam order. Consecutive usable beams ``i`` and ``j``, with
    ``k = (j - i - 1) mod N`` unusable beams between them, belong together when
    ``k <= floor(max_bridged_angle / spacing)`` and their points are at most ``breakpoint_gap + breakpoint_factor
    * (k + 1) * spacing * min(range_i, range_j)`` apart: the breakpoint scaled by the angle the
    two beams span. An unusable beam is never part of a segment.

    Args:
        points: ``(N, 2)`` points in beam order.
        ranges: ``(N,)`` readings, metres.
        usable: ``(N,)`` which readings carry a range.
        spacing: Angle between neighbouring beams, radians.
        settings: The breakpoint and bridging settings.
        wrap: The beams close a full turn, so the last beam neighbours the first.

    Returns:
        Each segment's beam indices in beam order, segments ordered by their first beam; a
        segment across the scan's start runs from its first beam before the end to its last beam
        after the start.
    """
    beams = np.flatnonzero(usable)
    if len(beams) == 0:
        return []
    following = np.roll(beams, -1)
    skipped = (following - beams - 1) % len(points)
    gaps = np.hypot(*(points[following] - points[beams]).T)
    spanned = (skipped + 1) * spacing  # the angle between the two beams
    limit = settings.breakpoint_gap + settings.breakpoint_factor * spanned * np.minimum(
        ranges[beams], ranges[following]
    )
    bridgeable = _bridgeable_beams(settings.max_bridged_angle, spacing)
    joined = (skipped <= bridgeable) & (gaps <= limit)  # beams[k] joins k + 1
    if not wrap:
        joined[-1] = False
    if joined.all():
        return [beams]
    # Start the usable beams after a break, so no segment runs across the array's end.
    shift = int(np.flatnonzero(~joined)[-1]) + 1
    beams, joined = np.roll(beams, -shift), np.roll(joined, -shift)
    segments = np.split(beams, np.flatnonzero(~joined[:-1]) + 1)
    return sorted(segments, key=lambda segment: int(segment[0]))


def _bridgeable_beams(max_angle: float, spacing: float) -> int:
    """How many unusable beams ``spacing`` apart fit in ``max_angle`` (0 without a spacing)."""
    if not spacing > 0.0:
        return 0
    return math.floor(max_angle / spacing + BRIDGE_TOLERANCE)


def incidence_weights(
    params: FloatArray,
    points: FloatArray,
    directions: FloatArray,
    noise_std: float,
    cos_floor: float,
) -> FloatArray:
    """``(n,)`` residual standard deviations ``noise_std * max(|u_i . n_i|, cos_floor)``, with
    ``n_i`` the circle ``params = (c_x, c_y, rho)``'s outward normal at point ``i``."""
    return _weights(_normals(params, points)[1], directions, noise_std, cos_floor)


def _normals(params: FloatArray, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Each point's distance from the circle's centre, and the outward unit normal there."""
    offsets = points - params[:2]
    distances = np.sqrt(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)
    return distances, offsets / distances[:, None]


def _weights(
    normals: FloatArray, directions: FloatArray, noise_std: float, cos_floor: float
) -> FloatArray:
    cosines = np.abs(directions[:, 0] * normals[:, 0] + directions[:, 1] * normals[:, 1])
    return np.asarray(noise_std * np.maximum(cosines, cos_floor))


def _weighted_system(
    params: FloatArray,
    points: FloatArray,
    directions: FloatArray,
    noise_std: float,
    cos_floor: float,
) -> tuple[FloatArray, FloatArray]:
    """:func:`circle_residuals` with :func:`incidence_weights` at ``params``, sharing one pass."""
    distances, normals = _normals(params, points)
    weights = _weights(normals, directions, noise_std, cos_floor)
    return _residuals(params, distances, normals, weights)


def _residuals(
    params: FloatArray, distances: FloatArray, normals: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    jacobian = np.empty((len(distances), 3))
    jacobian[:, :2] = -normals / weights[:, None]
    jacobian[:, 2] = -1.0 / weights
    return (distances - params[2]) / weights, jacobian


def circle_residuals(
    params: FloatArray, points: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Weighted residuals ``(|p_i - c| - rho) / w_i`` and their ``(n, 3)`` Jacobian with respect
    to ``(c_x, c_y, rho)``, the weights held fixed: ``[-n_i^T, -1] / w_i``."""
    distances, normals = _normals(params, points)
    return _residuals(params, distances, normals, weights)


def _algebraic_fit(points: FloatArray) -> FloatArray | None:
    """Kasa's fit: least squares of ``2 c . p + (rho^2 - |c|^2) = |p|^2``, or None if degenerate."""
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    solution, _, rank, _ = np.linalg.lstsq(design, (points**2).sum(axis=1), rcond=None)
    if rank < 3:
        return None
    centre, offset = solution[:2], solution[2]
    squared = offset + float(centre @ centre)
    if not squared > 0.0:
        return None
    return np.array([centre[0], centre[1], math.sqrt(squared)])


def _line_ssr(sums: FloatArray) -> FloatArray:
    """Orthogonal sum of squares of the best line through point sets given by their running sums
    ``(count, sx, sy, sxx, sxy, syy)`` along the last axis: the scatter matrix's smaller
    eigenvalue."""
    count, sx, sy, sxx, sxy, syy = np.moveaxis(sums, -1, 0)
    a = sxx - sx * sx / count
    b = sxy - sx * sy / count
    c = syy - sy * sy / count
    return np.asarray((a + c) / 2.0 - np.sqrt(((a - c) / 2.0) ** 2 + b * b))


def _corner_ssr(points: FloatArray) -> float:
    """The smallest unweighted sum of squares of one line, or of two lines sharing one of the
    points between them (each with at least two points)."""
    x, y = points[:, 0], points[:, 1]
    terms = np.column_stack([np.ones(len(points)), x, y, x * x, x * y, y * y])
    prefix = np.cumsum(terms, axis=0)  # prefix[i]: points 0..i
    total = prefix[-1]
    best = float(_line_ssr(total))
    if len(points) >= 3:
        split = np.arange(1, len(points) - 1)  # the shared point
        before = _line_ssr(prefix[split])
        after = _line_ssr(total - prefix[split - 1])
        best = min(best, float((before + after).min()))
    return best


def fit_circle(
    points: ArrayLike, directions: ArrayLike, noise_std: float, settings: FeatureSettings
) -> CircleFeature | None:
    """The circle a segment's points lie on, if they plausibly do.

    Args:
        points: ``(n, 2)`` points in the robot's frame, seen from the origin.
        directions: ``(n, 2)`` unit directions of their beams.
        noise_std: The range noise's standard deviation, metres (> 0).
        settings: The fit's settings.

    Returns:
        The feature (``beams`` numbered ``0 .. n-1``), or None when the segment is too short,
        degenerate, fails a test, or the result is not sound.
    """
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = len(points)
    if count < settings.min_points or not (
        np.isfinite(points).all() and np.isfinite(directions).all()
    ):
        return None
    params = _algebraic_fit(points)
    ceiling = DIVERGED_RADIUS_FACTOR * settings.radius_range[1]
    if params is None or params[2] > ceiling:
        return None
    with np.errstate(all="ignore"):
        for _ in range(settings.max_iterations):
            residuals, jacobian = _weighted_system(
                params, points, directions, noise_std, settings.cos_floor
            )
            try:
                step = np.linalg.solve(jacobian.T @ jacobian, -(jacobian.T @ residuals))
            except np.linalg.LinAlgError:
                return None
            params = params + step
            if not (np.isfinite(params).all() and abs(params[2]) <= ceiling):
                return None  # a straight surface: the radius runs away
            if np.abs(step).max() < STEP_TOLERANCE:
                break
        residuals, jacobian = _weighted_system(
            params, points, directions, noise_std, settings.cos_floor
        )
    covariance = _sound_covariance(jacobian, settings.max_fit_std)
    if covariance is None:
        return None
    ssr = float(residuals @ residuals)
    low, high = settings.radius_range
    mean_range = float(np.hypot(points[:, 0], points[:, 1]).mean())
    if (
        ssr > chi2_quantile(settings.fit_quantile, count - 3)
        or not low <= params[2] <= high
        or float(np.hypot(params[0], params[1])) <= mean_range
    ):
        return None
    distances = np.hypot(*(points - params[:2]).T) - params[2]
    if _corner_ssr(points) + settings.corner_penalty * noise_std**2 < float(distances @ distances):
        return None
    return CircleFeature(params, covariance, np.arange(count, dtype=np.intp), ssr)


def _sound_covariance(jacobian: FloatArray, max_std: float) -> FloatArray | None:
    """``(J^T J)^-1`` when it is finite, positive definite and no standard deviation exceeds
    ``max_std``; otherwise None."""
    if not np.isfinite(jacobian).all():
        return None
    try:
        factor = np.linalg.cholesky(jacobian.T @ jacobian)
        inverse_factor = np.linalg.solve(factor, np.eye(3))
    except np.linalg.LinAlgError:
        return None
    covariance = inverse_factor.T @ inverse_factor
    if not np.isfinite(covariance).all() or np.sqrt(np.diag(covariance)).max() > max_std:
        return None
    return np.asarray((covariance + covariance.T) / 2.0)


def _closes_full_turn(sensor: RangeSensor) -> tuple[float, bool]:
    """The angle between neighbouring beams, and whether they close a full turn."""
    angles = np.unwrap(sensor.mount_angles)
    if len(angles) < 2:
        return 0.0, False
    spacing = float(np.median(np.abs(np.diff(angles))))
    return spacing, abs(spacing * len(angles) - 2.0 * math.pi) <= FULL_TURN_TOLERANCE


def extract_circles(
    sensor: RangeSensor, readings: ArrayLike, settings: FeatureSettings | None = None
) -> list[CircleFeature]:
    """The circles in one scan, each fitted to a segment of neighbouring readings.

    Args:
        sensor: The sensor that read the scan; its ``noise_std`` is the fits' range noise.
        readings: ``(N,)`` readings in metres, per :class:`RangeSensor`'s reading contract.
        settings: The extraction settings; None uses the defaults.

    Returns:
        The features in the order of their segments' first beams, ``beams`` indexing
        ``readings``.

    Raises:
        ValueError: ``readings`` is not ``(sensor.num_readings,)``, or the sensor's noise is 0.
    """
    settings = FeatureSettings() if settings is None else settings
    if not sensor.noise_std > 0.0:
        raise ValueError("extract_circles needs a sensor with positive noise_std.")
    points, usable, directions = scan_points(sensor, readings)
    ranges = np.asarray(readings, dtype=np.float64)
    spacing, wrap = _closes_full_turn(sensor)
    features = []
    for segment in segment_scan(points, ranges, usable, spacing, settings, wrap):
        fit = fit_circle(points[segment], directions[segment], sensor.noise_std, settings)
        if fit is not None:
            features.append(fit._replace(beams=segment))
    return features
