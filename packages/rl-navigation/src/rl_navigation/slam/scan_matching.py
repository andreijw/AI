"""Lidar scan matching against the log-odds grid, and a scan-matching SLAM front end.

**The matcher** (:func:`match_points`) finds the pose that puts a scan's end points on the map's
occupied cells, as Hector SLAM does (Kohlbrecher et al., "A Flexible and Scalable SLAM System
with Full 3D Motion Estimation", 2011), on Thrun's likelihood field (*Probabilistic Robotics*,
2005, section 6.4): a :class:`~rl_navigation.mapping.fields.DistanceField` of the grid. With the
pose ``xi = (x, y, theta)``, the end points ``s_k`` in the robot's frame, ``p_k = R(theta) s_k +
t`` and ``D`` the field, it minimises over the beams ``B`` whose end points lie in the grid at
the prior (fixed for the whole match)

    E(xi) = 1/2 sigma_p^-2 sum_{k in B} min(D(p_k), d_max)^2 + 1/2 e^T Pbar^-1 e,
    e = xi (-) xibar (heading wrapped),

a truncated quadratic: a beam farther than ``d_max`` from every occupied cell is an outlier with
a constant cost. Gauss-Newton on the inliers (``D < d_max``), with
``J_k = [dD/dx, dD/dy, dD/dx (-(p_y - y)) + dD/dy (p_x - x)]``,
``H = sigma_p^-2 J^T J + Pbar^-1`` and ``b = sigma_p^-2 J^T r + Pbar^-1 e``, steps
``delta = -H^-1 b``, halving it while ``E`` rises. The field reaches ``d_max + res sqrt 2``, so
an inlier's four cell centres are never clamped and its gradient is exact. The covariance is
``H^-1`` at the result (Laplace), with ``J`` recomputed there. A match is accepted when it is
finite and positive definite, keeps enough inliers, fits them (inlier share and RMS) and moves
from the prior by at most the chi-square quantile in ``Pbar``'s metric. That gate is
conservative: the exact innovation covariance of a fused estimate is ``Pbar - P``, smaller, but
near-singular when the scan dominates.

**The front end** (:class:`ScanMatchingSlam`) is one pose and a grid: predict with the encoders
(:func:`~rl_navigation.estimation.ekf.predict_pose`, plus a pose-noise floor), match the scan
against the grid's field, and map the scan at the matched pose when the match is accepted. It has
no loop closure and no map uncertainty: its covariance is the pose's relative to the map, which
drifts unseen. The optional pose-noise floor keeps ``Pbar`` wide enough for the gate and the
prior after many scans of one map, whose information is not independent; it models no physical
process, and the tuned default is none (no scan was rejected without one).

Imports NumPy, ``rl_navigation.sim``, the ``estimation`` core and the ``mapping`` core.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.ekf import chi2_quantile, predict_pose
from rl_navigation.estimation.kalman import symmetrized
from rl_navigation.estimation.poses import checked_start, checked_wheel_travel, pose_difference
from rl_navigation.mapping.fields import DistanceField
from rl_navigation.mapping.grid import LIDAR_LOG_ODDS, LogOddsModel, OccupancyGrid
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import rotation, wrap_angle
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor, usable_readings
from rl_navigation.slam.ekf_slam import positive_definite

Termination = Literal["converged", "capped", "stalled"]


@dataclass(frozen=True)
class ScanMatchSettings:
    """The matcher's and the front end's settings.

    Args:
        resolution: Grid cell side in metres.
        max_distance: ``d_max``, the inlier truncation in metres.
        point_std: ``sigma_p``, one end point's distance noise in metres: the range noise, the
            cell's quantisation and the correlated half-cell bias of the map together.
        max_iterations: Gauss-Newton iterations at most.
        step_tolerance: A step below this in metres and radians on every axis has converged.
        max_halvings: Step halvings tried while the cost rises.
        min_inliers: Fewest inliers an accepted match keeps.
        min_inlier_share: Smallest share of the beams in the grid that are inliers.
        max_residual_rms: Largest RMS distance of the inliers in metres.
        gate_quantile: Chi-square quantile (3 degrees of freedom) of the move from the prior.
        pose_noise_floor: ``(3,)`` standard deviations in metres and radians added to the
            predicted covariance each step (see the module docstring).
        model: The grid's log-odds model.

    Raises:
        ValueError: A setting out of range.
    """

    resolution: float = 0.05
    max_distance: float = 0.15
    # SCAN-1's mean NEES is 3 at 3.3 cm on tuning seeds 1 to 50 (3.67 at 3 cm; it scales as
    # sigma^-2 under the wide prior).
    point_std: float = 0.033
    max_iterations: int = 10
    step_tolerance: float = 1e-4
    max_halvings: int = 4
    min_inliers: int = 30
    # On true-pose maps (SCAN-1, seeds 1 to 50) the inlier share is at least 0.971 and the RMS
    # at most 14.4 mm. The front end's own first map is one scan: at 30 % dropout it has gaps,
    # and the next scans' RMS is 30 to 38 mm (seed 1). A 30 mm limit rejected them all and 16 of
    # 50 runs were lost (SLAM-4 row b, seeds 1 to 50); 50 mm stays well under the RMS of a scan
    # that does not fit (about d_max / sqrt 3, 87 mm, for points spread over the band).
    min_inlier_share: float = 0.8
    max_residual_rms: float = 0.05
    gate_quantile: float = 0.999
    # Tuned on SLAM-4 row (a), seeds 1 to 50, for the smallest floor that keeps rejected scans
    # at 1 % or fewer: with none, 0 of 30,000 matches were rejected (ATE 10.6 mm mean; 13.3,
    # 13.8 and 14.1 mm with floors of 1, 2 and 5 mm or mrad).
    pose_noise_floor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    model: LogOddsModel = field(default=LIDAR_LOG_ODDS)

    def __post_init__(self) -> None:
        positive = {
            "resolution": self.resolution,
            "max_distance": self.max_distance,
            "point_std": self.point_std,
            "step_tolerance": self.step_tolerance,
        }
        for name, value in positive.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"ScanMatchSettings {name} must be positive, got {value}.")
        for name, count in (
            ("max_iterations", self.max_iterations),
            ("min_inliers", self.min_inliers),
        ):
            if count < 1:
                raise ValueError(f"ScanMatchSettings {name} must be at least 1, got {count}.")
        if self.max_halvings < 0:
            raise ValueError(
                f"ScanMatchSettings max_halvings must be at least 0, got {self.max_halvings}."
            )
        if not 0.0 <= self.min_inlier_share <= 1.0:
            raise ValueError(
                f"ScanMatchSettings min_inlier_share must be in [0, 1], got "
                f"{self.min_inlier_share}."
            )
        if not self.max_residual_rms > 0.0:
            raise ValueError(
                f"ScanMatchSettings max_residual_rms must be positive, got {self.max_residual_rms}."
            )
        if not 0.0 < self.gate_quantile < 1.0:
            raise ValueError(
                f"ScanMatchSettings gate_quantile must be in (0, 1), got {self.gate_quantile}."
            )
        floor = tuple(self.pose_noise_floor)
        if len(floor) != 3 or not all(math.isfinite(v) and v >= 0.0 for v in floor):
            raise ValueError(
                "ScanMatchSettings pose_noise_floor must be three finite values of at least 0, "
                f"got {self.pose_noise_floor}."
            )

    @property
    def field_reach(self) -> float:
        """How far the distance field is computed: ``max_distance + resolution sqrt 2``."""
        return self.max_distance + self.resolution * math.sqrt(2.0)


class MatchTerms(NamedTuple):
    """The residuals of ``K`` end points at one pose.

    Attributes:
        residuals: ``(K,)`` ``min(D(p_k), d_max)``; ``d_max`` for a point outside the grid.
        jacobian: ``(K, 3)`` their derivative with respect to ``(x, y, theta)``; zero rows for
            points that are not inliers.
        inliers: ``(K,)`` points inside the grid with ``D < d_max``.
    """

    residuals: FloatArray
    jacobian: FloatArray
    inliers: NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class ScanMatch:
    """One match's result: the matched pose and covariance, accepted or not.

    Attributes:
        pose: ``(3,)`` the matched pose (the prior when no beam lies in the grid).
        covariance: ``(3, 3)`` its covariance, ``H^-1`` (the prior's when no beam lies in it).
        accepted: Whether the match passed every check.
        reason: ``"accepted"`` or the first check it failed: ``"prior"`` (not positive definite),
            ``"no_beams"``, ``"not_finite"``, ``"few_inliers"``, ``"inlier_share"``,
            ``"residual_rms"``, ``"gate"``; the front end adds ``"off_bounds"``.
        termination: ``"converged"`` (the step taken, after any halving, was under the tolerance
            on every axis: a step halved down to that size also converges), ``"capped"``
            (``max_iterations`` reached) or ``"stalled"`` (no halving lowered the cost, or the
            normal equations could not be solved).
        iterations: Gauss-Newton steps taken.
        beams: Size of ``B``, the end points in the grid at the prior.
        inliers: ``(K,)`` which of the given points are inliers at the matched pose.
        residual_rms: RMS of the inliers' distances in metres (NaN without inliers).
        gate_distance: ``e^T Pbar^-1 e`` of the matched pose against the prior.
        costs: ``E`` at the prior and after each accepted step.
        halvings: How many times a step was halved because it raised ``E``, summed over the
            match's iterations (a stalled iteration's ``max_halvings + 1`` attempts count
            ``max_halvings + 1``).
    """

    pose: FloatArray
    covariance: FloatArray
    accepted: bool
    reason: str
    termination: Termination
    iterations: int
    beams: int
    inliers: NDArray[np.bool_]
    residual_rms: float
    gate_distance: float
    costs: tuple[float, ...]
    halvings: int = 0


def scan_endpoints(sensor: RangeSensor, readings: ArrayLike) -> tuple[FloatArray, NDArray[np.intp]]:
    """The usable readings' end points in the robot's frame, and which beams they are.

    Args:
        sensor: A sensor of one ray per mount (a lidar).
        readings: ``(N,)`` the scan (NaN allowed: unusable).

    Returns:
        ``(points, beams)``: ``(K, 2)`` end points ``mount + z (cos a, sin a)`` at the zero pose,
        and ``(K,)`` their indices in the scan.

    Raises:
        ValueError: A sensor with cones, or a scan of the wrong length.
    """
    if sensor.rays_per_cone != 1:
        raise ValueError(
            f"Scan matching needs one ray per mount (a lidar), got {sensor.rays_per_cone} rays "
            "per cone."
        )
    z = np.asarray(readings, dtype=np.float64)
    if z.shape != (sensor.num_readings,):
        raise ValueError(
            f"Scan matching needs readings of shape ({sensor.num_readings},), got {z.shape}."
        )
    beams = np.flatnonzero(usable_readings(z, sensor))
    zero = np.zeros((1, 3))
    origins = sensor.mount_origins(zero).reshape(-1, 2)[beams]
    angles = sensor.ray_angles(zero).reshape(-1)[beams]
    points = origins + z[beams, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    return points, beams


def match_residuals(
    field: DistanceField, points: ArrayLike, pose: ArrayLike, max_distance: float
) -> MatchTerms:
    """The truncated distances of end points placed at ``pose`` and their Jacobian.

    Args:
        field: The map's distance field, reaching at least ``max_distance``.
        points: ``(K, 2)`` end points in the robot's frame.
        pose: ``(3,)`` the pose to place them at.
        max_distance: ``d_max``.
    """
    local = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xi = np.asarray(pose, dtype=np.float64)
    rotated = local @ rotation(float(xi[2])).T  # (K, 2): R(theta) s_k = p_k - t
    sample = field.sample(rotated + xi[:2])
    inliers = sample.inside & (sample.values < max_distance)
    residuals = np.where(inliers, sample.values, max_distance)
    gx, gy = sample.gradients.T
    jacobian = np.column_stack([gx, gy, -gx * rotated[:, 1] + gy * rotated[:, 0]])
    jacobian[~inliers] = 0.0
    return MatchTerms(residuals, jacobian, inliers)


def match_points(
    field: DistanceField,
    points: ArrayLike,
    prior: ArrayLike,
    prior_covariance: ArrayLike,
    settings: ScanMatchSettings,
) -> ScanMatch:
    """Match end points to the map from a prior pose (see the module docstring).

    Args:
        field: The map's distance field, reaching at least ``settings.field_reach``.
        points: ``(K, 2)`` end points in the robot's frame.
        prior: ``(3,)`` the predicted pose.
        prior_covariance: ``(3, 3)`` its covariance.
        settings: The matcher's settings.

    Returns:
        The match. It never raises on the points' content.
    """
    local = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xibar = np.asarray(prior, dtype=np.float64).reshape(3)
    pbar = np.asarray(prior_covariance, dtype=np.float64).reshape(3, 3)
    d_max, weight = settings.max_distance, settings.point_std**-2
    none = np.zeros(len(local), dtype=bool)

    def unmatched(reason: str) -> ScanMatch:
        return ScanMatch(
            xibar.copy(), pbar.copy(), False, reason, "stalled", 0, 0, none, math.nan, 0.0, ()
        )

    if not (np.isfinite(xibar).all() and bool(positive_definite(pbar))):
        return unmatched("prior")
    information = symmetrized(np.linalg.inv(pbar))
    beams = field.sample(local @ rotation(float(xibar[2])).T + xibar[:2]).inside
    if not beams.any():
        return unmatched("no_beams")
    local = local[beams]

    def cost(xi: FloatArray) -> tuple[float, MatchTerms]:
        terms = match_residuals(field, local, xi, d_max)
        e = pose_difference(xi, xibar)
        return 0.5 * weight * float(terms.residuals @ terms.residuals) + 0.5 * float(
            e @ information @ e
        ), terms

    xi = xibar.copy()
    energy, terms = cost(xi)
    costs = [energy]
    termination: Termination = "capped"
    iterations = halvings = 0
    for _ in range(settings.max_iterations):
        iterations += 1
        J, r = terms.jacobian, terms.residuals
        H = weight * J.T @ J + information
        b = weight * J.T @ r + information @ pose_difference(xi, xibar)
        try:
            step = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            termination = "stalled"
            break
        for _ in range(settings.max_halvings + 1):
            candidate = xi + step
            candidate[2] = wrap_angle(candidate[2])
            candidate_energy, candidate_terms = cost(candidate)
            if candidate_energy <= energy:
                break
            step = 0.5 * step
            halvings += 1
        else:
            termination = "stalled"
            break
        xi, energy, terms = candidate, candidate_energy, candidate_terms
        costs.append(energy)
        if np.all(np.abs(step) < settings.step_tolerance):
            termination = "converged"
            break
    J = terms.jacobian[terms.inliers]
    H = weight * J.T @ J + information
    inliers = np.zeros(len(none), dtype=bool)
    inliers[beams] = terms.inliers
    count = int(terms.inliers.sum())
    rms = math.sqrt(float(np.mean(terms.residuals[terms.inliers] ** 2))) if count else math.nan
    e = pose_difference(xi, xibar)
    gate = float(e @ information @ e)
    try:
        covariance = symmetrized(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), np.nan)
    checks = (
        ("not_finite", np.isfinite(xi).all() and bool(positive_definite(covariance))),
        ("few_inliers", count >= settings.min_inliers),
        ("inlier_share", count >= settings.min_inlier_share * int(beams.sum())),
        ("residual_rms", rms <= settings.max_residual_rms),
        ("gate", gate <= chi2_quantile(settings.gate_quantile, 3)),
    )
    reason = next((name for name, ok in checks if not ok), "accepted")
    return ScanMatch(
        pose=xi,
        covariance=covariance,
        accepted=reason == "accepted",
        reason=reason,
        termination=termination,
        iterations=iterations,
        beams=int(beams.sum()),
        inliers=inliers,
        residual_rms=rms,
        gate_distance=gate,
        costs=tuple(costs),
        halvings=halvings,
    )


def match_scan(
    field: DistanceField,
    sensor: RangeSensor,
    readings: ArrayLike,
    prior: ArrayLike,
    prior_covariance: ArrayLike,
    settings: ScanMatchSettings,
) -> ScanMatch:
    """:func:`match_points` on a scan's usable end points (:func:`scan_endpoints`)."""
    points, _ = scan_endpoints(sensor, readings)
    return match_points(field, points, prior, prior_covariance, settings)


@dataclass(frozen=True, eq=False)
class ScanMatchUpdate:
    """What one :meth:`ScanMatchingSlam.update` did.

    Attributes:
        used: ``(N,)`` the beams that were inliers of an accepted match.
        match: The match, or None when there was nothing to match (the first mapped scan, or a
            scan without usable readings).
        mapped: Whether the scan was added to the grid.
    """

    used: NDArray[np.bool_]
    match: ScanMatch | None
    mapped: bool


class ScanMatchingSlam:
    """One pose and a log-odds grid, kept by scan matching (see the module docstring).

    It keeps the ``Localizer`` protocol (``pose``, ``covariance``, ``predict``, ``update``), so
    ``run_trial`` drives it like the localizers.

    Args:
        sensor: The lidar (one ray per mount).
        odometry: The encoders' model.
        start: ``(3,)`` the start estimate; the map is built in its frame.
        covariance: ``(3, 3)`` its covariance.
        shape: ``(rows, cols)`` of the grid.
        origin: ``(x0, y0)``, the grid's lower-left corner in metres.
        settings: The settings; None uses the defaults.
        bounds: ``(x_min, y_min, x_max, y_max)``; a match outside is rejected. None: no bounds.

    Raises:
        ValueError: A sensor with cones, a bad start or covariance, or a bad grid.
    """

    def __init__(
        self,
        sensor: RangeSensor,
        odometry: WheelOdometry,
        start: ArrayLike,
        covariance: ArrayLike,
        shape: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
        settings: ScanMatchSettings | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.settings = ScanMatchSettings() if settings is None else settings
        scan_endpoints(sensor, np.full(sensor.num_readings, np.nan))  # refuses cones
        self._pose, self._covariance = checked_start(start, covariance, "ScanMatchingSlam")
        self.sensor = sensor
        self.odometry = odometry
        self.bounds = bounds
        self._grid = OccupancyGrid(
            sensor, shape, self.settings.resolution, self.settings.model, origin
        )
        self._floor = np.diag(np.square(np.asarray(self.settings.pose_noise_floor, dtype=float)))
        self._field: DistanceField | None = None
        self._field_of = np.zeros(self._grid.shape, dtype=bool)
        self._mapped_any = False

    @property
    def pose(self) -> FloatArray:
        """``(3,)`` copy of the pose estimate."""
        return self._pose.copy()

    @property
    def covariance(self) -> FloatArray:
        """``(3, 3)`` copy of its covariance."""
        return self._covariance.copy()

    @property
    def grid(self) -> OccupancyGrid:
        """The map (read its arrays; :meth:`update` is the only writer)."""
        return self._grid

    @property
    def field(self) -> DistanceField | None:
        """The distance field of the map as last matched against, or None before any match."""
        return self._field

    def predict(self, wheel_travel: ArrayLike) -> None:
        """Move the pose by one encoder reading: ``P' = F P F^T + (G A) Sigma_w (G A)^T + Q_floor``.

        Raises:
            ValueError: ``wheel_travel`` is not two finite numbers.
        """
        travel = checked_wheel_travel(wheel_travel, "ScanMatchingSlam.predict")
        prediction = predict_pose(self._pose, travel, self.odometry)
        F = prediction.F
        self._pose = prediction.pose
        self._covariance = symmetrized(F @ self._covariance @ F.T + prediction.noise) + self._floor

    def _current_field(self) -> DistanceField:
        occupied = self._grid.occupied()
        if self._field is None or not np.array_equal(occupied, self._field_of):
            self._field = DistanceField.from_occupied(
                occupied, self.settings.resolution, self.settings.field_reach, self._grid.origin
            )
            self._field_of = occupied
        return self._field

    def _inside(self, pose: FloatArray) -> bool:
        if self.bounds is None:
            return True
        x_min, y_min, x_max, y_max = self.bounds
        return bool(x_min <= pose[0] <= x_max and y_min <= pose[1] <= y_max)

    def update(self, readings: ArrayLike) -> ScanMatchUpdate:
        """Match one scan, and map it at the matched pose when the match is accepted.

        The first scan with a usable reading is mapped at the current estimate. A scan without
        one, or a rejected match, keeps the prediction and leaves the map unchanged.

        Raises:
            ValueError: A scan of the wrong length (never on its content).
        """
        points, beams = scan_endpoints(self.sensor, readings)
        used = np.zeros(self.sensor.num_readings, dtype=bool)
        if len(points) == 0:
            return ScanMatchUpdate(used, None, False)
        if not self._mapped_any:
            self._grid.update(self._pose, readings)
            self._mapped_any = True
            return ScanMatchUpdate(used, None, True)
        match = match_points(
            self._current_field(), points, self._pose, self._covariance, self.settings
        )
        if match.accepted and not self._inside(match.pose):
            match = replace(match, accepted=False, reason="off_bounds")
        if not match.accepted:
            return ScanMatchUpdate(used, match, False)
        self._pose, self._covariance = match.pose.copy(), match.covariance.copy()
        self._grid.update(self._pose, readings)
        used[beams[match.inliers]] = True
        return ScanMatchUpdate(used, match, True)
