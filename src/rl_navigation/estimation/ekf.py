"""Localization on a known map with an extended Kalman filter over wheel odometry and ranges.

The filter follows the EKF localization of Thrun, Burgard and Fox, *Probabilistic Robotics*
(Table 7.2), with four differences:

- The control is the measured wheel travel ``(left, right)``, with Siegwart's per-wheel noise
  (:class:`~rl_navigation.sim.WheelOdometry`), not ``(v, omega)`` with Thrun's alpha noise,
  which is undefined for a turn in place.
- A measurement is the full ray-cast range of each beam against the map (:class:`RangeMap`),
  not a landmark feature. Its Jacobian is a central difference at ``fd_step``, because a beam's
  range is piecewise: a corner, a circle's edge or the cone's nearest ray switches it. Each beam
  is also cast at ``+-linearity_window`` along each pose axis; one whose casts leave the linear
  prediction by more than ``linearity_tol`` is not used (``nonlinear``).
- The update is a batch over the usable beams, each gated on its own normalised innovation,
  and the covariance is updated in Joseph form (:func:`kalman_update`).
- The update is iterated as Gauss-Newton (Bell and Cathey, 1993): from the prior ``(x0, P)``,
  ``x_{i+1} = x0 + K_i (z - h(x_i) - H_i (x0 - x_i))`` with ``K_i`` from ``H_i`` at ``x_i``, the
  beams chosen and gated at the first linearization. An iterate whose step from the last one
  exceeds ``linearity_window`` in any component is refused and the last one kept. Each iterate
  is linearized with the same check as the prior: a beam that fails it there (a corner within
  the window, or a non-finite cast) leaves the remaining iterations, since its central
  difference could be huge and collapse ``P``; with no beam left the last iterate stands. ``P``
  takes the last accepted ``K`` and ``H``. One iteration is the plain EKF. Relinearizing matters
  most after a large error: a single update linearized centimetres off leaves ``P``
  overconfident.
- An update is *held* when it did not converge (an iterate's step was refused) or when its
  normalised innovation squared exceeds the ``nis_quantile`` quantile of chi-square with as many
  degrees of freedom as used beams: its mean moves to the last accepted iterate, but ``P`` stays
  the prior, since a covariance from a linearization the scan contradicts would claim a precision
  it does not have (after a reset it collapsed to millimetres round a pose metres off).

**Frames and units.** The pose ``(x, y, theta)`` is the robot's axle midpoint (its centre) in the
map's frame, metres and radians, ``theta`` in ``[-pi, pi)``. Wheel travel is in metres. A scan
passed to :meth:`EkfLocalizer.update` is taken at the pose reached by the last
:meth:`EkfLocalizer.predict`.
"""

import statistics
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.kalman import kalman_update, symmetrized
from rl_navigation.estimation.maps import RangeMap, usable_readings
from rl_navigation.estimation.poses import (
    checked_axis_values,
    checked_start,
    checked_wheel_travel,
    pose_difference,
)
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.odometry import WheelOdometry

# Below this |Delta theta / 2| the derivative of sin(h) / h uses its series -h/3 + h^3/30 (error
# under h^5/840, about 1e-13), where the closed form (h cos h - sin h) / h^2 cancels badly.
SERIES_HALF_TURN = 1e-2

# A beam whose expected range is within this many noise deviations of the sensor's maximum would
# often read as censored (no echo), so it is not used.
CENSOR_MARGIN_STDS = 3.0

# +-1 along each pose axis in turn: (x+, x-, y+, y-, theta+, theta-).
_SIGNED_AXES = np.repeat(np.eye(3), 2, axis=0) * np.tile([1.0, -1.0], 3)[:, None]


def motion_jacobians(
    pose: ArrayLike, distance: float, turn: float
) -> tuple[FloatArray, FloatArray]:
    """Jacobians of ``unicycle_step(pose, distance, turn, 1.0)``, the exact arc of an odometry step.

    With ``h = turn / 2``, ``s = sin(h) / h``, ``m = theta + h`` and ``c = distance s``:
    ``F = [[1, 0, -c sin m], [0, 1, c cos m], [0, 0, 1]]`` with respect to the pose, and
    ``G = [[s cos m, distance (s' cos m - s/2 sin m)], [s sin m, distance (s' sin m + s/2 cos m)],
    [0, 1]]`` with respect to ``(distance, turn)``, where ``s' = ds / d turn``.

    Args:
        pose: ``(3,)`` pose before the step.
        distance: ``Delta s``, metres along the arc.
        turn: ``Delta theta``, radians.

    Returns:
        ``(F, G)``, shapes ``(3, 3)`` and ``(3, 2)``.
    """
    theta = float(np.asarray(pose, dtype=np.float64)[2])
    h = turn / 2.0
    s = float(np.sinc(h / np.pi))
    if abs(h) < SERIES_HALF_TURN:
        s_prime = 0.5 * (-h / 3.0 + h**3 / 30.0)
    else:
        s_prime = 0.5 * (h * np.cos(h) - np.sin(h)) / h**2
    cos_m, sin_m = np.cos(theta + h), np.sin(theta + h)
    chord = distance * s
    F = np.array([[1.0, 0.0, -chord * sin_m], [0.0, 1.0, chord * cos_m], [0.0, 0.0, 1.0]])
    G = np.array(
        [
            [s * cos_m, distance * (s_prime * cos_m - s / 2.0 * sin_m)],
            [s * sin_m, distance * (s_prime * sin_m + s / 2.0 * cos_m)],
            [0.0, 1.0],
        ]
    )
    return F, G


def chi2_quantile(q: float, dof: int) -> float:
    """The ``q`` quantile of chi-square with ``dof`` degrees of freedom, by Wilson and Hilferty's
    cube-root normal approximation: ``k (1 - 2/(9k) + z sqrt(2/(9k)))^3`` with ``z`` the normal
    quantile. At 0.999 it is 3.0 % above the exact value for one degree of freedom (11.16 against
    10.83) and closer for more (16.55 against 16.27 for three, 33.07 against 32.91 for twelve).

    Raises:
        ValueError: ``q`` not strictly between 0 and 1, or ``dof`` below 1.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"chi2_quantile needs a quantile strictly between 0 and 1, got {q}.")
    if dof < 1:
        raise ValueError(f"chi2_quantile needs at least 1 degree of freedom, got {dof}.")
    z = statistics.NormalDist().inv_cdf(q)
    spread = 2.0 / (9.0 * dof)
    return float(dof * (1.0 - spread + z * spread**0.5) ** 3)


class Linearization(NamedTuple):
    """The expected ranges at a pose and their Jacobian.

    ``h`` is ``(N,)``; ``H`` is ``(N, 3)``; ``nonlinear`` is ``(N,)``, True for a beam whose range
    is not linear within the window or whose casts are not finite.
    """

    h: FloatArray
    H: FloatArray
    nonlinear: NDArray[np.bool_]


def _central_jacobian(ranges: FloatArray, fd_step: float) -> FloatArray:
    """``(N, 3)`` central differences from the six ``fd_step`` probes in ``_SIGNED_AXES`` order."""
    return np.asarray(((ranges[0::2] - ranges[1::2]) / (2.0 * fd_step)).T)


def linearize_ranges(
    range_map: RangeMap,
    pose: ArrayLike,
    fd_step: float = 1e-4,
    linearity_window: float = 0.03,
    linearity_tol: float | None = None,
) -> Linearization:
    """Expected ranges at ``pose``, their Jacobian, and which beams are linear enough to use.

    One batched cast of 13 poses: the pose, ``+-fd_step`` along each axis for ``H``, and
    ``+-linearity_window`` along each axis, where a beam is ``nonlinear`` when any cast departs
    from ``h +- linearity_window H`` by more than ``linearity_tol``.

    Args:
        range_map: The map and sensor.
        pose: ``(3,)`` pose.
        fd_step: Finite-difference step in metres and radians.
        linearity_window: Distance in metres and radians over which a beam must stay linear.
        linearity_tol: Largest departure from linear in metres; None uses the map's noise.

    Returns:
        The linearization.
    """
    tol = range_map.noise_std if linearity_tol is None else linearity_tol
    pose = np.asarray(pose, dtype=np.float64).reshape(3)
    offsets = np.concatenate(
        [np.zeros((1, 3)), fd_step * _SIGNED_AXES, linearity_window * _SIGNED_AXES]
    )
    with np.errstate(invalid="ignore"):  # inf - inf where a probe pose sees nothing
        ranges = range_map.expected_ranges(pose + offsets)
        h = ranges[0]
        H = _central_jacobian(ranges[1:7], fd_step)
        predicted = h + linearity_window * (_SIGNED_AXES @ H.T)  # (6, N)
        linear = np.all(np.abs(ranges[7:] - predicted) <= tol, axis=0)
    # A non-finite cast or slope makes its departure inf or NaN, which fails the comparison.
    return Linearization(h, H, ~linear)


@dataclass(frozen=True)
class UpdateResult:
    """What one :meth:`EkfLocalizer.update` did with each beam.

    Attributes:
        used: ``(N,)`` beams in the last accepted iteration, which corrected the pose.
        gated: ``(N,)`` usable beams rejected as outliers by the innovation gate.
        nonlinear: ``(N,)`` beams not linear within the window, or with non-finite casts, at
            the prior or at an iterate.
        nis: Normalised innovation squared of the used beams at the first linearization (0 when
            none was used).
        iterations: Accepted iterations (0 when no beam was used).
        reset: Whether this scan completed ``reset_after`` consecutive counting scans (mostly
            gated, or held) and raised the covariance (see :class:`EkfLocalizer`).
        held: Whether the update was held: it did not converge, or its NIS exceeded the
            ``nis_quantile`` quantile, so the mean moved but the covariance was kept.
    """

    used: NDArray[np.bool_]
    gated: NDArray[np.bool_]
    nonlinear: NDArray[np.bool_]
    nis: float
    iterations: int
    reset: bool = False
    held: bool = False


class EkfLocalizer:
    """The robot's pose and its covariance, from wheel odometry and range scans on a known map.

    **The reset.** A still robot's covariance collapses to a fraction of a millimetre; bumped a
    few centimetres, every beam then fails the gate and the filter never moves again. A scan
    *counts* when it has at least ``min_candidates`` candidate beams (used after the last
    iteration, or gated) and either at least half of them are gated or its update was held (see
    the module docstring). After ``reset_after`` consecutive counting scans the filter raises
    each variance to at least ``reset_std**2`` (``P[i, i] <- max(P[i, i], reset_std[i]^2)``,
    off-diagonals kept), and the count starts again. The floor is idempotent, so a filter that
    keeps resetting does not grow ``P`` without bound. :attr:`recent_resets` counts the resets
    in the last ``reset_window`` scans: **2 or more means the filter is lost** (it keeps
    resetting and cannot find a pose that explains the scans), 0 that the last reset, if any,
    recovered it. Because a held scan counts, a filter whose scans keep contradicting it keeps
    resetting, and so keeps reporting itself lost, until a scan converges and agrees.

    Limits: in a symmetric room a reset can lock onto a pose that explains the scans as well as
    the true one (a 90 degree alias of a square room); its covariance shrinks again and
    :attr:`recent_resets` falls to 0, so the filter is wrong and quiet. The particle filter has
    the same limit. A reset needs a scan that contradicts the estimate: a heading bump of
    0.3 rad on the HC-SR04's five sonars fires none, because three of its sonars change by under
    3 cm and stay used, one turns nonlinear and only one is gated. A wrong association that a
    scan explains (a beam linearized on the wrong wall near a corner, consistent with its prior)
    is not caught either; the hold acts on the next scan that contradicts it.

    Args:
        range_map: The known map seen through the robot's range sensor; its ``noise_std`` is the
            range noise ``sigma_r`` (> 0).
        odometry: The wheel encoders' model (wheel base and noise).
        pose: ``(3,)`` initial pose estimate.
        covariance: ``(3, 3)`` symmetric positive-definite initial covariance.
        iterations: Gauss-Newton iterations of the update (1 is the plain EKF).
        fd_step: Finite-difference step of the range Jacobian, metres and radians.
        linearity_window: Half-width of the linearity check, and the longest accepted step of an
            iteration, metres and radians.
        linearity_tol: Largest departure from linear, metres; None uses ``sigma_r``.
        gate: A beam with ``innovation^2 / (H P H^T + sigma_r^2)`` at or above this is an outlier
            (9: three standard deviations).
        reset_after: Consecutive counting scans that trigger the reset (at least 1); None
            disables it.
        min_candidates: Fewest candidate beams for a scan to count (at least 1). Two keeps a
            sonar scan with dropouts, one beam of one gated, from counting.
        reset_std: ``(3,)`` positive floor of the standard deviations ``(x, y, theta)`` after a
            reset, metres and radians.
        reset_window: Scans over which :attr:`recent_resets` counts (at least 1).
        nis_quantile: The chi-square quantile above which an update's NIS holds it (in (0, 1));
            0.999 holds about 0.5 % of scans in tracking. None turns the hold off entirely, a
            refused iteration included: the filter as it was before the hold.

    Raises:
        ValueError: A pose or covariance of the wrong shape or not finite, a covariance that is
            not symmetric positive-definite, a sensor without noise, or a setting out of range.
    """

    def __init__(
        self,
        range_map: RangeMap,
        odometry: WheelOdometry,
        pose: ArrayLike,
        covariance: ArrayLike,
        *,
        iterations: int = 3,
        fd_step: float = 1e-4,
        linearity_window: float = 0.03,
        linearity_tol: float | None = None,
        gate: float = 9.0,
        reset_after: int | None = 3,
        min_candidates: int = 2,
        reset_std: ArrayLike = (0.2, 0.2, 0.1),
        reset_window: int = 20,
        nis_quantile: float | None = 0.999,
    ) -> None:
        pose, covariance = checked_start(pose, covariance, "EkfLocalizer")
        if not range_map.noise_std > 0.0:
            raise ValueError(
                f"EkfLocalizer needs a sensor with positive range noise_std, got "
                f"{range_map.noise_std}: the update divides by it."
            )
        settings = {"fd_step": fd_step, "linearity_window": linearity_window, "gate": gate}
        if linearity_tol is not None:
            settings["linearity_tol"] = linearity_tol
        if nis_quantile is not None and not 0.0 < nis_quantile < 1.0:
            raise ValueError(
                f"EkfLocalizer nis_quantile must be in (0, 1) or None, got {nis_quantile}."
            )
        self.nis_quantile = nis_quantile
        for name, value in settings.items():
            if not value > 0.0:
                raise ValueError(f"EkfLocalizer {name} must be positive, got {value}.")
        whole_numbers = {
            "iterations": iterations,
            "min_candidates": min_candidates,
            "reset_window": reset_window,
        }
        if reset_after is not None:
            whole_numbers["reset_after"] = reset_after
        for name, value in whole_numbers.items():
            if isinstance(value, bool) or not (isinstance(value, int | np.integer) and value >= 1):
                raise ValueError(
                    f"EkfLocalizer {name} must be an integer of at least 1, got {value}."
                )
        self.reset_std = checked_axis_values(reset_std, "EkfLocalizer", "reset_std")
        self.reset_after = None if reset_after is None else int(reset_after)
        self.min_candidates = int(min_candidates)
        self.reset_window = int(reset_window)
        self._counting_scans = 0
        self._recent: deque[bool] = deque(maxlen=self.reset_window)
        self.range_map = range_map
        self.odometry = odometry
        self.iterations = int(iterations)
        self.fd_step = float(fd_step)
        self.linearity_window = float(linearity_window)
        self.linearity_tol = linearity_tol
        self.gate = float(gate)
        self._pose = pose
        self._covariance = covariance

    @property
    def pose(self) -> FloatArray:
        """A copy of the estimated pose ``(x, y, theta)``."""
        return self._pose.copy()

    @property
    def covariance(self) -> FloatArray:
        """A copy of the ``(3, 3)`` pose covariance."""
        return self._covariance.copy()

    @property
    def recent_resets(self) -> int:
        """Resets within the last ``reset_window`` scans; 2 or more means the filter is lost."""
        return sum(self._recent)

    def _counted_reset(
        self, used: NDArray[np.bool_], gated: NDArray[np.bool_], held: bool = False
    ) -> bool:
        """Count this scan toward the reset and apply the floor if it completes the run."""
        if self.reset_after is None:
            return False
        num_gated = int(np.count_nonzero(gated))
        candidates = int(np.count_nonzero(used)) + num_gated
        counts = candidates >= self.min_candidates and (held or 2 * num_gated >= candidates)
        self._counting_scans = self._counting_scans + 1 if counts else 0
        reset = self._counting_scans >= self.reset_after
        if reset:
            self._counting_scans = 0
            covariance = self._covariance.copy()
            diagonal = np.diag_indices(3)
            covariance[diagonal] = np.maximum(covariance[diagonal], self.reset_std**2)
            self._covariance = covariance
        self._recent.append(reset)
        return reset

    def predict(self, wheel_travel: ArrayLike) -> None:
        """Move the estimate by one encoder reading and grow its covariance.

        ``P' = F P F^T + (G A) Sigma_w (G A)^T``, where ``A`` maps wheel travel to distance and
        turn and ``Sigma_w`` is the reading's covariance (see :func:`motion_jacobians`).

        Args:
            wheel_travel: ``(2,)`` finite ``(left, right)`` wheel travel in metres since the last
                call, per :class:`~rl_navigation.sim.WheelOdometry`'s reading contract.

        Raises:
            ValueError: ``wheel_travel`` is not two finite numbers.
        """
        travel = checked_wheel_travel(wheel_travel, "EkfLocalizer.predict")
        distance, turn = self.odometry.increments(travel)
        F, G = motion_jacobians(self._pose, distance, turn)
        GA = G @ self.odometry.increments_jacobian()
        covariance = F @ self._covariance @ F.T + GA @ self.odometry.wheel_covariance(travel) @ GA.T
        self._pose = unicycle_step(self._pose, distance, turn, 1.0)
        self._covariance = symmetrized(covariance)

    def update(self, readings: ArrayLike) -> UpdateResult:
        """Correct the estimate with one scan.

        A reading is unused when it is not finite (a driver's NaN is a dropout), at or beyond
        ``max_range`` (no echo), at or below ``min_range`` (clipped), when its expected range is
        within ``3 sigma_r`` of ``max_range``, when its beam is not linear, or when the gate
        rejects it. With nothing used the pose and covariance are left exactly as they were,
        unless the scan completes a reset, which raises the covariance's diagonal. A held update
        moves the pose but keeps the covariance (see the module docstring).

        Args:
            readings: ``(N,)`` scan in the sensor's mount order, metres.

        Returns:
            What happened to each beam.

        Raises:
            ValueError: The scan's length is not the sensor's.
        """
        z = np.asarray(readings, dtype=np.float64)
        rmap = self.range_map
        lin = linearize_ranges(
            rmap, self._pose, self.fd_step, self.linearity_window, self.linearity_tol
        )
        if z.shape != lin.h.shape:
            raise ValueError(
                f"EkfLocalizer.update got readings of shape {z.shape}; the sensor has "
                f"{len(lin.h)} beams."
            )
        sigma2 = rmap.noise_std**2
        candidates = (  # NaN compares False, so a NaN reading is never a candidate
            usable_readings(z, rmap)
            & ~lin.nonlinear
            & (
                np.where(lin.nonlinear, np.inf, lin.h)
                < rmap.max_range - CENSOR_MARGIN_STDS * rmap.noise_std
            )
        )
        innovation = np.where(candidates, z - np.where(candidates, lin.h, 0.0), 0.0)
        H = np.where(candidates[:, None], lin.H, 0.0)
        variance = np.einsum("ij,jk,ik->i", H, self._covariance, H) + sigma2
        gated = candidates & (innovation**2 >= self.gate * variance)
        used = candidates & ~gated
        if not used.any():
            return UpdateResult(
                used, gated, lin.nonlinear, 0.0, 0, self._counted_reset(used, gated)
            )
        x0, P0 = self._pose, self._covariance
        H_i, innovation_i = H[used], innovation[used]
        nis = float(
            innovation_i
            @ np.linalg.solve(H_i @ P0 @ H_i.T + sigma2 * np.eye(len(H_i)), innovation_i)
        )
        x_i, P_i, accepted = x0, P0, 0
        active = final = used
        refused = False
        while True:
            offset = pose_difference(x_i, x0)
            x_next, P_next = kalman_update(
                x0, P0, innovation_i + H_i @ offset, H_i, sigma2 * np.eye(len(H_i))
            )
            x_next[2] = wrap_angle(x_next[2])
            if accepted:
                step = pose_difference(x_next, x_i)
                if np.any(np.abs(step) > self.linearity_window):
                    refused = True
                    break
            x_i, P_i, accepted, final = x_next, P_next, accepted + 1, active
            if accepted == self.iterations:
                break
            # Each iterate is checked like the prior: a beam on a corner there would give a huge
            # central difference and collapse P, so it leaves the remaining iterations.
            relinearized = linearize_ranges(
                rmap, x_i, self.fd_step, self.linearity_window, self.linearity_tol
            )
            active = active & ~relinearized.nonlinear
            if not active.any():
                break
            H_i, innovation_i = relinearized.H[active], z[active] - relinearized.h[active]
        held = self.nis_quantile is not None and (
            refused or nis > chi2_quantile(self.nis_quantile, int(np.count_nonzero(used)))
        )
        self._pose, self._covariance = x_i, P0 if held else P_i
        return UpdateResult(
            final,
            gated,
            lin.nonlinear | (used & ~final),
            nis,
            accepted,
            self._counted_reset(final, gated, held),
            held,
        )
