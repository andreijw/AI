"""A pure-pursuit path follower in the robot's own commands, and their mapping to env actions.

The robot is a unicycle (:func:`~rl_navigation.sim.kinematics.unicycle_step`) commanded by a
forward speed ``v`` in ``[0, v_max]`` (it has no reverse) and a turn rate ``omega`` in
``[-omega_max, omega_max]``, the command :meth:`~rl_navigation.sim.world.World.step` takes and the
navigation environment's actions encode. Per step, from the pose ``(x, y, theta)``:

1. The robot's position is projected onto the polyline, searching only the arc lengths
   ``[s_prev, s_prev + L + v_max dt]``: progress never runs backwards and cannot jump across a
   U-shaped path to a leg the robot is merely near. The target ``T`` is the first point beyond
   the projection whose distance from the robot is at least the lookahead ``L``, or the path's
   end when none is.
2. ``alpha = wrap(atan2(T - p) - theta)``. If ``|alpha|`` exceeds ``turn_in_place`` the robot turns
   on the spot: ``v = 0``, ``omega = sign(alpha) omega_max``.
3. Otherwise pure pursuit (Coulter, CMU-RI-TR-92-01, 1992): curvature ``kappa = 2 sin(alpha) / d``
   with ``d = |T - p|``, ``v = v_max``, ``omega = kappa v``; if ``|omega|`` exceeds ``omega_max``,
   both are scaled by ``omega_max / |omega|``, which keeps the curvature. At the end of the path
   (``d`` below 1e-9 m) the command is zero.

The arc's bulge off its chord is ``(d / 2) tan(alpha / 2)``: at most 0.017 m with ``L`` 0.15 m and
``turn_in_place`` 25 degrees. A discrete action space gets the nearest of its rows that a short
look-ahead finds safe (:func:`safe_discrete_action`); a continuous one the exact inverse of the
environment's mapping (:func:`continuous_action`).

Imports NumPy, ``rl_navigation.sim`` and ``rl_navigation.planning.grid`` only.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.planning.grid import PlanningGrid
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle

# Below this distance to the target, in metres, the robot is at the path's end.
END_TOLERANCE = 1e-9


class Command(NamedTuple):
    """A velocity command: forward speed ``v`` in m/s and turn rate ``omega`` in rad/s."""

    v: float
    omega: float


@dataclass(frozen=True)
class FollowerParams:
    """The follower's settings. ``v_max``, ``omega_max`` and ``dt`` are the robot's (the
    environment's ``v_max``, ``omega_max`` and ``dt``), with no defaults here.

    Args:
        v_max: Top forward speed in m/s (> 0).
        omega_max: Top turn rate in rad/s (> 0).
        dt: Control period in seconds (> 0).
        lookahead: ``L``, the distance to the target point in metres (> 0). The default, 0.15 m,
            was tuned on layouts 10000 to 10199 (2026-11-13) with continuous actions; discrete
            actions used 0.15 m too.
        turn_in_place: Bearing error in radians above which the robot turns on the spot, in
            ``(0, pi]``. The default, 25 degrees, was tuned with continuous actions; discrete
            actions used 45 degrees.

    Raises:
        ValueError: A value out of range or not finite.
    """

    v_max: float
    omega_max: float
    dt: float
    lookahead: float = 0.15
    turn_in_place: float = math.radians(25.0)

    def __post_init__(self) -> None:
        for name in ("v_max", "omega_max", "dt", "lookahead"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"FollowerParams {name} must be positive, got {value}.")
        if not 0.0 < self.turn_in_place <= math.pi:
            raise ValueError(
                f"FollowerParams turn_in_place must be in (0, pi], got {self.turn_in_place}."
            )


class PurePursuit:
    """Follows one polyline with pure pursuit (see the module docstring).

    Args:
        path: ``(M, 2)`` the polyline's points, ``M >= 2``, start first.
        params: The follower's settings.

    Raises:
        ValueError: Fewer than two points, or points that are not finite.
    """

    def __init__(self, path: ArrayLike, params: FollowerParams) -> None:
        points = np.array(path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError(f"PurePursuit needs (M, 2) points with M >= 2, got {points.shape}.")
        if not np.isfinite(points).all():
            raise ValueError("PurePursuit needs finite points.")
        self.params = params
        self._points = points
        self._starts = points[:-1]
        self._steps = np.diff(points, axis=0)
        self._lengths = np.hypot(self._steps[:, 0], self._steps[:, 1])
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._lengths)])
        self._progress = 0.0

    @property
    def path(self) -> FloatArray:
        """A copy of the polyline."""
        return self._points.copy()

    @property
    def length(self) -> float:
        """The polyline's length in metres."""
        return float(self._cumulative[-1])

    @property
    def progress(self) -> float:
        """Arc length of the robot's last projection onto the path, in metres; never decreases."""
        return self._progress

    def remaining(self) -> FloatArray:
        """``(K, 2)`` the path from the last projection on: that point, then the later vertices."""
        later = self._points[1:][self._cumulative[1:] > self._progress]
        return np.vstack([self._point_at(self._progress), later])

    def _segment_at(self, s: float) -> int:
        """The index of the segment holding arc length ``s`` (the last one at the end)."""
        last = len(self._lengths) - 1
        return int(np.clip(np.searchsorted(self._cumulative, s, side="right") - 1, 0, last))

    def _point_at(self, s: float) -> FloatArray:
        k = self._segment_at(s)
        t = 0.0 if self._lengths[k] == 0.0 else (s - self._cumulative[k]) / self._lengths[k]
        point: FloatArray = self._starts[k] + min(max(t, 0.0), 1.0) * self._steps[k]
        return point

    def _project(self, p: FloatArray) -> float:
        """The arc length in the window ``[s_prev, s_prev + L + v_max dt]`` nearest to ``p``."""
        low = self._progress
        high = min(low + self.params.lookahead + self.params.v_max * self.params.dt, self.length)
        seg_low, seg_high = self._cumulative[:-1], self._cumulative[1:]
        overlaps = (seg_high >= low) & (seg_low <= high)
        best_s, best_d = low, math.inf
        for k in np.flatnonzero(overlaps):
            length = self._lengths[k]
            if length == 0.0:
                continue
            t = float(np.dot(p - self._starts[k], self._steps[k])) / length**2
            t_min = max((low - seg_low[k]) / length, 0.0)
            t_max = min((high - seg_low[k]) / length, 1.0)
            t = min(max(t, t_min), t_max)
            d = float(np.hypot(*(self._starts[k] + t * self._steps[k] - p)))
            if d < best_d:
                best_s, best_d = float(seg_low[k] + t * length), d
        return max(best_s, low)

    def _target(self, p: FloatArray) -> FloatArray:
        """The first point at or beyond the projection at least ``L`` from ``p``, else the end."""
        lookahead = self.params.lookahead
        here = self._point_at(self._progress)
        if np.hypot(*(here - p)) >= lookahead:
            return here
        k0 = self._segment_at(self._progress)
        for k in range(k0, len(self._lengths)):
            length = self._lengths[k]
            if length == 0.0:
                continue
            start, step = self._starts[k], self._steps[k]
            t_from = max((self._progress - self._cumulative[k]) / length, 0.0) if k == k0 else 0.0
            # |start + t step - p|^2 = L^2; the larger root is where the path leaves the circle.
            offset = start - p
            a, b, c = (
                length**2,
                2.0 * float(np.dot(offset, step)),
                float(np.dot(offset, offset)) - lookahead**2,
            )
            # The part of the path from the projection on starts inside the circle, so the line
            # meets it; max() only absorbs rounding.
            t = (-b + math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
            if t_from <= t <= 1.0:
                target: FloatArray = start + t * step
                return target
        end: FloatArray = self._points[-1]
        return end

    def command(self, pose: ArrayLike) -> Command:
        """The command for this step from the robot's pose ``(x, y, theta)``."""
        x, y, theta = np.asarray(pose, dtype=np.float64).reshape(3)
        p = np.array([x, y])
        self._progress = self._project(p)
        target = self._target(p)
        distance = float(np.hypot(*(target - p)))
        if distance < END_TOLERANCE:
            return Command(0.0, 0.0)
        alpha = float(wrap_angle(math.atan2(target[1] - y, target[0] - x) - theta))
        params = self.params
        if abs(alpha) > params.turn_in_place:
            return Command(0.0, math.copysign(params.omega_max, alpha))
        v = params.v_max
        omega = 2.0 * math.sin(alpha) / distance * v
        if abs(omega) > params.omega_max:
            v *= params.omega_max / abs(omega)
            omega = math.copysign(params.omega_max, omega)
        return Command(v, omega)


def continuous_action(command: Command, v_max: float, omega_max: float) -> NDArray[np.float32]:
    """The continuous action that commands ``command``: the exact inverse of the navigation
    environment's ``v = v_max (a0 + 1) / 2``, ``omega = omega_max a1``, clipped to ``[-1, 1]``."""
    action = np.array([2.0 * command.v / v_max - 1.0, command.omega / omega_max])
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def safe_discrete_action(
    command: Command,
    rows: ArrayLike,
    v_max: float,
    omega_max: float,
    pose: ArrayLike,
    grid: PlanningGrid,
    dt: float,
    horizon: int = 3,
) -> int:
    """The discrete action nearest to ``command`` whose next ``horizon`` steps stay on free cells.

    Rows are tried in order of their Euclidean distance from ``(v / v_max, omega / omega_max)``
    (ties to the lower index). A row passes when it does not translate (``v = 0``: a spin cannot
    collide) or when each of ``horizon`` steps of :func:`~rl_navigation.sim.kinematics.unicycle_step`
    at its command ends in a free cell of ``grid``, which should be inflated by the robot's radius
    and half a cell at least. A spin always passes, so the nearest spin is the fallback.

    The look-ahead uses the commanded speeds with no slip, and checks only where each step ends.
    Both are acceptable here: the environment's slip is 0 by default, and slip only scales a
    command down, so the robot goes no farther than predicted (its arc may bend a little
    differently); and a step is at most ``v_max dt`` = 5 cm, one cell, so a step between two free
    end cells can at most clip the corner of a blocked one, which the half cell in the grid's
    inflation allows for.

    Args:
        command: The follower's command.
        rows: ``(A, 2)`` each action's ``(v / v_max, omega / omega_max)`` (the environment's
            ``DISCRETE_ACTIONS``).
        v_max: Top forward speed.
        omega_max: Top turn rate.
        pose: The robot's pose ``(x, y, theta)``.
        grid: The configuration-space grid to check against.
        dt: Step length in seconds.
        horizon: Steps to look ahead.

    Returns:
        The index of the chosen row.

    Raises:
        ValueError: No row passes and none is a spin.
    """
    fractions = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    wanted = np.array([command.v / v_max, command.omega / omega_max])
    order = np.argsort(np.hypot(*(fractions - wanted).T), kind="stable")
    start = np.asarray(pose, dtype=np.float64).reshape(3)
    for index in order:
        v, omega = fractions[index, 0] * v_max, fractions[index, 1] * omega_max
        if v == 0.0 or _steps_free(start, v, omega, grid, dt, horizon):
            return int(index)
    raise ValueError(
        "safe_discrete_action: no row passed the safety check and none is a spin; the rows "
        f"{fractions.tolist()} need one with v = 0."
    )


def _steps_free(
    pose: FloatArray, v: float, omega: float, grid: PlanningGrid, dt: float, horizon: int
) -> bool:
    """Whether every one of ``horizon`` steps at ``(v, omega)`` from ``pose`` ends on a free cell."""
    current = pose
    for _ in range(horizon):
        current = unicycle_step(current, v, omega, dt)
        try:
            cell = grid.cell_of(current[:2])
        except ValueError:
            return False
        if grid.blocked[cell]:
            return False
    return True
