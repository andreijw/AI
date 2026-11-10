"""A log-odds occupancy grid built from range scans taken at known poses.

The update is ``occupancy_grid_mapping`` of Thrun, Burgard and Fox, *Probabilistic Robotics*
(2005), Table 9.1, with a fixed prior of 0.5 (log-odds 0): each cell adds the inverse sensor
model's evidence and is clamped, as OctoMap does (Hornung et al., *Autonomous Robots* 34(3), 2013):

    l_i <- clip(l_i + ism_i, l_min, l_max)

The inverse sensor model, per scan (:func:`rl_navigation.mapping.beams.endpoint_visits`): a cell
holding some usable ray's end point gets ``l_occ``; otherwise a cell some usable ray crosses wholly
before its end point gets ``l_free``; every other cell is unchanged. Each cell is updated at most
once per scan, and occupied wins over free, as in OctoMap. Each of a cone's rays is traced from its
mount with the mount's single reading. A reading without a range (a dropout, "nothing within
range", a reading clipped at ``min_range``, NaN; :func:`rl_navigation.sim.sensors.usable_readings`)
changes nothing: a dropout reads ``max_range`` too, so free space is never carved from one.

Imports NumPy and ``rl_navigation.sim`` only, so it runs on the robot.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.mapping.beams import endpoint_visits, sensor_rays, traverse_cells
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.sensors import RangeSensor, usable_readings


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class LogOddsModel:
    """How much one scan moves a cell, in log-odds ``ln(p / (1 - p))``.

    Args:
        l_occ: Evidence of a hit (> 0).
        l_free: Evidence of a miss (< 0).
        l_min: Lower clamp (< 0).
        l_max: Upper clamp (> 0).

    Raises:
        ValueError: A value of the wrong sign, or one that is not finite.
    """

    l_occ: float
    l_free: float
    l_min: float
    l_max: float

    def __post_init__(self) -> None:
        values = (self.l_occ, self.l_free, self.l_min, self.l_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"LogOddsModel needs finite values, got {values}.")
        for name, ok in (
            ("l_occ", self.l_occ > 0.0),
            ("l_free", self.l_free < 0.0),
            ("l_min", self.l_min < 0.0),
            ("l_max", self.l_max > 0.0),
        ):
            if not ok:
                raise ValueError(
                    f"LogOddsModel needs l_occ > 0, l_free < 0, l_min < 0 < l_max; {name} is "
                    f"{getattr(self, name)}."
                )

    @classmethod
    def from_probabilities(
        cls, p_hit: float, p_miss: float, p_min: float = 0.12, p_max: float = 0.97
    ) -> "LogOddsModel":
        """The model whose hit, miss and clamps are these occupancy probabilities.

        Args:
            p_hit: Occupancy probability of a cell a reading ends in, in (0.5, 1).
            p_miss: Occupancy probability of a cell a ray crosses, in (0, 0.5).
            p_min: Lower clamp, in (0, 0.5); OctoMap's 0.12 by default.
            p_max: Upper clamp, in (0.5, 1); OctoMap's 0.97 by default.

        Raises:
            ValueError: A probability outside its range.
        """
        for name, p, low, high in (
            ("p_hit", p_hit, 0.5, 1.0),
            ("p_miss", p_miss, 0.0, 0.5),
            ("p_min", p_min, 0.0, 0.5),
            ("p_max", p_max, 0.5, 1.0),
        ):
            if not low < p < high:
                raise ValueError(f"LogOddsModel {name} must be in ({low}, {high}), got {p}.")
        return cls(_logit(p_hit), _logit(p_miss), _logit(p_min), _logit(p_max))


# The lidar's single rays: OctoMap's hit and clamps, with a weaker miss (0.45 for its 0.4), because
# grazing beams on the way to farther surfaces otherwise free the cells a surface crosses
# (visible-surface recall 0.78 at 0.4 and 0.93 at 0.45 on tuning seeds 1 to 10, precision 1.00).
# One hit makes a cell occupied, 5 clamp it, and 18 misses free it again.
LIDAR_LOG_ODDS = LogOddsModel.from_probabilities(0.7, 0.45)
# The HC-SR04's 5 rays per cone share one reading, the nearest over the cone, so only about a third
# of their end points lie on a surface (31.9 to 38.7 % on seeds 1 to 5): a weaker hit (precision
# 0.84 and recall 0.82 on seeds 1 to 10; the maintainer's choice, 2026-11-09).
HC_SR04_LOG_ODDS = LogOddsModel.from_probabilities(0.6, 0.45)


class OccupancyGrid:
    """A log-odds occupancy grid of one robot's range sensor.

    Cell ``[i, j]`` spans ``x0 + [j, j + 1) resolution`` and ``y0 + [i, i + 1) resolution``, as in
    :func:`rl_navigation.sim.geometry.occupancy_grid`, so a grid at origin ``(0, 0)`` of that
    function's shape compares cell by cell with the true map.

    Args:
        sensor: The sensor whose scans :meth:`update` takes; its mounts and cones are the rays.
        shape: ``(rows, cols)``, both positive.
        resolution: Cell side in metres (> 0).
        model: The evidence per scan and the clamps.
        origin: ``(x0, y0)``, the grid's lower-left corner in metres.
        log_odds: An initial map of ``shape`` (a saved one), copied and clamped; its non-zero
            cells count as observed. None starts every cell unknown, at 0.

    Raises:
        ValueError: A bad shape, resolution, origin or initial map.
    """

    def __init__(
        self,
        sensor: RangeSensor,
        shape: tuple[int, int],
        resolution: float,
        model: LogOddsModel,
        origin: tuple[float, float] = (0.0, 0.0),
        log_odds: ArrayLike | None = None,
    ) -> None:
        if len(shape) != 2 or not all(int(n) == n and n > 0 for n in shape):
            raise ValueError(f"OccupancyGrid shape must be two positive integers, got {shape}.")
        if not (math.isfinite(resolution) and resolution > 0.0):
            raise ValueError(f"OccupancyGrid resolution must be positive, got {resolution}.")
        if len(origin) != 2 or not all(math.isfinite(value) for value in origin):
            raise ValueError(f"OccupancyGrid origin must be two finite numbers, got {origin}.")
        self._sensor = sensor
        self._shape = (int(shape[0]), int(shape[1]))
        self._resolution = float(resolution)
        self._origin = (float(origin[0]), float(origin[1]))
        self._model = model
        if log_odds is None:
            self._log_odds = np.zeros(self._shape)
        else:
            start = np.array(log_odds, dtype=np.float64)
            if start.shape != self._shape or not np.isfinite(start).all():
                raise ValueError(
                    f"OccupancyGrid log_odds must be finite of shape {self._shape}, got "
                    f"{start.shape}."
                )
            self._log_odds = np.clip(start, model.l_min, model.l_max)
        self._observed = self._log_odds != 0.0

    @property
    def sensor(self) -> RangeSensor:
        """The sensor whose scans the grid takes."""
        return self._sensor

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        return self._shape

    @property
    def resolution(self) -> float:
        """Cell side in metres."""
        return self._resolution

    @property
    def origin(self) -> tuple[float, float]:
        """``(x0, y0)``, the lower-left corner in metres."""
        return self._origin

    @property
    def model(self) -> LogOddsModel:
        """The evidence per scan and the clamps."""
        return self._model

    @property
    def log_odds(self) -> FloatArray:
        """``(rows, cols)`` read-only view of each cell's log-odds of being occupied."""
        view = self._log_odds.view()
        view.setflags(write=False)
        return view

    @property
    def observed(self) -> NDArray[np.bool_]:
        """``(rows, cols)`` read-only view: whether a scan has ever updated the cell."""
        view = self._observed.view()
        view.setflags(write=False)
        return view

    def probabilities(self) -> FloatArray:
        """``(rows, cols)`` occupancy probabilities, ``1 - 1 / (1 + exp(l))``."""
        probabilities: FloatArray = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
        return probabilities

    def occupied(self) -> NDArray[np.bool_]:
        """``(rows, cols)``: cells more likely occupied than not (``l > 0``)."""
        occupied: NDArray[np.bool_] = self._log_odds > 0.0
        return occupied

    def free(self) -> NDArray[np.bool_]:
        """``(rows, cols)``: cells more likely free than not (``l < 0``)."""
        free: NDArray[np.bool_] = self._log_odds < 0.0
        return free

    def update(self, pose: ArrayLike, readings: ArrayLike) -> None:
        """Add one scan taken at a known pose.

        Args:
            pose: ``(3,)`` the robot's pose ``(x, y, theta)`` when the scan was read.
            readings: ``(N,)`` the scan, in the sensor's mount order (NaN allowed: unusable).

        Raises:
            ValueError: A pose that is not three finite numbers, or a scan of the wrong length.
        """
        poses = np.asarray(pose, dtype=np.float64)
        if poses.shape != (3,) or not np.isfinite(poses).all():
            raise ValueError(f"OccupancyGrid.update needs a finite (3,) pose, got {poses}.")
        scan = np.asarray(readings, dtype=np.float64)
        if scan.shape != (self._sensor.num_readings,):
            raise ValueError(
                f"OccupancyGrid.update needs readings of shape ({self._sensor.num_readings},), "
                f"one per mount, got {scan.shape}."
            )
        z = np.repeat(scan, self._sensor.rays_per_cone)  # (N K,), one per ray
        usable = usable_readings(z, self._sensor)
        if not usable.any():
            return
        origins, angles = sensor_rays(self._sensor, poses[None])
        z = z[usable]
        # Follow each ray a cell past its reading, so a hit on a border reaches the cell behind.
        visits = traverse_cells(
            origins[usable],
            angles[usable],
            z + self._resolution,
            self._resolution,
            self._shape,
            self._origin,
        )
        occupied_visits, free_visits = endpoint_visits(visits, z)
        occupied = np.zeros(self._shape, dtype=bool)
        occupied[visits.rows[occupied_visits], visits.cols[occupied_visits]] = True
        free = np.zeros(self._shape, dtype=bool)
        free[visits.rows[free_visits], visits.cols[free_visits]] = True
        free &= ~occupied
        self._log_odds += self._model.l_occ * occupied + self._model.l_free * free
        np.clip(self._log_odds, self._model.l_min, self._model.l_max, out=self._log_odds)
        self._observed |= occupied | free
