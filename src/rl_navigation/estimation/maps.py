"""What an estimator needs of a map: the ranges a sensor would read from given poses."""

import dataclasses
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import Arena, FloatArray, clearance
from rl_navigation.sim.sensors import RangeSensor


class RangeMap(Protocol):
    """A known map seen through a range sensor.

    The localizers depend on this and nothing else of the map, so a known ``Arena`` (here), a
    particle filter's likelihood and an occupancy grid (Phase 8, step 3) plug in the same way.
    """

    @property
    def min_range(self) -> float:
        """Nearest distance the sensor reports, in metres."""
        ...

    @property
    def max_range(self) -> float:
        """Farthest distance the sensor reports, in metres; a lost reading reads this."""
        ...

    @property
    def noise_std(self) -> float:
        """Standard deviation of a reading's Gaussian noise, in metres."""
        ...

    def expected_ranges(self, poses: ArrayLike) -> FloatArray:
        """Noiseless, unclipped ranges from ``(P, 3)`` poses ``(x, y, theta)``: ``(P, N)``,
        ``inf`` where a beam meets nothing."""
        ...


class ParticleMap(RangeMap, Protocol):
    """A :class:`RangeMap` that a particle filter can also sample and cast a few beams of.

    Global localization draws poses over the map's free space, and the filter casts only the
    beams it uses from thousands of poses, so it needs these four more than the EKF does.
    """

    @property
    def num_readings(self) -> int:
        """Number of readings in a scan, one per beam."""
        ...

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` in metres: every free point lies inside."""
        ...

    def free(self, points: ArrayLike, margin: float) -> NDArray[np.bool_]:
        """``(P,)``: whether each of the ``(P, 2)`` points is more than ``margin`` metres from
        every obstacle and wall (a robot of that radius could stand there)."""
        ...

    def expected_beam_ranges(self, poses: ArrayLike, beams: ArrayLike) -> FloatArray:
        """:meth:`expected_ranges` of the ``(M,)`` beam indices only: ``(P, M)``."""
        ...


def usable_readings(readings: FloatArray, range_map: RangeMap) -> NDArray[np.bool_]:
    """``(N,)``: the readings that carry a range, ``min_range < z < max_range``.

    A dropout, a driver's NaN and "nothing within range" all read as unusable, as does a reading
    clipped at ``min_range``; NaN compares False, so it is never usable.
    """
    usable: NDArray[np.bool_] = (readings > range_map.min_range) & (readings < range_map.max_range)
    return usable


@dataclass(frozen=True, eq=False)
class SensorMap:
    """A :class:`ParticleMap` for a known :class:`Arena` and a :class:`RangeSensor`: the expected
    ranges are the simulator's own ray casts (mounts, cones, walls, circles and boxes).

    Args:
        arena: The known map.
        sensor: The sensor's geometry and noise; its ``noise_std`` is the filter's range noise.
    """

    arena: Arena
    sensor: RangeSensor

    @property
    def min_range(self) -> float:
        return self.sensor.min_range

    @property
    def max_range(self) -> float:
        return self.sensor.max_range

    @property
    def noise_std(self) -> float:
        return self.sensor.noise_std

    @property
    def num_readings(self) -> int:
        return self.sensor.num_readings

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return 0.0, 0.0, float(self.arena.width), float(self.arena.height)

    def expected_ranges(self, poses: ArrayLike) -> FloatArray:
        return self.sensor.true_ranges(self.arena, poses)

    def free(self, points: ArrayLike, margin: float) -> NDArray[np.bool_]:
        free: NDArray[np.bool_] = clearance(self.arena, points) > margin
        return free

    def expected_beam_ranges(self, poses: ArrayLike, beams: ArrayLike) -> FloatArray:
        """Casts only the chosen mounts, so a few beams cost a few beams' casts."""
        mounts = self.sensor.mount_angles[np.asarray(beams, dtype=np.intp)]
        return dataclasses.replace(self.sensor, mount_angles=mounts).true_ranges(self.arena, poses)
