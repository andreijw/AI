"""What an estimator needs of a map: the ranges a sensor would read from given poses."""

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import Arena, FloatArray
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


@dataclass(frozen=True, eq=False)
class SensorMap:
    """A :class:`RangeMap` for a known :class:`Arena` and a :class:`RangeSensor`: the expected
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

    def expected_ranges(self, poses: ArrayLike) -> FloatArray:
        return self.sensor.true_ranges(self.arena, poses)
