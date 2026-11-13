"""Timing wrappers for the comparison's filters: :class:`InstrumentedLocalizer` and its factory."""

import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.estimation.ekf import EkfLocalizer, UpdateResult
from rl_navigation.estimation.evaluation import Localizer, LocalizerFactory, ScanResult
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.sim import WheelOdometry
from rl_navigation.sim.geometry import FloatArray

TIMED_CALLS = ("predict", "update", "pose", "covariance")


class InstrumentedLocalizer:
    """A :class:`~rl_navigation.estimation.evaluation.Localizer` that times the one it wraps.

    Each call of ``predict``, ``update``, ``pose`` and ``covariance`` adds its wall time to
    :attr:`seconds`; ``pose`` and ``covariance`` are timed because MCL clusters its particles
    lazily, on the first of them after an update. For a wrapped :class:`EkfLocalizer` it also
    records the scans (counted from 0) on which the update reset the covariance and the largest
    :attr:`EkfLocalizer.recent_resets` seen.
    """

    def __init__(self, inner: Localizer) -> None:
        self.inner = inner
        self.seconds = dict.fromkeys(TIMED_CALLS, 0.0)
        self.scans = 0
        self.reset_steps: list[int] | None = [] if isinstance(inner, EkfLocalizer) else None
        self.max_recent_resets = 0

    def _timed(self, name: str, call: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        value = call()
        self.seconds[name] += time.perf_counter() - start
        return value

    @property
    def pose(self) -> FloatArray:
        """The wrapped filter's pose."""
        pose: FloatArray = self._timed("pose", lambda: self.inner.pose)
        return pose

    @property
    def covariance(self) -> FloatArray:
        """The wrapped filter's covariance."""
        covariance: FloatArray = self._timed("covariance", lambda: self.inner.covariance)
        return covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        """The wrapped filter's prediction."""
        self._timed("predict", lambda: self.inner.predict(wheel_travel))

    def update(self, readings: ArrayLike) -> ScanResult:
        """The wrapped filter's update, noting an EKF's reset."""
        result: ScanResult = self._timed("update", lambda: self.inner.update(readings))
        if self.reset_steps is not None and isinstance(self.inner, EkfLocalizer):
            if cast(UpdateResult, result).reset:
                self.reset_steps.append(self.scans)
            self.max_recent_resets = max(self.max_recent_resets, self.inner.recent_resets)
        self.scans += 1
        return result


class InstrumentedFactory:
    """A :class:`~rl_navigation.estimation.evaluation.LocalizerFactory` that wraps each filter
    the factory it is given builds in an :class:`InstrumentedLocalizer`, kept as :attr:`built`."""

    def __init__(self, factory: LocalizerFactory) -> None:
        self.factory = factory
        self.built: InstrumentedLocalizer | None = None

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
        self.built = InstrumentedLocalizer(
            self.factory(
                range_map=range_map,
                odometry=odometry,
                start=start,
                covariance=covariance,
                rng=rng,
                robot_radius=robot_radius,
            )
        )
        return self.built
