"""Wheel encoders on a differential-drive base: the wheel travel each step, with noise.

The noise model is Siegwart, Nourbakhsh and Scaramuzza, *Introduction to Autonomous Mobile
Robots* (2nd ed., section 5.2.4): each wheel's travel carries independent Gaussian noise whose
variance grows with the distance that wheel rolled, ``k |travel|``. A state estimator uses the
same object as its motion model.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import FloatArray


@dataclass(frozen=True)
class WheelOdometry:
    """Left and right wheel encoders of a differential-drive robot.

    Args:
        wheel_base: Distance between the wheels' contact points in metres (> 0). The default,
            0.2 m, puts the wheels at the rim of the default 0.1 m robot.
        noise_per_metre: ``k`` in metres: a wheel that rolls ``d`` metres reads it with variance
            ``k |d|`` (>= 0). The default, 2.5e-4 m (1.6 cm of standard deviation per metre per
            wheel, about 6 degrees of heading per metre), is a placeholder until the encoders
            are calibrated on the robot (Phase 8, step 5).

    **Reading contract.** A reading is a ``(2,)`` float array: the left and the right wheel's
    travel in metres since the previous reading, forward positive. An encoder driver on the robot
    returns the same.

    Raises:
        ValueError: A wheel base that is not positive or a negative noise.
    """

    wheel_base: float = 0.2
    noise_per_metre: float = 2.5e-4

    def __post_init__(self) -> None:
        if not self.wheel_base > 0.0:
            raise ValueError(f"WheelOdometry wheel_base must be positive, got {self.wheel_base}.")
        if not self.noise_per_metre >= 0.0:
            raise ValueError(
                f"WheelOdometry noise_per_metre must be at least 0, got {self.noise_per_metre}."
            )

    def wheel_travel(self, v: float, omega: float, dt: float) -> FloatArray:
        """Exact ``(left, right)`` travel for a step of ``dt`` at speed ``v`` and turn rate ``omega``:
        ``v dt -+ omega dt wheel_base / 2``."""
        half_turn = omega * dt * self.wheel_base / 2.0
        return np.array([v * dt - half_turn, v * dt + half_turn])

    def read(self, v: float, omega: float, dt: float, rng: np.random.Generator) -> FloatArray:
        """One noisy encoder reading of the step; two normals are drawn whatever the motion."""
        travel = self.wheel_travel(v, omega, dt)
        noise = rng.standard_normal(2)
        return travel + noise * np.sqrt(self.noise_per_metre * np.abs(travel))

    def increments(self, wheel_travel: ArrayLike) -> tuple[float, float]:
        """The distance and turn ``(Delta s, Delta theta)`` that a ``(left, right)`` travel means:
        ``((left + right) / 2, (right - left) / wheel_base)``."""
        left, right = np.asarray(wheel_travel, dtype=np.float64)
        return float((left + right) / 2.0), float((right - left) / self.wheel_base)

    def increments_jacobian(self) -> FloatArray:
        """``d (Delta s, Delta theta) / d (left, right)``, a constant ``(2, 2)``."""
        inverse_base = 1.0 / self.wheel_base
        return np.array([[0.5, 0.5], [-inverse_base, inverse_base]])

    def wheel_covariance(self, wheel_travel: ArrayLike) -> FloatArray:
        """``(2, 2)`` covariance of a reading: ``diag(k |left|, k |right|)``."""
        travel = np.abs(np.asarray(wheel_travel, dtype=np.float64))
        return np.diag(self.noise_per_metre * travel)
