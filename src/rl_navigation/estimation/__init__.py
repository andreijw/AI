"""State estimation for the robot: its pose from wheel odometry and range readings on a known map.

Imports only NumPy and ``rl_navigation.sim``, so it runs on the robot without Gymnasium or the
training framework.
"""

from rl_navigation.estimation.ekf import (
    EkfLocalizer,
    Linearization,
    UpdateResult,
    linearize_ranges,
    motion_jacobians,
    pose_difference,
)
from rl_navigation.estimation.kalman import kalman_update
from rl_navigation.estimation.maps import RangeMap, SensorMap

__all__ = [
    "EkfLocalizer",
    "Linearization",
    "RangeMap",
    "SensorMap",
    "UpdateResult",
    "kalman_update",
    "linearize_ranges",
    "motion_jacobians",
    "pose_difference",
]
