"""State estimation for the robot: its pose from wheel odometry and range readings on a known map,
with an extended Kalman filter (``EkfLocalizer``) or a particle filter (``MclLocalizer``).

Imports only NumPy and ``rl_navigation.sim``, so it runs on the robot without Gymnasium or the
training framework.

``rl_navigation.estimation.comparison`` (the EKF against MCL as a report and a command line) and
``rl_navigation.estimation.recording`` (one trial of both filters as frames) are tools for the desk and are
deliberately not imported here.
"""

from rl_navigation.estimation.ekf import (
    EkfLocalizer,
    Linearization,
    UpdateResult,
    linearize_ranges,
    motion_jacobians,
)
from rl_navigation.estimation.kalman import kalman_update
from rl_navigation.estimation.maps import ParticleMap, RangeMap, SensorMap, usable_readings
from rl_navigation.estimation.mcl import (
    MclLocalizer,
    MclSettings,
    MclUpdate,
    effective_sample_size,
    heaviest_cluster,
    low_variance_resample,
    measurement_log_likelihood,
    recovery_update,
    sample_free_poses,
)
from rl_navigation.estimation.poses import pose_difference

__all__ = [
    "EkfLocalizer",
    "Linearization",
    "MclLocalizer",
    "MclSettings",
    "MclUpdate",
    "ParticleMap",
    "RangeMap",
    "SensorMap",
    "UpdateResult",
    "effective_sample_size",
    "heaviest_cluster",
    "kalman_update",
    "linearize_ranges",
    "low_variance_resample",
    "measurement_log_likelihood",
    "motion_jacobians",
    "pose_difference",
    "recovery_update",
    "sample_free_poses",
    "usable_readings",
]
