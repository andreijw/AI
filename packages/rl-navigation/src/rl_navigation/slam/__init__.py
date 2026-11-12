"""Simultaneous localization and mapping for the robot.

``features`` cuts a lidar scan into segments and fits circles to them, the landmarks of the
landmark SLAM; ``ekf_slam`` estimates the robot's pose and a map of those circles in one Gaussian
(EKF-SLAM with first-estimate Jacobians). The core imports NumPy, ``rl_navigation.sim`` and the
``estimation`` core only, so it runs on the robot; ``feature_evaluation`` and ``evaluation``,
which score the features and the SLAM against the simulator's truth, are desk tools this package
does not import.
"""

from rl_navigation.slam.ekf_slam import (
    AMBIGUOUS,
    NEW_LANDMARK,
    EkfSlam,
    SlamSettings,
    SlamUpdate,
    associate_ml,
    fej_motion_jacobian,
    initialize_landmark,
    landmark_measurement,
)
from rl_navigation.slam.features import (
    CircleFeature,
    FeatureSettings,
    circle_residuals,
    extract_circles,
    fit_circle,
    incidence_weights,
    scan_points,
    segment_scan,
)

__all__ = [
    "AMBIGUOUS",
    "NEW_LANDMARK",
    "CircleFeature",
    "EkfSlam",
    "FeatureSettings",
    "SlamSettings",
    "SlamUpdate",
    "associate_ml",
    "circle_residuals",
    "extract_circles",
    "fej_motion_jacobian",
    "fit_circle",
    "incidence_weights",
    "initialize_landmark",
    "landmark_measurement",
    "scan_points",
    "segment_scan",
]
