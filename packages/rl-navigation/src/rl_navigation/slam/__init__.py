"""Simultaneous localization and mapping for the robot.

``features`` cuts a lidar scan into segments and fits circles to them, the landmarks of the
landmark SLAM; ``ekf_slam`` estimates the robot's pose and a map of those circles in one Gaussian
(EKF-SLAM with first-estimate Jacobians); ``scan_matching`` matches a lidar scan against the
log-odds grid and keeps one pose and the grid by it (a scan-matching front end). The core imports
NumPy, ``rl_navigation.sim`` and the ``estimation`` and ``mapping`` cores only, so it runs on the
robot; ``feature_evaluation``, ``evaluation`` and ``scan_evaluation``, which score the features
and the SLAM against the simulator's truth, are desk tools this package does not import.
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
from rl_navigation.slam.scan_matching import (
    MatchTerms,
    ScanMatch,
    ScanMatchingSlam,
    ScanMatchSettings,
    ScanMatchUpdate,
    match_points,
    match_residuals,
    match_scan,
    scan_endpoints,
)

__all__ = [
    "AMBIGUOUS",
    "NEW_LANDMARK",
    "CircleFeature",
    "EkfSlam",
    "FeatureSettings",
    "MatchTerms",
    "ScanMatch",
    "ScanMatchSettings",
    "ScanMatchUpdate",
    "ScanMatchingSlam",
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
    "match_points",
    "match_residuals",
    "match_scan",
    "scan_endpoints",
    "scan_points",
    "segment_scan",
]
