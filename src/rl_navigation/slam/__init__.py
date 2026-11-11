"""Simultaneous localization and mapping for the robot.

``features`` cuts a lidar scan into segments and fits circles to them, the landmarks of the
landmark SLAM. The core imports NumPy, ``rl_navigation.sim`` and the ``estimation`` core only, so
it runs on the robot; ``feature_evaluation``, which scores the features against the simulator's
truth, is a desk tool this package does not import.
"""

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
    "CircleFeature",
    "FeatureSettings",
    "circle_residuals",
    "extract_circles",
    "fit_circle",
    "incidence_weights",
    "scan_points",
    "segment_scan",
]
