"""Occupancy-grid mapping for the robot: a log-odds grid built from range scans at known poses.

The core (``beams``: the cells a ray crosses and which a reading marks; ``grid``:
``OccupancyGrid`` and its log-odds models) imports only NumPy and ``rl_navigation.sim``, so it runs
on the robot beside the estimators, a peer of ``rl_navigation.estimation`` rather than a layer over
it. ``evaluation``, the seeded runs scored against the true map, is a desk tool this package does
not import.
"""

from rl_navigation.mapping.beams import (
    TRAVERSE_TOLERANCE,
    CellVisits,
    endpoint_visits,
    sensor_rays,
    traverse_cells,
)
from rl_navigation.mapping.grid import (
    HC_SR04_LOG_ODDS,
    LIDAR_LOG_ODDS,
    LogOddsModel,
    OccupancyGrid,
)

__all__ = [
    "HC_SR04_LOG_ODDS",
    "LIDAR_LOG_ODDS",
    "TRAVERSE_TOLERANCE",
    "CellVisits",
    "LogOddsModel",
    "OccupancyGrid",
    "endpoint_visits",
    "sensor_rays",
    "traverse_cells",
]
