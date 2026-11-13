"""Path planning for the robot: A* on its configuration-space grid, and smoothing.

``grid`` (``PlanningGrid`` from the true map or a built one, inflated by the robot's radius and a
margin; ``reachable``), ``astar`` (``astar``, ``GridPath``, ``octile_distance``) and ``smoothing``
(``shortcut``, ``segment_free``, ``segments_free``, ``polyline_length``) import NumPy, ``rl_navigation.sim`` and
``rl_navigation.mapping.beams`` only, so the planner runs on the robot: a layer above the mapper,
never importing the estimators, SLAM, the Gymnasium layer or a desk tool.
"""

from rl_navigation.planning.astar import GridPath, astar, octile_distance
from rl_navigation.planning.grid import (
    Cell,
    PlanningGrid,
    from_arena,
    from_occupied,
    inflation_offsets,
    reachable,
)
from rl_navigation.planning.smoothing import polyline_length, segment_free, segments_free, shortcut

__all__ = [
    "Cell",
    "GridPath",
    "PlanningGrid",
    "astar",
    "from_arena",
    "from_occupied",
    "inflation_offsets",
    "octile_distance",
    "polyline_length",
    "reachable",
    "segment_free",
    "segments_free",
    "shortcut",
]
