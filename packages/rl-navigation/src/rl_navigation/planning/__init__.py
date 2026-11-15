"""Path planning for the robot: A* on its configuration-space grid, and smoothing.

``grid`` (``PlanningGrid`` from the true map or a built one, inflated by the robot's radius and a
margin; ``reachable``), ``astar`` (``astar``, ``GridPath``, ``octile_distance``) and ``smoothing``
(``shortcut``, ``segment_free``, ``segments_free``, ``polyline_length``) import NumPy, ``rl_navigation.sim`` and
``rl_navigation.mapping.beams`` only, so the planner runs on the robot: a layer above the mapper,
never importing the estimators, SLAM, the Gymnasium layer or a desk tool.
"""

from rl_navigation.planning.astar import GridPath, astar, octile_distance
from rl_navigation.planning.follower import (
    Command,
    FollowerParams,
    PurePursuit,
    continuous_action,
    safe_discrete_action,
)
from rl_navigation.planning.grid import (
    Cell,
    PlanningGrid,
    from_arena,
    from_occupied,
    inflation_offsets,
    reachable,
)
from rl_navigation.planning.navigator import (
    NO_PATH,
    Navigator,
    NavigatorParams,
    OnlineMap,
    PlanningMap,
    TrueMap,
)
from rl_navigation.planning.smoothing import (
    polyline_free,
    polyline_length,
    segment_free,
    segments_free,
    shortcut,
)

__all__ = [
    "NO_PATH",
    "Cell",
    "Command",
    "FollowerParams",
    "Navigator",
    "NavigatorParams",
    "OnlineMap",
    "PlanningMap",
    "PurePursuit",
    "TrueMap",
    "continuous_action",
    "safe_discrete_action",
    "GridPath",
    "PlanningGrid",
    "astar",
    "from_arena",
    "from_occupied",
    "inflation_offsets",
    "octile_distance",
    "polyline_free",
    "polyline_length",
    "reachable",
    "segment_free",
    "segments_free",
    "shortcut",
]
