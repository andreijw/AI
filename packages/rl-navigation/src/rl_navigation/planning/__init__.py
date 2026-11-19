"""Path planning for the robot: A* on its configuration-space grid, smoothing, a pure-pursuit
follower and a replanning navigator.

``grid`` (``PlanningGrid`` from the true map or a built one, inflated by the robot's radius and a
margin; ``reachable``), ``astar`` (``astar``, ``GridPath``, ``octile_distance``), ``smoothing``
(``shortcut``, ``segment_free``, ``segments_free``, ``polyline_free``, ``polyline_length``),
``follower`` (``PurePursuit``, ``FollowerParams``, ``Command``, and the navigation environment's
action mappings ``continuous_action`` and ``safe_discrete_action``) and ``navigator``
(``Navigator``, ``NavigatorParams``, the maps ``TrueMap`` and ``OnlineMap``, and
``TUNED_SETTINGS``, the margin and turn-in-place tuned per action kind) import NumPy,
``rl_navigation.sim`` and ``rl_navigation.mapping`` only, so the planner runs on the robot: a layer
above the mapper, never importing the estimators, SLAM, the Gymnasium layer or a desk tool (the
comparison harness, ``rl_navigation.planning_comparison``, imports this package, not the other
way round).
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
    DEFAULT_MAP_RANGE,
    NO_PATH,
    TUNED_SETTINGS,
    Navigator,
    NavigatorParams,
    OnlineMap,
    PlannerSettings,
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
    "DEFAULT_MAP_RANGE",
    "NO_PATH",
    "TUNED_SETTINGS",
    "PlannerSettings",
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
