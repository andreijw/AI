"""The robot-side navigation loop: plan on a map, follow the plan, replan when it goes stale.

A :class:`Navigator` turns a pose and a map into one :class:`~rl_navigation.planning.follower.Command`
per control step. The map is a :class:`PlanningMap`: :class:`TrueMap` (the arena itself, an oracle)
or :class:`OnlineMap` (a log-odds grid built from the robot's own scans). A plan is:

1. **Margin ladder.** The map inflated by ``robot_radius + resolution / 2 + m`` for ``m`` in
   ``(margin, margin / 2, 0)``, the first that holds a path; the last is where
   :func:`~rl_navigation.sim.layouts.generate_layout`'s guarantee of a path holds.
2. **Escape bubble.** When the robot's own cell is blocked, the cells within the inflation of its
   position that are blocked only by inflation (not in ``PlanningGrid.occupied``) are freed for
   this search, so a robot inside the band can leave it; a cell holding an obstacle never is.
3. **Substitute goal.** When the goal's cell is blocked, the nearest free cell whose centre lies
   within ``goal_tolerance`` (0.15 m, inside the environment's 0.2 m goal radius) is the target.
4. :func:`~rl_navigation.planning.grid.reachable` first (a flood fill, a few milliseconds), then
   :func:`~rl_navigation.planning.astar.astar` only where it passes, then
   :func:`~rl_navigation.planning.smoothing.shortcut`, then a new
   :class:`~rl_navigation.planning.follower.PurePursuit`.

Replanning happens every ``replan_every`` steps, and earlier when the map's ``version`` changed and
the rest of the path crosses a blocked cell. When no level holds a path the status is
``"no_path"`` and the robot turns on the spot to rescan: in one direction, chosen when
``"no_path"`` begins (towards the goal's bearing then) and held, whole turns at a time, until a
path is found, since turning towards the goal's bearing at every step only dithers about it (its
sign flips as the heading passes it). The search is retried only when the map's version changes
or ``replan_every`` steps have passed. A retry looks at the smallest inflation first and stops there if it holds no
path, so a failing retry costs one grid and one flood fill: an online map changes at almost every
scan while the robot turns, and retrying the whole ladder each time cost up to 20 ms a step over
a replan period on the tuning layouts (2026-11-13).

Imports NumPy, ``rl_navigation.sim``, ``rl_navigation.mapping`` (the online map's grid) and the
rest of ``rl_navigation.planning`` only.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.mapping.grid import OccupancyGrid
from rl_navigation.planning.astar import astar
from rl_navigation.planning.follower import Command, FollowerParams, PurePursuit
from rl_navigation.planning.grid import Cell, PlanningGrid, from_arena, from_occupied, reachable
from rl_navigation.planning.smoothing import polyline_free, shortcut
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.kinematics import wrap_angle

FOLLOWING, NO_PATH = "following", "no_path"


class PlanningMap(Protocol):
    """What the navigator plans on: a grid at any inflation, and a version that changes whenever
    the obstacles do."""

    @property
    def resolution(self) -> float: ...

    @property
    def version(self) -> int: ...

    def grid(self, inflation: float) -> PlanningGrid: ...


class TrueMap:
    """The arena as an oracle map: its exact inflated grid at origin ``(0, 0)``; the version never
    changes. Grids are kept per inflation.

    Args:
        arena: The arena.
        resolution: Cell side in metres.
    """

    def __init__(self, arena: Arena, resolution: float) -> None:
        self._arena = arena
        self._resolution = float(resolution)
        self._grids: dict[float, PlanningGrid] = {}

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def version(self) -> int:
        return 0

    def grid(self, inflation: float) -> PlanningGrid:
        if inflation not in self._grids:
            self._grids[inflation] = from_arena(self._arena, self._resolution, inflation)
        return self._grids[inflation]


class OnlineMap:
    """A map built from the robot's own scans: an :class:`~rl_navigation.mapping.OccupancyGrid`
    whose occupied cells (``l > 0``; unknown cells free) are inflated by
    :func:`~rl_navigation.planning.grid.from_occupied`.

    Readings farther than ``map_range`` are given to the grid as NaN, so they change nothing: a
    sonar's 15-degree cone puts a far or oblique hit's whole arc at the nearest range, marking free
    space as occupied, which closed the way to the goal on 4 of 20 tuning layouts at the HC-SR04's
    full 4 m. The default, 1.0 m, reached the goal on 194 and 193 of layouts 10000 to 10199
    (continuous and discrete actions) against 190 at 1.5 m (2026-11-13). The version goes up when
    the set of occupied cells changes.

    Args:
        occupancy: The log-odds grid to build; this map updates it.
        map_range: Farthest reading used, in metres (> 0).

    Raises:
        ValueError: A ``map_range`` that is not positive.
    """

    def __init__(self, occupancy: OccupancyGrid, map_range: float = 1.0) -> None:
        if not (math.isfinite(map_range) and map_range > 0.0):
            raise ValueError(f"OnlineMap map_range must be positive, got {map_range}.")
        self._occupancy = occupancy
        self._map_range = float(map_range)
        self._occupied = occupancy.occupied()
        self._version = 0
        self._grids: dict[float, PlanningGrid] = {}

    @property
    def occupancy(self) -> OccupancyGrid:
        """The log-odds grid."""
        return self._occupancy

    @property
    def resolution(self) -> float:
        return self._occupancy.resolution

    @property
    def version(self) -> int:
        return self._version

    def update(self, pose: ArrayLike, readings: ArrayLike) -> None:
        """Add one scan taken at ``pose``, readings beyond ``map_range`` left out."""
        scan = np.array(readings, dtype=np.float64)
        scan[scan > self._map_range] = np.nan
        self._occupancy.update(pose, scan)
        occupied = self._occupancy.occupied()
        if not np.array_equal(occupied, self._occupied):
            self._occupied = occupied
            self._version += 1
            self._grids = {}

    def grid(self, inflation: float) -> PlanningGrid:
        if inflation not in self._grids:
            self._grids[inflation] = from_occupied(
                self._occupied, self.resolution, inflation, self._occupancy.origin
            )
        return self._grids[inflation]


@dataclass(frozen=True)
class NavigatorParams:
    """The navigator's settings.

    Args:
        robot_radius: The robot's radius in metres (> 0).
        margin: Safety margin beyond the radius and half a cell, in metres (>= 0): at least the
            follower's worst cross-track error plus ``resolution (1 / sqrt(2) - 1 / 2)``, 0.0104 m
            at 5 cm. The default, 0.05 m, covers continuous actions (worst cross-track 0.033 to
            0.037 m on layouts 10000 to 10199, 2026-11-13); discrete actions need 0.06 m (0.041
            to 0.047 m).
        replan_every: Steps between plans, and between retries after ``"no_path"`` (>= 1).
        goal_tolerance: Farthest a substitute goal's centre may lie from the goal, in metres.

    Raises:
        ValueError: A value out of range.
    """

    robot_radius: float = 0.1
    margin: float = 0.05
    replan_every: int = 10
    goal_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if not (math.isfinite(self.robot_radius) and self.robot_radius > 0.0):
            raise ValueError(
                f"NavigatorParams robot_radius must be positive, got {self.robot_radius}."
            )
        if not (math.isfinite(self.margin) and self.margin >= 0.0):
            raise ValueError(f"NavigatorParams margin must be at least 0, got {self.margin}.")
        if (
            isinstance(self.replan_every, bool)
            or not isinstance(self.replan_every, int)
            or self.replan_every < 1
        ):
            raise ValueError(
                f"NavigatorParams replan_every must be an integer >= 1, got {self.replan_every!r}."
            )
        if not (math.isfinite(self.goal_tolerance) and self.goal_tolerance >= 0.0):
            raise ValueError(
                f"NavigatorParams goal_tolerance must be at least 0, got {self.goal_tolerance}."
            )

    def inflations(self, resolution: float) -> tuple[float, float, float]:
        """The margin ladder's inflations, largest first."""
        base = self.robot_radius + resolution / 2.0
        return base + self.margin, base + self.margin / 2.0, base


class Navigator:
    """Plans to a goal on a :class:`PlanningMap` and follows the plan (see the module docstring).

    Args:
        goal: ``(2,)`` the goal position.
        params: The navigator's settings.
        follower: The follower's settings (the robot's ``v_max``, ``omega_max`` and ``dt``).
    """

    def __init__(self, goal: ArrayLike, params: NavigatorParams, follower: FollowerParams) -> None:
        self.goal = np.array(goal, dtype=np.float64).reshape(2)
        self.params = params
        self.follower_params = follower
        self._follower: PurePursuit | None = None
        self._since = 0
        self._status = NO_PATH
        self._planned_version: int | None = None
        self._inflation: float | None = None
        self.searches = 0  # A* searches run
        self.plans = 0  # plan attempts, failed ones included
        self.reduced_margin_plans = 0  # plans found below the full margin (the ladder stepped down)
        self._spin_direction = 0.0

    @property
    def status(self) -> str:
        """``"following"`` a plan, or ``"no_path"``."""
        return self._status

    @property
    def inflation(self) -> float | None:
        """The ladder's inflation the current plan was made at, or None without one."""
        return self._inflation

    @property
    def path(self) -> FloatArray | None:
        """The current plan's polyline, or None."""
        return None if self._follower is None else self._follower.path

    def prepare(self, world_map: PlanningMap) -> tuple[PlanningGrid, ...]:
        """Build the map's grid at every level of the ladder now, before the first step, so that
        a fixed map's set-up (about 20 ms a level for the true map at 5 cm) is not paid inside a
        control step. Returns the grids, largest inflation first."""
        return tuple(world_map.grid(i) for i in self.params.inflations(world_map.resolution))

    def safety_grid(self, world_map: PlanningMap) -> PlanningGrid:
        """The map at the ladder's smallest inflation, for a discrete action's safety check."""
        return world_map.grid(self.params.inflations(world_map.resolution)[-1])

    def command(self, pose: ArrayLike, world_map: PlanningMap) -> Command:
        """This step's command from the robot's pose ``(x, y, theta)``."""
        state = np.asarray(pose, dtype=np.float64).reshape(3)
        if self._needs_plan(state, world_map):
            self._plan(state, world_map)
        self._since += 1
        if self._follower is None:
            return self._rescan(state)
        return self._follower.command(state)

    def _rescan(self, pose: FloatArray) -> Command:
        """A turn on the spot in one direction, held until a path is found (see the module
        docstring)."""
        if self._spin_direction == 0.0:
            bearing = float(wrap_angle(math.atan2(*(self.goal - pose[:2])[::-1]) - pose[2]))
            self._spin_direction = 1.0 if bearing >= 0.0 else -1.0
        return Command(0.0, self._spin_direction * self.follower_params.omega_max)

    def _needs_plan(self, pose: FloatArray, world_map: PlanningMap) -> bool:
        changed = world_map.version != self._planned_version
        due = self._since >= self.params.replan_every
        if self._planned_version is None or due:
            return True
        if self._follower is None:  # after "no_path", retry on a new map only
            return changed
        if not changed or self._inflation is None:
            return False
        grid = world_map.grid(self._inflation)
        rest = self._follower.remaining()
        start = grid.cell_of(rest[0])
        if grid.blocked[start]:
            # The robot's own place may lie in the band: check the first segment from where it
            # leaves the inflation around the robot.
            step = rest[1] - rest[0]
            length = float(np.hypot(*step))
            rest[0] = (
                rest[0] + step * min(self._inflation / length, 1.0) if length > 0.0 else rest[1]
            )
        return not polyline_free(grid, rest)

    def _plan(self, pose: FloatArray, world_map: PlanningMap) -> None:
        retry = self._status == NO_PATH and self._planned_version is not None
        self._since = 0
        self._planned_version = world_map.version
        self.plans += 1
        ladder = self.params.inflations(world_map.resolution)
        found: dict[float, FloatArray | None] = {}
        if retry:
            found[ladder[-1]] = self._search(pose, world_map.grid(ladder[-1]), ladder[-1])
            if found[ladder[-1]] is None:
                return  # still no path: one grid and one flood fill, not the whole ladder
        for inflation in ladder:
            if inflation in found:
                points = found[inflation]
            else:
                points = self._search(pose, world_map.grid(inflation), inflation)
            if points is not None:
                self._follower = PurePursuit(points, self.follower_params)
                self._inflation = inflation
                self._status = FOLLOWING
                self._spin_direction = 0.0
                self.reduced_margin_plans += inflation != ladder[0]
                return
        self._follower, self._inflation, self._status = None, None, NO_PATH

    def _search(self, pose: FloatArray, grid: PlanningGrid, inflation: float) -> FloatArray | None:
        """The smoothed path's points on one level of the ladder, or None."""
        try:
            start = grid.cell_of(pose[:2])
            goal_cell = grid.cell_of(self.goal)
        except ValueError:
            return None
        if grid.blocked[start]:
            grid = _escape(grid, pose[:2], inflation)
        goal_point = self.goal
        if grid.blocked[goal_cell]:
            substitute = _substitute(grid, self.goal, self.params.goal_tolerance)
            if substitute is None:
                return None
            goal_cell = substitute
            goal_point = grid.centres([substitute])[0]
        if not reachable(grid, start, goal_cell):
            return None
        self.searches += 1
        path = astar(grid, start, goal_cell)
        if path is None:  # pragma: no cover - reachable and A* agree (tested)
            return None
        points = np.vstack([pose[:2], grid.centres(path.cells[1:-1]), goal_point])
        if len(points) == 2 and np.array_equal(points[0], points[1]):
            return None
        return shortcut(grid, points)


def _cell_distances(grid: PlanningGrid, point: FloatArray) -> NDArray[np.float64]:
    """``(rows, cols)`` distance from each cell centre to ``point``."""
    rows, cols = grid.shape
    x = grid.origin[0] + (np.arange(cols) + 0.5) * grid.resolution
    y = grid.origin[1] + (np.arange(rows) + 0.5) * grid.resolution
    distances: NDArray[np.float64] = np.hypot(x[None, :] - point[0], y[:, None] - point[1])
    return distances


def _escape(grid: PlanningGrid, position: FloatArray, inflation: float) -> PlanningGrid:
    """``grid`` with the cells within ``inflation`` of ``position`` that hold no obstacle freed.
    A grid that does not know its obstacles (``occupied`` None) is returned unchanged."""
    if grid.occupied is None:
        return grid
    bubble = (_cell_distances(grid, position) <= inflation) & ~grid.occupied
    return PlanningGrid(grid.blocked & ~bubble, grid.resolution, grid.origin, grid.occupied)


def _substitute(grid: PlanningGrid, goal: FloatArray, tolerance: float) -> Cell | None:
    """The free cell nearest to ``goal`` whose centre lies within ``tolerance`` of it, or None."""
    distances = np.where(grid.blocked, np.inf, _cell_distances(grid, goal))
    index = int(np.argmin(distances))
    if not distances.flat[index] <= tolerance:
        return None
    i, j = divmod(index, grid.shape[1])
    return i, j
