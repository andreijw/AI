"""The robot's configuration space as a grid: which cells its centre may occupy.

A cell is *blocked* when the robot's centre there would come within the inflation ``I`` of an
obstacle; the planner uses ``I = robot_radius + resolution / 2 + margin``, where the half cell is
:func:`~rl_navigation.sim.layouts.path_exists`'s conservative allowance and the margin is the safety
the follower needs. Two sources give the same type:

- the true map, :func:`from_arena`: :func:`~rl_navigation.sim.geometry.occupancy_grid` inflated by
  ``I``, exact for the arena's circles, boxes and walls;
- a built map, :func:`from_occupied`: a boolean grid of occupied cells (an
  :class:`~rl_navigation.mapping.OccupancyGrid`'s ``occupied()``), dilated so that a cell is blocked
  when its centre lies within ``I`` of some occupied cell's *square* (not its centre, which would
  under-inflate along diagonals), and within ``I`` of the grid's edge (the walls).

The frame is ``occupancy_grid``'s: cell ``[i, j]`` spans ``x0 + [j, j + 1) resolution`` and
``y0 + [i, i + 1) resolution``, rows along y. Imports NumPy and ``rl_navigation.sim`` only.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import GRID_TOLERANCE, Arena, FloatArray, occupancy_grid
from rl_navigation.sim.layouts import connected

Cell = tuple[int, int]


@dataclass(frozen=True, eq=False)
class PlanningGrid:
    """A grid of cells the robot's centre may not enter.

    Args:
        blocked: ``(rows, cols)`` booleans, True where the centre may not go; copied, read-only.
        resolution: Cell side in metres (> 0).
        origin: ``(x0, y0)``, the grid's lower-left corner in metres.
        occupied: ``(rows, cols)`` booleans, the obstacles themselves before inflation (every one
            blocked), or None when unknown; copied, read-only. It tells a cell blocked only by
            the inflation, which a robot inside the band may cross to leave it, from an obstacle.

    Raises:
        ValueError: A ``blocked`` that is not a non-empty 2D array, a resolution that is not
            positive and finite, an origin that is not two finite numbers, or an ``occupied`` of
            another shape or with a cell that is not blocked.
    """

    blocked: NDArray[np.bool_]
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)
    occupied: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        blocked = np.array(self.blocked, dtype=bool)
        if blocked.ndim != 2 or blocked.size == 0:
            raise ValueError(
                f"PlanningGrid blocked must be a non-empty 2D array, got shape {blocked.shape}."
            )
        if not (math.isfinite(self.resolution) and self.resolution > 0.0):
            raise ValueError(f"PlanningGrid resolution must be positive, got {self.resolution}.")
        if len(self.origin) != 2 or not all(math.isfinite(v) for v in self.origin):
            raise ValueError(f"PlanningGrid origin must be two finite numbers, got {self.origin}.")
        blocked.setflags(write=False)
        object.__setattr__(self, "blocked", blocked)
        if self.occupied is not None:
            occupied = np.array(self.occupied, dtype=bool)
            if occupied.shape != blocked.shape or (occupied & ~blocked).any():
                raise ValueError(
                    f"PlanningGrid occupied must be of shape {blocked.shape} with every occupied "
                    f"cell blocked, got shape {occupied.shape}."
                )
            occupied.setflags(write=False)
            object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        rows, cols = self.blocked.shape
        return rows, cols

    def cell_of(self, point: ArrayLike) -> Cell:
        """The ``(row, col)`` holding a point; the grid's far edges belong to its last cells.

        Raises:
            ValueError: The point is not finite or lies outside the grid.
        """
        x, y = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
        rows, cols = self.shape
        j = (x - self.origin[0]) / self.resolution
        i = (y - self.origin[1]) / self.resolution
        if not (0.0 <= j <= cols and 0.0 <= i <= rows):  # NaN fails both
            raise ValueError(f"PlanningGrid point ({x}, {y}) is outside the grid or not finite.")
        return min(int(i), rows - 1), min(int(j), cols - 1)

    def centres(self, cells: ArrayLike) -> FloatArray:
        """``(K, 2)`` the ``(x, y)`` centres of ``(K, 2)`` ``(row, col)`` cells."""
        index = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
        centres: FloatArray = np.column_stack(
            [
                self.origin[0] + (index[:, 1] + 0.5) * self.resolution,
                self.origin[1] + (index[:, 0] + 0.5) * self.resolution,
            ]
        )
        return centres


def _checked_cell(grid: PlanningGrid, cell: ArrayLike, name: str) -> Cell:
    """``cell`` as a ``(row, col)`` of Python ints inside ``grid``.

    Raises:
        ValueError: Not two integers (a float such as 2.7, or a bool, is refused rather than
            truncated), or a cell outside the grid; ``name`` says which argument it was.
    """
    values = tuple(cell) if isinstance(cell, tuple | list | np.ndarray) else (cell,)
    if len(values) != 2 or not all(
        isinstance(v, int | np.integer) and not isinstance(v, bool | np.bool_) for v in values
    ):
        raise ValueError(f"{name} cell must be two integers (row, col), got {cell!r}.")
    i, j = int(values[0]), int(values[1])
    rows, cols = grid.shape
    if not (0 <= i < rows and 0 <= j < cols):
        raise ValueError(f"{name} cell ({i}, {j}) is outside the {rows} x {cols} grid.")
    return i, j


def _check_inflation(inflation: float) -> float:
    if not (math.isfinite(inflation) and inflation >= 0.0):
        raise ValueError(f"inflation must be finite and at least 0, got {inflation}.")
    return float(inflation)


def from_arena(arena: Arena, resolution: float, inflation: float) -> PlanningGrid:
    """The true map's configuration space: cells whose centre lies within ``inflation`` of an
    obstacle or a wall, at origin ``(0, 0)`` (:func:`~rl_navigation.sim.geometry.occupancy_grid`).

    Raises:
        ValueError: A negative or non-finite inflation.
    """
    inflate = _check_inflation(inflation)
    return PlanningGrid(
        occupancy_grid(arena, resolution, inflate=inflate),
        resolution,
        occupied=occupancy_grid(arena, resolution),
    )


def inflation_offsets(resolution: float, inflation: float) -> NDArray[np.intp]:
    """``(K, 2)`` the ``(di, dj)`` offsets whose cell centre lies within ``inflation`` of the
    square of the cell at ``(0, 0)``: ``resolution * hypot(max(|di| - 1/2, 0), max(|dj| - 1/2, 0))
    <= inflation`` (with :data:`~rl_navigation.sim.geometry.GRID_TOLERANCE`).

    Raises:
        ValueError: A negative or non-finite inflation, or a resolution that is not positive.
    """
    inflate = _check_inflation(inflation)
    if not (math.isfinite(resolution) and resolution > 0.0):
        raise ValueError(f"inflation_offsets needs a positive resolution, got {resolution}.")
    reach = int(math.ceil(inflate / resolution + 0.5))
    span = np.arange(-reach, reach + 1)
    di, dj = np.meshgrid(span, span, indexing="ij")
    gap = np.hypot(np.maximum(np.abs(di) - 0.5, 0.0), np.maximum(np.abs(dj) - 0.5, 0.0))
    inside = gap * resolution <= inflate + GRID_TOLERANCE
    offsets: NDArray[np.intp] = np.column_stack([di[inside], dj[inside]]).astype(np.intp)
    return offsets


def from_occupied(
    occupied: ArrayLike,
    resolution: float,
    inflation: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> PlanningGrid:
    """A built map's configuration space: ``occupied`` dilated by ``inflation``, plus the walls.

    A cell is blocked when its centre lies within ``inflation`` of an occupied cell's square
    (:func:`inflation_offsets`) or of the grid's edge, which stands for the arena's walls, as in
    :func:`~rl_navigation.sim.geometry.occupancy_grid`. Unknown cells are whatever ``occupied``
    says of them: an ``OccupancyGrid``'s ``occupied()`` leaves them free. The result's
    ``occupied`` is the mask itself.

    The walls lie at the grid's edges, ``rows * resolution`` and ``cols * resolution`` from the
    origin. ``occupancy_grid`` puts them at the arena's own size, which is the same only when the
    arena's sides are whole numbers of cells (a 5.02 m arena at 5 cm has 101 cells, whose far
    edge is at 5.05 m): build the grid from such an arena with :func:`from_arena`.

    Args:
        occupied: ``(rows, cols)`` booleans, True for an occupied cell.
        resolution: Cell side in metres.
        inflation: The inflation ``I`` in metres.
        origin: ``(x0, y0)``, the grid's lower-left corner.

    Raises:
        ValueError: A negative or non-finite inflation, or a bad grid (see :class:`PlanningGrid`).
    """
    mask = np.array(occupied, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"from_occupied needs a non-empty 2D mask, got shape {mask.shape}.")
    offsets = inflation_offsets(resolution, inflation)
    rows, cols = mask.shape
    reach = int(np.abs(offsets).max())
    padded = np.pad(mask, reach)
    blocked = np.zeros_like(mask)
    for di, dj in offsets:  # cell (i, j) is blocked by an occupied cell at (i + di, j + dj)
        blocked |= padded[reach + di : reach + di + rows, reach + dj : reach + dj + cols]
    i, j = np.arange(rows), np.arange(cols)
    to_edge_y = resolution * np.minimum(i + 0.5, rows - i - 0.5)
    to_edge_x = resolution * np.minimum(j + 0.5, cols - j - 0.5)
    walls = np.minimum.outer(to_edge_y, to_edge_x) <= inflation + GRID_TOLERANCE
    return PlanningGrid(blocked | walls, resolution, origin, occupied=mask)


def reachable(grid: PlanningGrid, start: Cell, goal: Cell) -> bool:
    """Whether A* can find a path from ``start`` to ``goal``, without searching.

    The start cell counts as free (the robot stands on it and may leave it). A diagonal move is
    allowed only between two free orthogonal neighbours, so it is two 4-connected moves and
    4-connected reachability is the planner's: :func:`~rl_navigation.sim.layouts.connected`'s
    vectorised flood fill, far cheaper than a search that expands every free cell to fail.

    Raises:
        ValueError: A cell that is not two integers or lies outside the grid.
    """
    start, goal = _checked_cell(grid, start, "start"), _checked_cell(grid, goal, "goal")
    free = ~grid.blocked
    free[start] = True
    return connected(free, start, goal)
