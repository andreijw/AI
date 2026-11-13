"""A* on a :class:`~rl_navigation.planning.grid.PlanningGrid`: 8-connected, never cutting a corner.

A move goes to one of the eight neighbours and costs ``resolution`` straight or
``resolution * sqrt(2)`` diagonally. A diagonal move is allowed only when both orthogonal
neighbours it passes are free, so the robot never squeezes between two diagonally blocked cells,
and 8-connected reachability equals 4-connected reachability
(:func:`~rl_navigation.planning.grid.reachable`).

The heuristic is the octile distance, ``resolution * (max(di, dj) + (sqrt(2) - 1) min(di, dj))``:
the exact cost on an empty grid, so it is admissible and consistent, and a cell is final when it is
first taken from the queue (Hart, Nilsson and Raphael, 1968). The queue is a binary heap with lazy
deletion; ties go to the lower f, then to the larger g (deeper along the path), then to the
earlier push, so a result is deterministic. The start cell alone may be blocked and is left; a
robot deeper inside an inflated band has no path, and a blocked goal other than the start has none
either. The escape bubble (crossing cells blocked only by inflation, ``PlanningGrid.occupied``) and
a substitute for a blocked goal belong to the navigator, not to this search.

Imports NumPy, the standard library and ``rl_navigation.planning.grid`` only.
"""

import heapq
import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from rl_navigation.planning.grid import Cell, PlanningGrid, _checked_cell

SQRT2 = math.sqrt(2.0)

# (row step, column step, cost in cells) of the eight moves.
_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)


class GridPath(NamedTuple):
    """A path found by :func:`astar`.

    Attributes:
        cells: ``(K, 2)`` the ``(row, col)`` cells from start to goal, both included.
        cost: The path's length in metres, the sum of its moves.
        expanded: How many cells the search took from its queue, the goal included.
    """

    cells: NDArray[np.intp]
    cost: float
    expanded: int


def octile_distance(a: Cell, b: Cell, resolution: float) -> float:
    """The cost of the cheapest path between two cells of an empty grid, in metres."""
    di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
    return resolution * (max(di, dj) + (SQRT2 - 1.0) * min(di, dj))


def astar(grid: PlanningGrid, start: Cell, goal: Cell) -> GridPath | None:
    """The cheapest path from ``start`` to ``goal`` through free cells, or None if there is none.

    Args:
        grid: The configuration-space grid.
        start: ``(row, col)`` of the robot's cell; it may be blocked (only it).
        goal: ``(row, col)`` of the goal's cell.

    Returns:
        The path, or None when the goal is blocked or cannot be reached.

    Raises:
        ValueError: A cell that is not two integers or lies outside the grid.
    """
    start, goal = _checked_cell(grid, start, "start"), _checked_cell(grid, goal, "goal")
    rows, cols = grid.shape
    blocked = grid.blocked.ravel().tolist()  # Python lists index far faster than NumPy scalars
    origin, target = start[0] * cols + start[1], goal[0] * cols + goal[1]
    if blocked[target] and target != origin:  # the robot already stands on its own goal
        return None
    gi, gj = goal

    def heuristic(index: int) -> float:
        di, dj = abs(index // cols - gi), abs(index % cols - gj)
        return di + dj + (SQRT2 - 2.0) * min(di, dj)  # in cells: max + (sqrt2 - 1) min

    best = {origin: 0.0}
    parent: dict[int, int] = {}
    closed = bytearray(rows * cols)
    pushes = 0
    queue = [(heuristic(origin), -0.0, pushes, origin)]
    expanded = 0
    while queue:
        _, negative_g, _, node = heapq.heappop(queue)
        if closed[node]:
            continue
        closed[node] = 1
        expanded += 1
        if node == target:
            return _path(parent, node, cols, -negative_g * grid.resolution, expanded)
        g, i, j = -negative_g, node // cols, node % cols
        for di, dj, step in _MOVES:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            neighbour = ni * cols + nj
            if closed[neighbour] or blocked[neighbour]:
                continue
            if di and dj and (blocked[ni * cols + j] or blocked[i * cols + nj]):
                continue  # a diagonal past a blocked corner
            cost = g + step
            if cost < best.get(neighbour, math.inf):
                best[neighbour] = cost
                parent[neighbour] = node
                pushes += 1
                heapq.heappush(queue, (cost + heuristic(neighbour), -cost, pushes, neighbour))
    return None


def _path(parent: dict[int, int], node: int, cols: int, cost: float, expanded: int) -> GridPath:
    """Walk the parents back from the goal."""
    chain = [node]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    flat = np.array(chain[::-1], dtype=np.intp)
    return GridPath(np.column_stack([flat // cols, flat % cols]), cost, expanded)
