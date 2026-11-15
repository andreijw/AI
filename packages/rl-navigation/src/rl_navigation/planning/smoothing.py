"""Path smoothing by line-of-sight shortcuts on the grid the path was planned on.

A grid path turns in steps of 45 degrees. :func:`shortcut` pulls it tight: from each kept point it
jumps to the farthest later point whose straight segment crosses only free cells
(:func:`segments_free`), so the result is an any-angle polyline whose points are a subsequence of
the input. On a grid inflated by ``I``, every point of such a segment lies in a free cell, within
``resolution / sqrt(2)`` of its centre, whose clearance exceeds ``I``: the segment keeps at least
``I - resolution / sqrt(2)`` from every obstacle. Not so for a segment that starts in a blocked
cell (a robot that A* lets leave its blocked start cell): such a segment is never free, so the
shortcut keeps the next point after it, and that first segment has no clearance bound.

The cells a segment crosses are :func:`~rl_navigation.mapping.beams.traverse_cells`' exact
traversal, which drops a corner sliver: a segment through the shared corner of two diagonally
blocked cells counts as free, although A* refuses that diagonal move. On an inflated grid both of
those cells lie within the inflation, so the bound above still holds.

Imports NumPy, ``rl_navigation.mapping.beams`` and ``rl_navigation.planning.grid`` only.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.mapping.beams import traverse_cells
from rl_navigation.planning.grid import PlanningGrid
from rl_navigation.sim.geometry import FloatArray


def segments_free(grid: PlanningGrid, starts: ArrayLike, ends: ArrayLike) -> NDArray[np.bool_]:
    """Whether each segment from a start to its end crosses only free cells inside the grid.

    All segments are traversed in one batched call. A segment of zero length is free when its
    cell is; a segment that leaves the grid is not.

    Args:
        grid: The grid.
        starts: ``(2,)``, one start for every segment, or ``(K, 2)``, one each.
        ends: ``(K, 2)`` the ends.

    Returns:
        ``(K,)`` booleans.
    """
    targets = np.asarray(ends, dtype=np.float64).reshape(-1, 2)
    origins = np.broadcast_to(np.asarray(starts, dtype=np.float64), targets.shape)
    delta = targets - origins
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    rows, cols = grid.shape
    cells, inside = grid.cells_of(origins)
    free = inside & ~grid.blocked[cells[:, 0], cells[:, 1]]  # a segment of zero length
    moving = np.flatnonzero(lengths > 0.0)
    if len(moving) == 0:
        return free
    visits = traverse_cells(
        origins[moving],
        np.arctan2(delta[moving, 1], delta[moving, 0]),
        lengths[moving],
        grid.resolution,
        grid.shape,
        grid.origin,
    )
    inside_blocked = grid.blocked[visits.rows.clip(0, rows - 1), visits.cols.clip(0, cols - 1)]
    bad = visits.kept & (~visits.inside | inside_blocked)
    free[moving] = ~bad.any(axis=1)
    return free


def polyline_free(grid: PlanningGrid, points: ArrayLike) -> bool:
    """Whether every segment of a ``(N, 2)`` polyline is free (:func:`segments_free`, one call)."""
    path = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return bool(segments_free(grid, path[:-1], path[1:]).all())


def segment_free(grid: PlanningGrid, a: ArrayLike, b: ArrayLike) -> bool:
    """Whether the segment from ``a`` to ``b`` crosses only free cells inside the grid
    (:func:`segments_free` for one segment)."""
    return bool(segments_free(grid, a, np.asarray(b, dtype=np.float64).reshape(1, 2))[0])


def shortcut(grid: PlanningGrid, points: ArrayLike) -> FloatArray:
    """The polyline pulled tight: from each kept point, the farthest later point in line of sight.

    From each kept point, the segments to every later point are checked in one batched
    :func:`segments_free` call and the farthest free one is kept (the next point when none is),
    so the cost is one NumPy traversal per kept point rather than one per pair of points.

    Args:
        grid: The grid the points were planned on.
        points: ``(N, 2)`` the path's points in order, start and goal included.

    Returns:
        ``(M, 2)`` the kept points, ``M <= N``, the first and last always kept.

    Raises:
        ValueError: ``points`` is not a non-empty ``(N, 2)`` array of finite numbers.
    """
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0 or not np.isfinite(path).all():
        raise ValueError(f"shortcut needs (N, 2) finite points with N >= 1, got {path.shape}.")
    kept = [0]
    while kept[-1] < len(path) - 1:
        here = kept[-1]
        visible = np.flatnonzero(segments_free(grid, path[here], path[here + 1 :]))
        kept.append(here + 1 + (int(visible[-1]) if len(visible) else 0))
    smooth: FloatArray = path[kept]
    return smooth


def polyline_length(points: ArrayLike) -> float:
    """The length of a ``(N, 2)`` polyline in metres (0 for a single point)."""
    path = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(np.hypot(*np.diff(path, axis=0).T).sum())
