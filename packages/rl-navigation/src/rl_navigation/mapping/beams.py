"""The cells a ray crosses on a grid, and which of them a range reading marks free or occupied.

The traversal is exact: it takes the distances at which each ray crosses the grid's vertical and
horizontal lines (after Amanatides and Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
1987), for every ray at once, with no loop over rays or cells. A ray through a cell corner crosses
both lines at almost the same distance, which leaves a sliver visit of a side cell a rounding error
long; visits shorter than :data:`TRAVERSE_TOLERANCE` are not kept.

The grid's frame is :func:`rl_navigation.sim.geometry.occupancy_grid`'s: cell ``[i, j]`` spans
``x0 + [j, j + 1) resolution`` and ``y0 + [i, i + 1) resolution``, rows along y. Imports NumPy and
``rl_navigation.sim`` only.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.sensors import RangeSensor

# A visit shorter than this, in metres, is a corner sliver and is not kept.
TRAVERSE_TOLERANCE = 1e-9


class CellVisits(NamedTuple):
    """The cells each of ``R`` rays crosses, in order, padded to ``M`` visits per ray.

    Every field is ``(R, M)``. A visit spans ``[t_in, t_out)`` metres along its ray; the
    intervals of a ray chain from 0 to its length, so padding visits have ``t_in == t_out``.

    Attributes:
        rows: Row of each visit's cell (along y); may lie outside the grid.
        cols: Column of each visit's cell (along x); may lie outside the grid.
        t_in: Distance along the ray where the visit starts.
        t_out: Distance where it ends.
        kept: The visit is longer than :data:`TRAVERSE_TOLERANCE` (not padding, not a sliver).
        inside: The visit's cell lies in the grid.
    """

    rows: NDArray[np.intp]
    cols: NDArray[np.intp]
    t_in: FloatArray
    t_out: FloatArray
    kept: NDArray[np.bool_]
    inside: NDArray[np.bool_]


def sensor_rays(sensor: RangeSensor, poses: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Every ray of every mount from many poses, flat, each starting at its mount.

    Args:
        sensor: The sensor (:meth:`RangeSensor.mount_origins`, :meth:`RangeSensor.ray_angles`).
        poses: ``(P, 3)`` poses ``(x, y, theta)``.

    Returns:
        ``(origins, angles)``: ``(P N K, 2)`` and ``(P N K,)``, pose by pose, mount by mount, then
        ray by ray, so ray ``r`` of a pose reads the scan's reading ``r // K``.
    """
    origins = np.repeat(sensor.mount_origins(poses).reshape(-1, 2), sensor.rays_per_cone, axis=0)
    return origins, sensor.ray_angles(poses).reshape(-1)


def traverse_cells(
    origins: ArrayLike,
    angles: ArrayLike,
    lengths: ArrayLike,
    resolution: float,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> CellVisits:
    """Every cell each ray crosses from its origin to its length, in order.

    A visit's cell is the one holding the midpoint of its interval, so a ray along a grid line,
    or from an origin on one, belongs to the cell above it or to its right.

    Args:
        origins: ``(R, 2)`` ray origins in metres.
        angles: ``(R,)`` ray directions in radians.
        lengths: ``(R,)`` how far to follow each ray, in metres (at least 0).
        resolution: Cell side in metres.
        shape: ``(rows, cols)`` of the grid.
        origin: ``(x0, y0)``, the grid's lower-left corner in metres.

    Returns:
        The visits, ``(R, M)`` per field.

    Raises:
        ValueError: Arrays of mismatched shapes, a value that is not finite, a negative length or
            a resolution that is not positive.
    """
    starts = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
    directions = np.asarray(angles, dtype=np.float64).reshape(-1)
    ends = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if not len(starts) == len(directions) == len(ends):
        raise ValueError(
            f"traverse_cells needs one origin, angle and length per ray, got {len(starts)} "
            f"origins, {len(directions)} angles and {len(ends)} lengths (rays)."
        )
    if not (np.isfinite(starts).all() and np.isfinite(directions).all()):
        raise ValueError("traverse_cells needs finite origins and angles.")
    if not (np.isfinite(ends).all() and (ends >= 0.0).all()):
        raise ValueError("traverse_cells needs finite lengths of at least 0.")
    if not resolution > 0.0:
        raise ValueError(f"traverse_cells needs a positive resolution, got {resolution}.")
    local = (starts - np.asarray(origin, dtype=np.float64)) / resolution  # in cells
    unit = np.stack([np.cos(directions), np.sin(directions)], axis=1)  # (R, 2)
    longest = float(ends.max(initial=0.0)) / resolution
    steps = np.arange(int(np.ceil(longest)) + 2)  # lines a ray can cross per axis, plus one
    crossings = [np.zeros((len(ends), 1)), ends[:, None]]
    for axis in (0, 1):
        start, step = local[:, axis], unit[:, axis]
        forward = step > 0.0
        first = np.where(forward, np.floor(start) + 1.0, np.ceil(start) - 1.0)
        lines = first[:, None] + np.where(forward[:, None], steps, -steps)
        # A zero component (sin 0) crosses no line of that family. A tiny one (cos(pi / 2) is
        # 6e-17) needs no tolerance: its crossings lie ~1e16 m away and the clip below drops them.
        flat = step == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (lines - start[:, None]) * resolution / step[:, None]
        crossings.append(np.where(flat[:, None], np.inf, t))
    t = np.sort(np.minimum(np.concatenate(crossings, axis=1), ends[:, None]), axis=1)
    t_in, t_out = t[:, :-1], t[:, 1:]
    middle = 0.5 * (t_in + t_out) / resolution
    cols = np.floor(local[:, 0, None] + middle * unit[:, 0, None]).astype(np.intp)
    rows = np.floor(local[:, 1, None] + middle * unit[:, 1, None]).astype(np.intp)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return CellVisits(rows, cols, t_in, t_out, t_out - t_in > TRAVERSE_TOLERANCE, inside)


def endpoint_visits(
    visits: CellVisits, ranges: ArrayLike
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Which visits a range reading marks occupied and which free, for each ray.

    This is the inverse sensor model of Thrun, Burgard and Fox, *Probabilistic Robotics*,
    Table 9.2, restricted to the cells a ray crosses, with the obstacle's depth (alpha) being the
    cell the reading ends in:

    - **occupied**: the ray's last kept visit with ``t_in <= z``, if its cell is in the grid. A
      reading ending exactly on a cell border marks the cell behind it; one ending in a dropped
      corner sliver marks the kept visit before it; one ending outside the grid marks nothing.
    - **free**: every other kept visit in the grid that ends at or before ``z``.

    Visits beyond ``z`` are neither.

    Args:
        visits: From :func:`traverse_cells`, followed at least ``z`` along each ray.
        ranges: ``(R,)`` each ray's reading ``z``, in metres.

    Returns:
        ``(occupied, free)``, each ``(R, M)`` booleans, disjoint; at most one occupied per ray.
    """
    z = np.asarray(ranges, dtype=np.float64).reshape(-1, 1)
    reached = visits.kept & (visits.t_in <= z)
    last = reached.shape[1] - 1 - np.argmax(reached[:, ::-1], axis=1)
    occupied = np.zeros_like(reached)
    ends = np.flatnonzero(reached.any(axis=1))
    occupied[ends, last[ends]] = True
    occupied &= visits.inside
    free = visits.kept & visits.inside & (visits.t_out <= z) & ~occupied
    return occupied, free
