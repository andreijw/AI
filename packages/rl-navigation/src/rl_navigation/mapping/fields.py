"""The distance field of an occupancy grid: how far each cell is from the nearest occupied one.

This is the map behind Thrun's likelihood field (*Probabilistic Robotics*, 2005, section 6.4) and
the scan matcher (:mod:`rl_navigation.slam.scan_matching`). A cell's value is the distance from its
centre to the centre of the nearest occupied cell, clamped at ``max_distance``: centre to centre,
because a reading marks the cell its end point falls in, so the cell centre is all the map knows
of where the surface lies (up to half a cell off, the same way along a whole surface). The
planner's inflation (:mod:`rl_navigation.planning.grid`) asks a different question, the distance
to an occupied cell's square, and keeps its own code.

:class:`DistanceField` samples the field between cell centres by bilinear interpolation, with its
gradient, so a point's distance is continuous and a Gauss-Newton step can follow it. The frame is
the grid's: cell ``[i, j]`` has its centre at ``origin + ((j + 0.5) res, (i + 0.5) res)``.

Imports NumPy and ``rl_navigation.sim`` only, so it runs on the robot.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import FloatArray


def distance_field(occupied: ArrayLike, resolution: float, max_distance: float) -> FloatArray:
    """``(rows, cols)`` distance from each cell's centre to the nearest occupied cell's centre,
    clamped at ``max_distance``.

    Separable, as Euclidean distance transforms are (Felzenszwalb and Huttenlocher, 2012), but
    over a window: first, along each row, the squared column offset ``dj^2`` to the nearest occupied
    cell within ``R = ceil(max_distance / resolution)`` cells; then, down each column, the least
    ``di^2 + dj^2`` over row offsets within ``R``. The window holds every cell nearer than
    ``max_distance``, so the result is exact; ``4 R + 2`` vectorised passes.

    Args:
        occupied: ``(rows, cols)`` booleans.
        resolution: Cell side in metres (> 0).
        max_distance: The clamp in metres (> 0); a cell with no occupied cell nearer reads it.

    Raises:
        ValueError: A mask that is not 2-D, or a resolution or clamp that is not positive.
    """
    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"distance_field needs a 2-D occupied mask, got shape {mask.shape}.")
    if not (math.isfinite(resolution) and resolution > 0.0):
        raise ValueError(f"distance_field needs a positive resolution, got {resolution}.")
    if not (math.isfinite(max_distance) and max_distance > 0.0):
        raise ValueError(f"distance_field needs a positive max_distance, got {max_distance}.")
    reach = math.ceil(max_distance / resolution)
    rows, cols = mask.shape
    along_rows = np.pad(mask, ((0, 0), (reach, reach)))
    columns = np.full(mask.shape, np.inf)  # dj^2 to the nearest occupied cell in the row
    for dj in range(-reach, reach + 1):
        hit = along_rows[:, reach + dj : reach + dj + cols]
        np.minimum(columns, np.where(hit, float(dj * dj), np.inf), out=columns)
    down = np.pad(columns, ((reach, reach), (0, 0)), constant_values=np.inf)
    squared = np.full(mask.shape, np.inf)  # di^2 + dj^2, in cells
    for di in range(-reach, reach + 1):
        np.minimum(squared, di * di + down[reach + di : reach + di + rows], out=squared)
    field: FloatArray = np.minimum(resolution * np.sqrt(squared), max_distance)
    return field


class FieldSample(NamedTuple):
    """The field at ``K`` points.

    Attributes:
        values: ``(K,)`` interpolated distances in metres; ``max_distance`` outside.
        gradients: ``(K, 2)`` their gradient in metres per metre (``x``, ``y``); zero outside.
        inside: ``(K,)`` whether the four cell centres around the point lie in the grid.
    """

    values: FloatArray
    gradients: FloatArray
    inside: NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class DistanceField:
    """A :func:`distance_field` on a grid's frame, sampled bilinearly between cell centres.

    Attributes:
        values: ``(rows, cols)`` read-only distances, clamped at ``max_distance``.
        resolution: Cell side in metres.
        max_distance: The clamp in metres.
        origin: ``(x0, y0)``, the grid's lower-left corner in metres.
    """

    values: FloatArray
    resolution: float
    max_distance: float
    origin: tuple[float, float]

    @classmethod
    def from_occupied(
        cls,
        occupied: ArrayLike,
        resolution: float,
        max_distance: float,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "DistanceField":
        """The field of an occupied mask on the frame of a grid at ``origin``."""
        values = distance_field(occupied, resolution, max_distance)
        values.setflags(write=False)
        return cls(values, float(resolution), float(max_distance), (origin[0], origin[1]))

    def sample(self, points: ArrayLike) -> FieldSample:
        """The field and its gradient at ``(K, 2)`` points.

        With ``g = (p - origin) / resolution - 1/2`` on the lattice of cell centres, ``(i0, j0)``
        its floor and ``(fy, fx)`` its fractions, the value is the bilinear blend of the four
        centres around it and the gradient is its derivative:
        ``dD/dx = ((1 - fy)(D01 - D00) + fy (D11 - D10)) / resolution`` and
        ``dD/dy = ((1 - fx)(D10 - D00) + fx (D11 - D01)) / resolution``.
        """
        at = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rows, cols = self.values.shape
        lattice = (at - np.asarray(self.origin)) / self.resolution - 0.5
        corner = np.floor(lattice)
        fx, fy = (lattice - corner).T
        j0, i0 = corner.astype(np.intp).T
        inside = (i0 >= 0) & (j0 >= 0) & (i0 + 1 < rows) & (j0 + 1 < cols)
        i0, j0 = np.clip(i0, 0, rows - 2), np.clip(j0, 0, cols - 2)
        d00, d01 = self.values[i0, j0], self.values[i0, j0 + 1]
        d10, d11 = self.values[i0 + 1, j0], self.values[i0 + 1, j0 + 1]
        values = (1.0 - fy) * ((1.0 - fx) * d00 + fx * d01) + fy * ((1.0 - fx) * d10 + fx * d11)
        gx = ((1.0 - fy) * (d01 - d00) + fy * (d11 - d10)) / self.resolution
        gy = ((1.0 - fx) * (d10 - d00) + fx * (d11 - d01)) / self.resolution
        return FieldSample(
            np.where(inside, values, self.max_distance),
            np.where(inside[:, None], np.column_stack([gx, gy]), 0.0),
            inside,
        )
