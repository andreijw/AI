"""Arena geometry: obstacles, batched ray casting, swept-disc collision and an occupancy grid.

Coordinates are metres in the arena frame: the origin is the lower-left corner, x points right and
y up, and angles are radians counter-clockwise from +x. Everything is float64.

Obstacles are circles, axis-aligned boxes and free wall segments of zero thickness. The arena's
four boundary walls, each box's four edges and the free walls together make :attr:`Arena.segments`,
so the ray caster and the collision check treat every straight surface the same way.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Coordinate = FloatArray | float

# |cross(direction, edge)| below this is a ray parallel to a segment: no hit.
PARALLEL_TOLERANCE = 1e-12
# A ray through a segment's end point within this (in units of the segment's length) still hits it,
# so a ray through the corner two edges share cannot slip between them by rounding.
ENDPOINT_TOLERANCE = 1e-12
# A cell centre this close to exactly ``inflate`` from an obstacle counts as within it, whatever
# the rounding of the centre's coordinates.
GRID_TOLERANCE = 1e-9
# Largest number of elements of one (origins, rays, surfaces) temporary in cast_rays: 2^20 float64
# values are 8 MiB, and the ray cast holds a handful of them at once.
MAX_RAY_ELEMENTS = 2**20


def _read_only(values: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    """``values`` as a read-only float64 copy of ``shape``, -1 standing for any length."""
    array = np.array(values, dtype=np.float64)
    if array.size == 0:
        array = np.zeros(tuple(0 if n == -1 else n for n in shape), dtype=np.float64)
    if array.ndim != len(shape) or any(
        n not in (-1, m) for n, m in zip(shape, array.shape, strict=True)
    ):
        wanted = "(" + ", ".join("N" if n == -1 else str(n) for n in shape) + ")"
        raise ValueError(f"Arena {name} should have shape {wanted}, got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Arena:
    """A rectangular arena ``[0, width] x [0, height]`` and the obstacles inside it.

    Args:
        width: Arena width in metres (> 0).
        height: Arena height in metres (> 0).
        circle_centers: ``(C, 2)`` circle centres.
        circle_radii: ``(C,)`` circle radii (> 0).
        boxes: ``(B, 4)`` axis-aligned boxes as ``(x_min, y_min, x_max, y_max)``.
        walls: ``(W, 2, 2)`` free wall segments, each two end points.

    Every array is stored as a read-only float64 copy. ``segments`` is derived: ``(S, 2, 2)`` with
    the four boundary walls first, then four edges per box, then the free walls. Arenas compare by
    identity (``eq=False``): two arenas with equal obstacles are not ``==``.

    Raises:
        ValueError: A non-positive size, a radius that is not positive, a box with no area, or an
            array of the wrong shape.
    """

    width: float
    height: float
    circle_centers: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    circle_radii: FloatArray = field(default_factory=lambda: np.zeros(0))
    boxes: FloatArray = field(default_factory=lambda: np.zeros((0, 4)))
    walls: FloatArray = field(default_factory=lambda: np.zeros((0, 2, 2)))
    segments: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (self.width > 0 and self.height > 0):
            raise ValueError(
                f"Arena width and height must be positive, got {self.width}, {self.height}."
            )
        centers = _read_only(self.circle_centers, (-1, 2), "circle_centers")
        radii = _read_only(self.circle_radii, (-1,), "circle_radii")
        boxes = _read_only(self.boxes, (-1, 4), "boxes")
        walls = _read_only(self.walls, (-1, 2, 2), "walls")
        if len(radii) != len(centers) or (radii <= 0).any():
            raise ValueError(
                f"Arena circle_radii should be {len(centers)} positive values, one per centre, "
                f"got {radii.tolist()}."
            )
        if ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
            raise ValueError("Arena boxes need x_min < x_max and y_min < y_max.")
        w, h = float(self.width), float(self.height)
        corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        segments = np.concatenate(
            [np.stack([corners, np.roll(corners, -1, axis=0)], axis=1), _box_edges(boxes), walls]
        )
        segments.setflags(write=False)
        for name, value in [
            ("width", w),
            ("height", h),
            ("circle_centers", centers),
            ("circle_radii", radii),
            ("boxes", boxes),
            ("walls", walls),
            ("segments", segments),
        ]:
            object.__setattr__(self, name, value)


def _box_edges(boxes: FloatArray) -> FloatArray:
    """The four edges of each box, ``(4 B, 2, 2)``, counter-clockwise from the lower-left corner."""
    x0, y0, x1, y1 = boxes.T
    corners = np.stack(
        [
            np.stack([x0, y0], -1),
            np.stack([x1, y0], -1),
            np.stack([x1, y1], -1),
            np.stack([x0, y1], -1),
        ],
        axis=1,
    )  # (B, 4, 2)
    return np.stack([corners, np.roll(corners, -1, axis=1)], axis=2).reshape(-1, 2, 2)


def cast_rays(
    arena: Arena, origins: ArrayLike, angles: ArrayLike, max_elements: int = MAX_RAY_ELEMENTS
) -> FloatArray:
    """Distance along each ray to the first surface it meets.

    Segments are hit where ``origin + t d = a + u (b - a)`` with ``t >= 0`` and ``0 <= u <= 1``,
    solved with 2D cross products; a ray parallel to a segment (collinear included) never hits it.
    Circles are hit at the nearer root ``t >= 0`` of ``|origin + t d - c|^2 = r^2``; a tangent ray
    (zero discriminant) hits. From inside a circle the ray reads the distance to its far side.

    Args:
        arena: The arena.
        origins: ``(P, 2)`` ray origins.
        angles: ``(P, R)`` ray directions in radians, ``R`` rays from each origin.
        max_elements: Origins are cast in chunks whose ``(chunk, R, surfaces)`` temporaries hold at
            most this many elements (at least one origin per chunk), bounding memory for many
            particles; the ranges do not depend on it.

    Returns:
        ``(P, R)`` distances; ``inf`` where a ray meets nothing (only possible from outside).

    Raises:
        ValueError: ``origins`` is not ``(P, 2)``, ``angles`` is not ``(P, R)``, or either holds a
            value that is not finite.
    """
    origins = np.asarray(origins, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if (
        origins.ndim != 2
        or origins.shape[1] != 2
        or angles.ndim != 2
        or len(angles) != len(origins)
    ):
        raise ValueError(
            f"cast_rays needs origins of shape (P, 2) and angles of shape (P, R), got origins "
            f"{origins.shape} and angles {angles.shape}."
        )
    if not (np.isfinite(origins).all() and np.isfinite(angles).all()):
        raise ValueError("cast_rays needs finite origins and angles.")
    surfaces = max(len(arena.segments), len(arena.circle_radii), 1)
    chunk = max(1, max_elements // max(angles.shape[1] * surfaces, 1))
    if len(origins) <= chunk:
        return _cast_chunk(arena, origins, angles)
    return np.concatenate(
        [
            _cast_chunk(arena, origins[i : i + chunk], angles[i : i + chunk])
            for i in range(0, len(origins), chunk)
        ]
    )


def _cast_chunk(arena: Arena, origins: FloatArray, angles: FloatArray) -> FloatArray:
    """:func:`cast_rays` on validated arrays, in one pass."""
    dx = np.cos(angles)[..., None]  # (P, R, 1)
    dy = np.sin(angles)[..., None]
    ox = origins[:, 0, None, None]  # (P, 1, 1)
    oy = origins[:, 1, None, None]

    a = arena.segments[:, 0]  # (S, 2)
    ex, ey = (arena.segments[:, 1] - a).T  # (S,)
    wx, wy = a[:, 0] - ox, a[:, 1] - oy  # (P, 1, S)
    denom = dx * ey - dy * ex  # (P, R, S)
    parallel = np.abs(denom) < PARALLEL_TOLERANCE
    denom = np.where(parallel, 1.0, denom)
    t = (wx * ey - wy * ex) / denom
    u = (wx * dy - wy * dx) / denom
    hit = ~parallel & (t >= 0.0) & (u >= -ENDPOINT_TOLERANCE) & (u <= 1.0 + ENDPOINT_TOLERANCE)
    nearest = np.where(hit, t, np.inf).min(axis=-1)

    if len(arena.circle_radii):
        fx = ox - arena.circle_centers[:, 0]  # (P, 1, C)
        fy = oy - arena.circle_centers[:, 1]
        b = fx * dx + fy * dy  # (P, R, C)
        disc = b * b - (fx * fx + fy * fy - arena.circle_radii**2)
        root = np.sqrt(np.maximum(disc, 0.0))
        near, far = -b - root, -b + root
        t_circle = np.where(near >= 0.0, near, far)
        hit = (disc >= 0.0) & (t_circle >= 0.0)
        nearest = np.minimum(nearest, np.where(hit, t_circle, np.inf).min(axis=-1))
    return nearest


def _distances_to_segments(
    px: Coordinate, py: Coordinate, ax: Coordinate, ay: Coordinate, ex: Coordinate, ey: Coordinate
) -> FloatArray:
    """Distances from points ``(px, py)`` to segments ``a + u e``, ``0 <= u <= 1``, broadcast.

    A zero-length segment is a point: its projection's numerator is 0 too, so the floor on the
    denominator gives ``u = 0``. (``np.maximum`` and ``np.minimum`` rather than ``np.where`` and
    ``np.clip``, which cost several times more on the few elements of a collision check.)
    """
    length2 = np.maximum(ex * ex + ey * ey, 1e-300)
    u = np.minimum(np.maximum(((px - ax) * ex + (py - ay) * ey) / length2, 0.0), 1.0)
    distances: FloatArray = np.hypot(px - ax - u * ex, py - ay - u * ey)
    return distances


def clearance(arena: Arena, points: ArrayLike) -> FloatArray:
    """Distance from each point to the nearest obstacle surface or arena wall.

    Args:
        arena: The arena.
        points: ``(N, 2)`` points.

    Returns:
        ``(N,)`` distances; 0 for a point inside a circle or a box, or outside the arena.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    (ax, ay), (ex, ey) = arena.segments[:, 0].T, (arena.segments[:, 1] - arena.segments[:, 0]).T
    gaps = _distances_to_segments(points[:, :1], points[:, 1:], ax, ay, ex, ey).min(axis=1)
    if len(arena.circle_radii):
        to_centre = np.linalg.norm(points[:, None, :] - arena.circle_centers, axis=-1)
        gaps = np.minimum(gaps, np.maximum(to_centre - arena.circle_radii, 0.0).min(axis=1))
    return np.where(_enclosed(arena, points), 0.0, gaps)


def _enclosed(arena: Arena, points: FloatArray) -> NDArray[np.bool_]:
    """``(N,)``: whether each point is inside a box or outside the arena."""
    x, y = points[:, 0, None], points[:, 1, None]
    boxes = arena.boxes
    inside_box = (
        (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])
    ).any(1)
    outside = (
        (points[:, 0] < 0)
        | (points[:, 0] > arena.width)
        | (points[:, 1] < 0)
        | (points[:, 1] > arena.height)
    )
    result: NDArray[np.bool_] = inside_box | outside
    return result


def segment_hits(arena: Arena, start: ArrayLike, end: ArrayLike, radius: float) -> bool:
    """Whether a disc of ``radius`` swept along the straight move ``start -> end`` touches anything.

    The swept disc touches a segment when the two segments come within ``radius`` (zero when they
    cross), and a circle when its centre comes within ``radius`` plus its radius of the move, so no
    move, however long, passes through a thin wall. A move that ends inside a box or outside the
    arena hits too, even far from every edge. A move of zero length is a point check. Touching
    (distance exactly ``radius``) counts as a hit.

    Args:
        arena: The arena.
        start: ``(2,)`` start of the move.
        end: ``(2,)`` end of the move.
        radius: The disc's radius in metres.

    Returns:
        True if the swept disc touches an obstacle, a free wall or the arena boundary.
    """
    x0, y0 = np.asarray(start, dtype=np.float64).reshape(2)
    x1, y1 = np.asarray(end, dtype=np.float64).reshape(2)
    boxes = arena.boxes
    if (
        not (0.0 <= x1 <= arena.width and 0.0 <= y1 <= arena.height)
        or (
            (x1 >= boxes[:, 0]) & (x1 <= boxes[:, 2]) & (y1 >= boxes[:, 1]) & (y1 <= boxes[:, 3])
        ).any()
    ):
        return True  # the move ends outside the arena or inside a box
    mx, my = x1 - x0, y1 - y0
    ax, ay = arena.segments[:, 0].T
    bx, by = arena.segments[:, 1].T
    ex, ey = bx - ax, by - ay
    # The move and a segment cross when each one's end points lie strictly on opposite sides of
    # the other (2D cross products of opposite signs); touching is left to the distances below.
    side_a, side_b = mx * (ay - y0) - my * (ax - x0), mx * (by - y0) - my * (bx - x0)
    side_0, side_1 = ex * (y0 - ay) - ey * (x0 - ax), ex * (y1 - ay) - ey * (x1 - ax)
    if ((side_a * side_b < 0.0) & (side_0 * side_1 < 0.0)).any():
        return True
    # Otherwise two segments are closest at an end point of one of them.
    ends = np.array([[x0], [x1]]), np.array([[y0], [y1]])
    if _distances_to_segments(*ends, ax, ay, ex, ey).min() <= radius:
        return True
    segment_ends = np.concatenate([ax, bx]), np.concatenate([ay, by])
    if _distances_to_segments(*segment_ends, x0, y0, mx, my).min() <= radius:
        return True
    cx, cy = arena.circle_centers.T
    return bool(
        (_distances_to_segments(cx, cy, x0, y0, mx, my) <= radius + arena.circle_radii).any()
    )


def occupancy_grid(arena: Arena, resolution: float, inflate: float = 0.0) -> NDArray[np.bool_]:
    """The arena as a boolean grid: True where a cell's centre is within ``inflate`` of an obstacle.

    Cell ``[i, j]`` has its centre at ``((j + 0.5) resolution, (i + 0.5) resolution)``: rows run
    along y, columns along x, row 0 at y = 0. A cell whose centre lies inside an obstacle is always
    occupied. Free walls and obstacles thinner than a cell can fall between cell centres; inflate by
    at least ``resolution / 2`` to keep them.

    Args:
        arena: The arena.
        resolution: Cell side in metres.
        inflate: Extra distance in metres (the robot's radius for a configuration-space map).

    Returns:
        ``(ceil(height / resolution), ceil(width / resolution))`` booleans.
    """
    rows = int(np.ceil(arena.height / resolution - GRID_TOLERANCE))
    cols = int(np.ceil(arena.width / resolution - GRID_TOLERANCE))
    ys, xs = np.meshgrid(
        (np.arange(rows) + 0.5) * resolution, (np.arange(cols) + 0.5) * resolution, indexing="ij"
    )
    centres = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return (clearance(arena, centres) <= inflate + GRID_TOLERANCE).reshape(rows, cols)
