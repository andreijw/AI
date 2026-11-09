"""A top-down picture of the arena as an RGB array, in NumPy alone.

The picture is ``pixels_per_metre`` pixels to the metre with y pointing up: row 0 is the top wall
(y = height) and column 0 the left wall (x = 0), so a point ``(x, y)`` falls in row
``rows - 1 - floor(y * pixels_per_metre)`` and column ``floor(x * pixels_per_metre)``.
:func:`arena_image` draws the obstacles once per layout; :func:`draw_scene` draws the beams, the
goal and the robot over a copy of it.

The overlay painters (:func:`paint_points`, :func:`paint_path` and :func:`paint_pose`) paint in
place in the colours they are given, for pictures other than the environment's frame (a
localization trial's particles, path and estimates). Each clips at the image's edge and skips
what is not finite.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import Arena, occupancy_grid

Image = NDArray[np.uint8]
Colour = tuple[int, int, int]

PIXELS_PER_METRE = 80
FREE_COLOUR = (255, 255, 255)
OBSTACLE_COLOUR = (60, 60, 60)
BEAM_COLOUR = (235, 120, 120)
GOAL_COLOUR = (40, 170, 70)
ROBOT_COLOUR = (40, 90, 200)
HEADING_COLOUR = (250, 200, 0)


def arena_image(arena: Arena, pixels_per_metre: float = PIXELS_PER_METRE) -> Image:
    """The arena's obstacles on a free background, ``(rows, cols, 3)`` uint8.

    A pixel is an obstacle when its centre is inside one (:func:`occupancy_grid` at one pixel per
    cell), so a free wall of zero thickness is not drawn.
    """
    occupied = occupancy_grid(arena, 1.0 / pixels_per_metre)[::-1]  # row 0 at the top
    image = np.empty(occupied.shape + (3,), dtype=np.uint8)
    image[...] = FREE_COLOUR
    image[occupied] = OBSTACLE_COLOUR
    return image


def _pixel_centres(image: Image, pixels_per_metre: float) -> tuple[NDArray[np.float64], ...]:
    """Arena coordinates ``(x, y)`` of every pixel centre, each ``(rows, cols)``."""
    rows, cols = image.shape[:2]
    x = (np.arange(cols) + 0.5) / pixels_per_metre
    y = (rows - 1 - np.arange(rows) + 0.5) / pixels_per_metre
    return np.meshgrid(x, y)


def _paint_disc(
    image: Image,
    centres: tuple[NDArray[np.float64], ...],
    centre: ArrayLike,
    radius: float,
    colour: tuple[int, int, int],
) -> None:
    cx, cy = np.asarray(centre, dtype=np.float64)[:2]
    xs, ys = centres
    image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2] = colour


def _paint_segments(
    image: Image,
    segments: NDArray[np.float64],
    pixels_per_metre: float,
    colour: tuple[int, int, int],
) -> None:
    """Paint ``(S, 2, 2)`` segments by sampling each at half-pixel steps."""
    rows, cols = image.shape[:2]
    for start, end in segments:
        samples = max(2, int(np.ceil(2.0 * np.linalg.norm(end - start) * pixels_per_metre)) + 1)
        points = np.linspace(start, end, samples)
        col = np.floor(points[:, 0] * pixels_per_metre).astype(int)
        row = rows - 1 - np.floor(points[:, 1] * pixels_per_metre).astype(int)
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        image[row[inside], col[inside]] = colour


def _clip_segments(
    segments: NDArray[np.float64], width: float, height: float
) -> NDArray[np.float64]:
    """The parts of ``(S, 2, 2)`` segments inside ``[0, width] x [0, height]``, by Liang-Barsky.

    A segment with a non-finite end, or wholly outside, is dropped; the rest are cut to the box,
    so painting one costs at most the box's diagonal in samples however long it was.
    """
    segments = segments[np.isfinite(segments).all(axis=(1, 2))]
    start, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    # Inside along each of the four edges while p * t <= q (x >= 0, x <= width, y >= 0, y <= height).
    p = np.stack([-delta[:, 0], delta[:, 0], -delta[:, 1], delta[:, 1]], axis=1)
    q = np.stack([start[:, 0], width - start[:, 0], start[:, 1], height - start[:, 1]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = q / p
    enter = np.where(p < 0, ratio, -np.inf).max(axis=1, initial=0.0)
    leave = np.where(p > 0, ratio, np.inf).min(axis=1, initial=1.0)
    keep = (enter <= leave) & ~((p == 0) & (q < 0)).any(axis=1)
    t = np.stack([enter, leave], axis=1)[keep]
    return start[keep, None] + t[..., None] * delta[keep, None]


def paint_points(
    image: Image,
    points: ArrayLike,
    colour: Colour,
    size: int = 1,
    pixels_per_metre: float = PIXELS_PER_METRE,
) -> None:
    """Paint a ``size`` x ``size`` pixel square at each point, in place.

    Args:
        image: ``(rows, cols, 3)`` uint8, changed in place.
        points: ``(P, 2)`` positions ``(x, y)`` in metres; non-finite ones are skipped.
        colour: The squares' colour.
        size: Side of each square in pixels: the point's pixel, then the next ``size - 1`` up and
            to the right split around it (one pixel for 1, a 3 x 3 block centred on it for 3).
        pixels_per_metre: Scale of the picture.
    """
    rows, cols = image.shape[:2]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    col = np.floor(points[:, 0] * pixels_per_metre)
    row = rows - 1 - np.floor(points[:, 1] * pixels_per_metre)
    near = (col >= -size) & (col < cols + size) & (row >= -size) & (row < rows + size)
    offsets = np.arange(size) - (size - 1) // 2
    row_grid = (row[near].astype(int)[:, None] + offsets)[:, :, None]
    col_grid = (col[near].astype(int)[:, None] + offsets)[:, None, :]
    row_grid, col_grid = np.broadcast_arrays(row_grid, col_grid)
    inside = (row_grid >= 0) & (row_grid < rows) & (col_grid >= 0) & (col_grid < cols)
    image[row_grid[inside], col_grid[inside]] = colour


def paint_path(
    image: Image,
    points: ArrayLike,
    colour: Colour,
    pixels_per_metre: float = PIXELS_PER_METRE,
) -> None:
    """Paint the polyline through ``(K, 2)`` points, in place.

    A non-finite point breaks the line (neither segment touching it is drawn), so a NaN row
    separates two pieces. Segments are cut to the image before they are sampled, so one far
    larger than the image (a lost filter's ellipse) costs no more than one across it.
    """
    rows, cols = image.shape[:2]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    clipped = _clip_segments(segments, cols / pixels_per_metre, rows / pixels_per_metre)
    _paint_segments(image, clipped, pixels_per_metre, colour)


def paint_pose(
    image: Image,
    pose: ArrayLike,
    radius: float,
    colour: Colour,
    heading_colour: Colour,
    pixels_per_metre: float = PIXELS_PER_METRE,
) -> None:
    """Paint a pose as a disc of ``radius`` metres with its heading from the centre, in place.

    The same pixels as :func:`draw_scene` paints for the robot, but only the window of pixels
    around the disc is tested, so a marker costs about ``(2 radius pixels_per_metre)^2``
    whatever the image's size. A non-finite pose paints nothing.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(pose).all():
        return
    rows, cols = image.shape[:2]
    x, y, theta = pose
    reach = radius * pixels_per_metre + 1
    col_lo = max(int(np.floor(x * pixels_per_metre - reach)), 0)
    col_hi = min(int(np.floor(x * pixels_per_metre + reach)) + 1, cols)
    row_lo = max(rows - 1 - int(np.floor(y * pixels_per_metre + reach)), 0)
    row_hi = min(rows - int(np.floor(y * pixels_per_metre - reach)), rows)
    if col_lo < col_hi and row_lo < row_hi:  # as _pixel_centres, on the window only
        xs, ys = np.meshgrid(
            (np.arange(col_lo, col_hi) + 0.5) / pixels_per_metre,
            (rows - 1 - np.arange(row_lo, row_hi) + 0.5) / pixels_per_metre,
        )
        window = image[row_lo:row_hi, col_lo:col_hi]
        window[(xs - x) ** 2 + (ys - y) ** 2 <= radius**2] = colour
    heading = pose[:2] + radius * np.array([np.cos(theta), np.sin(theta)])
    _paint_segments(image, np.array([[pose[:2], heading]]), pixels_per_metre, heading_colour)


def draw_scene(
    background: Image,
    pose: ArrayLike,
    goal: ArrayLike,
    robot_radius: float,
    goal_radius: float,
    beams: ArrayLike,
    pixels_per_metre: float = PIXELS_PER_METRE,
) -> Image:
    """A copy of ``background`` with the beams, then the goal, then the robot and its heading.

    Args:
        background: The arena from :func:`arena_image`, at the same ``pixels_per_metre``.
        pose: ``(3,)`` robot pose ``(x, y, theta)``.
        goal: ``(2,)`` goal position.
        robot_radius: Radius of the robot's disc in metres.
        goal_radius: Radius of the goal's disc in metres.
        beams: ``(B, 2, 2)`` sensor beams, each from its mount to where it read.
        pixels_per_metre: Scale of the picture.

    Returns:
        The frame, ``background``'s shape and dtype.
    """
    image = background.copy()
    pose = np.asarray(pose, dtype=np.float64)
    centres = _pixel_centres(image, pixels_per_metre)
    _paint_segments(
        image, np.asarray(beams, dtype=np.float64).reshape(-1, 2, 2), pixels_per_metre, BEAM_COLOUR
    )
    _paint_disc(image, centres, goal, goal_radius, GOAL_COLOUR)
    _paint_disc(image, centres, pose, robot_radius, ROBOT_COLOUR)
    heading = pose[:2] + robot_radius * np.array([np.cos(pose[2]), np.sin(pose[2])])
    _paint_segments(image, np.array([[pose[:2], heading]]), pixels_per_metre, HEADING_COLOUR)
    return image
