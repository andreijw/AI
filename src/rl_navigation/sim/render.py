"""A top-down picture of the arena as an RGB array, in NumPy alone.

The picture is ``pixels_per_metre`` pixels to the metre with y pointing up: row 0 is the top wall
(y = height) and column 0 the left wall (x = 0), so a point ``(x, y)`` falls in row
``rows - 1 - floor(y * pixels_per_metre)`` and column ``floor(x * pixels_per_metre)``.
:func:`arena_image` draws the obstacles once per layout; :func:`draw_scene` draws the beams, the
goal and the robot over a copy of it.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.sim.geometry import Arena, occupancy_grid

Image = NDArray[np.uint8]

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
