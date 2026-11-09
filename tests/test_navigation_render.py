"""rl_navigation.sim.render's overlay painters: points, paths and pose markers on small images.

Every image here is 20 x 20 pixels at 10 pixels per metre (a 2 m square), y up: a point
``(x, y)`` falls in row ``19 - floor(10 y)`` and column ``floor(10 x)``.
"""

import tracemalloc

import numpy as np
import pytest

from rl_navigation.sim import render

PPM = 10
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def blank(rows: int = 20, cols: int = 20) -> np.ndarray:
    return np.zeros((rows, cols, 3), dtype=np.uint8)


def painted(image: np.ndarray) -> set[tuple[int, int]]:
    """``(row, col)`` of every pixel that is not black."""
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(image.any(axis=2)), strict=True)}


# ---------------------------------------------------------------------------
# paint_points
# ---------------------------------------------------------------------------


def test_paint_points_paints_one_pixel_per_point_by_default():
    image = blank()
    render.paint_points(image, [[0.55, 0.55], [1.25, 1.95]], RED, pixels_per_metre=PPM)
    assert painted(image) == {(14, 5), (0, 12)}
    assert tuple(image[14, 5]) == RED


def test_paint_points_paints_a_square_of_size_pixels_around_each_point():
    image = blank()
    render.paint_points(image, [[0.55, 0.55]], RED, size=3, pixels_per_metre=PPM)
    assert painted(image) == {(r, c) for r in (13, 14, 15) for c in (4, 5, 6)}


def test_paint_points_clips_at_the_image_edge_and_skips_points_outside_or_not_finite():
    """A 3-pixel square on the top-left pixel keeps its 4 pixels inside; a point off the image and
    a NaN point paint nothing (and do not wrap round to the far edge)."""
    image = blank()
    points = [[0.05, 1.95], [-1.0, -1.0], [5.0, 0.5], [np.nan, 0.5]]
    render.paint_points(image, points, RED, size=3, pixels_per_metre=PPM)
    assert painted(image) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_paint_points_takes_no_points():
    image = blank()
    render.paint_points(image, np.empty((0, 2)), RED, pixels_per_metre=PPM)
    assert not painted(image)


# ---------------------------------------------------------------------------
# paint_path
# ---------------------------------------------------------------------------


def test_paint_path_joins_consecutive_points():
    image = blank()
    render.paint_path(image, [[0.05, 0.55], [1.95, 0.55], [1.95, 1.95]], RED, pixels_per_metre=PPM)
    expected = {(14, c) for c in range(20)} | {(r, 19) for r in range(15)}
    assert painted(image) == expected


def test_paint_path_breaks_at_a_non_finite_point():
    """A NaN row ends one piece and starts the next: nothing joins (0.95, 0.55) to (1.05, 1.55)."""
    image = blank()
    points = [[0.05, 0.55], [0.95, 0.55], [np.nan, np.nan], [1.05, 1.55], [1.95, 1.55]]
    render.paint_path(image, points, RED, pixels_per_metre=PPM)
    assert painted(image) == {(14, c) for c in range(10)} | {(4, c) for c in range(10, 20)}


def test_paint_path_culls_a_segment_wholly_off_the_image():
    image = blank()
    render.paint_path(
        image, [[5.0, 5.0], [6.0, 6.0], [3.0, -1.0], [-3.0, -1.0]], RED, pixels_per_metre=PPM
    )
    assert not painted(image)


def test_paint_path_clips_a_huge_segment_without_sampling_it_whole():
    """An ellipse far larger than the arena: 2e7 m at half-pixel steps would be 4e9 samples. Only
    the part across the image is sampled, so the row is painted and the memory stays small."""
    image = blank()
    tracemalloc.start()
    try:
        render.paint_path(image, [[-1e7, 0.55], [1e7, 0.55]], RED, pixels_per_metre=PPM)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert painted(image) == {(14, c) for c in range(20)}
    assert peak < 1_000_000, f"painting one clipped segment peaked at {peak} bytes"


def test_paint_path_with_fewer_than_two_points_paints_nothing():
    image = blank()
    render.paint_path(image, [[0.55, 0.55]], RED, pixels_per_metre=PPM)
    render.paint_path(image, np.empty((0, 2)), RED, pixels_per_metre=PPM)
    assert not painted(image)


# ---------------------------------------------------------------------------
# paint_pose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pose",
    [(1.0, 1.0, 0.7), (0.05, 1.95, -2.0), (1.97, 0.02, 3.0)],
    ids=["centre", "top-left corner", "bottom-right corner"],
)
def test_paint_pose_paints_the_robot_as_draw_scene_does(pose):
    """The same disc and heading as draw_scene's robot, near the edges too, though painted on a
    local window: draw_scene with no beams and a goal of radius 0 paints only the robot."""
    radius = 0.45
    background = np.full((20, 20, 3), render.FREE_COLOUR, dtype=np.uint8)
    expected = render.draw_scene(
        background, pose, (1.0, 1.0), radius, 0.0, np.empty((0, 2, 2)), pixels_per_metre=PPM
    )
    image = background.copy()
    render.paint_pose(
        image, pose, radius, render.ROBOT_COLOUR, render.HEADING_COLOUR, pixels_per_metre=PPM
    )
    np.testing.assert_array_equal(image, expected)
    assert (image == render.ROBOT_COLOUR).all(axis=2).sum() > 10


def test_paint_pose_off_the_image_or_not_finite_paints_nothing():
    image = blank()
    render.paint_pose(image, (5.0, 5.0, 0.0), 0.3, RED, BLUE, pixels_per_metre=PPM)
    render.paint_pose(image, (np.nan, 1.0, 0.0), 0.3, RED, BLUE, pixels_per_metre=PPM)
    assert not painted(image)
