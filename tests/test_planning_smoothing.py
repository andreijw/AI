"""rl_navigation.planning.smoothing: line-of-sight checks and shortcutting of a grid path."""

import math

import numpy as np
import pytest

from rl_navigation.mapping.beams import traverse_cells
from rl_navigation.planning.astar import astar
from rl_navigation.planning.grid import PlanningGrid, from_arena
from rl_navigation.planning.smoothing import (
    polyline_length,
    segment_free,
    segments_free,
    shortcut,
)
from rl_navigation.sim import LayoutParams, clearance, generate_layout


def test_segment_free_on_an_empty_grid_and_across_a_blocked_cell():
    blocked = np.zeros((4, 4), dtype=bool)
    grid = PlanningGrid(blocked, 1.0)
    assert segment_free(grid, [0.5, 0.5], [3.5, 3.5])
    blocked[2, 2] = True
    grid = PlanningGrid(blocked, 1.0)
    assert not segment_free(grid, [0.5, 0.5], [3.5, 3.5]), "the diagonal crosses cell (2, 2)"
    assert segment_free(grid, [0.5, 0.5], [3.5, 0.5]), "the bottom row is free"


def test_a_segment_leaving_the_grid_is_not_free():
    grid = PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0)
    assert not segment_free(grid, [0.5, 0.5], [2.5, 0.5])
    assert not segment_free(grid, [-0.5, 0.5], [1.5, 0.5])


def test_a_point_segment_is_free_exactly_when_its_cell_is():
    blocked = np.array([[False, True]])
    grid = PlanningGrid(blocked, 1.0)
    assert segment_free(grid, [0.5, 0.5], [0.5, 0.5])
    assert not segment_free(grid, [1.5, 0.5], [1.5, 0.5])
    assert not segment_free(grid, [2.5, 0.5], [2.5, 0.5]), "a point outside the grid is not free"


def test_a_segment_through_a_corner_of_two_blocked_cells_is_free_where_astar_is_not():
    """``traverse_cells`` drops the corner slivers of a ray through a grid vertex, so the segment
    from (0, 0) to (1, 1) through the shared corner of the two blocked cells crosses no blocked
    cell, while A* refuses the same diagonal move. Pinned on purpose: on an inflated grid both
    cells lie within the inflation, so the segment keeps the clearance the planner promised."""
    grid = PlanningGrid(np.array([[0, 1], [1, 0]], dtype=bool), 1.0)
    assert segment_free(grid, [0.5, 0.5], [1.5, 1.5])
    assert astar(grid, (0, 0), (1, 1)) is None


def test_shortcut_on_an_empty_grid_keeps_only_the_ends():
    grid = PlanningGrid(np.zeros((10, 10), dtype=bool), 0.1)
    points = np.array([[0.05, 0.05], [0.15, 0.05], [0.25, 0.15], [0.35, 0.25], [0.95, 0.95]])
    np.testing.assert_array_equal(shortcut(grid, points), points[[0, -1]])


def test_shortcut_keeps_the_corner_of_an_l_shaped_corridor():
    blocked = np.ones((5, 5), dtype=bool)
    blocked[0, :] = False  # the bottom row
    blocked[:, 4] = False  # the right column
    grid = PlanningGrid(blocked, 1.0)
    points = np.array(
        [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5], [4.5, 0.5], [4.5, 1.5], [4.5, 4.5]]
    )
    np.testing.assert_array_equal(shortcut(grid, points), [[0.5, 0.5], [4.5, 0.5], [4.5, 4.5]])


def test_shortcut_of_one_or_two_points_is_unchanged():
    grid = PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0)
    for points in ([[0.5, 0.5]], [[0.5, 0.5], [1.5, 1.5]]):
        np.testing.assert_array_equal(shortcut(grid, points), points)


def test_shortcut_refuses_points_that_are_not_a_sequence_of_2d_points():
    grid = PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0)
    for bad in (np.zeros((0, 2)), np.zeros((3, 3)), [[0.5, np.nan]]):
        with pytest.raises(ValueError, match="shortcut"):
            shortcut(grid, bad)


def test_polyline_length():
    assert polyline_length([[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]]) == pytest.approx(6.0)
    assert polyline_length([[1.0, 1.0]]) == 0.0


@pytest.mark.parametrize("seed", range(1, 21))
def test_shortcut_of_a_layout_path_is_a_shorter_free_subsequence_with_clearance(seed):
    params = LayoutParams()
    layout = generate_layout(np.random.default_rng(seed), params)
    resolution, margin = params.grid_resolution, 0.05
    inflation = params.robot_radius + resolution / 2.0 + margin
    grid = from_arena(layout.arena, resolution, inflation)
    start, goal = grid.cell_of(layout.start[:2]), grid.cell_of(layout.goal)
    path = astar(grid, start, goal)
    if path is None:  # the margin closed the way; the planner's ladder steps it down
        grid = from_arena(layout.arena, resolution, params.robot_radius + resolution / 2.0)
        margin, path = 0.0, astar(grid, start, goal)
    assert path is not None
    points = np.vstack([layout.start[:2], grid.centres(path.cells[1:-1]), layout.goal])
    smooth = shortcut(grid, points)
    np.testing.assert_array_equal(smooth[[0, -1]], points[[0, -1]])
    index = [int(np.flatnonzero((points == p).all(axis=1))[0]) for p in smooth]
    assert index == sorted(index), "the shortcut must keep the points' order"
    assert polyline_length(smooth) <= polyline_length(points) + 1e-12
    for a, b in zip(smooth[:-1], smooth[1:], strict=True):
        assert segment_free(grid, a, b)
        # Every point of a free segment lies in a free cell, within res/sqrt2 of its centre, whose
        # clearance exceeds the inflation: r + margin - (res/sqrt2 - res/2) at least.
        samples = a + np.linspace(0.0, 1.0, 200)[:, None] * (b - a)
        floor = params.robot_radius + margin - resolution * (np.sqrt(0.5) - 0.5)
        assert clearance(layout.arena, samples).min() > floor - 1e-9, f"layout seed {seed}"


def reference_segment_free(grid, a, b):
    """One segment at a time, as first written: its own traversal, its zero-length case apart."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.hypot(*(b - a)))
    if length == 0.0:
        try:
            return not grid.blocked[grid.cell_of(a)]
        except ValueError:
            return False
    visits = traverse_cells(
        a[None], [math.atan2(*(b - a)[::-1])], [length], grid.resolution, grid.shape, grid.origin
    )
    kept = visits.kept[0]
    if not visits.inside[0][kept].all():
        return False
    return not grid.blocked[visits.rows[0][kept], visits.cols[0][kept]].any()


def reference_shortcut(grid, points):
    """The shortcut as first written: from each kept point, test the later points one segment at
    a time from the farthest down, and keep the first in line of sight (or the next point)."""
    kept = [0]
    while kept[-1] < len(points) - 1:
        here, target = kept[-1], len(points) - 1
        while target > here + 1 and not reference_segment_free(grid, points[here], points[target]):
            target -= 1
        kept.append(target)
    return points[kept]


@pytest.mark.parametrize("seed", range(10000, 10200))
def test_the_batched_shortcut_equals_the_one_segment_at_a_time_version(seed):
    params = LayoutParams()
    layout = generate_layout(np.random.default_rng(seed), params)
    resolution = params.grid_resolution
    for margin in (0.08, 0.0):
        grid = from_arena(layout.arena, resolution, params.robot_radius + resolution / 2 + margin)
        path = astar(grid, grid.cell_of(layout.start[:2]), grid.cell_of(layout.goal))
        if path is not None:
            break
    assert path is not None
    points = np.vstack([layout.start[:2], grid.centres(path.cells[1:-1]), layout.goal])
    np.testing.assert_array_equal(shortcut(grid, points), reference_shortcut(grid, points))


def test_segments_free_matches_the_one_segment_reference():
    rng = np.random.default_rng(4)
    grid = PlanningGrid(rng.random((12, 12)) < 0.15, 0.5)
    start = np.array([3.1, 2.9])
    ends = np.vstack([rng.uniform(-0.5, 6.5, (300, 2)), start, start + [0.5, 0.0]])
    expected = [reference_segment_free(grid, start, end) for end in ends]
    assert segments_free(grid, start, ends).tolist() == expected
    assert segments_free(grid, start, np.zeros((0, 2))).shape == (0,)
