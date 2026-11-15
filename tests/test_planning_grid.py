"""rl_navigation.planning.grid: the configuration-space grid, its two inflations and reachability."""

import numpy as np
import pytest

from rl_navigation.planning.grid import (
    PlanningGrid,
    from_arena,
    from_occupied,
    inflation_offsets,
    reachable,
)
from rl_navigation.sim import Arena, LayoutParams, generate_layout, occupancy_grid, path_exists
from tests.planning_oracles import bellman_ford, distance_to_square


def offsets_set(resolution, inflation):
    return {tuple(int(v) for v in row) for row in inflation_offsets(resolution, inflation)}


@pytest.mark.parametrize(
    "inflation, count",
    [
        (0.0, 1),  # only the cell itself
        (0.5, 5),  # the four edge neighbours touch at 0.5; the diagonal corner is 0.707 away
        (0.75, 9),  # the diagonals' corners are at hypot(0.5, 0.5) = 0.707
        (1.5, 13),  # (2, 0) at 1.5; (2, 1) at hypot(1.5, 0.5) = 1.58 is out
    ],
)
def test_inflation_offsets_on_unit_cells_are_worked_out_by_hand(inflation, count):
    offsets = offsets_set(1.0, inflation)
    assert len(offsets) == count, f"I={inflation}: expected {count} offsets, got {sorted(offsets)}"
    assert (0, 0) in offsets


def test_inflation_offsets_measure_to_the_occupied_square_not_its_centre():
    """The review's diagonal case: at 5 cm cells and I = 0.18 m, offset (3, 3) has the occupied
    square's corner at 0.05 hypot(2.5, 2.5) = 0.177 m, inside I, while its centre is 0.212 m away,
    beyond even I + res/2 = 0.205 m; a centre-distance rule would leave the cell free."""
    offsets = offsets_set(0.05, 0.18)
    assert (3, 3) in offsets and (-3, 3) in offsets, "the diagonal at 0.177 m must be blocked"
    assert (4, 1) in offsets, "(4, 1) is 0.05 hypot(3.5, 0.5) = 0.177 m away: inside I"
    assert (4, 2) not in offsets, "(4, 2) is 0.05 hypot(3.5, 1.5) = 0.190 m away, beyond I"


@pytest.mark.parametrize("seed", range(6))
def test_from_occupied_blocks_exactly_the_cells_within_inflation_of_an_occupied_square(seed):
    rng = np.random.default_rng(seed)
    occupied = rng.random((9, 11)) < 0.08
    resolution, inflation = 0.05, float(rng.uniform(0.0, 0.2))
    grid = from_occupied(occupied, resolution, inflation, origin=(1.0, -2.0))
    rows, cols = occupied.shape
    for i in range(rows):
        for j in range(cols):
            centre = np.array([1.0 + (j + 0.5) * resolution, -2.0 + (i + 0.5) * resolution])
            near_obstacle = any(
                distance_to_square(
                    centre, (1.0 + m * resolution, -2.0 + k * resolution), resolution
                )
                <= inflation + 1e-9
                for k, m in zip(*np.nonzero(occupied), strict=True)
            )
            edge = min(
                centre[0] - 1.0,
                1.0 + cols * resolution - centre[0],
                centre[1] + 2.0,
                -2.0 + rows * resolution - centre[1],
            )
            expected = near_obstacle or edge <= inflation + 1e-9
            assert grid.blocked[i, j] == expected, (
                f"cell {(i, j)} at I={inflation:.3f}: blocked should be {expected} "
                f"(near an occupied square: {near_obstacle}, {edge:.3f} m from the edge)"
            )


def test_from_occupied_edge_band_matches_occupancy_grid_walls():
    """With nothing occupied, the blocked band is the walls, exactly as the true map inflates them."""
    arena = Arena(2.0, 1.5)
    for inflation in (0.0, 0.1, 0.175, 0.3):
        expected = occupancy_grid(arena, 0.05, inflate=inflation)
        got = from_occupied(np.zeros(expected.shape, dtype=bool), 0.05, inflation).blocked
        np.testing.assert_array_equal(got, expected, err_msg=f"wall band differs at I={inflation}")


def test_from_arena_is_the_simulators_inflated_grid():
    arena = Arena(
        3.0, 2.0, circle_centers=[[1.0, 1.0]], circle_radii=[0.3], boxes=[[2.0, 0.5, 2.4, 1.2]]
    )
    grid = from_arena(arena, 0.05, 0.175)
    np.testing.assert_array_equal(grid.blocked, occupancy_grid(arena, 0.05, inflate=0.175))
    assert grid.resolution == 0.05 and grid.origin == (0.0, 0.0) and grid.shape == (40, 60)


def test_a_larger_inflation_blocks_a_superset():
    rng = np.random.default_rng(3)
    occupied = rng.random((20, 20)) < 0.05
    small = from_occupied(occupied, 0.05, 0.1).blocked
    large = from_occupied(occupied, 0.05, 0.2).blocked
    assert (large | small == large).all() and large.sum() > small.sum()


def test_the_grid_is_read_only_and_a_copy():
    mask = np.zeros((3, 4), dtype=bool)
    grid = PlanningGrid(mask, 0.5)
    mask[0, 0] = True
    assert not grid.blocked[0, 0], "the grid must hold its own copy"
    with pytest.raises(ValueError):
        grid.blocked[0, 0] = True


def test_cell_of_and_centres_follow_occupancy_grid_convention():
    grid = PlanningGrid(np.zeros((4, 6), dtype=bool), 0.5, origin=(1.0, 2.0))
    assert grid.cell_of([1.0, 2.0]) == (0, 0)
    assert grid.cell_of([3.9, 3.6]) == (3, 5), "x picks the column, y the row"
    assert grid.cell_of([4.0, 4.0]) == (3, 5), "the far edge belongs to the last cell"
    np.testing.assert_allclose(grid.centres([(0, 0), (3, 5)]), [[1.25, 2.25], [3.75, 3.75]])
    for outside in ([0.99, 2.5], [2.0, 4.01], [np.nan, 2.5]):
        with pytest.raises(ValueError, match="outside"):
            grid.cell_of(outside)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocked": np.zeros(4, dtype=bool), "resolution": 1.0},
        {"blocked": np.zeros((0, 3), dtype=bool), "resolution": 1.0},
        {"blocked": np.zeros((2, 2), dtype=bool), "resolution": 0.0},
        {"blocked": np.zeros((2, 2), dtype=bool), "resolution": 1.0, "origin": (0.0, np.inf)},
    ],
)
def test_planning_grid_refuses_bad_arguments(kwargs):
    with pytest.raises(ValueError, match="PlanningGrid"):
        PlanningGrid(**kwargs)


def test_inflation_must_be_finite_and_not_negative():
    for bad in (-0.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="inflation"):
            inflation_offsets(0.05, bad)
        with pytest.raises(ValueError, match="inflation"):
            from_arena(Arena(1.0, 1.0), 0.05, bad)


def test_reachable_lets_the_robot_leave_a_blocked_start_and_refuses_a_blocked_goal():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[1, 1] = True
    grid = PlanningGrid(blocked, 1.0)
    assert reachable(grid, (1, 1), (0, 0)), "the robot stands on its own cell and may leave it"
    assert not reachable(grid, (0, 0), (1, 1)), "a blocked goal cannot be reached"
    assert reachable(grid, (0, 0), (0, 0))


def test_reachable_is_false_through_a_wall():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[:, 2] = True
    assert not reachable(PlanningGrid(blocked, 1.0), (2, 0), (2, 4))
    blocked[4, 2] = False
    assert reachable(PlanningGrid(blocked, 1.0), (2, 0), (2, 4))


@pytest.mark.parametrize("seed", range(40))
def test_reachable_agrees_with_eight_connected_search_without_corner_cutting(seed):
    """A diagonal move needs both orthogonal neighbours free, so it is two 4-connected moves:
    4-connected reachability equals the 8-connected planner's."""
    rng = np.random.default_rng(100 + seed)
    blocked = rng.random((8, 9)) < rng.uniform(0.1, 0.5)
    start, goal = tuple(rng.integers(0, 8, 2)), (int(rng.integers(0, 8)), int(rng.integers(0, 9)))
    start = (int(start[0]), int(start[1]))
    grid = PlanningGrid(blocked, 1.0)
    expected = bool(np.isfinite(bellman_ford(blocked, start, 1.0)[goal]))
    assert reachable(grid, start, goal) == expected, f"seed {seed}: {start} -> {goal}"


@pytest.mark.parametrize("seed", range(1, 11))
def test_reachable_on_the_inflated_true_map_agrees_with_path_exists(seed):
    params = LayoutParams()
    layout = generate_layout(np.random.default_rng(seed), params)
    resolution = params.grid_resolution
    grid = from_arena(layout.arena, resolution, params.robot_radius + resolution / 2.0)
    for goal in (layout.goal, np.array([0.3, 0.3]), np.array([4.7, 2.5])):
        expected = path_exists(
            layout.arena, layout.start[:2], goal, params.robot_radius, resolution
        )
        got = reachable(grid, grid.cell_of(layout.start[:2]), grid.cell_of(goal))
        assert got == expected, f"layout seed {seed}, goal {goal}: path_exists says {expected}"


def test_reachable_and_from_occupied_refuse_bad_input():
    grid = PlanningGrid(np.zeros((3, 3), dtype=bool), 1.0)
    with pytest.raises(ValueError, match="start cell"):
        reachable(grid, (3, 0), (0, 0))
    with pytest.raises(ValueError, match="goal cell"):
        reachable(grid, (0, 0), (0, -1))
    with pytest.raises(ValueError, match="from_occupied"):
        from_occupied(np.zeros(5, dtype=bool), 1.0, 0.1)
    with pytest.raises(ValueError, match="resolution"):
        inflation_offsets(0.0, 0.1)


def test_the_grids_keep_the_occupied_cells_apart_from_the_inflation():
    arena = Arena(2.0, 2.0, circle_centers=[[1.0, 1.0]], circle_radii=[0.3])
    grid = from_arena(arena, 0.05, 0.175)
    assert grid.occupied is not None
    np.testing.assert_array_equal(grid.occupied, occupancy_grid(arena, 0.05))
    assert (grid.blocked | grid.occupied == grid.blocked).all(), "occupied cells are blocked"
    assert grid.blocked.sum() > grid.occupied.sum()
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 4] = True
    built = from_occupied(mask, 0.05, 0.1)
    assert built.occupied is not None
    np.testing.assert_array_equal(built.occupied, mask)
    assert not built.occupied.flags.writeable and not built.blocked.flags.writeable
    mask[0, 0] = True
    assert not built.occupied[0, 0], "the grid must hold its own copy of the occupied cells"
    assert PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0).occupied is None


def test_an_occupied_mask_must_match_the_grid_and_be_blocked():
    with pytest.raises(ValueError, match="occupied"):
        PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0, occupied=np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="occupied"):
        PlanningGrid(np.zeros((2, 2), dtype=bool), 1.0, occupied=np.eye(2, dtype=bool))


def test_cells_of_agrees_with_cell_of_and_flags_points_outside():
    grid = PlanningGrid(np.zeros((4, 6), dtype=bool), 0.5, origin=(1.0, 2.0))
    points = np.array([[1.0, 2.0], [3.9, 3.6], [4.0, 4.0], [0.99, 2.5], [2.0, 4.01], [np.nan, 2.5]])
    cells, inside = grid.cells_of(points)
    assert inside.tolist() == [True, True, True, False, False, False]
    for point, cell, ok in zip(points, cells, inside, strict=True):
        if ok:
            assert tuple(cell) == grid.cell_of(point)
