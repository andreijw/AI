"""rl_navigation.planning.astar: A* on an 8-connected grid that never cuts a blocked corner."""

import itertools
import math

import numpy as np
import pytest

from rl_navigation.planning.astar import GridPath, astar, octile_distance
from rl_navigation.planning.grid import PlanningGrid, from_arena, reachable
from rl_navigation.sim import LayoutParams, generate_layout, path_exists
from tests.planning_oracles import MOVES, SQRT2, allowed, bellman_ford, brute_force_cost, step_cost


def assert_valid_path(grid: PlanningGrid, path: GridPath, start, goal):
    """The path starts and ends where asked, moves one cell at a time through free cells (the
    start excepted) without cutting a corner, and its cost is the sum of its moves."""
    cells = [tuple(int(v) for v in c) for c in path.cells]
    assert cells[0] == tuple(start) and cells[-1] == tuple(goal), f"ends {cells[0]}, {cells[-1]}"
    total = 0.0
    for a, b in itertools.pairwise(cells):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, f"{a} -> {b} is not one move"
        assert allowed(grid.blocked, a, b), f"{a} -> {b} enters a blocked cell or cuts a corner"
        total += step_cost(a, b, grid.resolution)
    assert path.cost == pytest.approx(total, rel=1e-12), "cost must equal the sum of its moves"


@pytest.mark.parametrize("shape", [(3, 3), (3, 4)])
@pytest.mark.parametrize("seed", range(12))
def test_astar_is_optimal_against_exhaustive_search_on_small_grids(shape, seed):
    rng = np.random.default_rng(seed)
    blocked = rng.random(shape) < 0.3
    grid = PlanningGrid(blocked, 0.5)
    for start in itertools.product(range(shape[0]), range(shape[1])):
        for goal in itertools.product(range(shape[0]), range(shape[1])):
            expected = brute_force_cost(blocked, start, goal, 0.5)
            path = astar(grid, start, goal)
            if math.isinf(expected):
                assert path is None, f"seed {seed}: {start} -> {goal} has no path"
                continue
            assert path is not None, f"seed {seed}: {start} -> {goal} costs {expected}, got None"
            assert path.cost == pytest.approx(expected, rel=1e-12), f"{start} -> {goal}"
            assert_valid_path(grid, path, start, goal)


@pytest.mark.parametrize("seed", range(200))
def test_astar_is_optimal_against_bellman_ford_on_random_grids(seed):
    rng = np.random.default_rng(1000 + seed)
    shape = (int(rng.integers(6, 13)), int(rng.integers(6, 13)))
    blocked = rng.random(shape) < rng.uniform(0.0, 0.4)
    grid = PlanningGrid(blocked, 0.05)
    start = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
    costs = bellman_ford(blocked, start, 0.05)
    for goal in itertools.product(range(shape[0]), range(shape[1])):
        path = astar(grid, start, goal)
        if np.isinf(costs[goal]):
            assert path is None, f"seed {seed}: {start} -> {goal} is unreachable"
        else:
            assert path is not None, f"seed {seed}: {start} -> {goal} costs {costs[goal]}"
            assert path.cost == pytest.approx(costs[goal], rel=1e-12), f"{start} -> {goal}"


def test_corner_to_corner_on_an_empty_grid_is_the_diagonal():
    grid = PlanningGrid(np.zeros((5, 5), dtype=bool), 0.05)
    path = astar(grid, (0, 0), (4, 4))
    assert path is not None
    assert path.cost == pytest.approx(4 * SQRT2 * 0.05)  # four diagonal moves of 5 cm
    np.testing.assert_array_equal(path.cells, [[k, k] for k in range(5)])


def test_on_an_empty_grid_only_the_path_is_expanded():
    """The octile heuristic is exact on an empty grid, and ties go to the larger g, so A* expands
    the ten cells of the diagonal and nothing else (Dijkstra would expand about 100)."""
    path = astar(PlanningGrid(np.zeros((10, 10), dtype=bool), 1.0), (0, 0), (9, 9))
    assert path is not None and path.expanded == 10, (
        f"expanded {None if path is None else path.expanded}"
    )


def test_a_corridor_and_a_u_trap():
    blocked = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    grid = PlanningGrid(blocked, 1.0)
    # From inside the U (open at the bottom) to the top middle: out of the U, along the bottom row,
    # up the outer column and back along the top row. Every diagonal on the way has the thin wall
    # as one of its sides, so all ten moves are straight: 2 out, 2 across, 4 up, 2 back.
    path = astar(grid, (2, 2), (0, 2))
    assert path is not None
    assert path.cost == pytest.approx(10.0), path.cells.tolist()
    assert_valid_path(grid, path, (2, 2), (0, 2))


def test_no_squeeze_between_two_diagonally_blocked_cells():
    blocked = np.array([[0, 1], [1, 0]], dtype=bool)
    grid = PlanningGrid(blocked, 1.0)
    assert astar(grid, (0, 0), (1, 1)) is None, "the diagonal between two blocked cells is shut"
    blocked[0, 1] = False
    path = astar(PlanningGrid(blocked, 1.0), (0, 0), (1, 1))
    assert path is not None and path.cost == pytest.approx(2.0), "one free side: around it, 2 moves"


def test_an_enclosed_goal_and_a_blocked_goal_have_no_path():
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[2:5, 2:5] = True
    blocked[3, 3] = False  # a free cell walled in
    grid = PlanningGrid(blocked, 1.0)
    assert astar(grid, (0, 0), (3, 3)) is None
    assert not reachable(grid, (0, 0), (3, 3))
    assert astar(grid, (0, 0), (2, 2)) is None, "a blocked goal"


def test_the_robot_may_leave_a_blocked_start_cell():
    blocked = np.zeros((3, 4), dtype=bool)
    blocked[1, 1] = True
    path = astar(PlanningGrid(blocked, 1.0), (1, 1), (1, 3))
    assert path is not None and path.cost == pytest.approx(2.0)


def test_start_equal_to_goal_costs_nothing():
    path = astar(PlanningGrid(np.zeros((3, 3), dtype=bool), 1.0), (1, 2), (1, 2))
    assert path is not None and path.cost == 0.0 and path.cells.tolist() == [[1, 2]]


def test_cells_outside_the_grid_are_refused():
    grid = PlanningGrid(np.zeros((3, 3), dtype=bool), 1.0)
    for start, goal in (((0, 0), (3, 0)), ((-1, 0), (0, 0)), ((0, 0), (0, 5))):
        with pytest.raises(ValueError, match="outside"):
            astar(grid, start, goal)


def test_ties_go_to_the_larger_g_so_the_path_takes_its_diagonals_first():
    """From (0, 0) to (2, 5) on an empty 3 x 6 grid every path of 2 diagonal and 3 straight moves
    costs 2 sqrt2 + 3, and every cell on one has the same f. Ties go to the larger g, so the
    diagonal neighbour (g = sqrt2) is expanded before the straight ones (g = 1): the path takes
    both diagonals first, then runs along the top row, and only its six cells are expanded."""
    path = astar(PlanningGrid(np.zeros((3, 6), dtype=bool), 1.0), (0, 0), (2, 5))
    assert path is not None
    assert path.cells.tolist() == [[0, 0], [1, 1], [2, 2], [2, 3], [2, 4], [2, 5]]
    assert path.expanded == 6 and path.cost == pytest.approx(2 * SQRT2 + 3)


@pytest.mark.parametrize("cell", [(2.7, 0), (0, -0.5), (1.0, 1), (True, 0), ("1", 0)])
def test_a_cell_that_is_not_a_pair_of_integers_is_refused(cell):
    grid = PlanningGrid(np.zeros((3, 3), dtype=bool), 1.0)
    with pytest.raises(ValueError, match="integers"):
        astar(grid, cell, (0, 0))
    with pytest.raises(ValueError, match="integers"):
        reachable(grid, (0, 0), cell)


def test_a_robot_deep_in_the_inflation_band_has_no_path_yet():
    """Only the start cell may be left: two cells inside a blocked band there is no way out. The
    escape bubble, which frees cells blocked only by inflation (``PlanningGrid.occupied`` tells
    them apart), comes with the navigator."""
    blocked = np.zeros((7, 7), dtype=bool)
    blocked[1:6, 1:6] = True
    grid = PlanningGrid(blocked, 1.0)
    assert astar(grid, (1, 1), (0, 0)) is not None, "from the band's edge the robot steps out"
    assert astar(grid, (3, 3), (0, 0)) is None, "two cells inside the band: no path"


@pytest.mark.parametrize("seed", range(30))
def test_octile_heuristic_is_admissible(seed):
    rng = np.random.default_rng(500 + seed)
    blocked = rng.random((9, 9)) < 0.3
    goal = (int(rng.integers(9)), int(rng.integers(9)))
    blocked[goal] = False
    # Cost-to-go of every cell is the cost from the goal on the reversed graph; the move rule is
    # symmetric, so it is the cost from the goal.
    to_go = bellman_ford(blocked, goal, 0.05)
    for cell in itertools.product(range(9), range(9)):
        if np.isfinite(to_go[cell]) and not blocked[cell]:
            h = octile_distance(cell, goal, 0.05)
            assert h <= to_go[cell] + 1e-12, f"h{cell} = {h} > cost-to-go {to_go[cell]}"


def test_octile_heuristic_is_consistent_and_exact_on_an_empty_grid():
    goal = (3, 5)
    for i, j in itertools.product(range(-2, 9), range(-2, 9)):
        for di, dj in MOVES:
            a, b = (i, j), (i + di, j + dj)
            assert (
                octile_distance(a, goal, 1.0)
                <= step_cost(a, b, 1.0) + octile_distance(b, goal, 1.0) + 1e-12
            )
    # 3 diagonal + 2 straight from (0, 0) to (3, 5).
    assert octile_distance((0, 0), goal, 0.05) == pytest.approx(0.05 * (3 * SQRT2 + 2))


@pytest.mark.parametrize("seed", range(1, 31))
def test_on_generated_layouts_astar_finds_a_path_exactly_when_path_exists_does(seed):
    params = LayoutParams()
    layout = generate_layout(np.random.default_rng(seed), params)
    resolution = params.grid_resolution
    grid = from_arena(layout.arena, resolution, params.robot_radius + resolution / 2.0)
    start, goal = grid.cell_of(layout.start[:2]), grid.cell_of(layout.goal)
    assert path_exists(layout.arena, layout.start[:2], layout.goal, params.robot_radius)
    path = astar(grid, start, goal)
    assert path is not None, f"layout seed {seed}: generate_layout guarantees a path"
    assert_valid_path(grid, path, start, goal)
    # Across a wall of the inflated map from the goal: the far corner may be cut off.
    corner = grid.cell_of([0.12, 4.88])
    expected = path_exists(layout.arena, layout.start[:2], [0.12, 4.88], params.robot_radius)
    assert (astar(grid, start, corner) is not None) == expected
