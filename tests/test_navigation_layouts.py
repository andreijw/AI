"""rl_navigation.sim.layouts: random obstacle layouts with a reachable goal."""

import math

import numpy as np
import pytest

from rl_navigation.sim.geometry import Arena, clearance
from rl_navigation.sim.layouts import LayoutParams, generate_layout, path_exists

SEEDS = range(20)


def layout(seed: int, **kwargs):
    return generate_layout(np.random.default_rng(seed), LayoutParams(**kwargs))


def test_same_seed_gives_the_same_layout():
    first, second, other = layout(3), layout(3), layout(4)
    np.testing.assert_array_equal(first.arena.segments, second.arena.segments)  # boxes included
    np.testing.assert_array_equal(first.arena.circle_centers, second.arena.circle_centers)
    np.testing.assert_array_equal(first.arena.circle_radii, second.arena.circle_radii)
    np.testing.assert_array_equal(first.start, second.start)
    np.testing.assert_array_equal(first.goal, second.goal)
    assert not np.array_equal(first.start, other.start)


def test_layout_has_the_requested_obstacles_within_their_size_ranges():
    params = LayoutParams()
    counts = {"circles": 0, "boxes": 0}
    for seed in SEEDS:
        arena = generate_layout(np.random.default_rng(seed), params).arena
        assert (arena.width, arena.height) == (params.width, params.height)
        assert len(arena.circle_radii) + len(arena.boxes) == params.num_obstacles
        lo, hi = params.circle_radius
        assert ((arena.circle_radii >= lo) & (arena.circle_radii <= hi)).all()
        sides = np.concatenate(
            [arena.boxes[:, 2] - arena.boxes[:, 0], arena.boxes[:, 3] - arena.boxes[:, 1]]
        )
        lo, hi = params.box_size
        assert ((sides >= lo - 1e-12) & (sides <= hi + 1e-12)).all()
        counts["circles"] += len(arena.circle_radii)
        counts["boxes"] += len(arena.boxes)
    assert counts["circles"] > 0 and counts["boxes"] > 0, "Both obstacle kinds should appear."


def test_start_and_goal_are_clear_of_every_obstacle():
    params = LayoutParams()
    for seed in SEEDS:
        result = generate_layout(np.random.default_rng(seed), params)
        gaps = clearance(result.arena, np.array([result.start[:2], result.goal]))
        assert (gaps >= params.robot_radius + params.margin).all(), (
            f"Seed {seed}: start and goal should be at least robot radius + margin from obstacles, "
            f"got {gaps}."
        )


def test_start_and_goal_are_far_enough_apart():
    for seed in SEEDS:
        result = layout(seed)
        assert np.linalg.norm(result.start[:2] - result.goal) >= 2.0, f"Seed {seed}: too close."


def test_start_heading_is_wrapped():
    headings = np.array([layout(seed).start[2] for seed in SEEDS])
    assert ((headings >= -math.pi) & (headings < math.pi)).all()
    assert np.ptp(headings) > 1.0, "Start headings should be drawn, not fixed."


def test_a_path_exists_from_start_to_goal():
    params = LayoutParams()
    for seed in SEEDS:
        result = generate_layout(np.random.default_rng(seed), params)
        assert path_exists(result.arena, result.start[:2], result.goal, params.robot_radius)


def test_an_arena_without_obstacles():
    result = layout(0, num_obstacles=0)
    assert len(result.arena.circle_radii) == 0 and len(result.arena.boxes) == 0


def test_path_exists_is_false_across_a_dividing_wall_and_true_through_a_gap():
    divided = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 5.0]])
    assert not path_exists(divided, np.array([1.0, 2.5]), np.array([4.0, 2.5]), 0.1)
    gap = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 2.0], [2.4, 2.5, 2.6, 5.0]])  # a 0.5 m gap
    assert path_exists(gap, np.array([1.0, 1.0]), np.array([4.0, 4.0]), 0.1)
    # The same gap is too narrow for a 0.3 m robot (0.6 m wide).
    assert not path_exists(gap, np.array([1.0, 1.0]), np.array([4.0, 4.0]), 0.3)


def test_path_exists_is_false_from_inside_an_obstacle():
    arena = Arena(5.0, 5.0, circle_centers=[[1.0, 1.0]], circle_radii=[0.5])
    assert not path_exists(arena, np.array([1.0, 1.0]), np.array([4.0, 4.0]), 0.1)


def test_an_infeasibly_dense_layout_is_rejected_up_front():
    with pytest.raises(ValueError, match="num_obstacles=60"):
        LayoutParams(num_obstacles=60)


def test_start_and_goal_that_cannot_be_far_enough_apart_are_rejected_up_front():
    with pytest.raises(ValueError, match="min_start_goal_distance"):
        LayoutParams(width=1.5, height=1.5, num_obstacles=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_obstacles": -1}, "num_obstacles"),
        ({"circle_radius": (0.5, 0.2)}, "circle_radius"),
        ({"box_size": (0.0, 1.0)}, "box_size"),
        ({"circle_fraction": 1.5}, "circle_fraction"),
        ({"max_attempts": 0}, "max_attempts"),
    ],
)
def test_layout_params_reject_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LayoutParams(**kwargs)


def test_generation_gives_up_after_the_attempt_limit():
    # The start-goal distance is a nanometre under the diagonal of the square start and goal are
    # drawn from: the up-front check allows it, but two random points are that far apart with
    # probability about 1e-18.
    params = LayoutParams(num_obstacles=0, max_attempts=5)
    side = params.width - 2 * (params.robot_radius + params.margin)
    reach = math.sqrt(2) * side - 1e-9
    with pytest.raises(RuntimeError, match="5 attempts"):
        layout(0, num_obstacles=0, max_attempts=5, min_start_goal_distance=reach)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid_resolution": 0.0}, "grid_resolution"),
        ({"robot_radius": 0.0}, "robot_radius"),
        ({"margin": -0.1}, "margin"),
    ],
)
def test_layout_params_reject_a_non_positive_grid_or_robot_and_a_negative_margin(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LayoutParams(**kwargs)


def test_path_exists_is_conservative_about_a_gap_just_wider_than_the_robot():
    # A 0.22 m gap (y 2.02 to 2.24) leaves a 0.1 m robot's centre a 2 cm corridor (y 2.12 to 2.14),
    # less than half a 5 cm cell. The cell centre at y = 2.125 lies in it, so inflating by the radius
    # alone would pass the gap on one cell's evidence; inflating by half a cell more refuses it.
    gap = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 2.02], [2.4, 2.24, 2.6, 5.0]])
    assert not path_exists(gap, np.array([1.0, 1.0]), np.array([4.0, 4.0]), 0.1)
