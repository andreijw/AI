"""rl_navigation.sim.geometry: the arena, ray casting, swept-disc collision and the occupancy grid.

Every expected value is worked out by hand in a comment; the arena is 5 m by 5 m unless a test says
otherwise, with its origin at the lower-left corner.
"""

import math

import numpy as np
import pytest

from rl_navigation.sim.geometry import Arena, cast_rays, clearance, occupancy_grid, segment_hits

EMPTY = Arena(width=5.0, height=5.0)


def cast_one(arena: Arena, origin: tuple[float, float], angle_deg: float) -> float:
    """The range of one ray, through the batched interface."""
    ranges = cast_rays(arena, np.array([origin]), np.radians([[angle_deg]]))
    assert ranges.shape == (1, 1)
    return float(ranges[0, 0])


@pytest.mark.parametrize(
    ("angle_deg", "expected"),
    # From (1, 1): the x = 5 wall is 4 m ahead at 0 deg, the y = 5 wall 4 m at 90 deg, the x = 0 and
    # y = 0 walls 1 m behind and below.
    [(0.0, 4.0), (90.0, 4.0), (180.0, 1.0), (-90.0, 1.0)],
)
def test_beam_reads_the_distance_to_the_arena_wall(angle_deg, expected):
    assert cast_one(EMPTY, (1.0, 1.0), angle_deg) == pytest.approx(expected, abs=1e-12), (
        f"A {angle_deg} deg beam from (1, 1) should reach the arena wall {expected} m away."
    )


def test_beam_reads_the_distance_to_a_circle():
    # Circle at (3, 1), r 0.5: its near side is at x = 2.5, 1.5 m from (1, 1).
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 1.0]], circle_radii=[0.5])
    assert cast_one(arena, (1.0, 1.0), 0.0) == pytest.approx(1.5, abs=1e-12)


def test_beam_hits_a_box_corner_at_45_degrees():
    # Box [3, 4] x [3, 4]: the 45 deg beam from (1, 1) meets its corner (3, 3) after 2 * sqrt(2) m.
    arena = Arena(5.0, 5.0, boxes=[[3.0, 3.0, 4.0, 4.0]])
    assert cast_one(arena, (1.0, 1.0), 45.0) == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-12)


@pytest.mark.parametrize(
    "wall",
    [
        [[2.0, 1.5], [4.0, 1.5]],  # parallel to the beam, 0.5 m above it
        [[2.0, 1.0], [4.0, 1.0]],  # collinear with the beam: a wall of zero thickness, edge on
    ],
)
def test_ray_parallel_to_a_wall_does_not_hit_it(wall):
    arena = Arena(5.0, 5.0, walls=[wall])
    with np.errstate(all="raise"):  # a parallel wall must not divide by zero
        reading = cast_one(arena, (1.0, 1.0), 0.0)
    assert reading == pytest.approx(4.0, abs=1e-12), (
        "A beam parallel to a wall should pass it and reach the x = 5 arena wall 4 m away."
    )


@pytest.mark.parametrize(
    "wall",
    [[[2.0, 0.2], [2.0, 0.8]], [[2.0, 0.8], [2.0, 0.2]]],  # both orientations: u > 1 and u < 0
    ids=["ends-below", "starts-below"],
)
def test_ray_past_the_end_of_a_wall_does_not_hit_it(wall):
    # The wall x = 2, y in [0.2, 0.8] stops short of the beam along y = 1.
    arena = Arena(5.0, 5.0, walls=[wall])
    assert cast_one(arena, (1.0, 1.0), 0.0) == pytest.approx(4.0, abs=1e-12), (
        "A beam passing beyond a wall's end point should not hit the wall's extension."
    )


def test_grazing_ray_hits_a_circle_at_its_tangent_point():
    # Circle at (3, 1), r 0.5; a beam along y = 1.5 touches its top at (3, 1.5), 2.5 m from (0.5, 1.5).
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 1.0]], circle_radii=[0.5])
    assert cast_one(arena, (0.5, 1.5), 0.0) == pytest.approx(2.5, abs=1e-12)
    # A millimetre higher it misses the circle and reaches the wall, 4.5 m away.
    assert cast_one(arena, (0.5, 1.501), 0.0) == pytest.approx(4.5, abs=1e-12)


def test_grazing_ray_hits_a_box_at_its_corner():
    # Box [2, 3] x [1, 2]: a beam along y = 1 from (1, 1) runs along the bottom edge and meets the
    # left edge at its end point (2, 1), 1 m away.
    arena = Arena(5.0, 5.0, boxes=[[2.0, 1.0, 3.0, 2.0]])
    assert cast_one(arena, (1.0, 1.0), 0.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "arena",
    [
        Arena(5.0, 5.0, circle_centers=[[0.5, 1.0]], circle_radii=[0.25]),
        Arena(5.0, 5.0, boxes=[[0.2, 0.5, 0.6, 1.5]]),
    ],
    ids=["circle", "box"],
)
def test_obstacle_behind_the_origin_is_not_hit(arena):
    assert cast_one(arena, (1.0, 1.0), 0.0) == pytest.approx(4.0, abs=1e-12), (
        "An obstacle behind the sensor should not shorten a forward beam."
    )


def test_beam_reads_the_nearest_of_several_obstacles():
    # Box at x = 2 is 1 m away; the circle at (4, 1) r 0.25 is 2.75 m away; the wall 4 m.
    arena = Arena(
        5.0, 5.0, circle_centers=[[4.0, 1.0]], circle_radii=[0.25], boxes=[[2.0, 0.5, 2.5, 1.5]]
    )
    assert cast_one(arena, (1.0, 1.0), 0.0) == pytest.approx(1.0, abs=1e-12)


def test_ray_from_outside_the_arena_pointing_away_reads_infinity():
    assert cast_one(EMPTY, (6.0, 1.0), 0.0) == math.inf


def test_cast_rays_is_batched_over_origins_and_angles():
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 3.0]], circle_radii=[0.4], boxes=[[1, 3, 2, 4]])
    origins = np.array([[1.0, 1.0], [2.5, 2.0], [4.0, 4.5]])
    angles = np.radians(np.linspace(-180.0, 170.0, 7))[None, :] + np.array([[0.0], [0.1], [0.2]])
    ranges = cast_rays(arena, origins, angles)
    assert ranges.shape == (3, 7), "cast_rays should return one range per (origin, angle) pair."
    assert ranges.dtype == np.float64
    for p in range(3):
        for r in range(7):
            single = cast_one(arena, tuple(origins[p]), math.degrees(angles[p, r]))
            assert ranges[p, r] == pytest.approx(single, abs=1e-12), (
                f"Batched range [{p}, {r}] should equal the same ray cast on its own."
            )


@pytest.mark.parametrize(
    ("origins", "angles"),
    [
        (np.zeros((3, 2)), np.zeros((2, 5))),
        (np.zeros((3, 3)), np.zeros((3, 5))),
        (np.zeros(2), np.zeros(5)),
    ],
    ids=["row-mismatch", "origin-not-2d", "unbatched"],
)
def test_cast_rays_rejects_mismatched_shapes(origins, angles):
    with pytest.raises(ValueError, match="origins"):
        cast_rays(EMPTY, origins, angles)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"circle_centers": [[1.0, 1.0]], "circle_radii": [0.0]}, "radii"),
        ({"circle_centers": [[1.0, 1.0]], "circle_radii": [0.5, 0.2]}, "radii"),
        ({"boxes": [[2.0, 1.0, 1.0, 2.0]]}, "boxes"),
        ({"walls": [[0.0, 0.0, 1.0, 1.0]]}, "walls"),
    ],
)
def test_arena_rejects_malformed_obstacles(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Arena(5.0, 5.0, **kwargs)


def test_arena_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="width"):
        Arena(0.0, 5.0)


def test_arena_arrays_are_read_only_float64():
    arena = Arena(5.0, 5.0, circle_centers=[[1, 1]], circle_radii=[1], boxes=[[2, 2, 3, 3]])
    # Four arena walls and four edges of the box.
    assert arena.segments.shape == (8, 2, 2)
    for array in (
        arena.circle_centers,
        arena.circle_radii,
        arena.boxes,
        arena.walls,
        arena.segments,
    ):
        assert array.dtype == np.float64
        with pytest.raises(ValueError):
            array[...] = 0.0


# --- segment_hits: the disc of the robot swept along a straight move -------------------------------

THIN_BOX = Arena(5.0, 5.0, boxes=[[2.495, 0.5, 2.505, 4.5]])  # a 1 cm thick wall
THIN_WALL = Arena(5.0, 5.0, walls=[[[2.5, 0.5], [2.5, 4.5]]])


@pytest.mark.parametrize("arena", [THIN_BOX, THIN_WALL], ids=["1cm-box", "segment-wall"])
def test_swept_disc_cannot_pass_a_thin_wall_in_one_step(arena):
    start, end = np.array([2.0, 2.5]), np.array([3.0, 2.5])
    assert (clearance(arena, np.array([start, end])) > 0.1).all(), (
        "Both ends of the move should be clear, so only the sweep can see the wall."
    )
    assert segment_hits(arena, start, end, 0.1), "A 1 m move across a thin wall should collide."


@pytest.mark.parametrize(
    ("arena", "start", "end", "expected"),
    [
        # Circle at (3, 2.5) r 0.5 with a 0.1 m robot: contact when the centre is 0.6 m away.
        (Arena(5, 5, circle_centers=[[3, 2.5]], circle_radii=[0.5]), (1, 2.5), (2.45, 2.5), True),
        (Arena(5, 5, circle_centers=[[3, 2.5]], circle_radii=[0.5]), (1, 2.5), (2.35, 2.5), False),
        (Arena(5, 5, circle_centers=[[3, 2.5]], circle_radii=[0.5]), (1, 2.5), (4.8, 2.5), True),
        # Box [2, 3] x [2, 3]: 0.05 m from its left edge collides, 0.15 m does not.
        (Arena(5, 5, boxes=[[2, 2, 3, 3]]), (1, 2.5), (1.95, 2.5), True),
        (Arena(5, 5, boxes=[[2, 2, 3, 3]]), (1, 2.5), (1.85, 2.5), False),
        # A free wall segment, approached end on.
        (Arena(5, 5, walls=[[[2, 2.5], [3, 2.5]]]), (1, 2.5), (1.92, 2.5), True),
        (Arena(5, 5, walls=[[[2, 2.5], [3, 2.5]]]), (1, 2.5), (1.85, 2.5), False),
        # The arena boundary at x = 5.
        (EMPTY, (4.5, 2.5), (4.95, 2.5), True),
        (EMPTY, (4.5, 2.5), (4.85, 2.5), False),
        # Turning on the spot: a move of zero length.
        (EMPTY, (4.95, 2.5), (4.95, 2.5), True),
        (EMPTY, (2.5, 2.5), (2.5, 2.5), False),
    ],
    ids=[
        "circle-contact",
        "circle-clear",
        "circle-swept-through",
        "box-contact",
        "box-clear",
        "wall-contact",
        "wall-clear",
        "boundary-contact",
        "boundary-clear",
        "spin-in-contact",
        "spin-clear",
    ],
)
def test_segment_hits_each_obstacle_kind(arena, start, end, expected):
    hit = segment_hits(arena, np.array(start, float), np.array(end, float), 0.1)
    assert hit is expected, f"A 0.1 m disc moving {start} -> {end} should collide: {expected}."


@pytest.mark.parametrize(("distance", "expected"), [(0.09, True), (0.11, False)])
def test_segment_hits_a_box_corner_passed_diagonally(distance, expected):
    # The move runs along direction (1, -1) and passes the corner (2, 2) of box [2, 3] x [2, 3] at
    # `distance`; both ends are 0.56 m from the box, so only the corner can be within reach.
    closest = np.array([2.0, 2.0]) - distance * np.array([1.0, 1.0]) / math.sqrt(2.0)
    start, end = closest + [-0.5, 0.5], closest + [0.5, -0.5]
    arena = Arena(5.0, 5.0, boxes=[[2.0, 2.0, 3.0, 3.0]])
    assert segment_hits(arena, start, end, 0.1) is expected


# --- clearance and the occupancy grid ---------------------------------------------------------------


def test_clearance_is_the_distance_to_the_nearest_surface():
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 2.5]], circle_radii=[0.5], boxes=[[0, 4, 1, 5]])
    points = np.array(
        [
            [1.5, 2.5],  # circle surface at x = 2.5: 1.0; nearest wall x = 0: 1.5
            [2.0, 2.5],  # circle surface 0.5 away
            [3.0, 2.5],  # inside the circle
            [0.5, 4.5],  # inside the box
            [1.5, 4.5],  # box edge x = 1: 0.5; top wall: 0.5
            [6.0, 1.0],  # outside the arena
        ]
    )
    np.testing.assert_allclose(clearance(arena, points), [1.0, 0.5, 0.0, 0.0, 0.5, 0.0], atol=1e-12)


def test_occupancy_grid_rows_are_y_and_columns_are_x():
    # 1 m arena at 0.25 m: box [0, 0.5] x [0, 0.25] covers the first two cells of row 0 only.
    arena = Arena(1.0, 1.0, boxes=[[0.0, 0.0, 0.5, 0.25]])
    grid = occupancy_grid(arena, resolution=0.25)
    assert grid.shape == (4, 4) and grid.dtype == np.bool_
    expected = np.zeros((4, 4), bool)
    expected[0, :2] = True
    np.testing.assert_array_equal(grid, expected)


def test_occupancy_grid_marks_a_circle_and_its_inflation():
    arena = Arena(1.0, 1.0, circle_centers=[[0.5, 0.5]], circle_radii=[0.2])
    grid = occupancy_grid(arena, resolution=0.1)
    # Cell (5, 5) has its centre at (0.55, 0.55), 0.07 m from the circle's centre.
    assert grid[5, 5] and not grid[0, 0] and not grid[5, 1]
    inflated = occupancy_grid(arena, resolution=0.1, inflate=0.2)
    # Every cell centre is 0.05 m or more from a wall; the border cells are within 0.2 m of one.
    assert inflated[0, :].all() and inflated[:, 0].all() and inflated[-1, :].all()
    assert inflated.sum() > grid.sum()


def test_occupancy_grid_marks_a_wall_thinner_than_a_cell_when_inflated_by_half_a_cell():
    # Cell centres sit at x = 0.45 and 0.55, 0.05 m either side of the wall: inflated by half a
    # cell, both columns are blocked; uninflated, no centre lies on the wall.
    arena = Arena(1.0, 1.0, walls=[[[0.5, 0.0], [0.5, 1.0]]])
    grid = occupancy_grid(arena, resolution=0.1, inflate=0.05)
    assert grid[:, 4].all() and grid[:, 5].all(), "The wall should block whole columns of cells."
    assert not occupancy_grid(arena, resolution=0.1)[:, 4].any()


def test_occupancy_grid_of_an_empty_arena_is_free():
    assert not occupancy_grid(EMPTY, resolution=0.25).any()


# --- review fixes -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [[3.0, 3.0, 4.0, 4.0], [3.0, 1.0, 4.0, 3.0], [1.0, 3.0, 3.0, 4.0]],
    ids=["corner-ahead", "corner-of-box-below-right", "corner-of-box-above-left"],
)
def test_beam_hits_a_box_corner_from_either_side(box):
    # Each box has a corner at (3, 3), on the 45 deg beam from (1, 1), 2 sqrt(2) m away: met head
    # on, or grazed with the box on the beam's right or on its left.
    arena = Arena(5.0, 5.0, boxes=[box])
    assert cast_one(arena, (1.0, 1.0), 45.0) == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-12)


@pytest.mark.parametrize(
    ("origins", "angles"),
    [
        (np.array([[np.nan, 1.0]]), np.zeros((1, 3))),
        (np.array([[1.0, 1.0]]), np.array([[0.0, np.inf, 1.0]])),
    ],
    ids=["nan-origin", "inf-angle"],
)
def test_cast_rays_rejects_non_finite_input(origins, angles):
    with pytest.raises(ValueError, match="finite"):
        cast_rays(EMPTY, origins, angles)


def test_cast_rays_in_chunks_matches_one_pass():
    # A budget of 50 elements forces one origin per chunk (5 rays x 10 segments), so P = 7 runs
    # in seven chunks; the ranges must be those of a single pass.
    arena = Arena(5.0, 5.0, circle_centers=[[3.0, 3.0]], circle_radii=[0.4], boxes=[[1, 3, 2, 4]])
    rng = np.random.default_rng(0)
    origins = rng.uniform(0.5, 4.5, (7, 2))
    angles = rng.uniform(-math.pi, math.pi, (7, 5))
    np.testing.assert_array_equal(
        cast_rays(arena, origins, angles, max_elements=50), cast_rays(arena, origins, angles)
    )


@pytest.mark.parametrize(
    ("arena", "start", "end"),
    [
        (Arena(5.0, 5.0, boxes=[[1.0, 1.0, 4.0, 4.0]]), (2.0, 2.5), (3.0, 2.5)),  # inside a box
        (EMPTY, (6.0, 2.5), (7.0, 2.5)),  # outside the arena
    ],
    ids=["inside-a-box", "outside-the-arena"],
)
def test_segment_hits_a_move_that_ends_inside_a_box_or_outside_the_arena(arena, start, end):
    # Both ends are more than the radius from every edge, so no edge or distance check sees it.
    assert segment_hits(arena, np.array(start), np.array(end), 0.1)
