"""rl_navigation.mapping.beams: the cells a ray crosses, and which of them a reading marks.

Hand cases use 1 m cells and binary-exact coordinates, so entry and exit distances compare exactly.
"""

import math

import numpy as np
import pytest

from rl_navigation.mapping.beams import TRAVERSE_TOLERANCE, endpoint_visits, traverse_cells


def visits_of(origin, angle, length, shape, resolution=1.0, origin_xy=(0.0, 0.0)):
    """The kept, in-grid visits of one ray as ``[(row, col, t_in, t_out), ...]`` in order."""
    visits = traverse_cells(
        np.array([origin], dtype=float),
        np.array([angle], dtype=float),
        np.array([length], dtype=float),
        resolution,
        shape,
        origin_xy,
    )
    keep = visits.kept[0] & visits.inside[0]
    return [
        (int(r), int(c), float(a), float(b))
        for r, c, a, b in zip(
            visits.rows[0, keep],
            visits.cols[0, keep],
            visits.t_in[0, keep],
            visits.t_out[0, keep],
            strict=True,
        )
    ]


def cells_of(origin, angle, length, shape, resolution=1.0):
    return [(r, c) for r, c, _, _ in visits_of(origin, angle, length, shape, resolution)]


def test_a_ray_along_a_row_of_three_cells_visits_each_with_its_entry_and_exit():
    """From (0.5, 0.5) along +x: cell 0 from 0 to 0.5, cell 1 to 1.5, cell 2 to 2.5; the last
    0.5 m lies outside the (1, 3) grid."""
    assert visits_of((0.5, 0.5), 0.0, 3.0, (1, 3)) == [
        (0, 0, 0.0, 0.5),
        (0, 1, 0.5, 1.5),
        (0, 2, 1.5, 2.5),
    ]


def test_a_ray_crosses_cell_borders_the_same_way_forward_and_backward():
    assert cells_of((0.5, 0.5), 0.0, 3.5, (1, 5)) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    backward = visits_of((4.5, 0.5), math.pi, 3.5, (1, 5))
    assert [(r, c) for r, c, _, _ in backward] == [(0, 4), (0, 3), (0, 2), (0, 1)]
    np.testing.assert_allclose([v[2] for v in backward], [0.0, 0.5, 1.5, 2.5], atol=1e-12)


@pytest.mark.parametrize(
    ("origin", "angle", "expected"),
    [
        ((0.5, 0.5), math.pi / 2, [(0, 0), (1, 0), (2, 0)]),  # cos(pi/2) = 6.1e-17
        ((0.5, 2.5), 3 * math.pi / 2, [(2, 0), (1, 0), (0, 0)]),  # cos(3 pi/2) = -1.8e-16
    ],
)
def test_vertical_rays_stay_in_their_column(origin, angle, expected):
    """A cosine of 1e-16 is no direction at all: the ray must not wander into another column."""
    assert cells_of(origin, angle, 2.5, (3, 1)) == expected


def test_a_negative_quadrant_diagonal_steps_down_and_left_through_the_corner():
    visits = visits_of((2.5, 2.5), -3 * math.pi / 4, 2.0, (3, 3))
    assert [(r, c) for r, c, _, _ in visits] == [(2, 2), (1, 1)]
    assert visits[1][2] == pytest.approx(math.sqrt(0.5), abs=1e-12)


def test_a_diagonal_through_a_corner_drops_the_slivers_of_the_side_cells():
    """Rounding gives one of (0, 1) or (1, 0) a visit about 1e-16 m long; it is not kept."""
    assert cells_of((0.5, 0.5), math.pi / 4, 2.0, (3, 3)) == [(0, 0), (1, 1)]


@pytest.mark.parametrize(
    ("origin", "angle", "shape", "expected"),
    [
        ((1.0, 1.0), 0.3, (3, 3), [(1, 1), (1, 2)]),  # on a corner: the cell up and right
        ((0.5, 1.0), 0.0, (2, 3), [(1, 0), (1, 1)]),  # along a horizontal line: above
        ((1.0, 0.5), math.pi / 2, (2, 3), [(0, 1), (1, 1)]),  # along a vertical line: right
    ],
)
def test_an_origin_on_a_corner_or_a_grid_line_belongs_to_the_cell_above_and_right(
    origin, angle, shape, expected
):
    assert cells_of(origin, angle, 1.5, shape) == expected


def test_a_ray_leaving_the_grid_keeps_only_its_cells_inside():
    assert cells_of((0.5, 0.5), 0.0, 10.0, (1, 3)) == [(0, 0), (0, 1), (0, 2)]


def test_a_ray_starting_outside_the_grid_enters_it_at_the_right_distance():
    visits = visits_of((-1.5, 0.5), 0.0, 3.0, (1, 3))
    assert visits[0] == (0, 0, 1.5, 2.5)


def test_the_grid_origin_shifts_the_cells():
    """A grid whose lower-left corner is at (-1, -1): the point (0.5, 0.5) is in cell (1, 1)."""
    assert visits_of((0.5, 0.5), 0.0, 0.4, (3, 3), origin_xy=(-1.0, -1.0))[0][:2] == (1, 1)


def test_every_field_has_one_row_per_ray_padded_for_the_longest():
    visits = traverse_cells(
        np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0.0, 0.0]), np.array([0.4, 2.2]), 1.0, (1, 3)
    )
    for field in visits:
        assert field.shape == visits.rows.shape and field.shape[0] == 2
    assert visits.kept[0].sum() == 1 and visits.kept[1].sum() == 3


def test_no_rays_give_empty_visits():
    visits = traverse_cells(np.zeros((0, 2)), np.zeros(0), np.zeros(0), 1.0, (2, 2))
    assert visits.rows.shape[0] == 0


def random_rays(seed: int, count: int = 500):
    rng = np.random.default_rng(seed)
    origins = rng.uniform(0.2, 4.8, (count, 2))
    angles = rng.uniform(-math.pi, math.pi, count)
    lengths = rng.uniform(0.01, 4.0, count)
    return origins, angles, lengths, rng


@pytest.mark.parametrize("seed", range(20))
def test_visits_are_contiguous_adjacent_and_add_up_to_the_ray(seed):
    """Over random rays on a 5 cm grid: the intervals chain, sum to the ray's length, and each
    kept visit's cell touches the previous one's."""
    origins, angles, lengths, _ = random_rays(seed)
    visits = traverse_cells(origins, angles, lengths, 0.05, (100, 100))
    np.testing.assert_allclose(
        (visits.t_out - visits.t_in).sum(axis=1), lengths, atol=1e-12, err_msg="lengths"
    )
    np.testing.assert_array_equal(visits.t_out[:, :-1], visits.t_in[:, 1:])
    for ray in range(len(origins)):
        kept = np.flatnonzero(visits.kept[ray])
        step_rows = np.abs(np.diff(visits.rows[ray, kept]))
        step_cols = np.abs(np.diff(visits.cols[ray, kept]))
        assert ((step_rows <= 1) & (step_cols <= 1) & (step_rows + step_cols >= 1)).all(), (
            f"ray {ray} jumped between cells that do not touch"
        )


@pytest.mark.parametrize("seed", range(20))
def test_the_occupied_visit_holds_the_end_point_away_from_borders(seed):
    origins, angles, lengths, rng = random_rays(seed)
    ranges = lengths * rng.uniform(0.0, 1.0, len(lengths))
    visits = traverse_cells(origins, angles, lengths, 0.05, (100, 100))
    occupied, free = endpoint_visits(visits, ranges)
    ends = origins + ranges[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    in_grid = ((ends >= 0.0) & (ends < 5.0)).all(axis=1)
    np.testing.assert_array_equal(
        occupied.sum(axis=1),
        in_grid.astype(int),
        err_msg="a ray ending in the grid marks exactly one cell; one ending outside marks none",
    )
    assert not (occupied & free).any()
    ends, occupied = ends[in_grid], occupied[in_grid]
    rows, cols = visits.rows[in_grid][occupied], visits.cols[in_grid][occupied]
    fraction = ends / 0.05 % 1.0
    away = ((fraction > 1e-6) & (fraction < 1 - 1e-6)).all(axis=1)
    np.testing.assert_array_equal(rows[away], np.floor(ends[away, 1] / 0.05))
    np.testing.assert_array_equal(cols[away], np.floor(ends[away, 0] / 0.05))


def endpoint_cells(origin, angle, length, z, shape):
    visits = traverse_cells(np.array([origin]), np.array([angle]), np.array([length]), 1.0, shape)
    occupied, free = endpoint_visits(visits, np.array([z]))

    def as_cells(mask):
        return list(zip(visits.rows[mask].tolist(), visits.cols[mask].tolist(), strict=True))

    return as_cells(occupied), as_cells(free)


def test_a_hit_on_a_border_marks_the_far_cell_and_frees_the_near_one():
    """z = 2.5 from x = 0.5 ends exactly on x = 3: cell 3 is behind the surface."""
    assert endpoint_cells((0.5, 0.5), 0.0, 3.5, 2.5, (1, 5)) == (
        [(0, 3)],
        [(0, 0), (0, 1), (0, 2)],
    )
    assert endpoint_cells((4.5, 0.5), math.pi, 3.5, 2.5, (1, 5)) == (
        [(0, 1)],
        [(0, 4), (0, 3), (0, 2)],
    )


def test_an_end_point_inside_a_dropped_sliver_marks_the_cell_before_it():
    """A ray 1e-10 m above the corner diagonal crosses cell (1, 0) for 1.4e-10 m, less than
    TRAVERSE_TOLERANCE. A reading ending in that sliver still marks a cell, the one before."""
    origin, angle = (0.5, 0.5 + 1e-10), math.pi / 4
    visits = traverse_cells(np.array([origin]), np.array([angle]), np.array([2.0]), 1.0, (3, 3))
    sliver = np.flatnonzero(~visits.kept[0] & (visits.t_out[0] > visits.t_in[0]))
    assert len(sliver) == 1
    assert visits.t_out[0, sliver[0]] - visits.t_in[0, sliver[0]] < TRAVERSE_TOLERANCE
    z = 0.5 * (visits.t_in[0, sliver[0]] + visits.t_out[0, sliver[0]])
    assert endpoint_cells(origin, angle, 2.0, z, (3, 3)) == ([(0, 0)], [])


def test_an_end_point_outside_the_grid_marks_no_cell_inside_it():
    assert endpoint_cells((0.5, 0.5), 0.0, 5.0, 4.2, (1, 3)) == (
        [],
        [(0, 0), (0, 1), (0, 2)],
    )


def test_cells_beyond_the_end_point_are_neither_free_nor_occupied():
    assert endpoint_cells((0.5, 0.5), 0.0, 4.5, 1.25, (1, 5)) == ([(0, 1)], [(0, 0)])


def test_traverse_cells_rejects_mismatched_or_non_finite_rays():
    with pytest.raises(ValueError, match="rays"):
        traverse_cells(np.zeros((2, 2)), np.zeros(3), np.ones(2), 1.0, (2, 2))
    with pytest.raises(ValueError, match="finite"):
        traverse_cells(np.zeros((1, 2)), np.zeros(1), np.array([np.inf]), 1.0, (2, 2))
    with pytest.raises(ValueError, match="resolution"):
        traverse_cells(np.zeros((1, 2)), np.zeros(1), np.ones(1), 0.0, (2, 2))


def test_a_shallow_ray_crosses_into_the_next_row_where_it_meets_the_line():
    """From (0.5, 0.9995) at 0.0005 rad the ray climbs 0.5 mm over 1 m: it meets y = 1 at
    t = 0.0005 / sin(0.0005), just past 1 m, inside column 1, and runs on in row 1. A tolerance that
    treats a direction component of 5e-4 as zero would keep it in row 0."""
    angle = 0.0005
    visits = visits_of((0.5, 0.9995), angle, 3.0, (2, 4))
    assert [(r, c) for r, c, _, _ in visits] == [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]
    t_row = 0.0005 / math.sin(angle)
    np.testing.assert_allclose(
        [v[2] for v in visits],
        [0.0, 0.5 / math.cos(angle), t_row, 1.5 / math.cos(angle), 2.5 / math.cos(angle)],
        rtol=1e-12,
    )
