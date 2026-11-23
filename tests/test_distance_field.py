"""The distance field of an occupancy grid: hand-worked distances, bilinear samples and their
gradients against finite differences."""

import math

import numpy as np
import pytest

from rl_navigation.mapping.fields import DistanceField, distance_field
from tests.helpers import assert_gradients_match

RESOLUTION = 0.1


def _one_cell(shape=(7, 7), cell=(3, 3)):
    occupied = np.zeros(shape, dtype=bool)
    occupied[cell] = True
    return occupied


def test_distances_from_one_occupied_cell_by_hand():
    field = distance_field(_one_cell(), RESOLUTION, 0.25)
    assert field[3, 3] == 0.0
    assert field[3, 4] == pytest.approx(0.1, abs=1e-15)
    assert field[4, 4] == pytest.approx(0.1 * math.sqrt(2.0), abs=1e-15)
    assert field[3, 5] == pytest.approx(0.2, abs=1e-15)
    assert field[1, 2] == pytest.approx(math.hypot(0.2, 0.1), abs=1e-15)
    assert field[5, 5] == 0.25, "0.28 m is beyond the clamp."
    assert field[3, 6] == 0.25, "0.3 m is beyond the clamp."
    assert field[0, 0] == 0.25


def test_an_empty_grid_is_the_clamp_everywhere():
    field = distance_field(np.zeros((4, 5), dtype=bool), RESOLUTION, 0.3)
    np.testing.assert_array_equal(field, np.full((4, 5), 0.3))


def test_a_clamp_that_is_no_multiple_of_the_resolution():
    field = distance_field(_one_cell(), RESOLUTION, 0.15)
    assert field[4, 4] == pytest.approx(0.1 * math.sqrt(2.0), abs=1e-15)
    assert field[3, 5] == 0.15
    assert field[3, 4] == pytest.approx(0.1, abs=1e-15)


def test_a_distance_equal_to_the_clamp_reads_the_clamp():
    field = distance_field(_one_cell(), RESOLUTION, 0.2)
    assert field[3, 5] == 0.2
    assert field[3, 4] == pytest.approx(0.1, abs=1e-15)


def test_the_nearest_of_several_cells_wins():
    occupied = _one_cell()
    occupied[3, 6] = True
    field = distance_field(occupied, RESOLUTION, 1.0)
    assert field[3, 5] == pytest.approx(0.1, abs=1e-15)
    assert field[3, 4] == pytest.approx(0.1, abs=1e-15)
    assert field[3, 1] == pytest.approx(0.2, abs=1e-15)


def test_distance_field_refuses_bad_arguments():
    with pytest.raises(ValueError, match="resolution"):
        distance_field(_one_cell(), 0.0, 0.2)
    with pytest.raises(ValueError, match="max_distance"):
        distance_field(_one_cell(), RESOLUTION, -0.1)
    with pytest.raises(ValueError, match="2-D"):
        distance_field(np.zeros(5, dtype=bool), RESOLUTION, 0.2)


def _field(occupied=None, max_distance=0.25, origin=(-0.3, 0.2)):
    occupied = _one_cell() if occupied is None else occupied
    return DistanceField.from_occupied(occupied, RESOLUTION, max_distance, origin)


def _centre(field, i, j):
    x0, y0 = field.origin
    return np.array([x0 + (j + 0.5) * RESOLUTION, y0 + (i + 0.5) * RESOLUTION])


def test_a_sample_at_a_cell_centre_is_that_cells_distance():
    field = _field()
    points = np.array([_centre(field, i, j) for i, j in ((3, 3), (3, 4), (4, 4), (2, 5), (1, 1))])
    sample = field.sample(points)
    np.testing.assert_allclose(
        sample.values, field.values[[3, 3, 4, 2, 1], [3, 4, 4, 5, 1]], atol=1e-12
    )
    assert sample.inside.all()


def test_a_sample_between_centres_is_bilinear():
    field = _field()
    between = 0.5 * (_centre(field, 3, 3) + _centre(field, 3, 4))
    sample = field.sample(between[None])
    assert sample.values[0] == pytest.approx(0.05, abs=1e-12)
    # On the line through the two centres the cell above is the one sampled: dD/dy blends the
    # rises to row 4 at both ends, (0.1 - 0) and (0.1 sqrt 2 - 0.1), half each, over 0.1 m.
    np.testing.assert_allclose(sample.gradients[0], [1.0, 0.5 * math.sqrt(2.0)], atol=1e-12)
    corner = 0.25 * sum(_centre(field, i, j) for i in (3, 4) for j in (3, 4))
    expected = 0.25 * (0.0 + 0.1 + 0.1 + 0.1 * math.sqrt(2.0))
    assert field.sample(corner[None]).values[0] == pytest.approx(expected, abs=1e-12)


def test_the_sampled_gradients_match_finite_differences():
    field = _field()
    rng = np.random.default_rng(5)
    # Away from the lines through cell centres, where the bilinear surface has creases.
    cells = rng.integers(1, 5, (12, 2))
    fractions = rng.uniform(0.1, 0.9, (12, 2))
    x0, y0 = field.origin
    points = np.column_stack(
        [
            x0 + (cells[:, 1] + 0.5 + fractions[:, 0]) * RESOLUTION,
            y0 + (cells[:, 0] + 0.5 + fractions[:, 1]) * RESOLUTION,
        ]
    )
    weights = rng.uniform(0.5, 2.0, 12)

    def loss():
        return float(weights @ field.sample(points).values)

    analytic = weights[:, None] * field.sample(points).gradients
    assert_gradients_match(loss, {"points": points}, {"points": analytic}, rtol=1e-6, atol=1e-9)


def test_a_point_without_four_cell_centres_around_it_is_outside():
    field = _field()
    x0, y0 = field.origin
    points = np.array(
        [[x0 + 0.01, y0 + 0.35], [x0 - 1.0, y0], [x0 + 0.69, y0 + 0.35], [x0 + 0.35, y0 + 0.35]]
    )
    sample = field.sample(points)
    np.testing.assert_array_equal(sample.inside, [False, False, False, True])
    np.testing.assert_array_equal(sample.values[:3], [0.25, 0.25, 0.25])
    np.testing.assert_array_equal(sample.gradients[:3], np.zeros((3, 2)))


def test_the_field_values_are_read_only():
    field = _field()
    with pytest.raises(ValueError):
        field.values[0, 0] = 1.0


def _brute_force(occupied, resolution, max_distance):
    """Every occupied cell against every cell: the definition, written out."""
    cells = np.argwhere(occupied)
    rows, cols = np.indices(occupied.shape)
    field = np.full(occupied.shape, max_distance)
    for i, j in cells:
        field = np.minimum(field, resolution * np.hypot(rows - i, cols - j))
    return np.minimum(field, max_distance)


@pytest.mark.parametrize("max_distance", [0.07, 0.15, 0.2207, 0.5])
def test_the_separable_field_is_the_brute_force_one(max_distance):
    rng = np.random.default_rng(int(max_distance * 1e4))
    occupied = rng.random((23, 31)) < 0.04
    np.testing.assert_allclose(
        distance_field(occupied, 0.05, max_distance),
        _brute_force(occupied, 0.05, max_distance),
        rtol=0,
        atol=1e-15,
    )
