"""rl_navigation.mapping.grid: the log-odds occupancy grid and its inverse sensor model.

Hand cases use 1 m cells, a one-beam sensor at the robot's centre and the lidar model, whose two
evidence values are written ``A`` (a hit, logit 0.7) and ``B`` (a miss, logit 0.45).
"""

import math

import numpy as np
import pytest

from rl_navigation.mapping.beams import endpoint_visits, traverse_cells
from rl_navigation.mapping.grid import (
    HC_SR04_LOG_ODDS,
    LIDAR_LOG_ODDS,
    LogOddsModel,
    OccupancyGrid,
)
from rl_navigation.sim import (
    Arena,
    LayoutParams,
    RangeSensor,
    cast_rays,
    clearance,
    generate_layout,
    lidar,
    occupancy_grid,
)
from tests.navigation_cases import free_poses, grid_from_mask


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


A, B = logit(0.7), logit(0.45)
L_MIN, L_MAX = logit(0.12), logit(0.97)
BEAM = RangeSensor(mount_angles=[0.0], min_range=0.05, max_range=10.0)
CENTRE_OF_CELL_0 = np.array([0.5, 0.5, 0.0])


def row_grid(cols: int, sensor: RangeSensor = BEAM, model: LogOddsModel = LIDAR_LOG_ODDS):
    return OccupancyGrid(sensor, (1, cols), 1.0, model)


def test_the_lidar_and_sonar_models_are_the_planned_probabilities():
    np.testing.assert_allclose(
        [LIDAR_LOG_ODDS.l_occ, LIDAR_LOG_ODDS.l_free, LIDAR_LOG_ODDS.l_min, LIDAR_LOG_ODDS.l_max],
        [A, B, L_MIN, L_MAX],
        rtol=1e-15,
    )
    assert HC_SR04_LOG_ODDS.l_occ == pytest.approx(logit(0.6), rel=1e-15)
    assert HC_SR04_LOG_ODDS.l_free == LIDAR_LOG_ODDS.l_free


def test_a_beam_along_a_row_frees_the_cells_before_its_end_and_marks_the_end():
    grid = row_grid(3)
    grid.update(CENTRE_OF_CELL_0, [2.25])
    np.testing.assert_allclose(grid.log_odds, [[B, B, A]], rtol=1e-15)
    np.testing.assert_array_equal(grid.observed, [[True, True, True]])


def test_a_hit_on_a_border_marks_the_cell_behind_it_in_either_direction():
    forward = row_grid(5)
    forward.update(CENTRE_OF_CELL_0, [2.5])
    np.testing.assert_allclose(forward.log_odds, [[B, B, B, A, 0.0]], rtol=1e-15)
    backward = row_grid(5)
    backward.update([4.5, 0.5, math.pi], [2.5])
    np.testing.assert_allclose(backward.log_odds, [[0.0, A, B, B, B]], rtol=1e-15)


@pytest.mark.parametrize("reading", [10.0, np.nan, 0.05, 0.01])
def test_a_reading_without_a_range_changes_nothing(reading):
    """A dropout or 'no echo' (max_range), a driver's NaN and a reading clipped at min_range."""
    grid = row_grid(3)
    grid.update(CENTRE_OF_CELL_0, [reading])
    np.testing.assert_array_equal(grid.log_odds, np.zeros((1, 3)))
    assert not grid.observed.any()


def test_a_scan_lost_entirely_to_dropouts_changes_nothing():
    sensor = lidar(num_beams=8)
    arena = Arena(4.0, 4.0)
    pose = np.array([2.0, 2.0, 0.3])
    readings, _ = sensor.read(arena, pose, np.random.default_rng(3), dropout_prob=1.0)
    grid = OccupancyGrid(sensor, (40, 40), 0.1, LIDAR_LOG_ODDS)
    grid.update(pose, readings)
    assert not grid.observed.any()


def test_a_mixed_scan_updates_only_its_usable_beams():
    sensor = RangeSensor(mount_angles=[0.0, math.pi / 2], min_range=0.05, max_range=10.0)
    grid = OccupancyGrid(sensor, (3, 3), 1.0, LIDAR_LOG_ODDS)
    grid.update(CENTRE_OF_CELL_0, [1.25, 10.0])  # the second beam (up) is a dropout
    np.testing.assert_allclose(grid.log_odds[0], [B, A, 0.0], rtol=1e-15)
    assert not grid.observed[1:].any()


def test_hits_clamp_at_the_upper_bound_and_misses_flip_the_cell_after_eighteen_scans():
    """Cell 2: 4 hits give 4A = 3.389, the 5th clamps at 3.476; then readings of 4.25 cross it:
    after 17 it is 3.476 + 17B = +0.064, after 18 it is -0.137."""
    grid = row_grid(5)
    for _ in range(4):
        grid.update(CENTRE_OF_CELL_0, [2.25])
    assert grid.log_odds[0, 2] == pytest.approx(4 * A, rel=1e-14)
    grid.update(CENTRE_OF_CELL_0, [2.25])
    assert grid.log_odds[0, 2] == L_MAX
    for _ in range(17):
        grid.update(CENTRE_OF_CELL_0, [4.25])
    assert grid.log_odds[0, 2] == pytest.approx(L_MAX + 17 * B, rel=1e-12)
    assert grid.occupied()[0, 2], "17 misses should not yet free a clamped lidar cell"
    grid.update(CENTRE_OF_CELL_0, [4.25])
    assert grid.free()[0, 2], "the 18th miss should free it"


@pytest.mark.parametrize(
    ("model", "hits_to_clamp", "hits_from_floor"),
    [(LIDAR_LOG_ODDS, 5, 3), (HC_SR04_LOG_ODDS, 9, 5)],
)
def test_walked_hit_counts_to_the_ceiling_and_back_from_the_floor(
    model, hits_to_clamp, hits_from_floor
):
    grid = row_grid(5, model=model)
    for _ in range(hits_to_clamp - 1):
        grid.update(CENTRE_OF_CELL_0, [2.25])
    assert grid.log_odds[0, 2] < model.l_max
    grid.update(CENTRE_OF_CELL_0, [2.25])
    assert grid.log_odds[0, 2] == model.l_max

    floor = row_grid(5, model=model)
    for _ in range(10):  # 10 misses reach l_min: 9 give -1.806
        floor.update(CENTRE_OF_CELL_0, [4.25])
    assert floor.log_odds[0, 2] == model.l_min
    for _ in range(hits_from_floor - 1):
        floor.update(CENTRE_OF_CELL_0, [2.25])
    assert floor.log_odds[0, 2] < 0.0
    floor.update(CENTRE_OF_CELL_0, [2.25])
    assert floor.log_odds[0, 2] > 0.0


def test_a_cell_is_updated_once_per_scan_however_many_rays_cross_it():
    """Two beams 0.1 rad apart cross cells 0 and 1 and both end in cell 2: one update each."""
    sensor = RangeSensor(mount_angles=[0.0, 0.1], min_range=0.05, max_range=10.0)
    grid = OccupancyGrid(sensor, (1, 3), 1.0, LIDAR_LOG_ODDS)
    grid.update(CENTRE_OF_CELL_0, [2.25, 2.25])
    np.testing.assert_allclose(grid.log_odds, [[B, B, A]], rtol=1e-15)


def test_a_hit_wins_over_a_miss_in_the_same_scan():
    """One beam ends in cell 2; the other passes through it to cell 4."""
    sensor = RangeSensor(mount_angles=[0.0, 0.1], min_range=0.05, max_range=10.0)
    grid = OccupancyGrid(sensor, (1, 5), 1.0, LIDAR_LOG_ODDS)
    grid.update(CENTRE_OF_CELL_0, [2.25, 4.25])
    np.testing.assert_allclose(grid.log_odds, [[B, B, A, B, A]], rtol=1e-15)


def test_cone_rays_start_at_the_mount_and_share_its_reading():
    """Mount 1 m ahead of the centre: its three rays start at x = 1.5, so cell 0 (the robot's)
    is untouched, and all three end in cell 3."""
    sensor = RangeSensor(
        mount_angles=[0.0],
        cone_width=0.2,
        rays_per_cone=3,
        mount_radius=1.0,
        min_range=0.05,
        max_range=10.0,
    )
    grid = OccupancyGrid(sensor, (1, 5), 1.0, LIDAR_LOG_ODDS)
    grid.update(CENTRE_OF_CELL_0, [2.25])
    np.testing.assert_allclose(grid.log_odds, [[0.0, B, B, A, 0.0]], rtol=1e-15)


def test_a_pose_outside_the_grid_still_maps_the_cells_inside():
    grid = row_grid(3)
    grid.update([-1.5, 0.5, 0.0], [3.25])
    np.testing.assert_allclose(grid.log_odds, [[B, A, 0.0]], rtol=1e-15)


@pytest.mark.parametrize(
    ("pose", "readings", "message"),
    [
        ([0.5, 0.5], [1.0], "pose"),
        ([0.5, np.nan, 0.0], [1.0], "pose"),
        (CENTRE_OF_CELL_0, [1.0, 2.0], "readings"),
        (CENTRE_OF_CELL_0, [], "readings"),
    ],
)
def test_update_rejects_a_bad_pose_or_a_scan_of_the_wrong_length(pose, readings, message):
    with pytest.raises(ValueError, match=message):
        row_grid(3).update(pose, readings)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (0, 3)}, "shape"),
        ({"shape": (2,)}, "shape"),
        ({"resolution": 0.0}, "resolution"),
        ({"origin": (np.inf, 0.0)}, "origin"),
        ({"log_odds": np.zeros((2, 2))}, "log_odds"),
    ],
)
def test_the_grid_rejects_bad_settings(kwargs, message):
    settings = {"shape": (1, 3), "resolution": 1.0, "origin": (0.0, 0.0)} | kwargs
    with pytest.raises(ValueError, match=message):
        OccupancyGrid(BEAM, model=LIDAR_LOG_ODDS, **settings)


def test_an_initial_map_is_copied_clipped_and_counts_as_observed_where_nonzero():
    start = np.array([[5.0, 0.0, -0.5]])
    grid = OccupancyGrid(BEAM, (1, 3), 1.0, LIDAR_LOG_ODDS, log_odds=start)
    start[0, 1] = 9.0
    np.testing.assert_allclose(grid.log_odds, [[L_MAX, 0.0, -0.5]])
    np.testing.assert_array_equal(grid.observed, [[True, False, True]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"l_occ": 0.0}, "l_occ"),
        ({"l_free": 0.1}, "l_free"),
        ({"l_min": 0.5}, "l_min"),
        ({"l_max": -0.5}, "l_max"),
    ],
)
def test_the_log_odds_model_rejects_evidence_of_the_wrong_sign(kwargs, message):
    settings = {"l_occ": A, "l_free": B, "l_min": L_MIN, "l_max": L_MAX} | kwargs
    with pytest.raises(ValueError, match=message):
        LogOddsModel(**settings)


@pytest.mark.parametrize("probabilities", [(1.0, 0.45), (0.7, 0.0), (0.4, 0.45), (0.7, 0.6)])
def test_from_probabilities_rejects_probabilities_that_give_no_evidence(probabilities):
    with pytest.raises(ValueError):
        LogOddsModel.from_probabilities(*probabilities)


def test_the_state_is_read_only_and_probabilities_follow_the_log_odds():
    grid = row_grid(3)
    grid.update(CENTRE_OF_CELL_0, [2.25])
    with pytest.raises(ValueError):
        grid.log_odds[0, 0] = 1.0
    with pytest.raises(ValueError):
        grid.observed[0, 0] = False
    np.testing.assert_allclose(grid.probabilities(), [[0.45, 0.45, 0.7]], rtol=1e-12)
    np.testing.assert_array_equal(grid.occupied(), [[False, False, True]])
    np.testing.assert_array_equal(grid.free(), [[True, True, False]])


def first_occupied_ranges(grid: OccupancyGrid, origins, angles) -> np.ndarray:
    """Test-only reference for the traversal: the midpoint of each ray's chord through the first
    occupied cell it crosses inside the grid, ``inf`` if it crosses none."""
    height, width = np.multiply(grid.shape, grid.resolution)
    lengths = np.full(len(origins), math.hypot(height, width))
    visits = traverse_cells(origins, angles, lengths, grid.resolution, grid.shape, grid.origin)
    rows = np.clip(visits.rows, 0, grid.shape[0] - 1)
    cols = np.clip(visits.cols, 0, grid.shape[1] - 1)
    hit = visits.kept & visits.inside & grid.occupied()[rows, cols]
    first = np.argmax(hit, axis=1)
    picked = np.arange(len(origins))
    middle = 0.5 * (visits.t_in[picked, first] + visits.t_out[picked, first])
    return np.where(hit.any(axis=1), middle, np.inf)


@pytest.mark.parametrize("seed", [5, 8])
def test_the_traversal_stops_at_the_surfaces_cast_rays_finds_on_a_rasterised_arena(seed):
    """On a 5 cm grid rasterised from a seeded layout (walls inflated by half a cell), each lidar
    ray from 20 free poses stops at its first occupied cell. That cell is never past the surface
    the simulator's own ray cast meets by more than a cell, and its chord's midpoint lies within
    1/2 + 1/sqrt(2) cells of an obstacle (the cell's centre is within half a cell, the midpoint
    within half a diagonal of it). A ray grazing an inflated wall may stop well before the cast's
    hit, by design, so the ranges themselves are not compared."""
    arena = generate_layout(np.random.default_rng(seed), LayoutParams()).arena
    resolution = 0.05
    sensor = lidar()
    grid = grid_from_mask(
        occupancy_grid(arena, resolution, inflate=resolution / 2), resolution, BEAM
    )
    poses = free_poses(arena, 20, seed=seed + 1, margin=0.2)
    origins = sensor.mount_origins(poses).reshape(-1, 2)
    angles = sensor.ray_angles(poses).reshape(-1)
    expected = cast_rays(arena, origins, angles[:, None])[:, 0]
    found = first_occupied_ranges(grid, origins, angles)
    assert np.isfinite(found).all(), "every ray in a closed arena meets a wall"
    overshoot = found - expected
    assert overshoot.max() <= resolution, (
        f"a ray passed {overshoot.max():.3f} m beyond the surface: the traversal skipped a cell"
    )
    points = origins + found[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    gaps = clearance(arena, points)
    assert gaps.max() <= (0.5 + math.sqrt(0.5)) * resolution + 1e-9, (
        f"a ray stopped {gaps.max():.3f} m from every obstacle: not at an occupied cell"
    )


def test_endpoint_visits_and_update_agree_on_one_ray():
    """The grid's update is the traversal's rule scattered to cells: nothing more."""
    origins, angles = np.array([[0.5, 0.5]]), np.array([0.0])
    visits = traverse_cells(origins, angles, np.array([3.25]), 1.0, (1, 5))
    occupied, free = endpoint_visits(visits, np.array([2.25]))
    grid = row_grid(5)
    grid.update(CENTRE_OF_CELL_0, [2.25])
    np.testing.assert_array_equal(grid.occupied()[0], np.isin(np.arange(5), visits.cols[occupied]))
    np.testing.assert_array_equal(grid.free()[0], np.isin(np.arange(5), visits.cols[free]))
