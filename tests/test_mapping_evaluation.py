"""rl_navigation.mapping.evaluation: mapping runs on the simulator's true trajectory, scored
against the true map."""

import math

import numpy as np
import pytest

from rl_navigation.estimation import evaluation as estimation_evaluation
from rl_navigation.estimation.evaluation import run_trial
from rl_navigation.mapping.evaluation import (
    MappingRun,
    build_map,
    map_trial,
    record_run,
    score_map,
    summarize_maps,
    true_maps,
)
from rl_navigation.mapping.grid import LIDAR_LOG_ODDS, OccupancyGrid
from rl_navigation.sim import Arena, LayoutParams, lidar, occupancy_grid
from tests.navigation_cases import grid_from_mask

RESOLUTION = 0.05
SCORES = (
    "observed_free_share",
    "cell_accuracy",
    "occupied_precision",
    "visible_surface_recall",
    "surface_recall",
    "false_free_cells",
)


BOX_ROOM = Arena(2.0, 2.0, boxes=[[0.8, 0.8, 1.2, 1.2]])
# The robot looks at the box from four sides, so every surface cell of it is visible.
AROUND_THE_BOX = np.array(
    [[0.4, 1.0, 0.0], [1.6, 1.0, math.pi], [1.0, 0.4, math.pi / 2], [1.0, 1.6, -math.pi / 2]]
)


def test_a_recorded_run_follows_run_trials_trajectory_exactly():
    run = record_run(3, lidar(), steps=50)
    trial = run_trial(3, lidar(), steps=50)
    np.testing.assert_array_equal(run.true_poses, trial.true_poses)
    np.testing.assert_array_equal(run.dead_reckoning, trial.dead_reckoning)
    assert run.readings.shape == (50, 64)
    assert not run.collided


def test_the_recorded_scans_are_the_ones_the_filter_would_read():
    """The same seed and sensor read the same scans whatever the filter: the recorder sees the
    scan at each recorded pose, so a noise-free re-read of the first pose agrees within noise."""
    sensor = lidar(noise_std=0.0)
    run = record_run(4, sensor, steps=5)
    expected = np.minimum(sensor.true_ranges(run.arena, run.true_poses), sensor.max_range)
    np.testing.assert_allclose(run.readings, np.maximum(expected, sensor.min_range), atol=1e-12)


def test_a_run_without_steps_is_refused():
    with pytest.raises(ValueError, match="no steps"):
        record_run(3, lidar(), steps=0)


def test_true_maps_reuse_the_simulators_occupancy_grid():
    maps = true_maps(BOX_ROOM, RESOLUTION)
    np.testing.assert_array_equal(
        maps.touching, occupancy_grid(BOX_ROOM, RESOLUTION, inflate=RESOLUTION / 2)
    )
    np.testing.assert_array_equal(maps.inside, occupancy_grid(BOX_ROOM, RESOLUTION))
    assert not (maps.deep & ~maps.inside).any()
    assert not (maps.surface & ~maps.touching).any()


def test_the_true_map_scores_perfectly():
    maps = true_maps(BOX_ROOM, RESOLUTION)
    scores = score_map(grid_from_mask(maps.touching, RESOLUTION), BOX_ROOM, AROUND_THE_BOX)
    assert set(scores) == set(SCORES)
    assert scores["cell_accuracy"] == 1.0
    assert scores["occupied_precision"] == 1.0
    assert scores["visible_surface_recall"] == 1.0
    assert scores["surface_recall"] == 1.0
    assert scores["false_free_cells"] == 0.0
    assert scores["observed_free_share"] == 1.0


def test_a_map_never_updated_has_seen_nothing_and_no_precision():
    grid = OccupancyGrid(lidar(), (40, 40), RESOLUTION, LIDAR_LOG_ODDS)
    scores = score_map(grid, BOX_ROOM, AROUND_THE_BOX)
    assert scores["observed_free_share"] == 0.0
    assert math.isnan(scores["occupied_precision"])
    assert scores["visible_surface_recall"] == 0.0


def test_a_phantom_obstacle_lowers_precision_by_one_cell_in_all():
    maps = true_maps(BOX_ROOM, RESOLUTION)
    mask = maps.touching.copy()
    mask[5, 30] = True  # (1.525, 0.275): far from the box and the walls
    scores = score_map(grid_from_mask(mask, RESOLUTION), BOX_ROOM, AROUND_THE_BOX)
    k = int(maps.touching.sum())
    assert scores["occupied_precision"] == pytest.approx(k / (k + 1))


def test_a_free_cell_deep_in_the_box_counts_as_false_free():
    maps = true_maps(BOX_ROOM, RESOLUTION)
    mask = maps.touching.copy()
    mask[20, 20] = False  # (1.025, 1.025): the middle of the box
    assert maps.deep[20, 20]
    assert (
        score_map(grid_from_mask(mask, RESOLUTION), BOX_ROOM, AROUND_THE_BOX)["false_free_cells"]
        == 1.0
    )


def test_the_pocket_between_a_circle_and_a_wall_is_not_false_free():
    """Cells whose centres lie between a circle and the wall are touching both by the half-cell
    inflation but inside neither: free there is right."""
    arena = Arena(2.0, 2.0, circle_centers=[[1.0, 1.8]], circle_radii=[0.17])
    maps = true_maps(arena, RESOLUTION)
    pocket = maps.touching & ~maps.inside
    free_everywhere_but_inside = grid_from_mask(maps.inside, RESOLUTION)
    assert pocket.any()
    poses = np.array([[1.0, 1.0, math.pi / 2]])
    assert score_map(free_everywhere_but_inside, arena, poses)["false_free_cells"] == 0.0


def test_cells_at_the_grid_edge_are_never_deep():
    """A circle cut by the arena's left wall: its column-0 cells may hold free space beyond the
    grid's edge, so they are not 'more than a cell inside'."""
    arena = Arena(2.0, 2.0, circle_centers=[[0.1, 1.0]], circle_radii=[0.3])
    maps = true_maps(arena, RESOLUTION)
    assert maps.inside[:, 0].any()
    assert not maps.deep[:, 0].any()
    assert maps.deep[:, 1:].any()


def test_score_map_needs_the_arena_frame():
    grid = OccupancyGrid(lidar(), (10, 10), RESOLUTION, LIDAR_LOG_ODDS)
    with pytest.raises(ValueError, match="frame"):
        score_map(grid, BOX_ROOM, AROUND_THE_BOX)


def test_a_short_lidar_run_maps_its_arena_and_dead_reckoning_maps_it_worse():
    """Seed 1, 200 steps: with the true poses nothing phantom and nothing free inside an
    obstacle; with the encoders' own estimate the map is clearly worse."""
    run = record_run(1, lidar(), steps=200)
    true_scores = score_map(build_map(run, LIDAR_LOG_ODDS), run.arena, run.true_poses)
    assert true_scores["occupied_precision"] >= 0.99
    assert true_scores["cell_accuracy"] >= 0.99
    assert true_scores["false_free_cells"] == 0.0
    dead = build_map(run, LIDAR_LOG_ODDS, poses="dead_reckoning")
    dead_scores = score_map(dead, run.arena, run.true_poses)
    assert dead_scores["occupied_precision"] < true_scores["occupied_precision"] - 0.05


def test_build_map_refuses_an_unknown_pose_source():
    run = record_run(1, lidar(), steps=3)
    with pytest.raises(ValueError, match="poses"):
        build_map(run, LIDAR_LOG_ODDS, poses="estimates")


def test_map_trial_reports_the_scores_steps_and_collision():
    result = map_trial(2, lidar(), LIDAR_LOG_ODDS, steps=20)
    assert set(SCORES) <= set(result)
    assert result["steps"] == 20.0
    assert result["collided"] == 0.0


def test_summarize_maps_gives_means_minima_spread_and_counts():
    scores = [
        {
            "cell_accuracy": 1.0,
            "occupied_precision": float("nan"),
            "false_free_cells": 0.0,
            "collided": 0.0,
        },
        {"cell_accuracy": 0.9, "occupied_precision": 0.8, "false_free_cells": 2.0, "collided": 1.0},
        {"cell_accuracy": 0.8, "occupied_precision": 0.6, "false_free_cells": 1.0, "collided": 0.0},
    ]
    summary = summarize_maps(scores)
    assert summary["cell_accuracy_mean"] == pytest.approx(0.9)
    assert summary["cell_accuracy_min"] == pytest.approx(0.8)
    assert summary["cell_accuracy_sd"] == pytest.approx(0.1)
    assert summary["occupied_precision_mean"] == pytest.approx(0.7), "NaN runs are skipped"
    assert summary["false_free_cells_max"] == 2.0
    assert summary["runs"] == 3.0
    assert summary["collisions"] == 1.0


def test_summarize_maps_refuses_no_runs():
    with pytest.raises(ValueError, match="runs"):
        summarize_maps([])


def test_a_mapping_run_is_a_plain_record():
    run = record_run(1, lidar(), steps=2)
    assert isinstance(run, MappingRun)
    assert run.sensor.num_readings == run.readings.shape[1]


SMALL_EMPTY_ROOM = LayoutParams(width=2.0, height=2.0, num_obstacles=0, min_start_goal_distance=0.5)


def test_a_run_ending_at_a_collision_is_mapped_scored_and_counted(monkeypatch):
    """The wander never collides on its own (its 0.6 m look-ahead covers every move), so the test
    blinds the look-ahead at its seam: the robot then drives into a wall of a 2 x 2 m room, the
    world's own collision check ends the run early, and the partial map is still scored and the
    collision counted."""
    monkeypatch.setattr(estimation_evaluation, "segment_hits", lambda *args: False)
    run = record_run(1, lidar(), steps=300, layout_params=SMALL_EMPTY_ROOM)
    assert run.collided, "driving blind across a 2 m room should hit a wall within 300 steps"
    assert 0 < len(run.true_poses) < 300
    assert len(run.readings) == len(run.true_poses)
    crashed = map_trial(1, lidar(), LIDAR_LOG_ODDS, steps=300, layout_params=SMALL_EMPTY_ROOM)
    assert crashed["collided"] == 1.0 and crashed["steps"] == len(run.true_poses)
    assert crashed["occupied_precision"] >= 0.99, "the partial map is scored like any other"
    monkeypatch.undo()
    clean = map_trial(1, lidar(), LIDAR_LOG_ODDS, steps=50, layout_params=SMALL_EMPTY_ROOM)
    assert clean["collided"] == 0.0
    summary = summarize_maps([crashed, clean])
    assert summary["collisions"] == 1.0 and summary["runs"] == 2.0
