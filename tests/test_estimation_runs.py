"""Recorded runs: ``run_trial``'s wandering robot with a filter that only records each scan."""

import numpy as np
import pytest

from rl_navigation.estimation import runs
from rl_navigation.estimation.evaluation import run_trial
from rl_navigation.estimation.runs import record_run
from rl_navigation.mapping import evaluation as mapping_evaluation
from rl_navigation.sim import lidar


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


def test_the_mapping_desk_tool_still_exports_the_recorded_run():
    """``record_run`` and ``MappingRun`` moved here from ``mapping.evaluation`` (released in
    0.2.0), which re-exports them: the old import path is the same object."""
    assert mapping_evaluation.record_run is runs.record_run
    assert mapping_evaluation.MappingRun is runs.MappingRun
