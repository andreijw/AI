"""Recorded runs: the scans of ``run_trial``'s wandering robot, with no filter in the loop.

A desk tool, not part of the library a robot imports (``rl_navigation.estimation`` does not import
it). :func:`record_run` drives the robot exactly as
:func:`~rl_navigation.estimation.evaluation.run_trial` does (the same layout, wander controller,
streams and scans), by handing ``run_trial`` a filter that only records each scan: the trajectory
does not depend on the filter, and running a real one would cost seconds per seed for nothing.
The mapping, the circle features' scoring and the SLAM oracle all read their scans and true poses
from it; ``rl_navigation.mapping.evaluation``, its first home, re-exports it.

Imports NumPy, ``rl_navigation.sim`` and ``rl_navigation.estimation.evaluation`` only.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.evaluation import FactoryContext, Localizer, run_trial
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.layouts import LayoutParams
from rl_navigation.sim.sensors import RangeSensor


@dataclass(frozen=True)
class _NoBeamsUsed:
    """The :class:`~rl_navigation.estimation.evaluation.ScanResult` of a filter using no beam."""

    used: NDArray[np.bool_]


class _ScanRecorder:
    """A :class:`~rl_navigation.estimation.evaluation.Localizer` that only records each scan.

    Its ``pose`` and ``covariance`` stay the trial's noisy start and initial covariance, so the
    trial's ``estimates``, ``covariances`` and ``beams_used`` mean nothing; :func:`record_run`
    passes on only the true poses, dead reckoning and the scans.
    """

    def __init__(self, start: FloatArray, covariance: FloatArray) -> None:
        self._start = np.array(start, dtype=np.float64)
        self._covariance = np.array(covariance, dtype=np.float64)
        self.scans: list[FloatArray] = []

    @property
    def pose(self) -> FloatArray:
        return self._start

    @property
    def covariance(self) -> FloatArray:
        return self._covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        """Nothing to predict: the poses come from the simulator."""

    def update(self, readings: ArrayLike) -> _NoBeamsUsed:
        scan = np.array(readings, dtype=np.float64)
        self.scans.append(scan)
        return _NoBeamsUsed(np.zeros(len(scan), dtype=bool))


class _RecorderFactory:
    """Builds the trial's :class:`_ScanRecorder` and keeps it and the trial's arena."""

    def __init__(self) -> None:
        self.recorder: _ScanRecorder | None = None
        self.arena: Arena | None = None

    def __call__(self, context: FactoryContext) -> Localizer:
        self.arena = context.range_map.arena
        self.recorder = _ScanRecorder(context.start, context.covariance)
        return self.recorder


@dataclass(frozen=True, eq=False)
class MappingRun:
    """One seeded run's scans and the poses they were read at.

    Attributes:
        arena: The run's true arena.
        sensor: The robot's sensor.
        true_poses: ``(T, 3)`` the simulator's pose after each step, where that step's scan was
            read.
        readings: ``(T, N)`` the scan of each step.
        dead_reckoning: ``(T, 3)`` the encoders integrated from the noisy start.
        collided: Whether the run ended at a collision before its last step.
    """

    arena: Arena
    sensor: RangeSensor
    true_poses: FloatArray
    readings: FloatArray
    dead_reckoning: FloatArray
    collided: bool


def record_run(
    seed: int,
    sensor: RangeSensor,
    *,
    steps: int = 600,
    dropout_prob: float | None = None,
    layout_params: LayoutParams | None = None,
) -> MappingRun:
    """The scans of a wandering robot on a seeded layout: ``run_trial``'s run, filter left out.

    For the same seed, sensor and settings the true poses are bit for bit those of
    :func:`~rl_navigation.estimation.evaluation.run_trial` with any filter; a dropout probability
    changes which readings are lost, not the trajectory. A collision ends the run early, as there.

    Args:
        seed: The run's seed.
        sensor: The robot's range sensor.
        steps: Steps to run.
        dropout_prob: Sensor dropout probability; None uses the sensor's own.
        layout_params: How the layout is drawn; None uses the defaults.

    Returns:
        The run.

    Raises:
        ValueError: The run has no steps (``steps`` 0, or a collision on the first step).
    """
    factory = _RecorderFactory()
    trial = run_trial(
        seed,
        sensor,
        steps=steps,
        dropout_prob=dropout_prob,
        layout_params=layout_params,
        make_localizer=factory,
    )
    if factory.recorder is None or factory.arena is None or len(trial.true_poses) == 0:
        raise ValueError(
            f"record_run: seed {seed} has no steps (steps={steps}, or a collision on the first "
            "step); nothing to map."
        )
    return MappingRun(
        arena=factory.arena,
        sensor=sensor,
        true_poses=trial.true_poses,
        readings=np.array(factory.recorder.scans, dtype=np.float64),
        dead_reckoning=trial.dead_reckoning,
        collided=trial.collided,
    )
