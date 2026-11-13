"""Seeded mapping runs on the simulator's true trajectory, scored against the true map.

A desk tool, not part of the library a robot imports (``rl_navigation.mapping`` does not import
it). :func:`record_run` drives the robot exactly as
:func:`~rl_navigation.estimation.evaluation.run_trial` does (the same layout, wander controller,
streams and scans), by handing ``run_trial`` a filter that only records each scan: the trajectory
does not depend on the filter, and running a real one would cost seconds per seed for nothing.
:func:`build_map` maps the run's scans from its true poses (or, as a control, from dead reckoning
on the same encoders), and :func:`score_map` compares the map with the arena rasterised by
:func:`~rl_navigation.sim.geometry.occupancy_grid`.

Imports NumPy, ``rl_navigation.sim``, ``rl_navigation.estimation`` (for ``run_trial``) and the
mapping core.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.averages import mean_or_nan
from rl_navigation.estimation.evaluation import Localizer, run_trial
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.mapping.beams import sensor_rays
from rl_navigation.mapping.grid import LogOddsModel, OccupancyGrid
from rl_navigation.sim.geometry import Arena, FloatArray, cast_rays, occupancy_grid
from rl_navigation.sim.layouts import LayoutParams
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor

PoseSource = Literal["true", "dead_reckoning"]
BoolGrid = NDArray[np.bool_]


@dataclass(frozen=True)
class _NoBeamsUsed:
    """The :class:`~rl_navigation.estimation.evaluation.ScanResult` of a filter that uses no beam."""

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

    def __call__(
        self,
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer:
        self.arena = range_map.arena
        self.recorder = _ScanRecorder(start, covariance)
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


def _arena_grid(
    arena: Arena, sensor: RangeSensor, resolution: float, model: LogOddsModel
) -> OccupancyGrid:
    """An empty grid on the arena's frame: origin ``(0, 0)`` and ``occupancy_grid``'s shape."""
    rows, cols = occupancy_grid(arena, resolution).shape
    return OccupancyGrid(sensor, (rows, cols), resolution, model)


def build_map(
    run: MappingRun,
    model: LogOddsModel,
    *,
    resolution: float = 0.05,
    poses: PoseSource = "true",
) -> OccupancyGrid:
    """The run's scans mapped on the arena's frame, from the true poses or from dead reckoning.

    Args:
        run: The run.
        model: The log-odds model.
        resolution: Cell side in metres.
        poses: ``"true"`` for the simulator's poses (mapping with known poses), or
            ``"dead_reckoning"`` for the encoders' own estimate (the control that shows why the
            pose must be estimated with the map).

    Raises:
        ValueError: An unknown ``poses``.
    """
    sources = {"true": run.true_poses, "dead_reckoning": run.dead_reckoning}
    if poses not in sources:
        raise ValueError(f"build_map poses must be one of {sorted(sources)}, got {poses!r}.")
    grid = _arena_grid(run.arena, run.sensor, resolution, model)
    for pose, readings in zip(sources[poses], run.readings, strict=True):
        grid.update(pose, readings)
    return grid


def _dilate(mask: BoolGrid, outside: bool = False) -> BoolGrid:
    """``mask`` grown by one cell in all 8 directions; cells beyond the edge count as
    ``outside``."""
    padded = np.pad(mask, 1, constant_values=outside)
    rows, cols = mask.shape
    grown = np.zeros_like(mask)
    for di in range(3):
        for dj in range(3):
            grown |= padded[di : di + rows, dj : dj + cols]
    return grown


class TrueMaps(NamedTuple):
    """The arena rasterised for scoring, each ``(rows, cols)`` on ``occupancy_grid``'s frame.

    Attributes:
        touching: Cells within half a cell of an obstacle (``inflate = resolution / 2``),
            boundary walls included: what a map should call occupied.
        inside: Cells whose centre is inside an obstacle (``inflate = 0``).
        surface: ``touching`` cells next to one that is not: the surfaces a sensor can see.
        band: Cells within one cell of the border between ``touching`` and the rest, where a map
            may call a cell either way.
        deep: ``inside`` cells whose 8 neighbours are all ``inside``, a neighbour beyond the
            grid's edge counting as not: more than a cell inside an obstacle, where free is wrong.
    """

    touching: BoolGrid
    inside: BoolGrid
    surface: BoolGrid
    band: BoolGrid
    deep: BoolGrid


def true_maps(arena: Arena, resolution: float) -> TrueMaps:
    """The arena's true maps for scoring, from :func:`occupancy_grid`."""
    touching = occupancy_grid(arena, resolution, inflate=resolution / 2.0)
    inside = occupancy_grid(arena, resolution)
    surface = touching & _dilate(~touching)
    band = surface | (~touching & _dilate(touching))
    deep = inside & ~_dilate(~inside, outside=True)
    return TrueMaps(touching, inside, surface, band, deep)


def visible_cells(
    arena: Arena, sensor: RangeSensor, poses: ArrayLike, shape: tuple[int, int], resolution: float
) -> BoolGrid:
    """Cells holding the noise-free end point of some ray, within ``max_range``, at some pose.

    Each of a cone's rays counts on its own, so these are the surfaces the sensor's rays reach.
    An end point at or below ``min_range`` counts as visible too, although such a reading carries
    no range and maps nothing: it is not filtered out, so that ``MAP-1``'s published
    ``visible_surface_recall`` keeps the definition it was judged with. They are rare: on
    ``MAP-1``'s runs (seeds 101 to 150) no lidar ray ended that close, and 15 of 750,000 HC-SR04
    rays did (sonars on the robot's edge passing within 1.2 cm of a surface).
    """
    origins, angles = sensor_rays(sensor, np.asarray(poses, dtype=np.float64).reshape(-1, 3))
    ranges = cast_rays(arena, origins, angles[:, None])[:, 0]
    reached = ranges < sensor.max_range
    ends = origins[reached] + ranges[reached, None] * np.column_stack(
        [np.cos(angles[reached]), np.sin(angles[reached])]
    )
    rows = np.clip(np.floor(ends[:, 1] / resolution).astype(np.intp), 0, shape[0] - 1)
    cols = np.clip(np.floor(ends[:, 0] / resolution).astype(np.intp), 0, shape[1] - 1)
    visible = np.zeros(shape, dtype=bool)
    visible[rows, cols] = True
    return visible


def score_map(grid: OccupancyGrid, arena: Arena, poses: ArrayLike) -> dict[str, float]:
    """A map against the arena's true maps (:func:`true_maps`).

    Args:
        grid: The map, on the arena's frame (origin ``(0, 0)``, ``occupancy_grid``'s shape).
        arena: The true arena.
        poses: ``(T, 3)`` the true poses the scans were read at (for the visible surfaces).

    Returns:
        - ``observed_free_share``: the share of free cells (not ``touching``) ever updated;
        - ``cell_accuracy``: over observed cells off the ``band``, the share whose label
          (occupied when ``l > 0``) matches ``touching``;
        - ``occupied_precision``: the share of occupied cells within one cell of ``touching``
          (NaN when none is occupied);
        - ``visible_surface_recall``: the share of ``surface`` cells within one cell of a visible
          one that have an occupied cell within one cell;
        - ``surface_recall``: the same over every ``surface`` cell, seen or not;
        - ``false_free_cells``: the number of ``deep`` cells mapped free (``l < 0``).

    Raises:
        ValueError: A grid not on the arena's frame.
    """
    truth = true_maps(arena, grid.resolution)
    if grid.origin != (0.0, 0.0) or grid.shape != truth.touching.shape:
        raise ValueError(
            f"score_map needs a grid on the arena's frame: origin (0, 0) and shape "
            f"{truth.touching.shape}, got {grid.origin} and {grid.shape}."
        )
    occupied, observed = grid.occupied(), grid.observed
    visible = visible_cells(arena, grid.sensor, poses, grid.shape, grid.resolution)
    seen_surface = truth.surface & _dilate(visible)
    near_occupied = _dilate(occupied)
    judged = observed & ~truth.band
    return {
        "observed_free_share": mean_or_nan(observed[~truth.touching]),
        "cell_accuracy": mean_or_nan((occupied == truth.touching)[judged]),
        "occupied_precision": mean_or_nan(_dilate(truth.touching)[occupied]),
        "visible_surface_recall": mean_or_nan(near_occupied[seen_surface]),
        "surface_recall": mean_or_nan(near_occupied[truth.surface]),
        "false_free_cells": float((grid.free() & truth.deep).sum()),
    }


def map_trial(
    seed: int,
    sensor: RangeSensor,
    model: LogOddsModel,
    *,
    steps: int = 600,
    dropout_prob: float | None = None,
    resolution: float = 0.05,
    poses: PoseSource = "true",
    layout_params: LayoutParams | None = None,
) -> dict[str, float]:
    """Record a run, map it and score it: :func:`score_map`'s figures plus ``steps`` (the run's
    length) and ``collided`` (1.0 for a run that ended at a collision, whose partial map is scored
    all the same)."""
    run = record_run(
        seed, sensor, steps=steps, dropout_prob=dropout_prob, layout_params=layout_params
    )
    grid = build_map(run, model, resolution=resolution, poses=poses)
    return {
        **score_map(grid, run.arena, run.true_poses),
        "steps": float(len(run.true_poses)),
        "collided": float(run.collided),
    }


def summarize_maps(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Each figure's mean, minimum and sample standard deviation over runs, NaN runs skipped.

    Returns:
        ``<name>_mean``, ``<name>_min`` and ``<name>_sd`` for every figure but ``collided``
        (NaN when no run has a value; the deviation also with a single one);
        ``false_free_cells_max``; ``runs``; and ``collisions``.

    Raises:
        ValueError: No runs.
    """
    if not scores:
        raise ValueError("summarize_maps needs runs; got no runs.")
    summary: dict[str, float] = {}
    for name in scores[0]:
        if name == "collided":
            continue
        values = np.array([run[name] for run in scores], dtype=np.float64)
        values = values[~np.isnan(values)]
        # Not mean_or_nan: the three lines keep one guard, parallel with _min and _sd.
        summary[f"{name}_mean"] = float(values.mean()) if len(values) else float("nan")
        summary[f"{name}_min"] = float(values.min()) if len(values) else float("nan")
        summary[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    if "false_free_cells" in scores[0]:
        summary["false_free_cells_max"] = float(max(run["false_free_cells"] for run in scores))
    summary["runs"] = float(len(scores))
    summary["collisions"] = float(sum(run.get("collided", 0.0) for run in scores))
    return summary
