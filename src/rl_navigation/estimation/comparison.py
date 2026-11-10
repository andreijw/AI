"""The EKF against MCL on the same seeded trials, with a Markdown report and a command line.

Each sensor and dropout is one row of four trial sets over the same seeds
(:func:`~rl_navigation.estimation.evaluation.run_trial`): both filters tracking from the noisy
start, MCL from anywhere and kidnapped, and the EKF from the noisy start and kidnapped. Every
trial can be saved as a :class:`TrialRecord`, so seeds run in separate processes merge into one
report::

    python -m rl_navigation.estimation.comparison --seeds 101-125 --records runs/a
    python -m rl_navigation.estimation.comparison --seeds 126-150 --records runs/b
    python -m rl_navigation.estimation.comparison --from-records runs/a runs/b --report out.md

Not imported by :mod:`rl_navigation.estimation`: it is a tool for the desk, not for the robot.
"""

import argparse
import json
import os
import platform
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.ekf import EkfLocalizer, UpdateResult
from rl_navigation.estimation.evaluation import (
    LOCALIZE_DEADLINE,
    LOCALIZE_HOLD,
    LOCALIZED_HEADING,
    LOCALIZED_POSITION,
    LOST_ERROR,
    NEES_UPPER_95,
    Localizer,
    LocalizerFactory,
    ScanResult,
    Trial,
    ekf,
    localization_steps,
    mcl,
    run_trial,
    summarize,
    summarize_localization,
)
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.estimation.mcl import MclSettings
from rl_navigation.estimation.poses import checked_axis_values, pose_difference
from rl_navigation.sim import RangeSensor, WheelOdometry, hc_sr04, lidar
from rl_navigation.sim.geometry import FloatArray

SensorFactory = Callable[[], RangeSensor]

# run_trial's World keeps its default step of 0.1 s.
STEP_SECONDS = 0.1

# CI's markdownlint (MD013) limit for lines outside tables.
MARKDOWN_LINE_LENGTH = 120

# The four trial sets of a row: (filter, kidnapped), keyed by the record's mode; the row's metrics
# carry each set's prefix.
MODES = {
    "ekf_tracking": ("ekf", False),
    "mcl_tracking": ("mcl", False),
    "mcl_kidnap": ("mcl", True),
    "ekf_kidnap": ("ekf", True),
}
PREFIXES = {
    "ekf_tracking": "ekf_",
    "mcl_tracking": "mcl_",
    "mcl_kidnap": "mcl_kidnap_",
    "ekf_kidnap": "ekf_kidnap_",
}
TIMED_CALLS = ("predict", "update", "pose", "covariance")
BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SENSORS: dict[str, SensorFactory] = {"hc_sr04": hc_sr04, "lidar": lidar}


@dataclass(frozen=True)
class EkfReset:
    """The EKF's still-robot reset and its hold, as :class:`EkfLocalizer` takes them.

    ``nis_quantile`` is the hold's setting rather than the reset's; it lives here because the
    records keep these settings as one ``ekf_reset`` entry, and a held scan counts toward the
    reset. A record written before the hold has no ``nis_quantile``: it ran without it (None).
    """

    reset_after: int | None = 3
    min_candidates: int = 2
    reset_std: tuple[float, float, float] = (0.2, 0.2, 0.1)
    reset_window: int = 20
    nis_quantile: float | None = 0.999


class InstrumentedLocalizer:
    """A :class:`~rl_navigation.estimation.evaluation.Localizer` that times the one it wraps.

    Each call of ``predict``, ``update``, ``pose`` and ``covariance`` adds its wall time to
    :attr:`seconds`; ``pose`` and ``covariance`` are timed because MCL clusters its particles
    lazily, on the first of them after an update. For a wrapped :class:`EkfLocalizer` it also
    records the scans (counted from 0) on which the update reset the covariance and the largest
    :attr:`EkfLocalizer.recent_resets` seen.
    """

    def __init__(self, inner: Localizer) -> None:
        self.inner = inner
        self.seconds = dict.fromkeys(TIMED_CALLS, 0.0)
        self.scans = 0
        self.reset_steps: list[int] | None = [] if isinstance(inner, EkfLocalizer) else None
        self.max_recent_resets = 0

    def _timed(self, name: str, call: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        value = call()
        self.seconds[name] += time.perf_counter() - start
        return value

    @property
    def pose(self) -> FloatArray:
        """The wrapped filter's pose."""
        pose: FloatArray = self._timed("pose", lambda: self.inner.pose)
        return pose

    @property
    def covariance(self) -> FloatArray:
        """The wrapped filter's covariance."""
        covariance: FloatArray = self._timed("covariance", lambda: self.inner.covariance)
        return covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        """The wrapped filter's prediction."""
        self._timed("predict", lambda: self.inner.predict(wheel_travel))

    def update(self, readings: ArrayLike) -> ScanResult:
        """The wrapped filter's update, noting an EKF's reset."""
        result: ScanResult = self._timed("update", lambda: self.inner.update(readings))
        if self.reset_steps is not None and isinstance(self.inner, EkfLocalizer):
            if cast(UpdateResult, result).reset:
                self.reset_steps.append(self.scans)
            self.max_recent_resets = max(self.max_recent_resets, self.inner.recent_resets)
        self.scans += 1
        return result


class InstrumentedFactory:
    """A :class:`~rl_navigation.estimation.evaluation.LocalizerFactory` that wraps each filter
    the factory it is given builds in an :class:`InstrumentedLocalizer`, kept as :attr:`built`."""

    def __init__(self, factory: LocalizerFactory) -> None:
        self.factory = factory
        self.built: InstrumentedLocalizer | None = None

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
        self.built = InstrumentedLocalizer(
            self.factory(
                range_map=range_map,
                odometry=odometry,
                start=start,
                covariance=covariance,
                rng=rng,
                robot_radius=robot_radius,
            )
        )
        return self.built


TRIAL_ARRAYS = ("true_poses", "estimates", "covariances", "dead_reckoning", "beams_used")


def _dropout_text(dropout: float | None) -> str:
    """``own`` for the sensor's own dropout, else the probability."""
    return "own" if dropout is None else f"{dropout:g}"


def record_file_name(sensor: str, dropout: float | None, mode: str, seed: int) -> str:
    """The file name of a trial's record: ``<sensor>_<dropout>_<mode>_seed<seed>.npz``."""
    return f"{sensor}_{_dropout_text(dropout)}_{mode}_seed{seed}.npz"


@dataclass(frozen=True, eq=False)
class TrialRecord:
    """One trial and what it was run with, saved as one ``.npz`` that loads without pickle.

    Attributes:
        trial: The trial.
        meta: JSON-ready values: ``seed``, ``sensor``, ``dropout`` (None for the sensor's own),
            ``mode`` (a key of :data:`MODES`), ``settings`` (``steps``, ``kidnap_at``,
            ``ekf_reset`` and ``mcl_settings`` with this sensor's ``covariance_scale``: what
            records merged into one report must share), ``resets`` (for the EKF, ``steps``: the
            scans that reset, and ``max_recent``: the largest ``recent_resets``; None for MCL)
            and ``timings`` (``seconds`` per timed call, ``machine``, ``blas_threads`` and
            ``jobs_note``; never compared).
    """

    trial: Trial
    meta: dict[str, Any]

    @property
    def file_name(self) -> str:
        """``<sensor>_<dropout>_<mode>_seed<seed>.npz``: one name per trial of a comparison."""
        meta = self.meta
        return record_file_name(meta["sensor"], meta["dropout"], meta["mode"], meta["seed"])

    def save(self, path: Path | str) -> None:
        """Write the trial's arrays, its ``collided`` and ``kidnap_at`` and the meta as JSON.

        The file appears whole or not at all: it is written beside ``path`` and then renamed.
        """
        path = Path(path)
        partial = path.with_name(path.name + ".partial")
        kidnap_at = [] if self.trial.kidnap_at is None else [self.trial.kidnap_at]
        with partial.open("wb") as file:
            self._write(file, kidnap_at)
        os.replace(partial, path)

    def _write(self, file: Any, kidnap_at: list[int]) -> None:
        np.savez(
            file,
            **{name: getattr(self.trial, name) for name in TRIAL_ARRAYS},
            collided=np.array(self.trial.collided),
            kidnap_at=np.array(kidnap_at, dtype=np.int64),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "TrialRecord":
        """Read a record written by :meth:`save` (``allow_pickle=False``)."""
        with np.load(path, allow_pickle=False) as data:
            kidnap_at = data["kidnap_at"]
            trial = Trial(
                **{name: data[name] for name in TRIAL_ARRAYS},
                collided=bool(data["collided"]),
                kidnap_at=int(kidnap_at[0]) if len(kidnap_at) else None,
            )
            meta = json.loads(str(data["meta"]))
        return cls(trial, meta)


def load_records(directories: Sequence[Path | str]) -> list[TrialRecord]:
    """Every record in the directories, in file-name order.

    Raises:
        ValueError: A directory holds no record.
    """
    records = []
    for directory in directories:
        paths = sorted(Path(directory).glob("*.npz"))
        if not paths:
            raise ValueError(f"No trial records (*.npz) in {directory}.")
        records += [TrialRecord.load(path) for path in paths]
    return records


@dataclass(frozen=True)
class ComparisonRow:
    """One sensor and dropout: the four trial sets' figures, keyed with the sets' prefixes
    (:data:`PREFIXES`)."""

    sensor: str
    dropout: float | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class Comparison:
    """The rows, the records they summarize and the report's settings (:func:`format_markdown`)."""

    rows: list[ComparisonRow]
    records: list[TrialRecord]
    meta: dict[str, Any]


def nees_per_step(trial: Trial) -> FloatArray:
    """``(T,)`` normalised estimation error squared ``e^T P^-1 e`` at each step of the trial."""
    errors = pose_difference(trial.estimates, trial.true_poses)
    solved = np.linalg.solve(trial.covariances, errors[..., None])[..., 0]
    return np.asarray(np.einsum("ti,ti->t", errors, solved))


def confidently_wrong(
    trial: Trial, reset_steps: Sequence[int] | None, window: int
) -> NDArray[np.bool_]:
    """For each step from the kidnapping on, whether the filter was wrong and did not show it.

    A step is confidently wrong when the position error exceeds :data:`LOST_ERROR`, the NEES
    exceeds :data:`NEES_UPPER_95` (the covariance does not cover the error) and, for a filter
    with resets (``reset_steps`` not None, the EKF), none of the last ``window`` scans, this one
    included, reset: a filter that keeps resetting is visibly lost, not confidently wrong.

    Raises:
        ValueError: A trial without a kidnapping.
    """
    if trial.kidnap_at is None:
        raise ValueError("confidently_wrong needs a kidnapped trial; this one has no kidnap_at.")
    errors = pose_difference(trial.estimates, trial.true_poses)
    nees = nees_per_step(trial)
    wrong = (np.hypot(errors[:, 0], errors[:, 1]) > LOST_ERROR) & (nees > NEES_UPPER_95)
    if reset_steps is not None:
        resets = np.zeros(len(wrong))
        resets[list(reset_steps)] = 1.0
        recent = np.convolve(resets, np.ones(window))[: len(wrong)] > 0.0
        wrong &= ~recent
    return wrong[trial.kidnap_at :]


def axis_ratios(trials: Sequence[Trial]) -> FloatArray:
    """``(3,)`` mean over every step of ``e_i^2 / P_ii`` for x, y and theta: 1 when consistent."""
    errors = np.concatenate([pose_difference(t.estimates, t.true_poses) for t in trials])
    variances = np.concatenate([np.diagonal(t.covariances, axis1=1, axis2=2) for t in trials])
    return np.asarray(np.mean(errors**2 / variances, axis=0))


def _identity(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    return meta["sensor"], meta["dropout"], meta["mode"], meta["seed"]


def _shared_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings every sensor shares: all but MCL's per-sensor covariance scale."""
    mcl_settings = {k: v for k, v in settings["mcl_settings"].items() if k != "covariance_scale"}
    return {**settings, "mcl_settings": mcl_settings}


def _is_ekf(record: TrialRecord) -> bool:
    """Whether the record is one of the EKF's trial sets."""
    return bool(MODES[record.meta["mode"]][0] == "ekf")


def _ekf_reset(record: TrialRecord) -> dict[str, Any]:
    """The record's EKF settings; a record from before the hold ran without it."""
    return {"nis_quantile": None, **record.meta["settings"]["ekf_reset"]}


def _ekf_settings(records: Sequence[TrialRecord]) -> dict[str, Any] | None:
    """The first EKF record's EKF settings (all agree once :func:`check_records` passed); None
    without an EKF record. MCL records are skipped: they may predate the hold."""
    return next((_ekf_reset(record) for record in records if _is_ekf(record)), None)


def check_records(records: Sequence[TrialRecord]) -> None:
    """Refuse records that cannot be merged into one comparison; timings are not compared.

    The EKF's settings (``ekf_reset``) are compared among EKF records only, since MCL does not
    read them; so MCL records from before the hold merge with EKF records run with it.

    Raises:
        ValueError: No records; two of the same (sensor, dropout, mode, seed); or settings that
            differ (all of them within a sensor, all but the MCL covariance scale across
            sensors, and the EKF's among the EKF records).
    """
    if not records:
        raise ValueError("A comparison needs trial records; got none.")
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        identity = _identity(record.meta)
        if identity in seen:
            raise ValueError(
                f"A duplicate trial record of (sensor, dropout, mode, seed) {identity}."
            )
        seen.add(identity)
    ekf_reset = _ekf_settings(records)
    shared: dict[str, Any] | None = None
    per_sensor: dict[str, dict[str, Any]] = {}
    for record in records:
        settings = {k: v for k, v in record.meta["settings"].items() if k != "ekf_reset"}
        shared = _shared_settings(settings) if shared is None else shared
        sensor_settings = per_sensor.setdefault(record.meta["sensor"], settings)
        if _shared_settings(settings) != shared or settings != sensor_settings:
            raise ValueError(
                f"Trial record {_identity(record.meta)} has other settings than the rest: "
                f"{settings} against {sensor_settings if settings != sensor_settings else shared}."
            )
        if _is_ekf(record) and _ekf_reset(record) != ekf_reset:
            raise ValueError(
                f"Trial record {_identity(record.meta)} has other EKF settings than the rest: "
                f"{_ekf_reset(record)} against {ekf_reset}."
            )


def _share(flags: Sequence[NDArray[np.bool_]]) -> float:
    """The share of True over every step of every array; NaN when there is no step."""
    pooled = np.concatenate([np.zeros(0, dtype=bool), *flags])
    return float(pooled.mean()) if len(pooled) else float("nan")


def _set_metrics(
    records: Sequence[TrialRecord], kidnapped: bool, deadline: int, hold: int, window: int
) -> dict[str, float]:
    """One trial set's figures: tracking (:func:`summarize`) or kidnapping
    (:func:`summarize_localization`), with the timing and, for the EKF, the resets."""
    trials = [record.trial for record in records]
    if kidnapped:
        summary = summarize_localization(trials, deadline, hold)
        summary["seconds_to_localize"] = summary["median_steps_to_localize"] * STEP_SECONDS
        summary["seconds_to_recover"] = summary["median_steps_to_recover"] * STEP_SECONDS
        per_trial = [
            confidently_wrong(
                record.trial,
                None if record.meta["resets"] is None else record.meta["resets"]["steps"],
                window,
            )
            for record in records
        ]
        localized = [
            wrong
            for wrong, trial in zip(per_trial, trials, strict=True)
            if localization_steps(trial, deadline, hold)[0] is not None
        ]
        summary["confidently_wrong"] = _share(per_trial)
        summary["confidently_wrong_localized"] = _share(localized)
    else:
        summary = summarize(trials)
        for axis, ratio in zip(("x", "y", "theta"), axis_ratios(trials), strict=True):
            summary[f"axis_ratio_{axis}"] = float(ratio)
        summary["trials"] = float(len(trials))
    seconds = sum(sum(record.meta["timings"]["seconds"].values()) for record in records)
    summary["ms_per_step"] = 1000.0 * seconds / sum(len(trial.true_poses) for trial in trials)
    if all(record.meta["resets"] is not None for record in records):
        summary["max_recent_resets"] = float(
            max(record.meta["resets"]["max_recent"] for record in records)
        )
    return summary


def _report_meta(records: Sequence[TrialRecord], deadline: int, hold: int) -> dict[str, Any]:
    """The settings :func:`format_markdown` prints, from records that passed the checks."""
    settings = records[0].meta["settings"]
    scales = {
        r.meta["sensor"]: r.meta["settings"]["mcl_settings"]["covariance_scale"] for r in records
    }
    timings = [record.meta["timings"] for record in records]
    return {
        "seeds": sorted({record.meta["seed"] for record in records}),
        "steps": settings["steps"],
        "kidnap_at": settings["kidnap_at"],
        "deadline": deadline,
        "hold": hold,
        "mcl_settings": _shared_settings(settings)["mcl_settings"],
        "mcl_scales": dict(sorted(scales.items())),
        "ekf_reset": _ekf_settings(records) or settings["ekf_reset"],
        "machines": sorted({timing["machine"] for timing in timings}),
        "blas_threads": sorted({timing["blas_threads"] for timing in timings}),
        "jobs_notes": sorted({timing["jobs_note"] for timing in timings}),
    }


def merge_records(records: Sequence[TrialRecord], *, deadline: int, hold: int) -> Comparison:
    """Summarize records into rows, one per sensor and dropout, sorted by sensor then dropout.

    Each set's trials are taken in seed order, so records of the same trials give the same
    figures however they were split across runs. ``deadline`` and ``hold`` are those of
    :func:`~rl_navigation.estimation.evaluation.localization_steps`.

    Raises:
        ValueError: As :func:`check_records`; a row without one of the four sets, or with sets
            of different seeds, or rows of different seeds; or a localization window past the
            kidnapping.
    """
    check_records(records)
    groups: dict[tuple[str, float | None], dict[str, list[TrialRecord]]] = {}
    for record in records:
        row = groups.setdefault((record.meta["sensor"], record.meta["dropout"]), {})
        row.setdefault(record.meta["mode"], []).append(record)
    ekf_reset = _ekf_settings(records)
    window = 0 if ekf_reset is None else int(ekf_reset["reset_window"])
    rows = []
    row_seeds: dict[str, list[int]] = {}
    for sensor, dropout in sorted(
        groups, key=lambda key: (key[0], key[1] is not None, key[1] or 0.0)
    ):
        sets = groups[sensor, dropout]
        missing = [mode for mode in MODES if mode not in sets]
        if missing:
            raise ValueError(
                f"Row {sensor}, dropout {_dropout_text(dropout)} has no {', '.join(missing)} "
                "trial records."
            )
        seeds = {mode: sorted(r.meta["seed"] for r in sets[mode]) for mode in MODES}
        if len({tuple(found) for found in seeds.values()}) > 1:
            raise ValueError(
                f"Row {sensor}, dropout {_dropout_text(dropout)} has sets of different seeds: "
                f"{seeds}."
            )
        if seeds["ekf_tracking"] != row_seeds.setdefault("seeds", seeds["ekf_tracking"]):
            raise ValueError(
                f"Row {sensor}, dropout {_dropout_text(dropout)} has seeds "
                f"{seeds['ekf_tracking']}, other rows {row_seeds['seeds']}: every row of a "
                "comparison needs the same seeds."
            )
        metrics: dict[str, float] = {}
        for mode, (_, kidnapped) in MODES.items():
            ordered = sorted(sets[mode], key=lambda record: record.meta["seed"])
            summary = _set_metrics(ordered, kidnapped, deadline, hold, window)
            metrics.update({f"{PREFIXES[mode]}{key}": value for key, value in summary.items()})
        rows.append(ComparisonRow(sensor, dropout, metrics))
    return Comparison(rows, list(records), _report_meta(records, deadline, hold))


def _jsonable(value: Any) -> Any:
    """``value`` as it reads back from JSON: tuples become lists."""
    return json.loads(json.dumps(value))


def _blas_threads() -> str:
    found = [f"{name}={os.environ[name]}" for name in BLAS_VARIABLES if name in os.environ]
    return ", ".join(found) if found else "BLAS threads not set"


def _run_record(
    seed: int,
    sensor: RangeSensor,
    dropout: float | None,
    factory: LocalizerFactory,
    *,
    steps: int,
    kidnap_at: int | None,
    meta: Mapping[str, Any],
) -> TrialRecord:
    """Run one instrumented trial; ``meta`` holds sensor, mode, settings and the timing notes."""
    instrumented = InstrumentedFactory(factory)
    trial = run_trial(
        seed,
        sensor,
        steps=steps,
        dropout_prob=dropout,
        make_localizer=instrumented,
        kidnap_at=kidnap_at,
    )
    built = instrumented.built
    if built is None:  # run_trial builds its filter before its first step
        raise RuntimeError("run_trial did not build a filter.")
    resets = (
        None
        if built.reset_steps is None
        else {"steps": built.reset_steps, "max_recent": built.max_recent_resets}
    )
    timing_notes = {key: meta[key] for key in ("machine", "blas_threads", "jobs_note")}
    return TrialRecord(
        trial,
        {
            "seed": seed,
            "sensor": meta["sensor"],
            "dropout": None if dropout is None else float(dropout),
            "mode": meta["mode"],
            "settings": meta["settings"],
            "resets": resets,
            "timings": {"seconds": dict(built.seconds), **timing_notes},
        },
    )


def run_records(
    seeds: Sequence[int],
    sensors: Mapping[str, SensorFactory],
    dropouts: Sequence[float | None] = (None, 0.3),
    *,
    steps: int = 600,
    kidnap_at: int = 300,
    mcl_settings: MclSettings = MclSettings(),  # noqa: B008 -- frozen, so sharing it is safe
    mcl_scales: Mapping[str, Sequence[float]] | None = None,
    ekf_reset: EkfReset = EkfReset(),  # noqa: B008 -- frozen, so sharing it is safe
    jobs_note: str = "not given",
    records_dir: Path | str | None = None,
) -> list[TrialRecord]:
    """Run the four trial sets of every sensor and dropout over ``seeds``, one record a trial.

    Args:
        seeds: The trials' seeds; every set runs all of them.
        sensors: Name and factory of each sensor; the name labels its rows and records.
        dropouts: Dropout probabilities; None is the sensor's own.
        steps: Steps per trial.
        kidnap_at: The step at which the kidnap sets carry the robot off.
        mcl_settings: MCL's settings; its ``covariance_scale`` is replaced per sensor.
        mcl_scales: MCL's covariance scale per sensor name; None, or a sensor left out, is
            ``(1, 1, 1)``.
        ekf_reset: The EKF's still-robot reset.
        jobs_note: What else ran on the machine meanwhile, printed beside the timings.
        records_dir: Save each record here as soon as its trial ends, so a run that fails
            keeps the trials it finished; a record already there is loaded instead of run
            again (:func:`merge_records` refuses it if its settings differ).

    Raises:
        ValueError: A scale for a sensor not in ``sensors``, or a setting out of range.
    """
    scales = {} if mcl_scales is None else dict(mcl_scales)
    unknown = sorted(set(scales) - set(sensors))
    if unknown:
        raise ValueError(f"MCL covariance scales given for sensors not compared: {unknown}.")
    timing_meta = {
        "machine": platform.platform(),
        "blas_threads": _blas_threads(),
        "jobs_note": jobs_note,
    }
    if records_dir is not None:
        Path(records_dir).mkdir(parents=True, exist_ok=True)
    reset_settings = asdict(ekf_reset)
    records = []
    sensor_settings = {}
    for name in sensors:  # every scale checked before the first trial runs
        sx, sy, stheta = checked_axis_values(
            scales.get(name, (1.0, 1.0, 1.0)), "run_records", f"mcl_scales[{name!r}]"
        )
        sensor_settings[name] = replace(
            mcl_settings, covariance_scale=(float(sx), float(sy), float(stheta))
        )
    for name, make_sensor in sensors.items():
        sensor_mcl = sensor_settings[name]
        settings = _jsonable(
            {
                "steps": steps,
                "kidnap_at": kidnap_at,
                "ekf_reset": reset_settings,
                "mcl_settings": asdict(sensor_mcl),
            }
        )
        factories = {
            "ekf_tracking": ekf(**reset_settings),
            "mcl_tracking": mcl(sensor_mcl),
            "mcl_kidnap": mcl(sensor_mcl, global_start=True),
            "ekf_kidnap": ekf(**reset_settings),
        }
        for dropout in dropouts:
            for mode, (_, kidnapped) in MODES.items():
                for seed in seeds:
                    path = None
                    if records_dir is not None:
                        path = Path(records_dir) / record_file_name(name, dropout, mode, seed)
                        if path.exists():
                            records.append(TrialRecord.load(path))
                            continue
                    record = _run_record(
                        int(seed),
                        make_sensor(),
                        dropout,
                        factories[mode],
                        steps=steps,
                        kidnap_at=kidnap_at if kidnapped else None,
                        meta={"sensor": name, "mode": mode, "settings": settings, **timing_meta},
                    )
                    if path is not None:
                        record.save(path)
                    records.append(record)
    return records


def compare_localizers(
    seeds: Sequence[int],
    sensors: Mapping[str, SensorFactory],
    dropouts: Sequence[float | None] = (None, 0.3),
    *,
    steps: int = 600,
    kidnap_at: int = 300,
    deadline: int = LOCALIZE_DEADLINE,
    hold: int = LOCALIZE_HOLD,
    mcl_settings: MclSettings = MclSettings(),  # noqa: B008 -- frozen, so sharing it is safe
    mcl_scales: Mapping[str, Sequence[float]] | None = None,
    ekf_reset: EkfReset = EkfReset(),  # noqa: B008 -- frozen, so sharing it is safe
    jobs_note: str = "not given",
    records_dir: Path | str | None = None,
) -> Comparison:
    """The EKF against MCL over ``seeds``: :func:`run_records`, then :func:`merge_records`.

    ``deadline`` and ``hold`` set when a filter counts as localized
    (:func:`~rl_navigation.estimation.evaluation.localization_steps`); the other arguments are
    :func:`run_records`'s.

    Raises:
        ValueError: ``deadline + hold`` past ``kidnap_at`` (checked before any trial runs), or
            as :func:`run_records`.
    """
    if deadline + hold > kidnap_at:
        raise ValueError(
            f"compare_localizers needs deadline + hold ({deadline} + {hold}) at most kidnap_at "
            f"({kidnap_at}): the first localization window must end by the kidnapping."
        )
    records = run_records(
        seeds,
        sensors,
        dropouts,
        steps=steps,
        kidnap_at=kidnap_at,
        mcl_settings=mcl_settings,
        mcl_scales=mcl_scales,
        ekf_reset=ekf_reset,
        jobs_note=jobs_note,
        records_dir=records_dir,
    )
    return merge_records(records, deadline=deadline, hold=hold)


def _axes(values: Sequence[float]) -> str:
    return ", ".join(f"{float(value):g}" for value in values)


def _seeds_text(seeds: Sequence[int]) -> str:
    """``101-150`` for a run of three or more consecutive seeds, else the seeds listed."""
    seeds = list(seeds)
    if len(seeds) > 2 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


def _median_text(steps: float, seconds: float) -> str:
    return "n/a" if np.isnan(steps) else f"{steps:g} steps ({seconds:.1f} s)"


def _table(header: Sequence[str], align: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [f"| {' | '.join(cells)} |" for cells in (header, align, *rows)]


def _tracking_rows(row: ComparisonRow) -> list[list[str]]:
    metrics = row.metrics
    label = [row.sensor, _dropout_text(row.dropout)]
    lines = []
    for name, prefix in (("EKF", "ekf_"), ("MCL", "mcl_")):

        def get(key: str, prefix: str = prefix) -> float:
            return metrics[prefix + key]

        resets = f"{get('max_recent_resets'):.0f}" if prefix == "ekf_" else "-"
        ratios = ", ".join(f"{get(f'axis_ratio_{axis}'):.2f}" for axis in ("x", "y", "theta"))
        lines.append(
            [
                *label,
                name,
                f"{get('position_rmse'):.4f}",
                f"{get('heading_rmse'):.4f}",
                f"{get('mean_nees'):.2f}",
                ratios,
                f"{get('nees_below_upper_95'):.3f}",
                f"{get('lost'):.0f}/{get('trials'):.0f}",
                resets,
                f"{get('ms_per_step'):.2f}",
            ]
        )
    lines.append(
        [
            *label,
            "Dead reckoning",
            f"{metrics['ekf_dead_reckoning_position_rmse']:.4f}",
            f"{metrics['ekf_dead_reckoning_heading_rmse']:.4f}",
            "-",
            "-",
            "-",
            f"{metrics['ekf_dead_reckoning_lost']:.0f}/{metrics['ekf_trials']:.0f}",
            "-",
            "-",
        ]
    )
    return lines


def _kidnap_rows(row: ComparisonRow) -> list[list[str]]:
    metrics = row.metrics
    lines = []
    for name, prefix, start in (
        ("MCL", "mcl_kidnap_", "anywhere"),
        ("EKF", "ekf_kidnap_", "noisy start"),
    ):

        def get(key: str, prefix: str = prefix) -> float:
            return metrics[prefix + key]

        resets = f"{get('max_recent_resets'):.0f}" if prefix == "ekf_kidnap_" else "-"
        wrong_localized = get("confidently_wrong_localized")
        lines.append(
            [
                row.sensor,
                _dropout_text(row.dropout),
                name,
                start,
                f"{get('global_share'):.2f}",
                _median_text(get("median_steps_to_localize"), get("seconds_to_localize")),
                f"{get('recovered_share'):.2f}",
                _median_text(get("median_steps_to_recover"), get("seconds_to_recover")),
                f"{get('confidently_wrong'):.3f}",
                "n/a" if np.isnan(wrong_localized) else f"{wrong_localized:.3f}",
                resets,
                f"{get('ms_per_step'):.2f}",
            ]
        )
    return lines


def format_markdown(rows: Sequence[ComparisonRow], meta: Mapping[str, Any]) -> str:
    """The comparison as a Markdown report: the settings, a tracking table, a global start and
    kidnapping table, and where the timings were taken.

    Args:
        rows: The rows, in the order to print them.
        meta: The settings, as :attr:`Comparison.meta` holds them: ``seeds``, ``steps``,
            ``kidnap_at``, ``deadline``, ``hold``, ``mcl_settings`` (``num_particles``,
            ``max_beams``, ``model_std``), ``mcl_scales`` (per sensor), ``ekf_reset`` (as
            :class:`EkfReset`), and the lists ``machines``, ``blas_threads`` and ``jobs_notes``.
    """
    mcl_settings, reset = meta["mcl_settings"], meta["ekf_reset"]
    scales = ", ".join(f"{sensor} ({_axes(scale)})" for sensor, scale in meta["mcl_scales"].items())
    if reset["reset_after"] is None:
        reset_text = "EKF reset: off."
    else:
        reset_text = (
            f"EKF reset: after {reset['reset_after']} counting scans of at least "
            f"{reset['min_candidates']} candidates, variance floor ({_axes(reset['reset_std'])}) "
            f"squared, window {reset['reset_window']} scans."
        )
    quantile = reset.get("nis_quantile")
    if quantile is None:
        hold_text = "EKF hold: off."
    else:
        hold_text = (
            f"EKF hold: an update that does not converge or has an NIS over the chi-square "
            f"{quantile:g} quantile keeps P."
        )
    lines = [
        "# Localization comparison: EKF and MCL",
        "",
        f"Seeds {_seeds_text(meta['seeds'])}; {meta['steps']} steps a run; the kidnap sets carry "
        f"the robot off at step {meta['kidnap_at']}.",
        f"Localized: within {LOCALIZED_POSITION:g} m and {LOCALIZED_HEADING:g} rad at each of the "
        f"{meta['hold']} steps that begin {meta['deadline']} steps after the start or the "
        f"kidnapping; a step is {STEP_SECONDS:g} s.",
        "",
        f"MCL: {mcl_settings['num_particles']} particles, {mcl_settings['max_beams']} beams, "
        f"model_std {mcl_settings['model_std']:g}; covariance scale {scales}.",
        reset_text,
        hold_text,
        "",
        "## Tracking from the noisy start",
        "",
        *_table(
            [
                "Sensor",
                "Dropout",
                "Filter",
                "Position RMSE (m)",
                "Heading RMSE (rad)",
                "Mean NEES",
                "Mean e_i^2/P_ii (x, y, theta)",
                f"NEES <= {NEES_UPPER_95:g}",
                "Lost",
                "Max recent resets",
                "ms/step",
            ],
            [":---"] * 3 + ["---:"] * 3 + [":---"] + ["---:"] * 4,
            [line for row in rows for line in _tracking_rows(row)],
        ),
        "",
        "## Global start and kidnapping",
        "",
        f"Confidently wrong: the share of steps after the kidnapping, over all kidnapped runs, "
        f"with position error over {LOST_ERROR:g} m, NEES over {NEES_UPPER_95:g} and, for the "
        f"EKF, no reset in the last {reset['reset_window']} scans.",
        "Localized runs: the same share over only the runs that had localized before the "
        "kidnapping (n/a when none had).",
        "",
        *_table(
            [
                "Sensor",
                "Dropout",
                "Filter",
                "Start",
                "Localized",
                "Median to localize",
                "Recovered",
                "Median to recover",
                "Confidently wrong (all runs)",
                "Confidently wrong (localized runs)",
                "Max recent resets",
                "ms/step",
            ],
            [":---"] * 4 + ["---:", ":---", "---:", ":---"] + ["---:"] * 4,
            [line for row in rows for line in _kidnap_rows(row)],
        ),
        "",
        f"Timings: {' / '.join(meta['machines'])}; {' / '.join(meta['blas_threads'])}; "
        f"concurrent jobs: {' / '.join(meta['jobs_notes'])}. ms/step: the filter's predict, "
        "update, pose and covariance per step.",
    ]
    return "\n".join(_wrap_prose(line) for line in lines) + "\n"


def _wrap_prose(line: str) -> str:
    """A prose line wrapped to Markdown's line length; table rows and headings stay whole."""
    if line.startswith(("|", "#")):
        return line
    return textwrap.fill(
        line, width=MARKDOWN_LINE_LENGTH, break_long_words=False, break_on_hyphens=False
    )


def _seed_list(text: str) -> list[int]:
    """``7`` or an inclusive range ``101-150``, low to high."""
    first, _, last = text.partition("-")
    try:
        low, high = int(first), int(last or first)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a seed or a range low-high: {text!r}") from error
    if high < low:
        raise argparse.ArgumentTypeError(f"seed range {text} runs backwards; write {high}-{low}")
    return list(range(low, high + 1))


def _dropout(text: str) -> float | None:
    """``own`` for the sensor's own dropout, else a probability."""
    return None if text == "own" else float(text)


# Options that set up a run, with their defaults: None in the parser, so that --from-records can
# tell that one was given and refuse it.
RUN_DEFAULTS: dict[str, Any] = {
    "sensors": sorted(SENSORS),
    "dropouts": [None, 0.3],
    "steps": 600,
    "kidnap_at": 300,
    "particles": MclSettings().num_particles,
    "mcl_scale": [],
    "jobs_note": "not given",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_navigation.estimation.comparison",
        description="Compare the EKF and MCL on seeded trials and print a Markdown report.",
    )
    parser.add_argument("--seeds", nargs="+", type=_seed_list, help="seeds or ranges, as 101-150")
    parser.add_argument("--sensors", nargs="+", choices=sorted(SENSORS), help="default: both")
    parser.add_argument(
        "--dropouts", nargs="+", type=_dropout, help="'own' or probabilities; default: own 0.3"
    )
    parser.add_argument("--steps", type=int, help="default: 600")
    parser.add_argument("--kidnap-at", type=int, help="default: 300")
    parser.add_argument("--deadline", type=int, default=LOCALIZE_DEADLINE)
    parser.add_argument("--hold", type=int, default=LOCALIZE_HOLD)
    parser.add_argument("--particles", type=int, help=f"default: {RUN_DEFAULTS['particles']}")
    parser.add_argument(
        "--mcl-scale",
        nargs=4,
        action="append",
        metavar=("SENSOR", "SX", "SY", "STH"),
        help="MCL covariance scale of one sensor (repeatable); (1, 1, 1) when not given",
    )
    parser.add_argument("--jobs-note", help="what else ran meanwhile, printed with the timings")
    parser.add_argument(
        "--records",
        type=Path,
        help="save one .npz record per trial here as it ends; records already there are reused",
    )
    parser.add_argument(
        "--from-records", nargs="+", type=Path, help="merge these record directories; run nothing"
    )
    parser.add_argument("--report", type=Path, help="also write the report here")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run (or merge records into) a comparison and print its report; ``--report`` saves it."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.from_records:
        given = [
            name for name in ("seeds", "records", *RUN_DEFAULTS) if getattr(args, name) is not None
        ]
        if given:
            flags = ", ".join("--" + name.replace("_", "-") for name in given)
            parser.error(
                f"--from-records runs nothing and takes its settings from the records: "
                f"leave out {flags}."
            )
        try:
            comparison = merge_records(
                load_records(args.from_records), deadline=args.deadline, hold=args.hold
            )
        except ValueError as error:  # records that do not merge, or a window past the kidnap
            parser.error(str(error))
    else:
        if not args.seeds:
            parser.error("--seeds is required unless --from-records is given.")
        for name, default in RUN_DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        seeds = [seed for seeds in args.seeds for seed in seeds]
        if len(set(seeds)) != len(seeds):
            parser.error(f"--seeds repeats a seed: {sorted(seeds)}.")
        scales = {}
        for sensor, *values in args.mcl_scale:
            if sensor not in args.sensors:
                parser.error(f"--mcl-scale names {sensor}, which is not in --sensors.")
            scales[sensor] = values
        if args.deadline + args.hold > args.kidnap_at:
            parser.error("--deadline plus --hold must be at most --kidnap-at.")
        try:
            scales = {sensor: [float(v) for v in values] for sensor, values in scales.items()}
            comparison = compare_localizers(
                seeds,
                {name: SENSORS[name] for name in args.sensors},
                args.dropouts,
                steps=args.steps,
                kidnap_at=args.kidnap_at,
                deadline=args.deadline,
                hold=args.hold,
                mcl_settings=MclSettings(num_particles=args.particles),
                mcl_scales=scales,
                jobs_note=args.jobs_note,
                records_dir=args.records,
            )
        except ValueError as error:  # a setting out of range, or reused records that differ
            parser.error(str(error))
    report = format_markdown(comparison.rows, comparison.meta)
    print(report, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
