"""The comparison's figures: records checked and merged into rows, one per sensor and dropout."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_navigation.estimation.comparison.records import (
    MODES,
    PREFIXES,
    TrialRecord,
    _dropout_text,
)
from rl_navigation.estimation.evaluation import (
    LOST_ERROR,
    NEES_UPPER_95,
    Trial,
    localization_steps,
    summarize,
    summarize_localization,
)
from rl_navigation.estimation.poses import pose_difference
from rl_navigation.sim.geometry import FloatArray

# run_trial's World keeps its default step of 0.1 s.
STEP_SECONDS = 0.1

# A confidently wrong step whose position sigma is under this has collapsed onto a wrong pose,
# rather than sitting at the EKF's reset floor (0.2 m): a quarter of that floor, in metres.
COLLAPSED_POSITION_STD = 0.05


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
    trial: Trial,
    reset_steps: Sequence[int] | None,
    window: int,
    max_position_std: float | None = None,
) -> NDArray[np.bool_]:
    """For each step from the kidnapping on, whether the filter was wrong and did not show it.

    A step is confidently wrong when the position error exceeds :data:`LOST_ERROR`, the NEES
    exceeds :data:`NEES_UPPER_95` (the covariance does not cover the error) and, for a filter
    with resets (``reset_steps`` not None, the EKF), none of the last ``window`` scans, this one
    included, reset: a filter that keeps resetting is visibly lost, not confidently wrong. With
    ``max_position_std`` it also needs a position sigma (the square root of the larger eigenvalue
    of ``P[:2, :2]``) under it: collapsed onto a wrong pose, not merely uncovered.

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
    if max_position_std is not None:
        largest = np.linalg.eigvalsh(trial.covariances[:, :2, :2])[:, -1]
        wrong &= np.sqrt(largest) < max_position_std
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
    """The record's EKF settings; a record from before the hold or the off-map count ran
    without it."""
    return {"nis_quantile": None, "count_off_map": False, **record.meta["settings"]["ekf_reset"]}


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
        resets = [
            None if record.meta["resets"] is None else record.meta["resets"]["steps"]
            for record in records
        ]
        per_trial = [
            confidently_wrong(record.trial, steps, window)
            for record, steps in zip(records, resets, strict=True)
        ]
        localized = [
            wrong
            for wrong, trial in zip(per_trial, trials, strict=True)
            if localization_steps(trial, deadline, hold)[0] is not None
        ]
        summary["confidently_wrong"] = _share(per_trial)
        summary["confidently_wrong_localized"] = _share(localized)
        summary["wrong_small_sigma"] = _share(
            [
                confidently_wrong(record.trial, steps, window, COLLAPSED_POSITION_STD)
                for record, steps in zip(records, resets, strict=True)
            ]
        )
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
