"""Running the comparison's trials: :func:`run_records` and :func:`compare_localizers`."""

import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rl_navigation.estimation.comparison.instrumentation import InstrumentedFactory
from rl_navigation.estimation.comparison.metrics import Comparison, merge_records
from rl_navigation.estimation.comparison.records import MODES, TrialRecord, record_file_name
from rl_navigation.estimation.evaluation import (
    LOCALIZE_DEADLINE,
    LOCALIZE_HOLD,
    LocalizerFactory,
    ekf,
    mcl,
    run_trial,
)
from rl_navigation.estimation.mcl import MclSettings
from rl_navigation.estimation.poses import checked_axis_values
from rl_navigation.sim import RangeSensor, hc_sr04, lidar

SensorFactory = Callable[[], RangeSensor]

BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SENSORS: dict[str, SensorFactory] = {"hc_sr04": hc_sr04, "lidar": lidar}


@dataclass(frozen=True)
class EkfReset:
    """The EKF's still-robot reset and its hold, as :class:`EkfLocalizer` takes them.

    ``nis_quantile`` is the hold's setting rather than the reset's; it lives here because the
    records keep these settings as one ``ekf_reset`` entry, and a held scan counts toward the
    reset. A record written before the hold has no ``nis_quantile``: it ran without it (None).
    Likewise a record written before the off-map count has no ``count_off_map`` (False).
    """

    reset_after: int | None = 3
    min_candidates: int = 2
    reset_std: tuple[float, float, float] = (0.2, 0.2, 0.1)
    reset_window: int = 20
    nis_quantile: float | None = 0.999
    count_off_map: bool = True


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
