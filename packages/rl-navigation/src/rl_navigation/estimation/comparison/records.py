"""A comparison's trials as records: one ``.npz`` per trial, saved and loaded without pickle."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl_navigation.estimation.evaluation import Trial

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
