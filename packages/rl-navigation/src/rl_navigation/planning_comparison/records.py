"""The comparison's rows as one JSON file, so that a run split into jobs is reported together.

A file holds a format number, a header (``(name, value)`` lines for the report) and the rows,
each with its episodes' :class:`~rl_navigation.planning_comparison.episodes.EpisodeRecord`
fields. :func:`merge_records` joins the files of one run's jobs.
"""

import dataclasses
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_navigation.planning_comparison.episodes import EpisodeRecord

FORMAT = 2  # 2 (2026-11-15): each row holds its robot_radius

Header = list[tuple[str, str]]


@dataclass(frozen=True)
class Row:
    """One row of the comparison: a controller on one environment setting.

    Args:
        name: The row's name (``"P-online-c"``, ``"TD3 e2-td3-s1"``).
        kind: ``"planner"`` or ``"policy"``.
        actions: ``"continuous"`` or ``"discrete"``.
        setting: The perturbed setting, or None for the nominal row.
        value: The setting's value, or None for the nominal row.
        episodes: The episodes, in seed order.
        robot_radius: The robot's radius in the environment played, in metres.
    """

    name: str
    kind: str
    actions: str
    setting: str | None
    value: float | None
    episodes: tuple[EpisodeRecord, ...]
    robot_radius: float

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.episodes]


def _row(data: dict[str, Any]) -> Row:
    episodes = tuple(
        EpisodeRecord(**{**e, "step_ms": tuple(e["step_ms"])}) for e in data["episodes"]
    )
    return Row(**{**data, "episodes": episodes})


def save_records(path: Path | str, rows: Sequence[Row], header: Sequence[tuple[str, str]]) -> None:
    """Write ``rows`` and ``header`` to ``path``, creating its folder. The file is written beside
    it and moved into place, so a run killed while saving keeps the rows saved before."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "format": FORMAT,
        "header": [list(line) for line in header],
        "rows": [dataclasses.asdict(row) for row in rows],
    }
    partial = f"{path}.partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def load_records(path: Path | str) -> tuple[list[Row], Header]:
    """The rows and header :func:`save_records` wrote.

    Raises:
        ValueError: If the file is of another format.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: records format {data.get('format')!r}, expected {FORMAT}")
    return [_row(row) for row in data["rows"]], [(name, value) for name, value in data["header"]]


def merge_records(paths: Sequence[Path | str]) -> tuple[list[Row], Header]:
    """The rows of every file in order, and their header lines, each once.

    Raises:
        ValueError: If the files' rows were played on different seeds, or a row (name, setting
            and value) is in two of them.
    """
    rows: list[Row] = []
    header: Header = []
    for path in paths:
        more, lines = load_records(path)
        rows += more
        header += [line for line in lines if line not in header]
    if len({tuple(row.seeds) for row in rows}) > 1:
        raise ValueError("the records were played on different seeds: merge one run's jobs only")
    keys = [(row.name, row.setting, row.value) for row in rows]
    repeated = sorted({key for key in keys if keys.count(key) > 1}, key=str)
    if repeated:
        raise ValueError(f"a row is in the records twice: {repeated[0]}")
    return rows, header
