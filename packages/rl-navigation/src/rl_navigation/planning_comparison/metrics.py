"""The comparison's statistics over episode records, and its registered gates.

- :func:`wilson_interval`: the Wilson score interval of a success rate.
- :func:`spl`: success weighted by path length (Anderson et al., 2018), the mean over episodes of
  ``S l / max(p, l)`` with ``S`` success, ``l`` the shortest length and ``p`` the length driven.
- :func:`common_successes`: median path length and time to goal of two rows on the layouts both
  reached, so that neither row's median is helped by the layouts it failed.
- :func:`mcnemar`: the exact McNemar test of two rows played on the same layouts, the two-sided
  binomial test (``p = 1/2``) on the discordant pairs.
- :func:`summarize`, :data:`GATES` and :func:`gate_verdicts`: the gates G1 to G3d registered with
  the plan (2026-11-13) before any judged run, and :func:`speed_verdict` for G4.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rl_navigation.planning import NavigatorParams
from rl_navigation.planning_comparison.episodes import EpisodeRecord
from rl_navigation.planning_comparison.records import Row

#: The two-sided 95 % normal quantile.
Z95 = 1.959963984540054

#: Steps in a replan period, the window of G4's sliding mean.
REPLAN_PERIOD = NavigatorParams().replan_every

#: Slack for comparing a measurement with a bound that is a sum of decimals (0.1 + 0.01).
_TOLERANCE = 1e-9


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of ``successes`` out of ``n``.

    Raises:
        ValueError: If ``n`` is not positive or ``successes`` is not in ``[0, n]``.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n] with n > 0, got {successes} of {n}")
    p, z2 = successes / n, z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def spl(records: Sequence[EpisodeRecord]) -> float:
    """Success weighted by path length: the mean of ``S l / max(p, l)``."""
    return float(
        np.mean(
            [r.success * r.shortest_length / max(r.path_length, r.shortest_length) for r in records]
        )
    )


def _paired(
    first: Sequence[EpisodeRecord], second: Sequence[EpisodeRecord]
) -> list[tuple[EpisodeRecord, EpisodeRecord]]:
    if [r.seed for r in first] != [r.seed for r in second]:
        raise ValueError("a pairing needs two rows played on the same seeds, in the same order")
    return list(zip(first, second, strict=True))


@dataclass(frozen=True)
class McNemar:
    """Discordant pairs and the exact two-sided p-value."""

    first_only: int
    second_only: int
    p_value: float


def mcnemar(first: Sequence[EpisodeRecord], second: Sequence[EpisodeRecord]) -> McNemar:
    """The exact McNemar test of two rows' successes, layout by layout.

    Raises:
        ValueError: If the rows were not played on the same seeds.
    """
    pairs = _paired(first, second)
    b = sum(a.success and not c.success for a, c in pairs)
    c = sum(not a.success and other.success for a, other in pairs)
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return McNemar(b, c, min(1.0, 2.0 * tail))


@dataclass(frozen=True)
class CommonSuccesses:
    """Medians on the layouts both rows reached (NaN when there are none)."""

    count: int
    first_path: float
    second_path: float
    first_time: float
    second_time: float


def common_successes(
    first: Sequence[EpisodeRecord], second: Sequence[EpisodeRecord]
) -> CommonSuccesses:
    """Median path length and time to goal of each row on the layouts both reached.

    Raises:
        ValueError: If the rows were not played on the same seeds.
    """
    both = [(a, b) for a, b in _paired(first, second) if a.success and b.success]
    if not both:
        return CommonSuccesses(0, math.nan, math.nan, math.nan, math.nan)
    return CommonSuccesses(
        len(both),
        float(np.median([a.path_length for a, _ in both])),
        float(np.median([b.path_length for _, b in both])),
        float(np.median([a.time for a, _ in both])),
        float(np.median([b.time for _, b in both])),
    )


def sliding_means(step_ms: Sequence[float], period: int = REPLAN_PERIOD) -> list[float]:
    """The mean of every window of ``period`` consecutive steps (one, of all, if shorter)."""
    if len(step_ms) <= period:
        return [float(np.mean(step_ms))]
    window = np.convolve(step_ms, np.ones(period) / period, mode="valid")
    return [float(v) for v in window]


@dataclass(frozen=True)
class RowSummary:
    """One row's outcomes, clearance and step times."""

    episodes: int
    successes: int
    collisions: int
    timeouts: int
    spl: float
    min_clearance: float
    clearance_p5: float
    reduced_margin: int
    step_ms_max: float
    sliding_ms_p99: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes

    @property
    def collision_rate(self) -> float:
        return self.collisions / self.episodes

    @property
    def success_interval(self) -> tuple[float, float]:
        """The success rate's Wilson 95 % interval."""
        return wilson_interval(self.successes, self.episodes)


def summarize(records: Sequence[EpisodeRecord], period: int = REPLAN_PERIOD) -> RowSummary:
    """A row's summary. ``reduced_margin`` counts the episodes whose plans stepped down the
    margin ladder; the sliding means of every episode are pooled before their 99th percentile.

    Raises:
        ValueError: If there are no records.
    """
    if not records:
        raise ValueError("a row needs at least one episode")
    clearances = [r.min_clearance for r in records]
    means = [m for r in records for m in sliding_means(r.step_ms, period)]
    return RowSummary(
        episodes=len(records),
        successes=sum(r.success for r in records),
        collisions=sum(r.collision for r in records),
        timeouts=sum(r.timeout for r in records),
        spl=spl(records),
        min_clearance=min(clearances),
        clearance_p5=float(np.quantile(clearances, 0.05)),
        reduced_margin=sum(r.ladder_level is not None and r.ladder_level > 0 for r in records),
        step_ms_max=max(r.plan_ms_max for r in records),
        sliding_ms_p99=float(np.quantile(means, 0.99)),
    )


@dataclass(frozen=True)
class Gate:
    """A registered gate on one planner row's nominal episodes.

    Args:
        name: The gate's name.
        row: The planner row it judges.
        min_success: The least success rate.
        max_collision_rate: The largest share of episodes ending in a collision.
        min_clearance: The least clearance over the robot's radius in metres, or None.
    """

    name: str
    row: str
    min_success: float
    max_collision_rate: float
    min_clearance: float | None

    def bound_text(self) -> str:
        parts = [
            f"success ≥ {self.min_success * 100:g}%",
            "no collision"
            if self.max_collision_rate == 0
            else f"collisions ≤ {self.max_collision_rate * 100:g}%",
        ]
        if self.min_clearance is not None:
            parts.append(f"min clearance ≥ r + {self.min_clearance:g} m")
        return ", ".join(parts)


#: The gates registered with the plan (2026-11-13), judged on layouts 30000 to 30499.
GATES = (
    Gate("G1", "P-true-c", 0.98, 0.0, 0.01),
    Gate("G2", "P-true-d", 0.97, 0.005, 0.005),
    Gate("G3c", "P-online-c", 0.93, 0.02, None),
    Gate("G3d", "P-online-d", 0.93, 0.02, None),
)


@dataclass(frozen=True)
class GateVerdict:
    gate: Gate
    summary: RowSummary
    passed: bool


def gate_verdicts(nominal: Mapping[str, Row]) -> list[GateVerdict]:
    """Each gate whose row is in ``nominal`` (planner row name to its nominal row), judged with
    that row's robot radius."""
    verdicts = []
    for gate in GATES:
        if gate.row not in nominal:
            continue
        row = nominal[gate.row]
        summary = summarize(row.episodes)
        passed = (
            summary.success_rate >= gate.min_success
            and summary.collision_rate <= gate.max_collision_rate
            and (
                gate.min_clearance is None
                or summary.min_clearance >= row.robot_radius + gate.min_clearance - _TOLERANCE
            )
        )
        verdicts.append(GateVerdict(gate, summary, passed))
    return verdicts


#: G4's bounds in milliseconds (the maintainer's decision, 2026-11-13): the 99th percentile of
#: the sliding mean over one replan period, and every single step.
SLIDING_P99_MS, STEP_MAX_MS = 15.0, 100.0

#: The rows G4 judges: ``"nominal"``, each planner's nominal row, or ``"every"``, every planner row
#: of the sweep too. Nominal: the four planner rows at nominal sensors, the perturbed rows' timings
#: reported, not gated (the maintainer's decision, 2026-11-15).
G4_SCOPES = ("nominal", "every")
G4_SCOPE = "nominal"


@dataclass(frozen=True)
class SpeedVerdict:
    sliding_ms_p99: float
    step_ms_max: float
    passed: bool


def speed_verdict(summary: RowSummary) -> SpeedVerdict:
    """G4 on one planner row."""
    passed = (
        summary.sliding_ms_p99 <= SLIDING_P99_MS + _TOLERANCE and summary.step_ms_max <= STEP_MAX_MS
    )
    return SpeedVerdict(summary.sliding_ms_p99, summary.step_ms_max, passed)
