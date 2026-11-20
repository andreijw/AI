"""The desk harness that plays the planner and trained policies on the same seeded layouts.

``controllers`` (a :class:`Policy` protocol, :class:`PolicyController`, :class:`PlannerController`
on the true or an online map), ``episodes`` (:func:`play` gives an :class:`EpisodeRecord`),
``metrics`` (Wilson intervals, SPL, medians on common successes, the exact McNemar test, the
registered gates), ``records`` (JSON) and ``report`` (Markdown).

It imports the Gymnasium layer (``rl_navigation.gym_env``), ``planning``, ``mapping`` and ``sim``,
never ``rl_core``: ``compare_planner.py`` at the repository root loads configs and checkpoints and
sweeps the sensor settings. ``rl_navigation.planning`` never imports this package.
"""

from rl_navigation.planning_comparison.controllers import (
    MAP_SOURCES,
    PLANNER_ROWS,
    RESOLUTION,
    Controller,
    PlannerController,
    Policy,
    PolicyController,
    navigation_env,
)
from rl_navigation.planning_comparison.episodes import EpisodeRecord, play, shortest_length
from rl_navigation.planning_comparison.metrics import (
    G4_SCOPE,
    G4_SCOPES,
    GATES,
    REPLAN_PERIOD,
    CommonSuccesses,
    Gate,
    GateVerdict,
    McNemar,
    RowSummary,
    SpeedVerdict,
    common_successes,
    gate_verdicts,
    mcnemar,
    sliding_means,
    speed_verdict,
    spl,
    summarize,
    wilson_interval,
)
from rl_navigation.planning_comparison.records import Row, load_records, merge_records, save_records
from rl_navigation.planning_comparison.report import render_report

__all__ = [
    "G4_SCOPE",
    "G4_SCOPES",
    "GATES",
    "MAP_SOURCES",
    "PLANNER_ROWS",
    "REPLAN_PERIOD",
    "RESOLUTION",
    "CommonSuccesses",
    "Controller",
    "EpisodeRecord",
    "Gate",
    "GateVerdict",
    "McNemar",
    "PlannerController",
    "Policy",
    "PolicyController",
    "Row",
    "RowSummary",
    "SpeedVerdict",
    "common_successes",
    "gate_verdicts",
    "load_records",
    "mcnemar",
    "merge_records",
    "navigation_env",
    "play",
    "render_report",
    "save_records",
    "shortest_length",
    "sliding_means",
    "speed_verdict",
    "spl",
    "summarize",
    "wilson_interval",
]
