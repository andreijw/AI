"""The comparison's Markdown report: every row, the registered gates, and each policy paired with
each planner on the same setting, the like-for-like pairings (a policy against the online planner
of its own action kind) marked primary."""

import math
from collections.abc import Sequence

from rl_navigation.planning_comparison.controllers import PLANNER_ROWS
from rl_navigation.planning_comparison.metrics import (
    G4_SCOPE,
    GATES,
    REPLAN_PERIOD,
    SLIDING_P99_MS,
    STEP_MAX_MS,
    common_successes,
    gate_verdicts,
    mcnemar,
    speed_verdict,
    summarize,
)
from rl_navigation.planning_comparison.records import Row


def _setting(row: Row) -> str:
    if row.setting is None:
        return "nominal | –"
    return f"{row.setting} | {row.value:g}"


def _number(value: float, digits: int) -> str:
    return "–" if math.isnan(value) else f"{value:.{digits}f}"


def _rows_table(rows: Sequence[Row]) -> list[str]:
    lines = [
        "| Row | Setting | Value | Episodes | Success (95% CI) | Collisions | Timeouts | SPL "
        "| Min Clearance (m) | Clearance p5 (m) | Reduced Margin | Step Max (ms) "
        f"| Sliding {REPLAN_PERIOD}-Step p99 (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: "
        "| :---: | :---: |",
    ]
    for row in rows:
        s = summarize(row.episodes)
        low, high = s.success_interval
        reduced = "–" if row.kind == "policy" else str(s.reduced_margin)
        lines.append(
            f"| {row.name} | {_setting(row)} | {s.episodes} | {s.success_rate * 100:.1f}% "
            f"({low * 100:.1f}–{high * 100:.1f}) | {s.collisions} | {s.timeouts} | {s.spl:.3f} "
            f"| {s.min_clearance:.3f} | {s.clearance_p5:.3f} | {reduced} | {s.step_ms_max:.1f} "
            f"| {s.sliding_ms_p99:.2f} |"
        )
    return lines


def _robot_radius(rows: Sequence[Row]) -> float:
    radii = {row.robot_radius for row in rows}
    if len(radii) != 1:
        raise ValueError(f"a report needs one robot radius across its rows, got {sorted(radii)}")
    return radii.pop()


def _g4_rows(planners: Sequence[Row]) -> list[Row | str]:
    """The planner rows G4 judges, by :data:`~.metrics.G4_SCOPE`: each planner's nominal row in
    :data:`PLANNER_ROWS` order, or its name where the run has none, then (``"every"``) the
    perturbed rows."""
    nominal = {r.name: r for r in planners if r.setting is None}
    judged: list[Row | str] = [nominal.get(name, name) for name in PLANNER_ROWS]
    judged += [r for r in nominal.values() if r.name not in PLANNER_ROWS]
    if G4_SCOPE == "every":
        judged += [r for r in planners if r.setting is not None]
    return judged


def _gates(
    rows: Sequence[Row], timing_rows: Sequence[Row] | None, timing_source: str | None
) -> list[str]:
    planners = [r for r in rows if r.kind == "planner"]
    verdicts = {
        v.gate.name: v for v in gate_verdicts({r.name: r for r in planners if r.setting is None})
    }
    radius = _robot_radius(rows)
    scope = "each planner's nominal row" if G4_SCOPE == "nominal" else "every planner row"
    lines = [
        "## Gates",
        "",
        f"Registered with the plan (2026-11-13) on the nominal rows, with r = {radius:g} m:",
        "",
        *(f"- **{g.name}** ({g.row}): {g.bound_text()}" for g in GATES),
        f"- **G4** ({scope}): the 99th percentile of the {REPLAN_PERIOD}-step sliding mean "
        f"≤ {SLIDING_P99_MS:g} ms and every step ≤ {STEP_MAX_MS:g} ms",
        "",
        "| Gate | Row | Success | Collisions | Min Clearance (m) | Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for gate in GATES:
        if gate.name not in verdicts:
            lines.append(f"| {gate.name} | {gate.row} | – | – | – | not played |")
            continue
        v = verdicts[gate.name]
        s = v.summary
        lines.append(
            f"| {gate.name} | {gate.row} | {s.success_rate * 100:.1f}% "
            f"({s.successes}/{s.episodes}) | {s.collisions} ({s.collision_rate * 100:.1f}%) "
            f"| {s.min_clearance:.3f} | {'pass' if v.passed else 'FAIL'} |"
        )
    if timing_rows is None:
        timed = planners
        source = (
            "G4 below is from this run's step times, measured beside whatever else the machine "
            "was running (the header's load averages): not the solo timing run G4 is judged from."
        )
    else:
        timed = [r for r in timing_rows if r.kind == "planner"]
        source = (
            f"G4 is judged from the solo timing run, {timing_source}: the planner rows played "
            "alone on the machine. This run's step times are in the rows table, reported only."
        )
    lines += [
        "",
        source,
        "Perturbed rows' step times are reported, not gated.",
        "",
        f"| G4 Row | Setting | Value | Sliding {REPLAN_PERIOD}-Step p99 (ms) | Step Max (ms) "
        "| Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in _g4_rows(timed):
        if isinstance(row, str):
            lines.append(f"| {row} | nominal | – | – | – | not played |")
            continue
        speed = speed_verdict(summarize(row.episodes))
        lines.append(
            f"| {row.name} | {_setting(row)} | {speed.sliding_ms_p99:.2f} "
            f"| {speed.step_ms_max:.1f} | {'pass' if speed.passed else 'FAIL'} |"
        )
    return lines


def _primary(policy: Row, planner: Row) -> bool:
    """Like for like: the online planner of the policy's own action kind."""
    return PLANNER_ROWS.get(planner.name) == ("online", policy.actions)


def _pairs(rows: Sequence[Row]) -> list[tuple[Row, Row]]:
    """Each policy row with each planner row of the same setting and value."""
    planners = [r for r in rows if r.kind == "planner"]
    return [
        (policy, planner)
        for policy in rows
        if policy.kind == "policy"
        for planner in planners
        if (planner.setting, planner.value) == (policy.setting, policy.value)
    ]


def _pairings(rows: Sequence[Row]) -> list[str]:
    lines = [
        "## Policies against planners",
        "",
        "Paired layout by layout: both reached the goal, only the policy, only the planner; the "
        "exact McNemar test on the discordant pairs; medians on the layouts both reached. "
        "Reported, not gated. The primary pairings are like for like: each policy against the "
        "online planner of its own action kind (TD3 against P-online-c, DQN against "
        "P-online-d), which sees only the readings the policy sees; the true-map planners are "
        "an oracle upper bound.",
        "",
        "| Policy | Planner | Setting | Value | Both | Policy Only | Planner Only | McNemar p "
        "| Median Path (m), Policy / Planner | Median Time (s), Policy / Planner | Pairing |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for policy, planner in _pairs(rows):
        test = mcnemar(policy.episodes, planner.episodes)
        both = common_successes(policy.episodes, planner.episodes)
        lines.append(
            f"| {policy.name} | {planner.name} | {_setting(policy)} | {both.count} "
            f"| {test.first_only} | {test.second_only} | {test.p_value:.3f} "
            f"| {_number(both.first_path, 2)} / {_number(both.second_path, 2)} "
            f"| {_number(both.first_time, 1)} / {_number(both.second_time, 1)} "
            f"| {'primary' if _primary(policy, planner) else '–'} |"
        )
    return lines


def _spread(rows: Sequence[Row]) -> list[str]:
    """Per agent type with several checkpoints, against each planner on each setting: the range
    of the checkpoints' success rates and of their differences from the planner's. There is no
    pooled claim."""
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for policy, planner in _pairs(rows):
        rate = summarize(policy.episodes).success_rate
        difference = rate - summarize(planner.episodes).success_rate
        key = (policy.name.split()[0], planner.name, _setting(policy))
        groups.setdefault(key, []).append((rate, difference))
    lines = [
        f"| {agent} | {planner_name} | {setting} | {len(values)} "
        f"| {min(r for r, _ in values) * 100:.1f} to {max(r for r, _ in values) * 100:.1f}% "
        f"| {min(d for _, d in values) * 100:+.1f} to {max(d for _, d in values) * 100:+.1f} |"
        for (agent, planner_name, setting), values in groups.items()
        if len(values) > 1
    ]
    if not lines:
        return []
    return [
        "## Spread across checkpoints",
        "",
        "Each agent type with several checkpoints: the range of their success rates, and of each "
        "one's success minus the planner's in points.",
        "",
        "| Agent | Planner | Setting | Value | Checkpoints | Success | Difference (points) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
        *lines,
        "",
    ]


def render_report(
    rows: Sequence[Row],
    *,
    title: str,
    header: Sequence[tuple[str, str]],
    timing_rows: Sequence[Row] | None = None,
    timing_source: str | None = None,
) -> str:
    """The Markdown report of ``rows``, with ``header`` as its opening list. G4 is judged on
    ``timing_rows`` (a solo timing run of the planner rows, named by ``timing_source``) when
    given, else on ``rows``' own step times, and the report says which.

    Raises:
        ValueError: If the rows were played with different robot radii.
    """
    return "\n".join(
        [
            f"# {title}",
            "",
            *(f"- **{name}**: {value}" for name, value in header),
            "",
            "Episode i of every row resets with the same seed, so every row plays the same layouts. "
            "SPL weights each success by the shortest path (the planner's on the true map with no "
            "margin) over the longer of it and the path driven. Two biases go with it: that "
            "shortest path is A* on a grid inflated by r plus half a cell, then smoothed, so it "
            "overestimates the true shortest path, which raises every row's SPL a little; and it "
            "runs to the goal's centre while an episode succeeds on entering the goal radius "
            "(0.2 m), so a robot can stop short of the shortest length and its ratio is capped "
            "at 1. Both apply alike to every row. Min clearance is from the robot's centre; "
            "Reduced Margin counts episodes whose plans stepped down the margin ladder.",
            "",
            "## Rows",
            "",
            *_rows_table(rows),
            "",
            *_gates(rows, timing_rows, timing_source),
            "",
            *_pairings(rows),
            "",
            *_spread(rows),
        ]
    )
