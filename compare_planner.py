"""Play the A*-plus-pure-pursuit planner and trained navigation policies on the same seeded
layouts, nominal and with the sensors' noise and dropout swept, and report how they compare.

The rows: ``P-true-c`` and ``P-true-d`` plan on the true map (an oracle, nominal only, since they
ignore the sensors), ``P-online-c`` and ``P-online-d`` on a map built from the observation's own
readings (the like-for-like rows), with continuous and discrete actions; and each ``--agent``
checkpoint, played greedily. Every row plays layouts ``--first-seed`` to ``--first-seed +
--episodes - 1``; each ``--sweep`` value is played by the online planners and the agents. The
harness is ``rl_navigation.planning_comparison``; this script holds every ``rl_core`` and
``train.py`` use: the configs, the checkpoints and the robustness sweep's settings.

Records are saved after every row; ``--resume`` continues a killed run from its records file,
playing only the rows it does not hold. A run split into jobs (``--records`` each) is reported
together with ``--from-records``. The header records the commit, whether the working tree was
clean (else a hash of its changes against the commit, untracked files included), the settings, and the load average at the start and the
end, since the step times are measured in the run.
"""

import argparse
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import run_provenance
from rl_core.challenges import discover_challenges
from rl_core.robustness import Perturbation, parse_sweep, perturbed_section
from rl_core.training.checkpoints import file_sha256
from rl_navigation.gym_env import MAX_EPISODE_STEPS
from rl_navigation.planning import DEFAULT_MAP_RANGE, TUNED_SETTINGS
from rl_navigation.planning_comparison import (
    PLANNER_ROWS,
    Controller,
    EpisodeRecord,
    PlannerController,
    PolicyController,
    Row,
    load_records,
    merge_records,
    navigation_env,
    play,
    render_report,
    save_records,
)
from train import load_run_config, numpy_agent_config, select_run_challenge

ROOT = os.path.dirname(os.path.abspath(__file__))
TITLE = "Planning comparison: A* and pure pursuit against trained policies"


class GreedyAgent:
    """An agent as robustness_rows plays it: each observation through the agent's normalisation,
    then a greedy action."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def select_action(self, observation: Any, training: bool = False) -> Any:
        return self.agent.select_action(
            self.agent.normalize_observation(observation, training=False), training=False
        )


@dataclass(frozen=True)
class PlannedRow:
    """A row to play: who drives, on which environment sections, for how many steps at most."""

    name: str
    kind: str
    actions: str
    challenge: Any
    sections: list[tuple[str | None, float | None, dict[str, Any]]]
    controller: Callable[[], Controller]
    max_steps: int
    robot_radius: float


def play_row(
    challenge: Any,
    section: Mapping[str, Any],
    controller: Controller,
    seeds: Iterable[int],
    *,
    max_steps: int,
    shortest_cache: dict[int, float],
) -> tuple[list[EpisodeRecord], float]:
    """One row: ``controller`` on a new environment built from ``section``, once per seed; and
    the environment's robot radius."""
    env = challenge.make_env(section, render_mode=None)
    try:
        records = [
            play(env, controller, seed, max_steps=max_steps, shortest_cache=shortest_cache)
            for seed in seeds
        ]
        return records, navigation_env(env).layout_params.robot_radius
    finally:
        env.close()


def _sections(parser, challenge, config, sweeps, swept):
    """The nominal section, then (when ``swept``) one per perturbation, refusing a setting the
    wrapper cannot sweep, a value it refuses, or a setting in two sweeps."""
    section = {k: v for k, v in config["environment"].items() if k != "curriculum"}
    env = challenge.make_env(section, render_mode=None)
    try:
        sections = [(None, None, section)]
        seen = set()
        for sweep in sweeps:
            try:
                perturbed = [(*perturbed_section(section, env, p), p.value) for p in sweep]
            except ValueError as error:
                parser.error(str(error))
            setting = perturbed[0][0]
            if setting in seen:
                parser.error(f"{setting} is swept twice: give its values in one --sweep")
            seen.add(setting)
            if swept:
                sections += [(name, value, s) for name, s, value in perturbed]
        raw = navigation_env(env)
        return sections, raw.continuous, raw.layout_params.robot_radius
    finally:
        env.close()


def _max_steps(config):
    return int(config["training"].get("max_steps_per_episode") or MAX_EPISODE_STEPS)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-seed", type=int, help="The first layout's reset seed")
    parser.add_argument("--episodes", type=int, help="Layouts per row, from --first-seed on")
    parser.add_argument(
        "--planners",
        nargs="*",
        choices=list(PLANNER_ROWS),
        default=list(PLANNER_ROWS),
        help="Planner rows to play (default: all four; give none for the agents only)",
    )
    parser.add_argument(
        "--agent",
        nargs=2,
        action="append",
        default=[],
        metavar=("CONFIG", "CHECKPOINT"),
        help="A trained agent: the config it trained with and its checkpoint; repeatable. The "
        "row is named after its type and the checkpoint's folder",
    )
    for kind, default in (
        ("continuous", "navigation_continuous_td3.yaml"),
        ("discrete", "navigation_dqn.yaml"),
    ):
        parser.add_argument(
            f"--{kind}-config",
            default=os.path.join(ROOT, "configs", default),
            help=f"Config whose environment the {kind} planner rows play (default: "
            f"configs/{default})",
        )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="SETTING=V1,V2,...",
        help="A sensor setting and the values to play the online planners and the agents at; "
        "repeatable, one per setting (e.g. dropout_prob=0,0.02,0.1)",
    )
    parser.add_argument("--records", help="JSON file to write every episode to, after each row")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run whose --records file exists: play only the rows it does not hold",
    )
    parser.add_argument("--report", help="Markdown file to write the report to (default: print)")
    parser.add_argument(
        "--timing-records",
        metavar="RECORDS",
        help="With --from-records: the solo timing run's records (the planner rows alone on the "
        "machine), which G4 is judged from",
    )
    parser.add_argument(
        "--from-records",
        nargs="+",
        metavar="RECORDS",
        help="Report these records files of one run together, playing nothing",
    )
    args = parser.parse_args()
    if args.timing_records and not args.from_records:
        parser.error("--timing-records goes with --from-records: report a finished run")
    if args.from_records:
        if args.agent or args.sweep or args.first_seed is not None or args.episodes is not None:
            parser.error("--from-records plays nothing: give it only --report")
        return parser, args, []
    if args.first_seed is None or args.episodes is None:
        parser.error("--first-seed and --episodes are required unless --from-records is given")
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")
    sweeps: list[list[Perturbation]] = []
    for text in args.sweep:
        try:
            sweeps.append(parse_sweep(text))
        except ValueError as error:
            parser.error(str(error))
    if not args.planners and not args.agent:
        parser.error("nothing to play: give --planners or --agent")
    return parser, args, sweeps


def _plan(parser, args, sweeps):
    """Every row to play and the header lines, with each config and checkpoint loaded and
    checked before anything is played."""
    discover_challenges()
    plan, header = [], []
    configs = {}
    for name in args.planners:
        map_source, actions = PLANNER_ROWS[name]
        path = getattr(args, f"{actions}_config")
        if path not in configs:
            config, agent_type = load_run_config(path)
            configs[path] = (config, select_run_challenge(config, agent_type))
        config, challenge = configs[path]
        sections, _, radius = _sections(
            parser, challenge, config, sweeps, swept=map_source == "online"
        )
        plan.append(
            PlannedRow(
                name,
                "planner",
                actions,
                challenge,
                sections,
                lambda m=map_source: PlannerController(m),
                MAX_EPISODE_STEPS,
                radius,
            )
        )
    for path in sorted({getattr(args, f"{PLANNER_ROWS[n][1]}_config") for n in args.planners}):
        header.append(("Planner environment", f"`{path}`"))
    for config_path, checkpoint in args.agent:
        config, agent_type = load_run_config(config_path)
        challenge = select_run_challenge(config, agent_type)
        sections, continuous, radius = _sections(parser, challenge, config, sweeps, swept=True)
        env = challenge.make_env(sections[0][2], render_mode=None)
        try:
            agent = challenge.make_agent(agent_type, numpy_agent_config(config), env)
            agent.load(checkpoint)
        finally:
            env.close()
        label = os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))
        name = f"{agent_type.upper()} {label}"
        plan.append(
            PlannedRow(
                name,
                "policy",
                "continuous" if continuous else "discrete",
                challenge,
                sections,
                lambda a=agent: PolicyController(GreedyAgent(a)),
                _max_steps(config),
                radius,
            )
        )
        header.append(
            (
                name,
                f"`{config_path}`, `{os.path.basename(checkpoint)}` "
                f"(sha256 {file_sha256(checkpoint)})",
            )
        )
    return plan, header


def _load_average() -> str:
    return " ".join(f"{load:.2f}" for load in os.getloadavg())


def _provenance(job: str) -> list[tuple[str, str]]:
    """The commit and working tree the run plays, and the load average now (the start)."""
    return [
        ("Commit", run_provenance.commit(ROOT)),
        ("Working tree", run_provenance.working_tree(ROOT)),
        (f"Load average at start ({job})", _load_average()),
    ]


def _plan_lines(header):
    """The header without its load averages: what a resumed run must share with its records."""
    return [line for line in header if not line[0].startswith("Load average")]


def _check_same(parser, what, saved, ours):
    """Refuse ``what`` unless every line of ``saved`` and ``ours`` agrees (``unknown`` never
    does)."""
    saved_lines, our_lines = dict(saved), dict(ours)
    for key in dict.fromkeys([*saved_lines, *our_lines]):
        if not run_provenance.same(saved_lines.get(key), our_lines.get(key)):
            parser.error(
                f"{what} is not this run: its {key!r} is {saved_lines.get(key)!r}, this run's "
                f"{our_lines.get(key)!r}"
            )


def _resumed(parser, args, header, plan):
    """The rows already in ``--records`` and the header to go on with (the saved one, with a
    resume load average), or nothing to resume. Refuses an existing file without --resume, and a
    file whose header (load averages aside) differs from this run's or that holds a row this
    run's plan does not."""
    if not args.records or not os.path.exists(args.records):
        return [], header
    if not args.resume:
        parser.error(f"{args.records} exists: give --resume to continue that run, or another path")
    try:
        rows, saved = load_records(args.records)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    _check_same(parser, args.records, _plan_lines(saved), _plan_lines(header))
    planned = {(p.name, setting, value) for p in plan for setting, value, _ in p.sections}
    for row in rows:
        if (row.name, row.setting, row.value) not in planned:
            parser.error(
                f"{args.records} holds {row.name} at {row.setting or 'nominal'} "
                f"{'' if row.value is None else row.value}, not in this run's plan"
            )
    job = os.path.basename(args.records)
    kept = [line for line in saved if not line[0].startswith("Load average at end")]
    return rows, [*kept, (f"Load average at resume ({job})", _load_average())]


def _play(parser, args, sweeps):
    """Play every row not already in the records, saving after each; the rows and header."""
    plan, lines = _plan(parser, args, sweeps)
    job = os.path.basename(args.records) if args.records else "this run"
    last = args.first_seed + args.episodes - 1
    radii = sorted({planned.robot_radius for planned in plan})
    header = [
        ("Seeds", f"{args.first_seed} to {last} ({args.episodes} layouts)"),
        *_provenance(job),
        ("Planners", ", ".join(args.planners) or "none"),
        ("Sweeps", " ".join(args.sweep) or "none"),
        *lines,
        (
            "Planner settings",
            "; ".join(
                f"{kind}: margin {s.margin:g} m, turn in place {s.turn_in_place:.4g} rad"
                for kind, s in TUNED_SETTINGS.items()
            ),
        ),
        ("Online map range", f"{DEFAULT_MAP_RANGE:g} m"),
        ("Robot radius", ", ".join(f"{r:g} m" for r in radii)),
    ]
    rows, header = _resumed(parser, args, header, plan)
    done = {(row.name, row.setting, row.value) for row in rows}

    def current_header():
        return [*header, (f"Load average at end ({job})", _load_average())]

    seeds = range(args.first_seed, last + 1)
    shortest: dict[int, float] = {}
    for planned in plan:
        for setting, value, section in planned.sections:
            at = "nominal" if setting is None else f"{setting}={value:g}"
            if (planned.name, setting, value) in done:
                print(f"Kept from the records: {planned.name}: {at}", flush=True)
                continue
            print(f"Playing {planned.name}: {at}", flush=True)
            records, radius = play_row(
                planned.challenge,
                section,
                planned.controller(),
                seeds,
                max_steps=planned.max_steps,
                shortest_cache=shortest,
            )
            row = Row(
                planned.name, planned.kind, planned.actions, setting, value, tuple(records), radius
            )
            rows.append(row)
            if args.records:
                save_records(args.records, rows, current_header())
    if args.records:
        print(f"Records saved to '{args.records}'")
    return rows, current_header()


def _merged(parser, paths):
    """The rows and header of one run's records files, refusing files of different or unknown
    code."""
    try:
        rows, header = merge_records(paths)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for key in ("Commit", "Working tree"):
        values = [value for name, value in header if name == key]
        # One value across the files, and a known one: "unknown" never matches itself.
        agree = len(values) == 1 and run_provenance.same(values[0], values[0])
        if len(paths) > 1 and not agree:
            parser.error(f"the records were played with different or unknown {key!r}: {values}")
    return rows, header


#: The header lines solo timing records must share with the run they judge G4 for.
_TIMING_KEYS = ("Commit", "Working tree", "Seeds")


def _timing(parser, path, header):
    """The solo timing run's rows and header: the same code and seeds as ``header``'s run, and
    exactly the four planner rows at nominal sensors."""
    try:
        rows, timing_header = load_records(path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    ours, theirs = dict(header), dict(timing_header)
    _check_same(
        parser,
        path,
        [(k, theirs.get(k)) for k in _TIMING_KEYS],
        [(k, ours.get(k)) for k in _TIMING_KEYS],
    )
    played = sorted(((r.name, r.kind, r.setting) for r in rows), key=str)
    if played != sorted(((name, "planner", None) for name in PLANNER_ROWS), key=str):
        parser.error(
            f"{path} must hold the four planner rows at nominal sensors only "
            f"({', '.join(PLANNER_ROWS)}), got {played}"
        )
    return rows, timing_header


def main():
    """Play every row, or report records already played, and write what was asked for."""
    parser, args, sweeps = _parse_args()
    if args.from_records:
        rows, header = _merged(parser, args.from_records)
    else:
        rows, header = _play(parser, args, sweeps)
    timing_rows = None
    if args.timing_records:
        timing_rows, timing_header = _timing(parser, args.timing_records, header)
        header = [
            *header,
            ("G4 timing", f"`{args.timing_records}`"),
            *((f"G4 timing: {name}", value) for name, value in timing_header),
        ]
    report = render_report(
        rows,
        title=TITLE,
        header=header,
        timing_rows=timing_rows,
        timing_source=f"`{args.timing_records}`",
    )
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to '{args.report}'")


if __name__ == "__main__":
    main()
