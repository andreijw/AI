"""The comparison's command line: ``python -m rl_navigation.estimation.comparison``."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rl_navigation.estimation.comparison.metrics import merge_records
from rl_navigation.estimation.comparison.records import load_records
from rl_navigation.estimation.comparison.report import format_markdown
from rl_navigation.estimation.comparison.runner import SENSORS, compare_localizers
from rl_navigation.estimation.evaluation import LOCALIZE_DEADLINE, LOCALIZE_HOLD
from rl_navigation.estimation.mcl import MclSettings


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
