"""The comparison as a Markdown report: :func:`format_markdown`."""

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rl_navigation.estimation.comparison.metrics import (
    COLLAPSED_POSITION_STD,
    STEP_SECONDS,
    ComparisonRow,
)
from rl_navigation.estimation.comparison.records import _dropout_text
from rl_navigation.estimation.evaluation import (
    LOCALIZED_HEADING,
    LOCALIZED_POSITION,
    LOST_ERROR,
    NEES_UPPER_95,
)

_SMALL_SIGMA = f"Wrong with σ < {100 * COLLAPSED_POSITION_STD:g} cm"

# CI's markdownlint (MD013) limit for lines outside tables.
MARKDOWN_LINE_LENGTH = 120


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
                f"{get('wrong_small_sigma'):.3f}",
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
    if reset.get("count_off_map", False):
        off_map_text = (
            "EKF off-map count: a scan with no candidate beam while the estimate is not in free "
            "space counts toward the reset."
        )
    else:
        off_map_text = "EKF off-map count: off."
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
        off_map_text,
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
        f"{_SMALL_SIGMA}: the confidently wrong steps whose position standard deviation (larger "
        f"axis) is under {COLLAPSED_POSITION_STD:g} m.",
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
                f"{_SMALL_SIGMA} (all runs)",
                "Max recent resets",
                "ms/step",
            ],
            [":---"] * 4 + ["---:", ":---", "---:", ":---"] + ["---:"] * 5,
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
