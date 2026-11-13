"""The EKF against MCL on the same seeded trials, with a Markdown report and a command line.

Each sensor and dropout is one row of four trial sets over the same seeds
(:func:`~rl_navigation.estimation.evaluation.run_trial`): both filters tracking from the noisy
start, MCL from anywhere and kidnapped, and the EKF from the noisy start and kidnapped. Every
trial can be saved as a :class:`TrialRecord`, so seeds run in separate processes merge into one
report::

    python -m rl_navigation.estimation.comparison --seeds 101-125 --records runs/a
    python -m rl_navigation.estimation.comparison --seeds 126-150 --records runs/b
    python -m rl_navigation.estimation.comparison --from-records runs/a runs/b --report out.md

Not imported by :mod:`rl_navigation.estimation`: it is a tool for the desk, not for the robot.

The package's modules: ``instrumentation`` (timing the filters), ``records`` (a trial as a
``.npz``), ``metrics`` (records merged into rows), ``runner`` (running the trials), ``report``
(the Markdown) and ``cli`` (the command line). Every name is re-exported here.
"""

from rl_navigation.estimation.comparison.cli import RUN_DEFAULTS, main
from rl_navigation.estimation.comparison.instrumentation import (
    TIMED_CALLS,
    InstrumentedFactory,
    InstrumentedLocalizer,
)
from rl_navigation.estimation.comparison.metrics import (
    COLLAPSED_POSITION_STD,
    STEP_SECONDS,
    Comparison,
    ComparisonRow,
    axis_ratios,
    check_records,
    confidently_wrong,
    merge_records,
    nees_per_step,
)
from rl_navigation.estimation.comparison.records import (
    MODES,
    PREFIXES,
    TRIAL_ARRAYS,
    TrialRecord,
    load_records,
    record_file_name,
)
from rl_navigation.estimation.comparison.report import MARKDOWN_LINE_LENGTH, format_markdown
from rl_navigation.estimation.comparison.runner import (
    BLAS_VARIABLES,
    SENSORS,
    EkfReset,
    SensorFactory,
    compare_localizers,
    run_records,
)

__all__ = [
    "BLAS_VARIABLES",
    "COLLAPSED_POSITION_STD",
    "MARKDOWN_LINE_LENGTH",
    "MODES",
    "PREFIXES",
    "RUN_DEFAULTS",
    "SENSORS",
    "STEP_SECONDS",
    "TIMED_CALLS",
    "TRIAL_ARRAYS",
    "Comparison",
    "ComparisonRow",
    "EkfReset",
    "InstrumentedFactory",
    "InstrumentedLocalizer",
    "SensorFactory",
    "TrialRecord",
    "axis_ratios",
    "check_records",
    "compare_localizers",
    "confidently_wrong",
    "format_markdown",
    "load_records",
    "main",
    "merge_records",
    "nees_per_step",
    "record_file_name",
    "run_records",
]
