"""Deprecated: the robustness benchmark moved to :mod:`rl_core.robustness` in rl-core 0.2.0, and this path
is removed in 0.3.0.

Each name is handed out by a module ``__getattr__`` (PEP 562) that warns, so ``from
rl_core.training.robustness import x`` and ``import *`` keep working with a DeprecationWarning. The new module
is imported only when a name is used: importing the training package loads no challenge. Patching a name here
changes nothing; patch ``rl_core.robustness``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils.deprecation import warn_deprecated

if TYPE_CHECKING:  # so type checkers still see the names at the old path
    from ..robustness import (
        Perturbation,
        RobustnessRow,
        canonical_setting,
        difficulty_reached,
        parse_sweep,
        perturbed_section,
        physics_defaults,
        robustness_rows,
        training_range,
    )

__all__ = [
    "Perturbation",
    "RobustnessRow",
    "canonical_setting",
    "difficulty_reached",
    "parse_sweep",
    "perturbed_section",
    "physics_defaults",
    "robustness_rows",
    "training_range",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        f"rl_core.training.robustness.{name}",
        f"import {name} from rl_core.robustness",
        dist="rl-core",
        since="0.2.0",
        removed_in="0.3.0",
    )
    from .. import robustness

    return getattr(robustness, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
