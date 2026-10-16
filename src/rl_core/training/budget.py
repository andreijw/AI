"""How long a run trains: in episodes (``training.num_episodes``) or environment steps
(``training.total_steps``), and the one rule for overriding either from the command line."""

from __future__ import annotations

from typing import Any


def apply_budget_override(
    training: dict[str, Any], episodes: int | None, steps: int | None
) -> dict[str, Any]:
    """
    Apply a command-line budget to a ``training`` config section, in place.

    An episode budget replaces any step budget (``total_steps`` is removed, since it would take
    precedence); a step budget is set alongside ``num_episodes``, which the Trainer then ignores.

    Args:
        training: The config's ``training`` mapping.
        episodes: Episodes for every run, or None.
        steps: Environment steps for every run, or None.

    Returns:
        ``training``, for chaining.

    Raises:
        ValueError: If both are given.
    """
    if episodes is not None and steps is not None:
        raise ValueError("Give either an episode or a step budget, not both")
    if episodes is not None:
        training["num_episodes"] = episodes
        training.pop("total_steps", None)
    if steps is not None:
        training["total_steps"] = steps
    return training


def budget_of(training: dict[str, Any]) -> tuple[int, str]:
    """
    The budget a ``training`` config section asks for, and its unit: ``total_steps`` steps when
    set (the Trainer then ignores ``num_episodes``), else ``num_episodes`` episodes.
    """
    if training.get("total_steps") is not None:
        return int(training["total_steps"]), "steps"
    return int(training["num_episodes"]), "episodes"
