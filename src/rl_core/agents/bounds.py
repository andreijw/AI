"""Continuous action bounds read from an agent config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _bound(raw: Any, action_dim: int, name: str) -> np.ndarray:
    value = (
        np.full(action_dim, raw, dtype=np.float32)
        if np.ndim(raw) == 0
        else np.asarray(raw, dtype=np.float32)
    )
    if value.shape != (action_dim,):
        raise ValueError(
            f"{name} must be a scalar or have {action_dim} entries, got shape {value.shape}"
        )
    return value


def action_bounds(
    config: Mapping[str, Any], action_dim: int, require_finite: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``action_low`` and ``action_high`` (default -1 and 1) as float32 arrays of shape (action_dim,).

    A scalar applies to every action dimension. The Pendulum challenge fills both from the
    environment's action space.

    Args:
        config: Agent config.
        action_dim: Number of action dimensions.
        require_finite: Also require finite bounds with ``action_low < action_high``, as an agent
            that scales a tanh output to the bounds does.

    Raises:
        ValueError: If a bound has the wrong length, or ``require_finite`` is not met.
    """
    low = _bound(config.get("action_low", -1.0), action_dim, "action_low")
    high = _bound(config.get("action_high", 1.0), action_dim, "action_high")
    if require_finite:
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError(
                f"action_low and action_high must be finite, got {low.tolist()} and {high.tolist()}"
            )
        if (low >= high).any():
            raise ValueError(
                f"action_low must be below action_high, got {low.tolist()} and {high.tolist()}"
            )
    return low, high
