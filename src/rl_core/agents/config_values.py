"""Strict readers for agent config values, so a mistyped setting fails at construction."""

from typing import Any

import numpy as np


def positive_int(config: dict[str, Any], key: str, default: int) -> int:
    """``config[key]`` (or ``default``) as an int, rejecting booleans, non-integers and values <= 0."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return int(value)
