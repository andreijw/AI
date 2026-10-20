"""Strict readers for agent config values, so a mistyped setting fails at construction."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """``config[key]`` (or ``default``) as an int, rejecting booleans, non-integers and values <= 0."""
    return _checked_positive_int(key, config.get(key, default))


def optional_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    """``config[key]`` as an int, or None if unset; otherwise checked as :func:`positive_int` does."""
    value = config.get(key)
    return None if value is None else _checked_positive_int(key, value)


def _checked_positive_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return int(value)
