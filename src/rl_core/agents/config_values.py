"""Strict readers for agent config values, so a mistyped setting fails at construction."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """``config[key]`` (or ``default``) as an int, rejecting booleans, non-integers and values <= 0."""
    return _checked_int(key, config.get(key, default), minimum=1)


def optional_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    """``config[key]`` as an int, or None if unset; otherwise checked as :func:`positive_int` does."""
    value = config.get(key)
    return None if value is None else _checked_int(key, value, minimum=1)


def non_negative_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """``config[key]`` (or ``default``) as an int, rejecting booleans, non-integers and values < 0."""
    return _checked_int(key, config.get(key, default), minimum=0)


def flag(config: Mapping[str, Any], key: str, default: bool) -> bool:
    """``config[key]`` (or ``default``) as a bool, rejecting anything but a Python or NumPy boolean.

    A quoted ``"false"`` from a hand-written YAML file would otherwise be truthy.
    """
    value = config.get(key, default)
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return bool(value)


def check_positive_int(name: str, value: Any) -> int:
    """One value (such as a list entry) as an int, checked as :func:`positive_int` does under ``name``."""
    return _checked_int(name, value, minimum=1)


_KIND_BY_MINIMUM = {0: "non-negative", 1: "positive"}


def _checked_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be a {_KIND_BY_MINIMUM[minimum]} integer, got {value!r}")
    return int(value)
