"""Configuration loading and management."""

import json
import os
from collections.abc import Mapping
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not supported
        OSError: If config file cannot be read
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()
    if file_ext not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported config format: {file_ext}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) if file_ext in {".yaml", ".yml"} else json.load(f)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid configuration file '{config_path}': {exc}") from exc
    except OSError as exc:
        exc.add_note(f"Failed to read config file '{config_path}'.")
        raise

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{config_path}' must contain a top-level mapping/object."
        )

    return config


def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Nested dictionaries are merged depth-first so that individual leaf values
    in *override_config* override their counterparts in *base_config* without
    discarding sibling keys that are absent from the override.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base

    Returns:
        Merged configuration dictionary (a new dict; inputs are not mutated)
    """
    merged = {**base_config}
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_config(config: dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        save_path: Path to save the configuration
    """
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def config_get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    """``config[key]``, or ``default`` when the key is missing or set to ``null`` in YAML."""
    value = config.get(key)
    return default if value is None else value


# Environment settings every challenge accepts without a default: the scripts read them, and a
# challenge passes them through as given (null meaning null).
PASS_THROUGH_ENV_SETTINGS = frozenset({"render_mode", "seed"})


def read_env_settings(
    config: Mapping[str, Any], defaults: Mapping[str, Any], owner: str
) -> dict[str, Any]:
    """
    A challenge's environment settings from a config's ``environment`` section.

    Each key of ``defaults`` is read with :func:`config_get`. ``render_mode`` and ``seed`` are
    allowed and left to the caller. Any other key set to a value is rejected; one left off (null,
    0, false or empty) is accepted, so a config copied from another challenge with its noise off
    still loads. A misspelt setting left off, such as ``obs_nosie_std: 0.0``, is accepted too.

    Args:
        config: The ``environment`` section.
        defaults: Each setting the challenge supports, keyed by its config name, with its default.
        owner: The challenge's environment id, named in the error.

    Raises:
        ValueError: If ``config`` sets a setting that is not in ``defaults``.
    """
    unsupported = sorted(
        key
        for key, value in config.items()
        if key not in defaults and key not in PASS_THROUGH_ENV_SETTINGS and value
    )
    if unsupported:
        supported = ", ".join(sorted(defaults))
        passed_through = ", ".join(sorted(PASS_THROUGH_ENV_SETTINGS))
        raise ValueError(
            f"{owner} does not support environment setting(s) {', '.join(unsupported)}; "
            f"supported: {supported} (and {passed_through})"
        )
    return {key: config_get(config, key, default) for key, default in defaults.items()}
