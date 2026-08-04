"""Configuration loading and management."""

import json
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
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
            if file_ext in {".yaml", ".yml"}:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
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


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
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


def save_config(config: Dict[str, Any], save_path: str) -> None:
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
