"""Tests for configuration utilities."""

import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

import pytest
import yaml

from rl_cartpole.utils.config import load_config, merge_configs, save_config


def test_load_yaml_config():
    """Test loading YAML configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"key": "value", "number": 42}, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config["key"] == "value"
        assert config["number"] == 42
    finally:
        os.unlink(temp_path)


def test_load_json_config():
    """Test loading JSON configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value", "number": 42}, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config["key"] == "value"
        assert config["number"] == 42
    finally:
        os.unlink(temp_path)


def test_load_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_unsupported_format():
    """Test loading unsupported config format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("some text")
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_merge_configs():
    """Test merging configurations."""
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20, "d": 4}

    merged = merge_configs(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 20  # Overridden
    assert merged["c"] == 3
    assert merged["d"] == 4  # New key


def test_save_config():
    """Test saving configuration."""
    config = {"key": "value", "number": 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "subdir", "config.yaml")
        save_config(config, save_path)

        assert os.path.exists(save_path)

        loaded = load_config(save_path)
        assert loaded == config


def test_load_yml_extension():
    """Test loading a .yml file (alternate YAML extension)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump({"foo": "bar"}, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config["foo"] == "bar"
    finally:
        os.unlink(temp_path)


def test_merge_configs_does_not_mutate_base():
    """merge_configs should not modify the original base config."""
    base = {"a": 1, "b": 2}
    original_base = base.copy()
    merge_configs(base, {"b": 99, "c": 3})

    assert base == original_base


def test_merge_configs_empty_override():
    """Merging with an empty override should return a copy of base."""
    base = {"x": 10}
    merged = merge_configs(base, {})
    assert merged == base


def test_merge_configs_empty_base():
    """Merging an empty base with an override should return the override."""
    override = {"y": 20}
    merged = merge_configs({}, override)
    assert merged == override


def test_save_config_no_subdir():
    """save_config should work when the target dir already exists (no subdir)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "config.yaml")
        config = {"answer": 42}
        save_config(config, save_path)

        assert os.path.exists(save_path)
        loaded = load_config(save_path)
        assert loaded == config
