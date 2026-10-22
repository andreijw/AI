"""Tests for configuration utilities."""

import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from unittest.mock import patch

import pytest
import yaml

from rl_core.utils.config import load_config, merge_configs, read_env_settings, save_config


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


def test_load_config_invalid_yaml_raises_value_error():
    """Invalid YAML content should raise ValueError with context."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("foo: [1, 2")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid configuration file"):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_config_invalid_json_raises_value_error():
    """Invalid JSON content should raise ValueError with context."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"foo": 1,,}')
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid configuration file"):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_config_requires_top_level_mapping():
    """Top-level config should be an object/mapping."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(["not", "a", "mapping"], f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="top-level mapping/object"):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_config_preserves_oserror_subclass_and_errno(tmp_path):
    """Read failures should keep the original OSError subclass and metadata."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: bar", encoding="utf-8")

    with (
        patch(
            "builtins.open",
            side_effect=PermissionError(13, "Permission denied", str(config_path)),
        ),
        pytest.raises(PermissionError) as exc_info,
    ):
        load_config(str(config_path))

    exc = exc_info.value
    assert exc.errno == 13
    assert exc.filename == str(config_path)
    assert exc.__notes__
    assert f"Failed to read config file '{config_path}'." in exc.__notes__


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


def test_merge_configs_deep_merge_nested_dicts():
    """Nested dicts should be merged recursively, not replaced wholesale."""
    base = {"agent": {"lr": 0.001, "gamma": 0.99}, "training": {"episodes": 1000}}
    override = {"agent": {"lr": 0.003}}  # only change lr; gamma should be preserved

    merged = merge_configs(base, override)

    assert merged["agent"]["lr"] == 0.003
    assert merged["agent"]["gamma"] == 0.99  # preserved from base
    assert merged["training"]["episodes"] == 1000  # untouched nested section


def test_merge_configs_deep_merge_does_not_mutate_inputs():
    """Deep merge should not mutate either input dict."""
    base = {"nested": {"a": 1, "b": 2}}
    override = {"nested": {"b": 99}}

    base_copy = {"nested": {"a": 1, "b": 2}}
    override_copy = {"nested": {"b": 99}}

    merge_configs(base, override)

    assert base == base_copy
    assert override == override_copy


def test_merge_configs_override_replaces_non_dict_with_dict():
    """If override value is a dict but base value is not, override wins."""
    base = {"key": "scalar"}
    override = {"key": {"nested": 42}}

    merged = merge_configs(base, override)
    assert merged["key"] == {"nested": 42}


def test_save_config_no_subdir():
    """save_config should work when the target dir already exists (no subdir)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "config.yaml")
        config = {"answer": 42}
        save_config(config, save_path)

        assert os.path.exists(save_path)
        loaded = load_config(save_path)
        assert loaded == config


ENV_DEFAULTS = {"name": "Toy-v0", "max_episode_steps": 50, "obs_noise_std": 0.0}


def test_read_env_settings_fills_defaults_and_treats_null_as_unset():
    config = {"name": None, "max_episode_steps": 9}
    settings = read_env_settings(config, ENV_DEFAULTS, "Toy-v0")
    assert settings == {"name": "Toy-v0", "max_episode_steps": 9, "obs_noise_std": 0.0}
    assert config == {"name": None, "max_episode_steps": 9}, "the config must not be changed"


def test_read_env_settings_leaves_render_mode_and_seed_to_the_caller():
    settings = read_env_settings({"render_mode": "rgb_array", "seed": 3}, ENV_DEFAULTS, "Toy-v0")
    assert "render_mode" not in settings and "seed" not in settings, (
        "render_mode and seed are allowed but passed through by the caller, null meaning null"
    )


@pytest.mark.parametrize("key", ["action_noise_prob", "obs_nosie_std", "domain_randomization"])
def test_read_env_settings_rejects_a_setting_the_owner_does_not_support(key):
    with pytest.raises(ValueError, match=f"Toy-v0 does not support .*{key}") as error:
        read_env_settings({key: 0.1}, ENV_DEFAULTS, "Toy-v0")
    assert "max_episode_steps, name, obs_noise_std" in str(error.value), "list what is supported"


@pytest.mark.parametrize("value", [None, 0, 0.0, False, {}, []])
def test_read_env_settings_accepts_an_unsupported_setting_left_off(value):
    """A config copied from another challenge with its noise off still loads."""
    assert (
        read_env_settings({"action_noise_prob": value}, ENV_DEFAULTS, "Toy-v0")["name"] == "Toy-v0"
    )
