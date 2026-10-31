"""Tests for the strict agent config readers."""

import numpy as np
import pytest

from rl_core.agents.config_values import (
    async_envs_setting,
    check_positive_int,
    flag,
    non_negative_int,
    optional_positive_int,
    positive_int,
)


@pytest.mark.parametrize("value", [1, 256, np.int64(8)])
def test_positive_int_accepts_positive_integers(value):
    result = positive_int({"rollout_steps": value}, "rollout_steps", 1)
    assert result == int(value)
    assert type(result) is int


def test_positive_int_falls_back_to_the_default():
    assert positive_int({}, "rollout_steps", 3) == 3


def _required(config, key):
    return positive_int(config, key, 1)


@pytest.mark.parametrize("read", [_required, optional_positive_int])
@pytest.mark.parametrize("value", [0, -1, 2.5, 4.0, True, "4"])
def test_both_readers_reject_everything_else_naming_the_key(read, value):
    with pytest.raises(ValueError, match="rollout_steps must be a positive integer"):
        read({"rollout_steps": value}, "rollout_steps")


def test_positive_int_rejects_an_explicit_none():
    with pytest.raises(ValueError, match="rollout_steps must be a positive integer"):
        positive_int({"rollout_steps": None}, "rollout_steps", 1)


@pytest.mark.parametrize("config", [{}, {"minibatch_size": None}])
def test_optional_positive_int_is_none_when_unset(config):
    assert optional_positive_int(config, "minibatch_size") is None


def test_optional_positive_int_accepts_positive_integers():
    result = optional_positive_int({"minibatch_size": np.int32(64)}, "minibatch_size")
    assert result == 64
    assert type(result) is int


@pytest.mark.parametrize("value", [0, 1, 10_000, np.int64(5)])
def test_non_negative_int_accepts_zero_and_positive_integers(value):
    result = non_negative_int({"learning_starts": value}, "learning_starts", 7)
    assert result == int(value)
    assert type(result) is int


def test_non_negative_int_falls_back_to_the_default():
    assert non_negative_int({}, "learning_starts", 7) == 7


@pytest.mark.parametrize("value", [-1, 2.5, 4.0, True, "4", None])
def test_non_negative_int_rejects_everything_else_naming_the_key(value):
    with pytest.raises(ValueError, match="learning_starts must be a non-negative integer"):
        non_negative_int({"learning_starts": value}, "learning_starts", 7)


def test_check_positive_int_checks_one_value_under_the_given_name():
    assert check_positive_int("hidden_sizes[0]", np.int32(64)) == 64
    with pytest.raises(ValueError, match=r"hidden_sizes\[1\] must be a positive integer, got 2.5"):
        check_positive_int("hidden_sizes[1]", 2.5)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_flag_accepts_a_boolean(value):
    assert flag({"double_q": value}, "double_q", not value) is bool(value)


@pytest.mark.parametrize("default", [True, False])
def test_flag_falls_back_to_the_default(default):
    assert flag({}, "double_q", default) is default


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, np.int64(1)])
def test_flag_rejects_everything_else_naming_the_key(value):
    """A quoted "false" in a YAML file would otherwise be truthy and switch the setting on."""
    with pytest.raises(ValueError, match="double_q must be true or false"):
        flag({"double_q": value}, "double_q", False)


@pytest.mark.parametrize("default", [True, False])
def test_async_envs_setting_falls_back_to_the_default_when_unset(default):
    assert async_envs_setting({}, default) is default


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (True, True), (False, False), (np.bool_(True), True), (np.bool_(False), False)],
)
def test_async_envs_setting_reads_null_as_synchronous(value, expected):
    """null meant synchronous before the strict reader, and still does."""
    assert async_envs_setting({"async_envs": value}, True) is expected


@pytest.mark.parametrize("value", ["false", "no", "true", 0, 1])
def test_async_envs_setting_rejects_everything_else_naming_the_key(value):
    """A quoted "false" would otherwise be truthy and start one process per environment."""
    with pytest.raises(ValueError, match=f"async_envs must be true, false or null, got {value!r}"):
        async_envs_setting({"async_envs": value}, True)
