"""Tests for the strict agent config readers."""

import numpy as np
import pytest

from rl_core.agents.config_values import (
    check_positive_int,
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
