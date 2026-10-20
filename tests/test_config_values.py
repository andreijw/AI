"""Tests for the strict agent config readers."""

import numpy as np
import pytest

from rl_core.agents.config_values import positive_int


@pytest.mark.parametrize("value", [1, 256, np.int64(8)])
def test_positive_int_accepts_positive_integers(value):
    result = positive_int({"rollout_steps": value}, "rollout_steps", 1)
    assert result == int(value)
    assert type(result) is int


def test_positive_int_falls_back_to_the_default():
    assert positive_int({}, "rollout_steps", 3) == 3


@pytest.mark.parametrize("value", [0, -1, 2.5, 4.0, True, "4", None])
def test_positive_int_rejects_everything_else_naming_the_key(value):
    with pytest.raises(ValueError, match="rollout_steps must be a positive integer"):
        positive_int({"rollout_steps": value}, "rollout_steps", 1)
