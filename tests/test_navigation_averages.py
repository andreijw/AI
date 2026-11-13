"""The desk tools' mean and median that give NaN, without a warning, when there is nothing to
average (rl_navigation.estimation.averages)."""

import math
import warnings

import numpy as np
import pytest

from rl_navigation.estimation.averages import mean_or_nan, median_or_nan


@pytest.mark.parametrize(
    "values, mean, median",
    [
        ([1.0, 2.0, 6.0], 3.0, 2.0),
        (np.array([4.0, 1.0, 2.0, 3.0]), 2.5, 2.5),
        (np.array([[True, False], [True, True]]), 0.75, 1.0),
    ],
)
def test_the_mean_and_median_of_every_element(values, mean, median):
    assert mean_or_nan(values) == mean
    assert median_or_nan(values) == median
    assert type(mean_or_nan(values)) is float and type(median_or_nan(values)) is float


@pytest.mark.parametrize(
    "values", [[], np.zeros(0), np.zeros((0, 3), dtype=bool), np.zeros((4, 0), dtype=bool)]
)
def test_nothing_to_average_is_nan_without_a_warning(values):
    """A ``(4, 0)`` grid has rows but no element: it counts as empty too."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(mean_or_nan(values))
        assert math.isnan(median_or_nan(values))
