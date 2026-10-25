"""Tests for rl_core.utils.smoothing, the rolling average behind the solve numbers and the plots."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from rl_core.utils.smoothing import rolling_average


@pytest.mark.parametrize(
    ("data", "window", "expected"),
    [
        ([5.0], 3, [5.0]),  # one value averages to itself
        ([1.0, 2.0, 3.0], 1, [1.0, 2.0, 3.0]),  # a window of 1 changes nothing
        # [1], [1, 2], [2, 4], [4, 8] -> 1, 1.5, 3, 6: trailing, fewer values at the start
        ([1.0, 2.0, 4.0, 8.0], 2, [1.0, 1.5, 3.0, 6.0]),
        ([1.0, 2.0, 3.0, 4.0], 4, [1.0, 1.5, 2.0, 2.5]),  # window equal to the length
        ([2.0, 4.0], 100, [2.0, 3.0]),  # window longer than the data uses what there is
    ],
)
def test_rolling_average_trails_over_the_window(data, window, expected):
    """Each value averages itself and up to window - 1 values before it."""
    assert rolling_average(data, window) == expected


@pytest.mark.parametrize("window", [0, -1])
def test_rolling_average_rejects_a_window_below_one(window):
    """A window below 1 has no average; it must fail loudly instead of returning NaN."""
    with pytest.raises(ValueError, match="window must be >= 1"):
        rolling_average([1.0, 2.0], window)
