"""Rolling averages of reward curves, shared by the benchmark's solve numbers and the plots.

Kept apart from the plotting module so that a change to how plots look cannot change a published
solve episode, and so that computing one does not import matplotlib.
"""

from collections.abc import Sequence

import numpy as np


def rolling_average(data: Sequence[float], window: int) -> list[float]:
    """
    Compute a trailing rolling average.

    Args:
        data: Sequence of float values.
        window: Look-back window size (must be >= 1).

    Returns:
        List of rolling-average values the same length as *data*.

    Raises:
        ValueError: If *window* is below 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    result: list[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start : i + 1])))
    return result
