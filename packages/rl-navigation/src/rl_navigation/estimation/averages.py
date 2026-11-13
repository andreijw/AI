"""A mean and a median over every element that are NaN, without a warning, when there is none.

The evaluation desk tools (``estimation.evaluation``, ``mapping.evaluation``, ``slam.evaluation``
and ``slam.feature_evaluation``) report a figure over no sample as NaN. A leaf: NumPy only.
"""

import numpy as np
from numpy.typing import ArrayLike


def mean_or_nan(values: ArrayLike) -> float:
    """The mean of every element of ``values``; NaN when it has none."""
    array = np.asarray(values)
    return float(array.mean()) if array.size else float("nan")


def median_or_nan(values: ArrayLike) -> float:
    """The median of every element of ``values``; NaN when it has none."""
    array = np.asarray(values)
    return float(np.median(array)) if array.size else float("nan")
