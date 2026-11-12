"""Running mean and variance of an agent's observations, for ``normalize_observations``.

Follows Stable-Baselines3's ``RunningMeanStd`` and ``VecNormalize``: the statistics start at mean 0,
variance 1 and a count of 1e-4, each observation is folded in with Chan et al.'s parallel update (a
batch of one), and an observation is normalised as ``(x - mean) / sqrt(var + 1e-8)``, clipped to
±10. An entry that is always 0 (a masked velocity) keeps mean 0 and normalises to exactly 0; its
variance only falls towards 0 (the initial count keeps it at about 1e-4 / n).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

EPSILON = 1e-8
CLIP = 10.0
INITIAL_COUNT = 1e-4


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(x - mean) / scale`` clipped to ±10, as a new float64 array; ``scale`` is
    ``sqrt(var + 1e-8)``."""
    return np.clip((np.asarray(x, dtype=np.float64) - mean) / scale, -CLIP, CLIP)


class RunningMeanStd:
    """The running mean and variance of ``dim``-entry observations, in float64."""

    def __init__(self, dim: int) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = INITIAL_COUNT

    def update(self, x: np.ndarray) -> None:
        """Fold one observation into the statistics."""
        x = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        total = self.count + 1.0
        self.mean = self.mean + delta / total
        self.var = (self.var * self.count + delta**2 * self.count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """``x`` centred and scaled by the statistics, clipped to ±10, as a new float64 array."""
        return standardize(x, self.mean, self.scale)

    @property
    def scale(self) -> np.ndarray:
        """``sqrt(var + 1e-8)``, what :meth:`normalize` divides by."""
        return np.sqrt(self.var + EPSILON)

    def state(self) -> dict[str, np.ndarray]:
        """The statistics as arrays, by name (``mean``, ``var``, ``count``), for a checkpoint."""
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": np.array(self.count)}

    @staticmethod
    def check_state(state: Mapping[str, Any], dim: int) -> None:
        """
        Raise ValueError, naming the entry, unless ``state`` holds statistics :meth:`load_state`
        can take for ``dim``-entry observations: finite ``mean`` and ``var`` of shape ``(dim,)``,
        ``var`` not negative, and one positive ``count``.
        """
        for name in ("mean", "var"):
            value = np.asarray(state[name])
            if value.shape != (dim,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must hold {dim} finite values, got shape {value.shape}: {value!r}"
                )
        if np.any(np.asarray(state["var"]) < 0):
            raise ValueError(f"var must not be negative, got {state['var']!r}")
        count = np.asarray(state["count"])
        if count.shape != () or not np.isfinite(count) or count <= 0:
            raise ValueError(f"count must be one positive number, got {count!r}")

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Take the statistics :meth:`check_state` accepted; this step cannot fail."""
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(np.asarray(state["count"]))
