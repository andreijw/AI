"""Exploration noise for deterministic policies (DDPG; TD3 could reuse it).

The noise is in the scaled [-1, 1] action space; the agent multiplies it by its action scale. A
noise object holds no generator: the agent passes its exploration generator to every
:meth:`~ExplorationNoise.sample`, so the agent's checkpoint saves the generator as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

NOISE_TYPES = ("gaussian", "ornstein_uhlenbeck")
_OU_KEYS = ("ou_theta", "ou_dt")


class ExplorationNoise(Protocol):
    """Noise added to a deterministic policy's training actions."""

    def sample(self, rng: np.random.Generator) -> Array:
        """The noise for the next training action, shape (action_dim,)."""
        ...

    def reset(self) -> None:
        """Start a new episode."""
        ...


class GaussianNoise:
    """Independent ``N(0, std²)`` noise at every step."""

    def __init__(self, std: float, action_dim: int) -> None:
        self.std = std
        self.action_dim = action_dim

    def sample(self, rng: np.random.Generator) -> Array:
        return rng.normal(0.0, self.std, self.action_dim)

    def reset(self) -> None:
        """Nothing to reset: every draw is independent."""


class OrnsteinUhlenbeckNoise:
    """
    Temporally correlated noise, as SB3's ``OrnsteinUhlenbeckActionNoise`` with mean 0:
    ``x_{t+1} = x_t - theta * x_t * dt + sigma * sqrt(dt) * eps`` with ``eps ~ N(0, I)``, from
    ``x_0 = 0``.

    The state decays by ``1 - theta * dt`` per step, so with SB3's theta 0.15 and dt 0.01 one push
    direction lasts about 1 / (theta * dt) ≈ 667 steps; the stationary standard deviation is about
    ``sigma / sqrt(2 * theta)``. :meth:`reset` returns the state to zero; DDPG calls it at the end
    of every episode.
    """

    def __init__(
        self, sigma: float, action_dim: int, theta: float = 0.15, dt: float = 0.01
    ) -> None:
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self._state: Array = np.zeros(action_dim)

    def sample(self, rng: np.random.Generator) -> Array:
        increment = self.sigma * np.sqrt(self.dt) * rng.standard_normal(self._state.shape)
        self._state = self._state - self.theta * self._state * self.dt + increment
        return self._state.copy()

    def reset(self) -> None:
        self._state = np.zeros_like(self._state)


def make_exploration_noise(
    config: Mapping[str, Any], action_dim: int, default_std: float
) -> ExplorationNoise:
    """
    The noise a config asks for: ``noise_type`` ("gaussian" by default, or "ornstein_uhlenbeck"),
    ``noise_std`` (the Gaussian std or the OU sigma; ``default_std`` when unset), and for OU
    ``ou_theta`` (default 0.15) and ``ou_dt`` (default 0.01).

    Raises:
        ValueError: For an unknown ``noise_type``, a negative ``noise_std``, a non-positive
            ``ou_theta`` or ``ou_dt``, ``ou_theta * ou_dt`` above 1 (the state would overshoot
            zero), or an ``ou_*`` setting without ``noise_type: ornstein_uhlenbeck``.
    """
    noise_type = str(config.get("noise_type", "gaussian")).lower()
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
    std = float(config.get("noise_std", default_std))
    if std < 0.0:
        raise ValueError(f"noise_std must be at least 0, got {std!r}")
    if noise_type == "gaussian":
        stray = [key for key in _OU_KEYS if key in config]
        if stray:
            raise ValueError(f"{stray[0]} applies only with noise_type 'ornstein_uhlenbeck'")
        return GaussianNoise(std, action_dim)

    theta = float(config.get("ou_theta", 0.15))
    dt = float(config.get("ou_dt", 0.01))
    for name, value in (("ou_theta", theta), ("ou_dt", dt)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if theta * dt > 1.0:
        raise ValueError(f"ou_theta * ou_dt must be at most 1, got {theta} * {dt} = {theta * dt}")
    return OrnsteinUhlenbeckNoise(std, action_dim, theta=theta, dt=dt)
