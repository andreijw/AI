"""Frame stacking and velocity masking of a flat observation vector.

:class:`BaseEnvWrapper` applies them after observation noise: the mask, then the stack, so a frame
keeps the noise it was observed with and a masked entry is exactly zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np

from rl_core.agents.config_values import check_positive_int


class ObservationOptions(TypedDict, total=False):
    """The observation settings every challenge's wrapper takes and passes to
    :class:`BaseEnvWrapper` unchanged, as keyword arguments."""

    frame_stack: int
    mask_velocity: bool


# Their config defaults, which every challenge's environment settings include: both off.
OBSERVATION_DEFAULTS: dict[str, Any] = {"frame_stack": 1, "mask_velocity": False}


class FrameStack:
    """
    The last ``size`` observations, oldest first, as one flat vector.

    :meth:`reset` fills every slot with the episode's first observation, as Gymnasium's
    ``FrameStackObservation`` does by default: zeros would be a state the environment never shows.
    Each call returns a new array, so a stack an agent keeps (in a buffer, say) never changes.
    """

    def __init__(self, size: Any) -> None:
        """
        Args:
            size: How many observations to stack, 1 or more.

        Raises:
            ValueError: If ``size`` is not a positive integer.
        """
        self.size = check_positive_int("frame_stack", size)
        self._frames: np.ndarray | None = None

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Start a new episode from ``obs`` and return the stack: ``obs`` repeated ``size`` times."""
        self._frames = np.repeat(np.asarray(obs)[np.newaxis], self.size, axis=0)
        return self._frames.reshape(-1).copy()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """
        Add ``obs`` as the newest frame, drop the oldest, and return the stack.

        Raises:
            RuntimeError: Before the first :meth:`reset`.
            ValueError: If ``obs`` has another shape than the episode's first observation.
        """
        if self._frames is None:
            raise RuntimeError("FrameStack.push called before reset")
        obs = np.asarray(obs)
        if obs.shape != self._frames.shape[1:]:
            raise ValueError(
                f"Observation shape {obs.shape} does not match the stacked frames' shape "
                f"{self._frames.shape[1:]}"
            )
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = obs
        return self._frames.reshape(-1).copy()


def _flat_box(space: gym.Space[Any]) -> gym.spaces.Box:
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise ValueError(f"Frame stacking and velocity masking need a 1-D Box space, got {space}")
    return space


def stacked_space(space: gym.Space[Any], size: int) -> gym.spaces.Box:
    """The space of a :class:`FrameStack` of ``size`` observations from ``space``: its bounds tiled."""
    box = _flat_box(space)
    size = check_positive_int("frame_stack", size)
    return gym.spaces.Box(low=np.tile(box.low, size), high=np.tile(box.high, size), dtype=box.dtype)


def masked_space(space: gym.Space[Any], indices: Sequence[int]) -> gym.spaces.Box:
    """
    ``space`` with the entries at ``indices`` pinned at zero, as :func:`mask_entries` leaves them.

    Raises:
        ValueError: If ``space`` is not a 1-D Box, or ``indices`` is empty or outside it.
    """
    box = _flat_box(space)
    if not indices or any(not 0 <= i < box.shape[0] for i in indices):
        raise ValueError(
            f"Velocity mask indices must be a non-empty list inside an observation of "
            f"{box.shape[0]} entries, got {tuple(indices)}"
        )
    low, high = box.low.copy(), box.high.copy()
    low[list(indices)] = 0
    high[list(indices)] = 0
    return gym.spaces.Box(low=low, high=high, dtype=box.dtype)


def mask_entries(obs: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """A copy of ``obs`` with the entries at ``indices`` set to zero."""
    masked = np.array(obs, copy=True)
    masked[list(indices)] = 0
    return masked
