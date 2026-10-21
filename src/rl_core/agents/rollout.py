"""Whole episodes buffered until a PPO rollout is full, and saved with the agent's checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np


class RolloutColumns(NamedTuple):
    """One episode, or a rollout of joined episodes, as PPO trains on it."""

    observations: np.ndarray  # (n, observation_dim)
    actions: np.ndarray  # (n,) discrete or (n, action_dim) continuous
    rewards: np.ndarray  # (n,) as processed for training: scaled, a cut's last one bootstrapped
    dones: np.ndarray  # (n,) True where the advantage estimate stops


class EpisodeRollout:
    """
    Whole episodes buffered until they hold at least ``rollout_steps`` transitions.

    Training waits for whole episodes, so a rollout can exceed ``rollout_steps`` by up to one
    episode. A partly filled rollout is saved in checkpoints (``ppo_rollout_<column>`` keys,
    none while the buffer is empty) and restored on load. The steps are saved as processed, so
    a resume with a different ``reward_scale`` keeps the old scale for them. When training ends,
    a partly filled rollout is not trained on.
    """

    KEY_PREFIX = "ppo_rollout_"

    def __init__(self, rollout_steps: int) -> None:
        self.rollout_steps = rollout_steps
        self._episodes: list[RolloutColumns] = []
        self._length = 0

    def __len__(self) -> int:
        """Transitions buffered."""
        return self._length

    def add(self, episode: RolloutColumns) -> RolloutColumns | None:
        """
        Buffer one episode.

        Returns:
            The full rollout, its episodes joined in order, once it holds ``rollout_steps``
            transitions (the buffer then starts again); otherwise None.
        """
        self._episodes.append(episode)
        self._length += len(episode.rewards)
        if self._length < self.rollout_steps:
            return None
        rollout = self._joined()
        self._episodes, self._length = [], 0
        return rollout

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        """The buffered steps as checkpoint arrays; empty while nothing is buffered."""
        if not self._episodes:
            return {}
        return {f"{self.KEY_PREFIX}{k}": v for k, v in self._joined()._asdict().items()}

    def restore(
        self, data: Mapping[str, Any], observation_dim: int, action_shape: tuple[int, ...]
    ) -> None:
        """
        Replace the buffer with a checkpoint's; a checkpoint without rollout keys empties it.

        Args:
            data: The checkpoint's arrays by name.
            observation_dim: Width of each observation.
            action_shape: Shape of one action: ``()`` for discrete, ``(action_dim,)`` continuous.

        Raises:
            ValueError: If only some rollout keys are present, or a column's shape is not
                ``(n, *step_shape)`` for one common ``n``. The buffer is then left unchanged.
        """
        keys = [f"{self.KEY_PREFIX}{name}" for name in RolloutColumns._fields]
        missing = [key for key in keys if key not in data]
        if len(missing) == len(keys):
            self._episodes, self._length = [], 0
            return
        if missing:
            raise ValueError(f"Checkpoint holds part of a rollout; missing {', '.join(missing)}")
        columns = RolloutColumns(*(np.asarray(data[key]) for key in keys))
        n_steps = len(columns.observations) if columns.observations.ndim else 0
        step_shapes = ((observation_dim,), action_shape, (), ())
        for key, column, step_shape in zip(keys, columns, step_shapes, strict=True):
            if column.shape != (n_steps, *step_shape):
                raise ValueError(
                    f"{key} has shape {column.shape}, expected {(n_steps, *step_shape)}"
                )
        self._episodes, self._length = ([columns], n_steps) if n_steps else ([], 0)

    def _joined(self) -> RolloutColumns:
        return RolloutColumns(*(np.concatenate(c) for c in zip(*self._episodes, strict=True)))
