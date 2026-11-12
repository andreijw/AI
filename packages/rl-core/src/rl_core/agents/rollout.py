"""Whole episodes buffered until a PPO rollout is full, and saved with the agent's checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

    def extend(self, episodes: Sequence[RolloutColumns]) -> RolloutColumns | None:
        """
        Buffer several parts (one window of several environments' steps) and check the size only
        after the last: the full rollout then holds all of them, so none is trained on after an
        update its steps did not see.

        Returns:
            The full rollout, as :meth:`add` returns it, or None.
        """
        self._episodes.extend(episodes)
        self._length += sum(len(episode.rewards) for episode in episodes)
        if not self._episodes or self._length < self.rollout_steps:
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
        columns = self._checked_columns(data, observation_dim, action_shape)
        n_steps = 0 if columns is None else len(columns.rewards)
        self._episodes, self._length = (
            ([columns], n_steps) if columns is not None and n_steps else ([], 0)
        )

    def check(
        self, data: Mapping[str, Any], observation_dim: int, action_shape: tuple[int, ...]
    ) -> None:
        """Raise the ValueError :meth:`restore` would, changing nothing."""
        self._checked_columns(data, observation_dim, action_shape)

    def _checked_columns(
        self, data: Mapping[str, Any], observation_dim: int, action_shape: tuple[int, ...]
    ) -> RolloutColumns | None:
        """The checkpoint's rollout columns once validated, or None if it holds no rollout."""
        keys = [f"{self.KEY_PREFIX}{name}" for name in RolloutColumns._fields]
        missing = [key for key in keys if key not in data]
        if len(missing) == len(keys):
            return None
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
        return columns

    def _joined(self) -> RolloutColumns:
        return RolloutColumns(*(np.concatenate(c) for c in zip(*self._episodes, strict=True)))


class RolloutWindows:
    """
    PPO's collection from several environments (``training.num_envs`` with ``update_every``).

    Each call of :meth:`update_segments` is one window of every environment's steps; the parts are
    buffered together and the rollout is trained on only at the end of a call, so every rollout is
    whole windows collected by the weights it starts from, its old log-probabilities are those of
    collection, and a part cut at the window's end bootstraps from the value before the update (as
    SB3's ``last_values``). ``rollout_steps`` must be a multiple of ``update_every x num_envs``.
    Whole episodes handed one at a time from several environments would straddle updates while
    PPO's ratio counted them as current, so several environments need ``update_every``; and one
    environment keeps training on whole episodes.

    The class using it provides ``rollout_steps``, ``_rollout`` (an :class:`EpisodeRollout`),
    ``_rollout_episode(batch)`` and ``_train_on_rollout(*columns)``.
    """

    rollout_steps: int
    _rollout: EpisodeRollout
    _rollout_episode: Callable[[Any], RolloutColumns | None]
    _train_on_rollout: Callable[..., dict[str, float]]
    _num_envs: int = 1

    def enable_parallel_envs(self, num_envs: int, update_every: int | None = None) -> None:
        """
        Raises:
            ValueError: Without ``update_every``, or if ``rollout_steps`` is not a multiple of
                ``update_every x num_envs``.
        """
        name = type(self).__name__
        if update_every is None:
            raise ValueError(
                f"{name} with num_envs {num_envs} needs update_every: whole episodes from several "
                "environments would be trained on after updates their steps did not see, and "
                "PPO's ratio would not notice"
            )
        if self.rollout_steps % (update_every * num_envs) != 0:
            raise ValueError(
                f"{name}: rollout_steps ({self.rollout_steps}) must be a multiple of update_every "
                f"x num_envs ({update_every} x {num_envs}), so that every rollout is whole windows"
            )
        self._num_envs = num_envs

    def enable_mid_episode_updates(self, update_every: int) -> None:
        """
        Raises:
            ValueError: With one environment (it trains on whole episodes), that is unless
                :meth:`enable_parallel_envs` came first.
        """
        if self._num_envs == 1:
            raise ValueError(
                f"{type(self).__name__} trains on whole episodes: update_every is for its "
                "collection from several environments (num_envs above 1)"
            )

    def update_segments(self, segments: Sequence[Any]) -> list[dict[str, float]]:
        """Buffer every part of the window, then train once if the rollout is full."""
        parts = [part for part in map(self._rollout_episode, segments) if part is not None]
        rollout = self._rollout.extend(parts)
        return [] if rollout is None else [self._train_on_rollout(*rollout)]
