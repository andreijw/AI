"""Experience replay buffer for off-policy reinforcement learning."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from rl_core.seeding import (
    AGENT_REPLAY_STREAM,
    check_generator_state,
    derive_rng,
    generator_state,
    restore_generator,
)

_COLUMNS = ("observations", "actions", "rewards", "next_observations", "dones")


class ReplayBuffer:
    """
    Fixed-capacity FIFO ring buffer storing transition tuples for off-policy RL.

    Stores transitions in pre-allocated NumPy arrays to ensure O(1) insertions
    and efficient uniform batch sampling without python list overhead.

    Transitions are represented as:
        (observation, action, reward, next_observation, done)

    Actions are discrete (int64 scalars) by default, or continuous float64 vectors of length
    ``action_dim`` when it is given.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        seed: int | None = None,
        *,
        action_dim: int | None = None,
    ) -> None:
        """
        Initialise the ReplayBuffer.

        Args:
            capacity: Maximum number of transitions to store.
            observation_dim: Dimensionality of observation vectors.
            seed: Optional RNG seed for reproducible random sampling.
            action_dim: Length of a continuous action vector, or None for discrete actions.

        Raises:
            ValueError: If capacity, observation_dim or action_dim is non-positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")

        if action_dim is not None and action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")

        self.capacity: int = int(capacity)
        self.action_dim: int | None = None if action_dim is None else int(action_dim)
        self.observation_dim: int = int(observation_dim)
        # Sampling on the agent's replay stream of the seed, apart from the environment's numbers.
        self._rng: np.random.Generator = derive_rng(seed, AGENT_REPLAY_STREAM)

        self.observations: npt.NDArray[np.float64] = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float64
        )
        self.actions: npt.NDArray[Any] = (
            np.zeros(self.capacity, dtype=np.int64)
            if self.action_dim is None
            else np.zeros((self.capacity, self.action_dim), dtype=np.float64)
        )
        self.rewards: npt.NDArray[np.float64] = np.zeros(self.capacity, dtype=np.float64)
        self.next_observations: npt.NDArray[np.float64] = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float64
        )
        self.dones: npt.NDArray[np.bool_] = np.zeros(self.capacity, dtype=bool)

        self._idx: int = 0
        self._size: int = 0

    def add(
        self,
        obs: npt.ArrayLike,
        action: int | npt.ArrayLike,
        reward: float,
        next_obs: npt.ArrayLike,
        done: bool,
    ) -> None:
        """
        Insert a single transition into the ring buffer.

        Overwrites the oldest transition when the buffer reaches capacity.

        Args:
            obs: Observation vector before transition.
            action: Discrete action taken, or a continuous action vector of length ``action_dim``.
            reward: Scalar reward received.
            next_obs: Observation vector after transition.
            done: Whether the transition is terminal, so its Bellman target must not bootstrap
                from ``next_obs``. Store a time-limit cut as not terminal.
        """
        self.observations[self._idx] = np.asarray(obs, dtype=np.float64)
        self.actions[self._idx] = self._actions_array(action, single=True)
        self.rewards[self._idx] = float(reward)
        self.next_observations[self._idx] = np.asarray(next_obs, dtype=np.float64)
        self.dones[self._idx] = bool(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        obs: npt.ArrayLike,
        actions: npt.ArrayLike,
        rewards: npt.ArrayLike,
        next_obs: npt.ArrayLike,
        dones: npt.ArrayLike,
    ) -> None:
        """
        Insert a batch of transitions into the ring buffer.

        Args:
            obs: Array of observations of shape (N, observation_dim).
            actions: Discrete actions of shape (N,), or continuous ones of shape (N, action_dim).
            rewards: Array of scalar rewards of shape (N,).
            next_obs: Array of next observations of shape (N, observation_dim).
            dones: Terminal flags of shape (N,): True means no bootstrap from the next
                observation (a true termination, not a time-limit cut).

        Raises:
            ValueError: If input array lengths do not match, or continuous actions are not
                shaped (N, action_dim).
        """
        obs_arr = np.asarray(obs, dtype=np.float64)
        actions_arr = self._actions_array(actions, single=False)
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        next_obs_arr = np.asarray(next_obs, dtype=np.float64)
        dones_arr = np.asarray(dones, dtype=bool)

        n = len(obs_arr)
        if not (len(actions_arr) == len(rewards_arr) == len(next_obs_arr) == len(dones_arr) == n):
            raise ValueError(
                f"Inconsistent batch lengths: obs={len(obs_arr)}, actions={len(actions_arr)}, "
                f"rewards={len(rewards_arr)}, next_obs={len(next_obs_arr)}, dones={len(dones_arr)}"
            )

        if n == 0:
            return

        # If incoming batch exceeds buffer capacity, keep the most recent elements
        if n > self.capacity:
            obs_arr = obs_arr[-self.capacity :]
            actions_arr = actions_arr[-self.capacity :]
            rewards_arr = rewards_arr[-self.capacity :]
            next_obs_arr = next_obs_arr[-self.capacity :]
            dones_arr = dones_arr[-self.capacity :]
            n = self.capacity

        first_slice_len = min(n, self.capacity - self._idx)
        second_slice_len = n - first_slice_len

        # First slice: from current _idx towards the end of pre-allocated array
        self.observations[self._idx : self._idx + first_slice_len] = obs_arr[:first_slice_len]
        self.actions[self._idx : self._idx + first_slice_len] = actions_arr[:first_slice_len]
        self.rewards[self._idx : self._idx + first_slice_len] = rewards_arr[:first_slice_len]
        self.next_observations[self._idx : self._idx + first_slice_len] = next_obs_arr[
            :first_slice_len
        ]
        self.dones[self._idx : self._idx + first_slice_len] = dones_arr[:first_slice_len]

        # Second slice (if wraparound occurs): from index 0
        if second_slice_len > 0:
            self.observations[:second_slice_len] = obs_arr[first_slice_len:]
            self.actions[:second_slice_len] = actions_arr[first_slice_len:]
            self.rewards[:second_slice_len] = rewards_arr[first_slice_len:]
            self.next_observations[:second_slice_len] = next_obs_arr[first_slice_len:]
            self.dones[:second_slice_len] = dones_arr[first_slice_len:]

        self._idx = (self._idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def add_transitions(self, batch: dict[str, Any]) -> int:
        """
        Store a trainer batch: a transition is terminal only if the episode really ended there.

        Uses the batch's ``terminated`` flags when present (the trainer sends them), else its
        ``dones``, so a cut at the time limit still bootstraps from the next state. A batch
        without ``next_observations`` or without either flag is not stored.

        Returns:
            The number of transitions stored (0 if the batch was not stored).

        Raises:
            ValueError: If ``terminated`` has a different length from ``actions``.
        """
        terminal = batch.get("terminated", batch.get("dones"))
        if "terminated" in batch and len(batch["terminated"]) != len(batch.get("actions", [])):
            raise ValueError(
                f"terminated has {len(batch['terminated'])} flags for "
                f"{len(batch.get('actions', []))} actions"
            )
        if "next_observations" not in batch or terminal is None:
            return 0
        self.add_batch(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            terminal,
        )
        return len(batch["actions"])

    def _actions_array(self, actions: Any, single: bool) -> npt.NDArray[Any]:
        """
        Actions as stored: int64 when discrete; float64 of shape (action_dim,) for one continuous
        action or (N, action_dim) for a batch, never reshaped to fit.

        Raises:
            ValueError: If a continuous action array has any other shape.
        """
        if self.action_dim is None:
            return np.asarray(actions, dtype=np.int64)
        arr = np.asarray(actions, dtype=np.float64)
        batch_length = arr.shape[0] if arr.ndim else 0
        expected = (self.action_dim,) if single else (batch_length, self.action_dim)
        if arr.shape != expected:
            kind = "action" if single else "actions"
            raise ValueError(
                f"Continuous {kind} must have shape {expected} (action_dim={self.action_dim}), "
                f"got {arr.shape}"
            )
        return arr

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a random batch of transitions uniformly without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys 'observations', 'actions', 'rewards',
            'next_observations', and 'dones'.

        Raises:
            ValueError: If batch_size <= 0 or batch_size > len(self).
        """
        self._check_sample_size(batch_size)
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return self._gather(indices)

    def _check_sample_size(self, batch_size: int) -> None:
        """Refuse a batch size that is not positive or exceeds the transitions stored."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} samples from buffer with size {self._size}"
            )

    def _gather(self, indices: npt.NDArray[np.int64]) -> dict[str, Any]:
        """The stored columns at ``indices``, keyed as :meth:`sample` returns them."""
        return {name: getattr(self, name)[indices] for name in _COLUMNS}

    def can_sample(self, batch_size: int) -> bool:
        """Return True if the buffer contains at least batch_size transitions."""
        return self._size >= batch_size

    KEY_PREFIX = "replay_"

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        """
        The buffer as checkpoint arrays under ``replay_`` keys: the filled part of each column
        (sampling draws indices below the size, so the rest is never read), the capacity, the
        next write index, the size and the sampling generator. The columns are views of the
        buffer, valid until the next write.
        """
        state = {f"{self.KEY_PREFIX}{name}": getattr(self, name)[: self._size] for name in _COLUMNS}
        state[f"{self.KEY_PREFIX}capacity"] = np.array(self.capacity)
        state[f"{self.KEY_PREFIX}idx"] = np.array(self._idx)
        state[f"{self.KEY_PREFIX}size"] = np.array(self._size)
        state[f"{self.KEY_PREFIX}rng"] = generator_state(self._rng)
        return state

    def restore(self, data: Mapping[str, Any]) -> None:
        """
        Replace the contents with a checkpoint's; a checkpoint without ``replay_`` keys empties
        the buffer.

        Raises:
            ValueError: Naming the key, if the saved capacity differs from this buffer's, the
                write index or size is inconsistent, or a column's shape or dtype does not fit.
                The buffer is then unchanged.
        """
        key = self.KEY_PREFIX
        if f"{key}size" not in data:
            self.clear()
            return
        names = ("capacity", "idx", "rng", *_COLUMNS)
        missing = [f"{key}{name}" for name in names if f"{key}{name}" not in data]
        if missing:
            raise ValueError(
                f"Checkpoint holds part of a replay buffer; missing {', '.join(missing)}"
            )
        idx, size = self._checked_position(data)
        columns = {name: np.asarray(data[f"{key}{name}"]) for name in _COLUMNS}
        for name, column in columns.items():
            current = getattr(self, name)
            if column.shape != (size, *current.shape[1:]) or column.dtype != current.dtype:
                raise ValueError(
                    f"{key}{name} has shape {column.shape} and dtype {column.dtype}, expected "
                    f"{(size, *current.shape[1:])} and {current.dtype}"
                )
        check_generator_state(self._rng, data[f"{key}rng"], f"{key}rng")
        restore_generator(self._rng, data[f"{key}rng"])
        for name, column in columns.items():
            getattr(self, name)[:size] = column
        self._idx, self._size = idx, size

    def _checked_position(self, data: Mapping[str, Any]) -> tuple[int, int]:
        """The saved write index and size, once they fit this buffer's capacity."""
        key = self.KEY_PREFIX
        for name in ("capacity", "idx", "size"):
            if np.asarray(data[f"{key}{name}"]).shape != ():
                raise ValueError(f"{key}{name} must be a single integer")
        capacity, idx, size = (int(data[f"{key}{n}"]) for n in ("capacity", "idx", "size"))
        if capacity != self.capacity:
            raise ValueError(
                f"{key}capacity is {capacity}, this buffer holds {self.capacity}: a wrapped "
                "buffer cannot move to another capacity"
            )
        if not 0 <= size <= capacity:
            raise ValueError(f"{key}size {size} is outside [0, {capacity}]")
        if not 0 <= idx < capacity or (size < capacity and idx != size):
            raise ValueError(f"{key}idx {idx} does not fit size {size} and capacity {capacity}")
        return idx, size

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size


def importance_weights(
    probs: npt.NDArray[np.float64], p_min: float, beta: float
) -> npt.NDArray[np.float64]:
    """
    Importance-sampling weights of samples drawn with probabilities ``probs``:
    ``(N·P(i))^-β`` over the buffer's largest such weight ``(N·P_min)^-β`` (Schaul et al. 2016,
    section 3.4, as OpenAI Baselines scales them). N cancels, leaving ``(P(i) / P_min)^-β``, so
    every weight is at most 1 and the scale is the same for every batch drawn from one buffer.
    """
    return np.asarray((probs / p_min) ** -beta, dtype=np.float64)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional prioritised replay (Schaul et al. 2016, section 3.3): the ring buffer of
    :class:`ReplayBuffer` with a priority per slot.

    A priority is stored already raised to ``alpha``: ``(|δ| + epsilon)^alpha`` once
    :meth:`update_priorities` has seen the transition's TD error δ, and before that the largest
    priority so far (1 in an empty buffer), so every new transition is likely to be replayed.
    :meth:`sample` draws one index from each of ``batch_size`` equal slices of the cumulative
    priority (stratified, with replacement), and weights each sample with
    :func:`importance_weights`. ``alpha`` 0 is uniform sampling with replacement.

    Checkpoints add ``replay_priorities`` and ``replay_max_priority``; a uniform buffer's
    checkpoint restores with every priority at 1, and a uniform buffer loading this one's drops
    them.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        seed: int | None = None,
        *,
        alpha: float,
        epsilon: float,
        action_dim: int | None = None,
    ) -> None:
        """
        Raises:
            ValueError: If ``alpha`` is outside [0, 1] or ``epsilon`` is not positive, besides
                :class:`ReplayBuffer`'s checks.
        """
        super().__init__(capacity, observation_dim, seed, action_dim=action_dim)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        if not epsilon > 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}")
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.priorities: npt.NDArray[np.float64] = np.zeros(self.capacity, dtype=np.float64)
        self.max_priority: float = 1.0

    def add(
        self,
        obs: npt.ArrayLike,
        action: int | npt.ArrayLike,
        reward: float,
        next_obs: npt.ArrayLike,
        done: bool,
    ) -> None:
        """Insert one transition at the largest priority so far (see :meth:`ReplayBuffer.add`)."""
        start = self._idx
        super().add(obs, action, reward, next_obs, done)
        self.priorities[start] = self.max_priority

    def add_batch(
        self,
        obs: npt.ArrayLike,
        actions: npt.ArrayLike,
        rewards: npt.ArrayLike,
        next_obs: npt.ArrayLike,
        dones: npt.ArrayLike,
    ) -> None:
        """Insert transitions at the largest priority so far (see :meth:`ReplayBuffer.add_batch`)."""
        start = self._idx
        super().add_batch(obs, actions, rewards, next_obs, dones)
        written = min(len(np.asarray(obs)), self.capacity)
        self.priorities[(start + np.arange(written)) % self.capacity] = self.max_priority

    def sample(self, batch_size: int, *, beta: float = 1.0) -> dict[str, Any]:
        """
        Draw ``batch_size`` transitions in proportion to their priorities, stratified.

        Args:
            batch_size: Number of transitions to draw (at most the number stored).
            beta: Strength of the importance-sampling correction in [0, 1]; 1 corrects fully.

        Returns:
            :class:`ReplayBuffer`'s columns plus ``indices`` (the slots drawn, for
            :meth:`update_priorities`) and ``weights`` (their importance weights, at most 1).

        Raises:
            ValueError: If ``beta`` is outside [0, 1], or as :meth:`ReplayBuffer.sample`.
        """
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta!r}")
        self._check_sample_size(batch_size)
        stored = self.priorities[: self._size]
        cumulative = np.cumsum(stored)
        total = cumulative[-1]
        points = (np.arange(batch_size) + self._rng.random(batch_size)) * (total / batch_size)
        indices = np.minimum(np.searchsorted(cumulative, points, side="right"), self._size - 1)
        batch = self._gather(indices)
        batch["indices"] = indices
        batch["weights"] = importance_weights(stored[indices] / total, stored.min() / total, beta)
        return batch

    def update_priorities(self, indices: npt.ArrayLike, td_errors: npt.ArrayLike) -> None:
        """Set the priorities of the slots ``indices`` to ``(|td_errors| + epsilon)^alpha``; a
        slot drawn twice takes the last of its (equal) errors."""
        priorities = (np.abs(np.asarray(td_errors, dtype=np.float64)) + self.epsilon) ** self.alpha
        self.priorities[np.asarray(indices)] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max(initial=0.0)))

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        """:meth:`ReplayBuffer.checkpoint_state` plus the stored priorities and the largest."""
        state = super().checkpoint_state()
        state[f"{self.KEY_PREFIX}priorities"] = self.priorities[: self._size]
        state[f"{self.KEY_PREFIX}max_priority"] = np.array(self.max_priority)
        return state

    def restore(self, data: Mapping[str, Any]) -> None:
        """
        :meth:`ReplayBuffer.restore`, with the priorities when the checkpoint has them and every
        priority at 1 when it comes from a uniform buffer.

        Raises:
            ValueError: Naming the key, if only one of the priority keys is present, or the
                priorities are not one positive float64 per stored transition; or as
                :meth:`ReplayBuffer.restore`. The buffer is then unchanged.
        """
        priorities, max_priority = self._checked_priorities(data)
        super().restore(data)
        if priorities is None:
            self.priorities[: self._size] = 1.0
            self.max_priority = 1.0
        else:
            self.priorities[: self._size] = priorities
            self.max_priority = max_priority

    def _checked_priorities(
        self, data: Mapping[str, Any]
    ) -> tuple[npt.NDArray[np.float64] | None, float]:
        """The saved priorities and their maximum, once they fit the saved size; (None, 1) for a
        checkpoint without them."""
        key = self.KEY_PREFIX
        names = (f"{key}priorities", f"{key}max_priority")
        present = [name in data for name in names]
        if not any(present):
            return None, 1.0
        if not all(present):
            missing = [name for name, there in zip(names, present, strict=True) if not there]
            raise ValueError(
                f"Checkpoint holds part of a prioritised replay buffer; missing {missing[0]}"
            )
        size = int(data[f"{key}size"]) if f"{key}size" in data else 0
        priorities = np.asarray(data[names[0]])
        if (
            priorities.shape != (size,)
            or priorities.dtype != np.float64
            or not np.all(priorities > 0.0)
        ):
            raise ValueError(
                f"{names[0]} must hold {size} positive float64 values, got shape "
                f"{priorities.shape} and dtype {priorities.dtype}"
            )
        max_priority = np.asarray(data[names[1]])
        if max_priority.shape != () or not float(max_priority) > 0.0:
            raise ValueError(f"{names[1]} must be a single positive number")
        return priorities, float(max_priority)

    def clear(self) -> None:
        """Empty the buffer and reset the largest priority to 1."""
        super().clear()
        self.max_priority = 1.0
