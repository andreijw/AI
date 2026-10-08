"""Experience replay buffer for off-policy reinforcement learning."""

from typing import Any

import numpy as np
import numpy.typing as npt


class ReplayBuffer:
    """
    Fixed-capacity FIFO ring buffer storing transition tuples for off-policy RL.

    Stores transitions in pre-allocated NumPy arrays to ensure O(1) insertions
    and efficient uniform batch sampling without python list overhead.

    Transitions are represented as:
        (observation, action, reward, next_observation, done)
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        seed: int | None = None,
    ) -> None:
        """
        Initialise the ReplayBuffer.

        Args:
            capacity: Maximum number of transitions to store.
            observation_dim: Dimensionality of observation vectors.
            seed: Optional RNG seed for reproducible random sampling.

        Raises:
            ValueError: If capacity or observation_dim is non-positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")

        self.capacity: int = int(capacity)
        self.observation_dim: int = int(observation_dim)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.observations: npt.NDArray[np.float64] = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float64
        )
        self.actions: npt.NDArray[np.int64] = np.zeros(self.capacity, dtype=np.int64)
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
        action: int,
        reward: float,
        next_obs: npt.ArrayLike,
        done: bool,
    ) -> None:
        """
        Insert a single transition into the ring buffer.

        Overwrites the oldest transition when the buffer reaches capacity.

        Args:
            obs: Observation vector before transition.
            action: Discrete action taken.
            reward: Scalar reward received.
            next_obs: Observation vector after transition.
            done: Whether the episode terminated/truncated at this step.
        """
        self.observations[self._idx] = np.asarray(obs, dtype=np.float64)
        self.actions[self._idx] = int(action)
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
            actions: Array of discrete actions of shape (N,).
            rewards: Array of scalar rewards of shape (N,).
            next_obs: Array of next observations of shape (N, observation_dim).
            dones: Array of boolean termination flags of shape (N,).

        Raises:
            ValueError: If input array lengths do not match.
        """
        obs_arr = np.asarray(obs, dtype=np.float64)
        actions_arr = np.asarray(actions, dtype=np.int64)
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
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} samples from buffer with size {self._size}"
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def can_sample(self, batch_size: int) -> bool:
        """Return True if the buffer contains at least batch_size transitions."""
        return self._size >= batch_size

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size
