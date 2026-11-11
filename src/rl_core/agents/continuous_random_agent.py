"""Continuous random agent baseline implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from rl_core.seeding import AGENT_EVAL_STREAM, AGENT_EXPLORATION_STREAM, derive_rng

from .base_agent import BaseAgent
from .bounds import action_bounds


class ContinuousRandomAgent(BaseAgent):
    """
    Random agent for continuous action spaces.

    Samples actions uniformly at random from [action_low, action_high].
    Useful as an unbiased baseline and for smoke-testing environments.
    """

    # Checkpoints from before 2026-10-21 stored the generators as "rng_state" and "eval_rng_state".
    _LEGACY_GENERATOR_KEYS: ClassVar[dict[str, str]] = {
        "_rng": "rng_state",
        "_eval_rng": "eval_rng_state",
    }

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        """
        Initialize the continuous random agent.

        Args:
            observation_dim: Dimension of observation space.
            action_dim: Dimension of continuous action space.
            config: Configuration dictionary. Supported keys:
                - ``seed`` (int | None): Random seed.
                - ``action_low`` (float | list[float]): Lower action bound (default: -1.0).
                - ``action_high`` (float | list[float]): Upper action bound (default: 1.0).
        """
        super().__init__(observation_dim, action_dim, config)
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim!r}")

        self.action_low, self.action_high = action_bounds(config, action_dim)

        seed: int | None = config.get("seed")
        # Training actions on the exploration stream, apart from the environment's numbers.
        self._rng = derive_rng(seed, AGENT_EXPLORATION_STREAM)
        # Evaluation actions draw from their own stream, so evaluating does not shift training actions.
        self._eval_rng = derive_rng(seed, AGENT_EVAL_STREAM)

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Sample a random action within [action_low, action_high].

        Args:
            observation: Current observation (ignored).
            training: Whether in training mode; evaluation draws from a separate generator.

        Returns:
            Continuous action vector of shape (action_dim,).
        """
        rng = self._rng if training else self._eval_rng
        return rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def enable_mid_episode_updates(self, update_every: int) -> None:
        """Accepted: it learns nothing, so a benchmark floor can share a config with update_every."""

    def enable_parallel_envs(self, num_envs: int, update_every: int | None = None) -> None:
        """Accepted: it learns nothing, so a benchmark floor can share a config with num_envs."""

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        No-op update for random agent.

        Args:
            batch: Trajectory data (ignored).

        Returns:
            Empty metrics dictionary.
        """
        return {}

    def _checkpoint_arrays(self) -> dict[str, Any]:
        return {
            "observation_dim": np.int64(self.observation_dim),
            "action_dim": np.int64(self.action_dim),
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

    def _check_checkpoint_arrays(self, data: Mapping[str, Any], path: str) -> None:
        """
        Raises:
            ValueError: Naming the entry, unless the checkpoint was saved for this agent's spaces:
                the same dimensions and action bounds. Taking over other ones would act outside
                this environment's bounds.
        """
        super()._check_checkpoint_arrays(data, path)
        for key, own in self._checkpoint_arrays().items():
            if not np.array_equal(data[key], own):
                raise ValueError(
                    f"{path}: {key} is {np.asarray(data[key]).tolist()} in the checkpoint, this "
                    f"agent's is {np.asarray(own).tolist()}: it was saved for other observation "
                    "or action spaces"
                )

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """Nothing to assign: :meth:`_check_checkpoint_arrays` found every entry equal to this
        agent's (only the generators, which :meth:`load` restores, carry state)."""
