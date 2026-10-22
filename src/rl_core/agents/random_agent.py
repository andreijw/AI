"""Random agent for baseline testing."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from rl_core.seeding import AGENT_EVAL_STREAM, AGENT_EXPLORATION_STREAM, derive_rng

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Simple random agent that selects actions uniformly at random.

    Useful as a baseline and for testing the training pipeline.
    Action selection is reproducible when a seed is provided via config.
    """

    # Checkpoints from before 2026-10-21 stored the generators as "rng_state" and "eval_rng_state".
    _LEGACY_GENERATOR_KEYS: ClassVar[dict[str, str]] = {
        "_rng": "rng_state",
        "_eval_rng": "eval_rng_state",
    }

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
        """
        Initialize the random agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            config: Configuration dictionary. Supports optional key:
                - ``seed`` (int | None): Random seed for reproducible action selection.
        """
        super().__init__(observation_dim, action_dim, config)
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")

        seed: int | None = config.get("seed")
        # Training actions on the exploration stream, apart from the environment's numbers.
        self._rng = derive_rng(seed, AGENT_EXPLORATION_STREAM)
        # Evaluation actions draw from their own stream, so evaluating does not shift training actions.
        self._eval_rng = derive_rng(seed, AGENT_EVAL_STREAM)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select a random action.

        Args:
            observation: Current observation (ignored)
            training: Whether in training mode; evaluation draws from a separate generator

        Returns:
            Random action
        """
        rng = self._rng if training else self._eval_rng
        return int(rng.integers(0, self.action_dim))

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        No-op update for random agent.

        Args:
            batch: Trajectory data (ignored)

        Returns:
            Empty metrics dictionary
        """
        return {}

    def _checkpoint_arrays(self) -> dict[str, Any]:
        return {
            "observation_dim": np.int64(self.observation_dim),
            "action_dim": np.int64(self.action_dim),
        }

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        saved_observation_dim = int(data["observation_dim"])
        saved_action_dim = int(data["action_dim"])
        if saved_observation_dim != self.observation_dim or saved_action_dim != self.action_dim:
            raise ValueError(
                "Checkpoint dimensions do not match this agent instance: "
                f"checkpoint(observation_dim={saved_observation_dim}, "
                f"action_dim={saved_action_dim}) vs "
                f"agent(observation_dim={self.observation_dim}, action_dim={self.action_dim})."
            )
