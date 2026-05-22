"""Random agent for baseline testing."""

import json
from typing import Any, Dict, Optional

import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Simple random agent that selects actions uniformly at random.

    Useful as a baseline and for testing the training pipeline.
    Action selection is reproducible when a seed is provided via config.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
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

        seed: Optional[int] = config.get("seed")
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select a random action.

        Args:
            observation: Current observation (ignored)
            training: Whether in training mode (ignored)

        Returns:
            Random action
        """
        return int(self._rng.integers(0, self.action_dim))

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        No-op update for random agent.

        Args:
            batch: Trajectory data (ignored)

        Returns:
            Empty metrics dictionary
        """
        return {}

    def save(self, path: str) -> None:
        """Save random-agent metadata and RNG state to a NumPy ``.npz`` checkpoint."""
        save_path = self._resolve_save_path(path)
        state_json = json.dumps(self._rng.bit_generator.state)
        np.savez(
            save_path,
            observation_dim=np.int64(self.observation_dim),
            action_dim=np.int64(self.action_dim),
            rng_state=state_json,
        )

    def load(self, path: str) -> None:
        """Load random-agent metadata and RNG state from a NumPy ``.npz`` checkpoint."""
        load_path = self._resolve_load_path(path)
        with np.load(load_path) as data:
            saved_observation_dim = int(data["observation_dim"])
            saved_action_dim = int(data["action_dim"])
            if saved_observation_dim != self.observation_dim or saved_action_dim != self.action_dim:
                raise ValueError(
                    "Checkpoint dimensions do not match this agent instance: "
                    f"checkpoint(observation_dim={saved_observation_dim}, "
                    f"action_dim={saved_action_dim}) vs "
                    f"agent(observation_dim={self.observation_dim}, action_dim={self.action_dim})."
                )
            rng_state_raw = data["rng_state"]
            rng_state = json.loads(str(rng_state_raw.item() if hasattr(rng_state_raw, "item") else rng_state_raw))

        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = rng_state
