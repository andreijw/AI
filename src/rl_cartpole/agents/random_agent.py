"""Random agent for baseline testing."""

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
        """
        No-op save for random agent.

        Args:
            path: Path to save (ignored)
        """
        pass

    def load(self, path: str) -> None:
        """
        No-op load for random agent.

        Args:
            path: Path to load (ignored)
        """
        pass
