"""Random agent for baseline testing."""

from typing import Any, Dict

import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Simple random agent that selects actions uniformly at random.

    Useful as a baseline and for testing the training pipeline.
    """

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select a random action.

        Args:
            observation: Current observation (ignored)
            training: Whether in training mode (ignored)

        Returns:
            Random action
        """
        return np.random.randint(0, self.action_dim)

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
