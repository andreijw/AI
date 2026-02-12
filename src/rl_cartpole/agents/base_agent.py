"""Base agent interface for reinforcement learning."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.

    Defines the interface that all agent implementations must follow.
    This allows for easy swapping of different RL algorithms.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize the agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            config: Configuration dictionary for the agent
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config

    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        Args:
            observation: Current observation from environment
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experience.

        Args:
            batch: Dictionary containing trajectory data

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Args:
            path: Path to save the agent
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Args:
            path: Path to load the agent from
        """
        pass
