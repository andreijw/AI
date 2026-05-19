"""Base agent interface for reinforcement learning."""

import os
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

    # ------------------------------------------------------------------
    # Shared utility helpers
    # ------------------------------------------------------------------

    def _select_action_from_probs(self, probs: np.ndarray, training: bool) -> int:
        """
        Sample or greedily select an action from a probability distribution.

        Lazily initialises a per-agent RNG on the first call.

        Args:
            probs:    Action probability distribution.
            training: If True, sample stochastically; otherwise act greedily.

        Returns:
            Selected action index.
        """
        if not hasattr(self, "_rng"):
            seed = self.config.get("seed") if isinstance(self.config, dict) else None
            self._rng = np.random.default_rng(seed)
        if training:
            return int(self._rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    @staticmethod
    def _resolve_save_path(path: str) -> str:
        """
        Validate and resolve the checkpoint save path.

        Raises ``ValueError`` for unsupported extensions. Appends ``.npz``
        when *path* has no extension.

        Args:
            path: Requested destination path.

        Returns:
            Resolved path to write to.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )
        return path if extension else f"{path}.npz"

    @staticmethod
    def _resolve_load_path(path: str) -> str:
        """
        Validate and resolve the checkpoint load path.

        Raises ``ValueError`` for unsupported extensions and
        ``FileNotFoundError`` when the file cannot be located.

        Args:
            path: Requested source path.

        Returns:
            Resolved path to read from.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )
        has_exact_path = os.path.exists(path)
        if extension:
            # Backward-compatible fallback for older checkpoints written as "<name>.pt.npz".
            legacy_npz_path = f"{path}.npz"
            if has_exact_path or extension == ".npz":
                return path
            if os.path.exists(legacy_npz_path):
                return legacy_npz_path
            raise FileNotFoundError(
                f"Checkpoint not found at '{path}' (or legacy fallback '{legacy_npz_path}')."
            )
        # No extension: prefer the bare path if it exists, then try appending ".npz".
        if has_exact_path:
            return path
        npz_path = f"{path}.npz"
        if os.path.exists(npz_path):
            return npz_path
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}' (or fallback '{npz_path}')."
        )

    @staticmethod
    def _validate_dims(observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        """
        Validate common network dimension hyperparameters.

        Args:
            observation_dim: Observation space dimensionality.
            action_dim:      Number of discrete actions.
            hidden_dim:      Hidden layer width.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim!r}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

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

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experience.

        Args:
            batch: Dictionary containing trajectory data

        Returns:
            Dictionary of training metrics
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Args:
            path: Path to save the agent
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Args:
            path: Path to load the agent from
        """
