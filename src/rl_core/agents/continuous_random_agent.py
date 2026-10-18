"""Continuous random agent baseline implementation."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .base_agent import BaseAgent
from .bounds import action_bounds


class ContinuousRandomAgent(BaseAgent):
    """
    Random agent for continuous action spaces.

    Samples actions uniformly at random from [action_low, action_high].
    Useful as an unbiased baseline and for smoke-testing environments.
    """

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
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Sample a random action within [action_low, action_high].

        Args:
            observation: Current observation (ignored).
            training: Whether in training mode (ignored).

        Returns:
            Continuous action vector of shape (action_dim,).
        """
        return self._rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        No-op update for random agent.

        Args:
            batch: Trajectory data (ignored).

        Returns:
            Empty metrics dictionary.
        """
        return {}

    def save(self, path: str) -> None:
        """Save random-agent metadata and RNG state to a NumPy ``.npz`` checkpoint."""
        save_path = self._resolve_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        state_json = json.dumps(self._rng.bit_generator.state)
        with open(save_path, "wb") as f:
            np.savez_compressed(
                f,
                observation_dim=np.int64(self.observation_dim),
                action_dim=np.int64(self.action_dim),
                action_low=self.action_low,
                action_high=self.action_high,
                rng_state=state_json,
            )

    def load(self, path: str) -> None:
        """Load random-agent metadata and RNG state from a NumPy ``.npz`` checkpoint."""
        resolved_path = self._resolve_load_path(path)
        with np.load(resolved_path) as data:
            self.observation_dim = int(data["observation_dim"])
            self.action_dim = int(data["action_dim"])
            self.action_low = np.asarray(data["action_low"], dtype=np.float32)
            self.action_high = np.asarray(data["action_high"], dtype=np.float32)
            if "rng_state" in data:
                rng_state_raw = data["rng_state"]
                rng_state_text = np.asarray(rng_state_raw).item()
                if isinstance(rng_state_text, str):
                    self._rng = np.random.default_rng()
                    self._rng.bit_generator.state = json.loads(rng_state_text)
