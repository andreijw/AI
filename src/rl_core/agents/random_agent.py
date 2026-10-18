"""Random agent for baseline testing."""

import json
from typing import Any

import numpy as np

from rl_core.seeding import AGENT_EVAL_STREAM, derive_rng

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Simple random agent that selects actions uniformly at random.

    Useful as a baseline and for testing the training pipeline.
    Action selection is reproducible when a seed is provided via config.
    """

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
        self._rng = np.random.default_rng(seed)
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

    def save(self, path: str) -> None:
        """Save random-agent metadata and RNG state to a NumPy ``.npz`` checkpoint."""
        save_path = self._resolve_save_path(path)
        state_json = json.dumps(self._rng.bit_generator.state)
        with open(save_path, "wb") as checkpoint_file:
            np.savez(
                checkpoint_file,
                observation_dim=np.int64(self.observation_dim),
                action_dim=np.int64(self.action_dim),
                rng_state=state_json,
                eval_rng_state=json.dumps(self._eval_rng.bit_generator.state),
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
            rng_state_text = np.asarray(rng_state_raw).item()
            if not isinstance(rng_state_text, str):
                raise ValueError(
                    "RandomAgent checkpoint rng_state must be stored as a JSON string."
                )
            rng_state = json.loads(rng_state_text)
            eval_state = (  # checkpoints from before 2026-10-18 have none
                json.loads(str(np.asarray(data["eval_rng_state"]).item()))
                if "eval_rng_state" in data
                else None
            )

        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = rng_state
        if eval_state is not None:
            self._eval_rng = np.random.default_rng()
            self._eval_rng.bit_generator.state = eval_state
