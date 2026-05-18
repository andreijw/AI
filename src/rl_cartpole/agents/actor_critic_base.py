"""Shared base class for actor-critic style agents."""

import os
from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from .base_agent import BaseAgent


class ActorCriticBase(BaseAgent):
    """
    Shared base for actor-critic style agents (A2C / PPO) with a two-headed MLP.

    Provides a common shared-trunk / dual-head network architecture, forward
    pass, discounted-return computation, stochastic/greedy action selection,
    and NumPy-npz checkpoint save/load.

    Subclasses must implement :meth:`update` and must set ``self.learning_rate``
    (with their own default) before finishing ``__init__``.

    Config keys shared by all subclasses (all optional):
        gamma        (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim   (int):   Shared hidden layer width. Default: 128.
        value_coef   (float): Critic-loss weight. Default: 0.5.
        entropy_coef (float): Entropy-bonus weight. Default: 0.01.
        seed         (int):   RNG seed for weight initialisation. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(observation_dim, action_dim, config)

        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = int(config.get("hidden_dim", 128))
        self.value_coef: float = float(config.get("value_coef", 0.5))
        self.entropy_coef: float = float(config.get("entropy_coef", 0.01))

        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {self.hidden_dim!r}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in the interval (0, 1], got {self.gamma!r}")
        if self.value_coef < 0.0:
            raise ValueError(f"value_coef must be non-negative, got {self.value_coef!r}")
        if self.entropy_coef < 0.0:
            raise ValueError(f"entropy_coef must be non-negative, got {self.entropy_coef!r}")

        # He-initialised weights for ReLU activations.
        rng = np.random.default_rng(config.get("seed"))
        scale1 = np.sqrt(2.0 / observation_dim)
        scale_head = np.sqrt(2.0 / self.hidden_dim)

        # Shared trunk: obs → hidden
        self._W1: np.ndarray = rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        self._b1: np.ndarray = np.zeros(self.hidden_dim)

        # Policy head (actor): hidden → action_dim (softmax)
        self._W_pi: np.ndarray = rng.standard_normal((self.hidden_dim, action_dim)) * scale_head
        self._b_pi: np.ndarray = np.zeros(action_dim)

        # Value head (critic): hidden → scalar
        self._W_v: np.ndarray = rng.standard_normal(self.hidden_dim) * scale_head
        self._b_v: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass through the shared trunk and both heads.

        Args:
            obs: Single observation vector of shape (observation_dim,).

        Returns:
            Tuple of (action_probabilities, state_value, hidden_activations).
        """
        h = np.maximum(0.0, obs @ self._W1 + self._b1)  # ReLU hidden layer

        # Policy head
        logits = h @ self._W_pi + self._b_pi
        logits = logits - np.max(logits)  # numerically stable softmax
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Value head (scalar)
        value = float(h @ self._W_v) + self._b_v

        return probs, value, h

    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns G_t = Σ_{k≥t} γ^(k-t) r_k for each step t.

        Args:
            rewards: 1-D array of per-step rewards for one episode.

        Returns:
            1-D array of discounted returns, same length as *rewards*.
        """
        n_steps = len(rewards)
        returns = np.empty(n_steps, dtype=np.float64)
        cumulative = 0.0
        for t in range(n_steps - 1, -1, -1):
            cumulative = float(rewards[t]) + self.gamma * cumulative
            returns[t] = cumulative
        return returns

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        During training the action is sampled from the policy distribution.
        During evaluation the greedy (highest-probability) action is returned.

        Args:
            observation: Current observation vector.
            training:    If True, sample stochastically; otherwise act greedily.

        Returns:
            Selected action index.
        """
        probs, _, _ = self._forward(observation)

        # Lazily initialize a per-agent RNG to avoid using NumPy's global RNG.
        if not hasattr(self, "_rng"):
            seed = None
            if isinstance(self.config, dict):
                seed = self.config.get("seed")
            self._rng = np.random.default_rng(seed)

        if training:
            return int(self._rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent parameters using collected trajectory data."""

    def save(self, path: str) -> None:
        """
        Save network weights to disk in NumPy ``.npz`` format.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has one of
        these extensions, parameters are written exactly to that filename. If
        *path* has no extension, ``.npz`` is appended.

        Args:
            path: Destination file path.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )
        save_path = path if extension else f"{path}.npz"
        with open(save_path, "wb") as f:
            np.savez(
                f,
                W1=self._W1,
                b1=self._b1,
                W_pi=self._W_pi,
                b_pi=self._b_pi,
                W_v=self._W_v,
                b_v=np.array(self._b_v),
            )

    def load(self, path: str) -> None:
        """
        Load network weights from disk.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has no
        extension and does not exist, ``.npz`` is appended and that file is
        loaded. Loading a ``.pt`` path also supports legacy
        ``<name>.pt.npz`` checkpoints.

        Args:
            path: Source file path.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )

        has_exact_path = os.path.exists(path)
        if extension:
            legacy_npz_path = f"{path}.npz"
            if has_exact_path or extension == ".npz":
                load_path = path
            elif os.path.exists(legacy_npz_path):
                load_path = legacy_npz_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at '{path}' (or legacy fallback '{legacy_npz_path}')."
                )
        else:
            load_path = path if has_exact_path else f"{path}.npz"

        with np.load(load_path) as data:
            self._W1 = data["W1"]
            self._b1 = data["b1"]
            self._W_pi = data["W_pi"]
            self._b_pi = data["b_pi"]
            self._W_v = data["W_v"]
            self._b_v = float(data["b_v"])
