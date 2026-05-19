"""Shared base class for actor-critic style agents."""

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

        self._validate_dims(observation_dim, action_dim, self.hidden_dim)
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

    def _parse_trajectory(
        self, batch: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Extract and validate a trajectory batch into typed arrays.

        Raises ``ValueError`` if the lengths of observations, actions, and
        rewards do not all match.

        Args:
            batch: Dictionary with keys "observations", "actions", "rewards".

        Returns:
            Tuple of (observations, actions, rewards, n_steps).
        """
        observations = np.asarray(batch["observations"], dtype=np.float64)
        actions = np.asarray(batch["actions"], dtype=int)
        rewards = np.asarray(batch["rewards"], dtype=np.float64)
        n_steps = len(rewards)
        if not (len(observations) == len(actions) == n_steps):
            raise ValueError(
                "Inconsistent trajectory lengths: "
                f"observations={len(observations)}, "
                f"actions={len(actions)}, "
                f"rewards={n_steps}"
            )
        return observations, actions, rewards, n_steps

    def _zero_ac_gradients(self) -> Dict[str, Any]:
        """
        Return a zeroed gradient dictionary matching the network parameters.

        Returns:
            Dict with keys ``W1``, ``b1``, ``W_pi``, ``b_pi``, ``W_v``, ``b_v``.
        """
        return {
            "W1": np.zeros_like(self._W1),
            "b1": np.zeros_like(self._b1),
            "W_pi": np.zeros_like(self._W_pi),
            "b_pi": np.zeros_like(self._b_pi),
            "W_v": np.zeros_like(self._W_v),
            "b_v": 0.0,
        }

    def _accumulate_ac_gradients(
        self,
        grads: Dict[str, Any],
        obs: np.ndarray,
        h: np.ndarray,
        d_pi_logits: np.ndarray,
        d_v_out: float,
    ) -> None:
        """
        Accumulate actor-critic gradients for one time-step in-place.

        Args:
            grads:       Gradient accumulator dict (modified in-place).
            obs:         Observation vector for this step.
            h:           Hidden activations from the forward pass.
            d_pi_logits: Gradient w.r.t. policy logits.
            d_v_out:     Gradient w.r.t. value head output.
        """
        grads["W_pi"] += np.outer(h, d_pi_logits)
        grads["b_pi"] += d_pi_logits
        grads["W_v"] += d_v_out * h
        grads["b_v"] += d_v_out

        # Backpropagate through the shared trunk
        d_h = self._W_pi @ d_pi_logits + d_v_out * self._W_v
        d_pre_h = d_h * (h > 0)  # ReLU derivative

        grads["W1"] += np.outer(obs, d_pre_h)
        grads["b1"] += d_pre_h

    def _apply_ac_gradients(self, grads: Dict[str, Any], scale: float) -> None:
        """
        Apply accumulated gradients to the network parameters.

        Args:
            grads: Gradient accumulator dict (from :meth:`_zero_ac_gradients`).
            scale: Multiplier applied to each gradient before subtraction
                   (typically ``learning_rate / n_steps``).
        """
        self._W1 -= scale * grads["W1"]
        self._b1 -= scale * grads["b1"]
        self._W_pi -= scale * grads["W_pi"]
        self._b_pi -= scale * grads["b_pi"]
        self._W_v -= scale * grads["W_v"]
        self._b_v -= scale * grads["b_v"]

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
        return self._select_action_from_probs(probs, training)

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
        save_path = self._resolve_save_path(path)
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
        load_path = self._resolve_load_path(path)
        with np.load(load_path) as data:
            self._W1 = data["W1"]
            self._b1 = data["b1"]
            self._W_pi = data["W_pi"]
            self._b_pi = data["b_pi"]
            self._W_v = data["W_v"]
            self._b_v = float(data["b_v"])
