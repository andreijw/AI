"""Continuous action space Actor-Critic base and agent implementations (Gaussian policy)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base_agent import BaseAgent
from .optim import Adam
from .returns import discounted_returns


class ContinuousActorCriticBase(BaseAgent):
    """
    Base class for continuous action space actor-critic agents (Gaussian policy).

    Uses a shared neural trunk with two output heads:
      1. Gaussian Actor Head: outputs state-dependent mean mu(s) and trainable log_std
      2. Critic Head: outputs state-value estimate V(s)

    Actions are sampled from N(mu, sigma^2) and clipped to [action_low, action_high].
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)

        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = int(config.get("hidden_dim", 128))
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.value_coef: float = float(config.get("value_coef", 0.5))
        self.entropy_coef: float = float(config.get("entropy_coef", 0.001))
        self.max_grad_norm: float = float(config.get("max_grad_norm", 0.5))

        # Action bounds
        raw_low = config.get("action_low", -1.0)
        raw_high = config.get("action_high", 1.0)
        self.action_low: np.ndarray = (
            np.full(action_dim, raw_low, dtype=np.float32)
            if np.isscalar(raw_low)
            else np.asarray(raw_low, dtype=np.float32)
        )
        self.action_high: np.ndarray = (
            np.full(action_dim, raw_high, dtype=np.float32)
            if np.isscalar(raw_high)
            else np.asarray(raw_high, dtype=np.float32)
        )

        self._validate_dims(observation_dim, action_dim, self.hidden_dim)

        # Initialize network parameters using He initialization
        self._rng = np.random.default_rng(config.get("seed"))
        scale1 = np.sqrt(2.0 / observation_dim)
        scale_head = np.sqrt(2.0 / self.hidden_dim)

        # Trunk: obs -> hidden
        self._W1: np.ndarray = (
            self._rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        )
        self._b1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        # Actor head: hidden -> mu
        self._W_mu: np.ndarray = (
            self._rng.standard_normal((self.hidden_dim, action_dim)) * scale_head
        )
        self._b_mu: np.ndarray = np.zeros(action_dim, dtype=np.float64)

        # Trainable log standard deviation: initialized to -0.5 (std ~ 0.6)
        init_log_std = float(config.get("init_log_std", -0.5))
        self._log_std: np.ndarray = np.full(action_dim, init_log_std, dtype=np.float64)

        # Critic head: hidden -> value
        self._W_v: np.ndarray = self._rng.standard_normal(self.hidden_dim) * scale_head
        self._b_v: float = 0.0

        # Adam with global gradient-norm clipping
        self._optimizer = Adam(
            {
                "W1": self._W1,
                "b1": self._b1,
                "W_mu": self._W_mu,
                "b_mu": self._b_mu,
                "log_std": self._log_std,
                "W_v": self._W_v,
                "b_v": self._b_v,
            },
            max_grad_norm=self.max_grad_norm,
        )

    @staticmethod
    def _validate_dims(obs_dim: int, act_dim: int, hidden_dim: int) -> None:
        if obs_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {obs_dim}")
        if act_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {act_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

    def _forward(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float | np.ndarray, np.ndarray]:
        """
        Forward pass through trunk and heads.

        Returns:
            mu: Action mean vector(s)
            log_std: Action log standard deviation
            v: State value estimate(s)
            h: Hidden layer activation(s) (post-ReLU)
        """
        s = np.asarray(state, dtype=np.float64)
        h = np.maximum(0.0, s @ self._W1 + self._b1)  # ReLU
        mu = h @ self._W_mu + self._b_mu
        log_std = np.clip(self._log_std, -20.0, 2.0)
        v = (h @ self._W_v + self._b_v).item() if s.ndim == 1 else h @ self._W_v + self._b_v
        return mu, log_std, v, h

    def select_action(
        self, state: np.ndarray, training: bool = True, evaluate: bool | None = None
    ) -> np.ndarray:
        """
        Select continuous action for a single environment state.

        Args:
            state: Observation vector of shape (observation_dim,).
            training: If True, samples stochastically; if False, acts deterministically.
            evaluate: Legacy alias; if True, acts deterministically. Overrides training if passed.

        Returns:
            Action vector of shape (action_dim,) clipped to [action_low, action_high].
        """
        mu, log_std, _, _ = self._forward(state)
        is_eval = evaluate if evaluate is not None else not training
        if is_eval:
            action = mu
        else:
            std = np.exp(log_std)
            noise = self._rng.standard_normal(self.action_dim)
            action = mu + std * noise
        return np.clip(action, self.action_low, self.action_high)

    def _log_prob(self, action: np.ndarray, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        """Compute Gaussian log-likelihood log N(action; mu, sigma^2)."""
        std = np.exp(log_std)
        variance = std**2
        log_2pi = np.log(2.0 * np.pi)
        residual = action - mu
        log_probs = -0.5 * ((residual**2) / (variance + 1e-8) + 2.0 * log_std + log_2pi)
        return np.sum(log_probs, axis=-1)

    def _entropy(self, log_std: np.ndarray) -> float:
        """Differential entropy of multivariate Gaussian with diagonal covariance."""
        log_2pi_e = np.log(2.0 * np.pi * np.e)
        return float(np.sum(log_std + 0.5 * log_2pi_e))

    def _adam_step(self, grads: dict[str, np.ndarray | float], lr: float) -> None:
        """Apply one step of Adam optimization with gradient clipping."""
        for k, step in self._optimizer.step(grads, lr).items():
            if k == "W1":
                self._W1 -= step
            elif k == "b1":
                self._b1 -= step
            elif k == "W_mu":
                self._W_mu -= step
            elif k == "b_mu":
                self._b_mu -= step
            elif k == "log_std":
                self._log_std -= step
            elif k == "W_v":
                self._W_v -= step
            elif k == "b_v":
                self._b_v -= float(step)

    def save(self, path: str) -> None:
        """
        Save network weights to disk in NumPy ``.npz`` format.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has one of
        these extensions, parameters are written exactly to that filename. If
        *path* has no extension, ``.npz`` is appended.
        """
        save_path = self._resolve_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            arrays: dict[str, Any] = {
                "W1": self._W1,
                "b1": self._b1,
                "W_mu": self._W_mu,
                "b_mu": self._b_mu,
                "log_std": self._log_std,
                "W_v": self._W_v,
                "b_v": np.array(self._b_v),
                **self._optimizer.state(),
            }
            np.savez_compressed(f, **arrays)

    def load(self, path: str) -> None:
        """
        Load network weights from disk.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has no
        extension, ``.npz`` is preferred.
        """
        resolved_path = self._resolve_load_path(path)
        with np.load(resolved_path) as data:
            self._W1 = data["W1"]
            self._b1 = data["b1"]
            self._W_mu = data["W_mu"]
            self._b_mu = data["b_mu"]
            self._log_std = data["log_std"]
            self._W_v = data["W_v"]
            self._b_v = float(data["b_v"])
            self._optimizer.load_state(data)


class ContinuousActorCriticAgent(ContinuousActorCriticBase):
    """
    Advantage Actor-Critic (A2C) agent for continuous control.

    Updates parameters using discounted returns and advantage estimates:
      A(s, a) = G_t - V(s)
      L_policy = -log_prob(a|s) * A(s, a) - entropy_coef * H
      L_critic = 0.5 * (G_t - V(s))^2
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))

    def update(self, trajectory: Any) -> dict[str, float]:
        """
        Update policy and value networks on an episode trajectory or batch.

        Args:
            trajectory: List of transition dicts or batch dictionary.

        Returns:
            Dictionary with training loss metrics.
        """
        if isinstance(trajectory, dict):
            if len(trajectory.get("observations", [])) == 0:
                return {}
            states = np.asarray(trajectory["observations"], dtype=np.float64)
            actions = np.asarray(trajectory["actions"], dtype=np.float64)
            rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
        elif isinstance(trajectory, list):
            if not trajectory:
                return {}
            states = np.array([t["state"] for t in trajectory], dtype=np.float64)
            actions = np.array([t["action"] for t in trajectory], dtype=np.float64)
            rewards = np.array([t["reward"] for t in trajectory], dtype=np.float64)
        else:
            return {}

        returns = discounted_returns(rewards, self.gamma)

        components, grads = self._loss_and_grads(states, actions, returns)

        self._adam_step(grads, self.learning_rate)

        return {
            "loss": components["total"],
            "policy_loss": components["policy"],
            "value_loss": components["value"],
            "entropy": components["entropy"],
        }

    def _loss_and_grads(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Compute the actor-critic loss and its gradients for one episode, changing no state.

        The loss is ``−mean(log π(a_t|s_t)·A_t) + value_coef·½·mean((V(s_t) − G_t)²)
        − entropy_coef·H``, where ``H`` is the entropy of the Gaussian policy summed over action
        dimensions. Returns and advantages are held constant when differentiating.

        Args:
            states:     Array of shape (T, observation_dim).
            actions:    Array of shape (T, action_dim).
            returns:    Array of shape (T,): the Monte-Carlo return G_t of each step.
            advantages: Array of shape (T,) to use as A_t, exactly as given: it is not
                        normalised again. If omitted, ``G_t − V(s_t)`` from the current weights,
                        normalised over the episode, is used, which lets :meth:`update` make a
                        single forward pass. Passing that same array explicitly gives identical
                        results.

        Returns:
            Tuple of (components, gradients). Components holds the ``policy``, ``value`` and
            ``entropy`` terms and their weighted ``total``. Gradients are those of ``total``,
            keyed ``W1``, ``b1``, ``W_mu``, ``b_mu``, ``log_std``, ``W_v``, ``b_v``.

        Raises:
            ValueError: If ``advantages`` is given with a different length from ``returns``.
        """
        n_steps = len(returns)
        if advantages is not None and len(advantages) != n_steps:
            raise ValueError(
                f"advantages has {len(advantages)} entries but the batch has {n_steps} steps"
            )

        # Forward pass over batch
        h = np.maximum(0.0, states @ self._W1 + self._b1)  # (n_steps, hidden_dim)
        mu = h @ self._W_mu + self._b_mu  # (n_steps, action_dim)
        log_std = np.clip(self._log_std, -20.0, 2.0)
        values = h @ self._W_v + self._b_v  # (n_steps,)

        if advantages is None:
            advantages = returns - values
            # Normalize advantages for training stability
            if n_steps > 1:
                adv_std = np.std(advantages)
                advantages = (advantages - np.mean(advantages)) / (adv_std + 1e-8)

        # Gaussian log probs
        variance = np.exp(2.0 * log_std)
        residual = actions - mu  # (n_steps, action_dim)

        # Policy gradients
        # d(log_prob)/d(mu) = residual / variance
        d_mu = (residual / (variance + 1e-8)) * advantages[:, np.newaxis]
        d_log_std = (((residual**2) / (variance + 1e-8)) - 1.0) * advantages[:, np.newaxis]

        # Negative log-likelihood objective: L_pi = -sum(log_prob * A)
        grad_w_mu = -(h.T @ d_mu) / n_steps
        grad_b_mu = -np.mean(d_mu, axis=0)
        grad_log_std = -np.mean(d_log_std, axis=0) - self.entropy_coef

        # Critic loss: 0.5 * (V - G)^2
        v_diff = values - returns  # (n_steps,)
        grad_w_v = (h.T @ v_diff) * self.value_coef / n_steps
        grad_b_v = float(np.mean(v_diff) * self.value_coef)

        # Backpropagation into hidden layer
        d_h_actor = -d_mu @ self._W_mu.T  # (n_steps, hidden_dim)
        d_h_critic = (v_diff[:, np.newaxis] * self.value_coef) @ self._W_v[np.newaxis, :]
        d_h = (d_h_actor + d_h_critic) * (h > 0) / n_steps  # ReLU gradient

        grad_w1 = states.T @ d_h
        grad_b1 = np.sum(d_h, axis=0)

        grads = {
            "W1": grad_w1,
            "b1": grad_b1,
            "W_mu": grad_w_mu,
            "b_mu": grad_b_mu,
            "log_std": grad_log_std,
            "W_v": grad_w_v,
            "b_v": grad_b_v,
        }

        policy_loss = float(-np.mean(self._log_prob(actions, mu, log_std) * advantages))
        value_loss = float(0.5 * np.mean(np.square(v_diff)))
        entropy = self._entropy(log_std)

        components = {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
            "total": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
        }
        return components, grads
