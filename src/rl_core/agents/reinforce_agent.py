"""REINFORCE (Monte Carlo Policy Gradient) agent for CartPole."""

from typing import Any

import numpy as np

from rl_core.seeding import AGENT_WEIGHTS_STREAM, derive_rng

from .base_agent import BaseAgent
from .optim import Adam, optimizer_from_config
from .returns import discounted_returns


class ReinforceAgent(BaseAgent):
    """
    REINFORCE agent implementing the Monte Carlo Policy Gradient algorithm.

    Uses a two-layer MLP policy network (obs → ReLU hidden → softmax output)
    trained entirely with NumPy.  After each episode the policy parameters are
    updated using the REINFORCE gradient estimate with a mean-return baseline to
    reduce variance.

    This is the natural first step beyond a random agent: the agent observes its
    own returns and adjusts action probabilities accordingly, without needing a
    value-function approximator or importance sampling.

    References:
        Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms
        for Connectionist Reinforcement Learning. Machine Learning, 8, 229–256.

    Config keys (all optional):
        learning_rate (float): Step size. Default: 1e-3.
        gamma        (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim   (int):   Width of the single hidden layer. Default: 128.
        seed         (int):   RNG seed for weight initialisation. Default: None.
        optimizer    (str):   "sgd" or "adam" (see :mod:`rl_core.agents.optim`). Default: "sgd".
        max_grad_norm (float): Global gradient-norm clip; requires optimizer "adam". Default: none.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
        """
        Initialise the REINFORCE agent.

        Args:
            observation_dim: Dimension of the observation vector.
            action_dim:      Number of discrete actions.
            config:          Hyperparameter dictionary (see class docstring).
        """
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 1e-3))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = int(config.get("hidden_dim", 128))

        # Validate hyperparameters early to avoid numerical issues later.
        self._validate_dims(observation_dim, action_dim, self.hidden_dim)
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in the interval (0, 1], got {self.gamma!r}")

        # He-initialised weights for ReLU activations
        rng = derive_rng(config.get("seed"), AGENT_WEIGHTS_STREAM)
        scale1 = np.sqrt(2.0 / observation_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)

        self._W1: np.ndarray = rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        self._b1: np.ndarray = np.zeros(self.hidden_dim)
        self._W2: np.ndarray = rng.standard_normal((self.hidden_dim, action_dim)) * scale2
        self._b2: np.ndarray = np.zeros(action_dim)

        self._optimizer: Adam | None = optimizer_from_config(
            config, {"W1": self._W1, "b1": self._b1, "W2": self._W2, "b2": self._b2}
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _policy(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the policy network.

        Args:
            obs: Single observation vector of shape (observation_dim,).

        Returns:
            Tuple of (action_probabilities, hidden_activations).
        """
        h = np.maximum(0.0, obs @ self._W1 + self._b1)  # ReLU
        logits = h @ self._W2 + self._b2
        # Numerically stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return probs, h

    def _loss_and_grads(
        self, observations: np.ndarray, actions: np.ndarray, return_weights: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """
        Compute the REINFORCE loss and its gradients for one episode, changing no state.

        The loss is ``Σ_t −return_weights[t] · log π(a_t | s_t)``, summed over the steps.
        ``return_weights`` are held constant when differentiating.

        Args:
            observations:   Array of shape (T, observation_dim).
            actions:        Integer array of shape (T,).
            return_weights: Array of shape (T,): the (standardised) return credited to each step.

        Returns:
            Tuple of (loss, gradients). Gradients are keyed ``W1``, ``b1``, ``W2``, ``b2`` and are
            those of the summed loss; :meth:`update` divides by the episode length.
        """
        grad_w1 = np.zeros_like(self._W1)
        grad_b1 = np.zeros_like(self._b1)
        grad_w2 = np.zeros_like(self._W2)
        grad_b2 = np.zeros_like(self._b2)

        total_loss = 0.0

        for obs, action, disc_return in zip(observations, actions, return_weights, strict=True):
            probs, h = self._policy(obs)

            # d/d_logits log π(a|s) = one_hot(a) - π(·|s)
            d_logits = -probs.copy()
            d_logits[action] += 1.0

            # Negate for gradient *descent* on the negative expected return
            d_logits *= -disc_return

            # Layer-2 gradients
            grad_w2 += np.outer(h, d_logits)
            grad_b2 += d_logits

            # Layer-1 gradients (through ReLU)
            d_h = d_logits @ self._W2.T
            d_h[h <= 0] = 0.0
            grad_w1 += np.outer(obs, d_h)
            grad_b1 += d_h

            total_loss += -np.log(np.clip(probs[action], 1e-8, 1.0)) * disc_return

        return float(total_loss), {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        During training the action is sampled from the policy distribution,
        preserving the stochasticity required for exploration.  During
        evaluation the greedy (highest-probability) action is returned.

        Args:
            observation: Current observation vector.
            training:    If True, sample stochastically; otherwise act greedily.

        Returns:
            Selected action index.
        """
        probs, _ = self._policy(observation)
        return self._select_action_from_probs(probs, training)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Update policy parameters using the REINFORCE gradient estimate.

        The update applies a mean-return baseline (subtract the episode mean)
        before scaling gradients, which reduces variance without introducing
        bias.

        Args:
            batch: Dictionary with keys:
                   - "observations": array of shape (T, observation_dim)
                   - "actions":      integer array of shape (T,)
                   - "rewards":      float array of shape (T,)

        Returns:
            Dictionary with key "policy_loss" (mean policy-gradient loss over episode).
        """
        observations: np.ndarray = np.asarray(batch["observations"], dtype=np.float64)
        actions: np.ndarray = np.asarray(batch["actions"], dtype=int)
        rewards: np.ndarray = np.asarray(batch["rewards"], dtype=np.float64)

        n_steps = len(rewards)
        if n_steps == 0:
            return {"policy_loss": 0.0}

        # Discounted returns with mean baseline (only normalise when std > ε to
        # avoid producing all-zeros or NaN when every return is identical)
        returns = discounted_returns(rewards, self.gamma)
        std = returns.std()
        returns = (returns - returns.mean()) / std if std > 1e-8 else returns - returns.mean()

        if not (len(observations) == len(actions) == len(returns)):
            raise ValueError(
                "Inconsistent trajectory lengths: "
                f"observations={len(observations)}, "
                f"actions={len(actions)}, "
                f"returns={len(returns)}"
            )

        total_loss, grads = self._loss_and_grads(observations, actions, returns)

        if self._optimizer is None:
            # Gradient step (averaged over episode length)
            self._W1 -= self.learning_rate * grads["W1"] / n_steps
            self._b1 -= self.learning_rate * grads["b1"] / n_steps
            self._W2 -= self.learning_rate * grads["W2"] / n_steps
            self._b2 -= self.learning_rate * grads["b2"] / n_steps
        else:
            mean = {name: gradient / n_steps for name, gradient in grads.items()}
            steps = self._optimizer.step(mean, self.learning_rate)
            self._W1 -= steps["W1"]
            self._b1 -= steps["b1"]
            self._W2 -= steps["W2"]
            self._b2 -= steps["b2"]

        return {"policy_loss": float(total_loss / n_steps)}

    def save(self, path: str) -> None:
        """
        Save policy network weights to disk in NumPy ``.npz`` format.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has one of
        these extensions, parameters are written exactly to that filename. If
        *path* has no extension, ``.npz`` is appended.

        Args:
            path: Destination file path.
        """
        save_path = self._resolve_save_path(path)
        with open(save_path, "wb") as f:
            arrays: dict[str, Any] = {
                "W1": self._W1,
                "b1": self._b1,
                "W2": self._W2,
                "b2": self._b2,
            }
            if self._optimizer is not None:
                arrays.update(self._optimizer.state())
            np.savez(f, **arrays)

    def load(self, path: str) -> None:
        """
        Load policy network weights from disk.

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
            self._W2 = data["W2"]
            self._b2 = data["b2"]
            if self._optimizer is not None:
                self._optimizer.load_state(data)
