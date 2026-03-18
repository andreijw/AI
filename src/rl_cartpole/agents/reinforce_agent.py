"""REINFORCE (Monte Carlo Policy Gradient) agent for CartPole."""

from typing import Any, Dict, Tuple

import numpy as np

from .base_agent import BaseAgent


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
        learning_rate (float): Step size for SGD updates. Default: 1e-3.
        gamma        (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim   (int):   Width of the single hidden layer. Default: 128.
        seed         (int):   RNG seed for weight initialisation. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
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

        # He-initialised weights for ReLU activations
        rng = np.random.default_rng(config.get("seed"))
        scale1 = np.sqrt(2.0 / observation_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)

        self._W1: np.ndarray = rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        self._b1: np.ndarray = np.zeros(self.hidden_dim)
        self._W2: np.ndarray = rng.standard_normal((self.hidden_dim, action_dim)) * scale2
        self._b2: np.ndarray = np.zeros(action_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _policy(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        # Lazily initialize a per-agent RNG to avoid using NumPy's global RNG.
        if not hasattr(self, "_rng"):
            seed = getattr(self, "seed", None)
            if seed is None:
                config = getattr(self, "config", None)
                if isinstance(config, dict):
                    seed = config.get("seed")
            self._rng = np.random.default_rng(seed)

        if training:
            return int(self._rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
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
        returns = self._compute_returns(rewards)
        std = returns.std()
        returns = (
            (returns - returns.mean()) / std if std > 1e-8 else returns - returns.mean()
        )

        # Accumulate REINFORCE gradients over the episode
        grad_w1 = np.zeros_like(self._W1)
        grad_b1 = np.zeros_like(self._b1)
        grad_w2 = np.zeros_like(self._W2)
        grad_b2 = np.zeros_like(self._b2)

        total_loss = 0.0

        for obs, action, disc_return in zip(observations, actions, returns):
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

        # Gradient step (averaged over episode length)
        self._W1 -= self.learning_rate * grad_w1 / n_steps
        self._b1 -= self.learning_rate * grad_b1 / n_steps
        self._W2 -= self.learning_rate * grad_w2 / n_steps
        self._b2 -= self.learning_rate * grad_b2 / n_steps

        return {"policy_loss": float(total_loss / n_steps)}

    def save(self, path: str) -> None:
        """
        Save policy network weights to disk in NumPy .npz format.

        If *path* does not end with ``.npz``, the extension is appended
        explicitly, so the file on disk will be ``<path>.npz``.

        Args:
            path: Destination file path (with or without ``.npz`` extension).
        """
        save_path = path if path.endswith(".npz") else f"{path}.npz"
        np.savez(
            save_path,
            W1=self._W1,
            b1=self._b1,
            W2=self._W2,
            b2=self._b2,
        )

    def load(self, path: str) -> None:
        """
        Load policy network weights from disk.

        Handles the ``.npz`` extension automatically whether or not it was
        included in *path*.

        Args:
            path: Source file path (with or without ``.npz`` extension).

        Raises:
            FileNotFoundError: If neither *path* nor ``<path>.npz`` exists.
        """
        load_path = path if path.endswith(".npz") else f"{path}.npz"
        data = np.load(load_path)
        self._W1 = data["W1"]
        self._b1 = data["b1"]
        self._W2 = data["W2"]
        self._b2 = data["b2"]
