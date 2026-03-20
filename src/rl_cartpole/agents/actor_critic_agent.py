"""Actor-Critic (A2C) agent for CartPole."""

import os
from typing import Any, Dict, Tuple

import numpy as np

from .base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic agent (A2C style) for CartPole.

    Uses a two-headed MLP with a shared trunk: the *actor* head outputs a
    softmax policy over discrete actions while the *critic* head estimates
    the state-value function V(s).  Policy parameters are updated with a
    REINFORCE-style gradient scaled by the advantage

        A_t = G_t - V(s_t)

    where G_t is the discounted Monte-Carlo return and V(s_t) is the
    current critic estimate.  The critic is trained via mean-squared-error
    regression towards G_t.  An optional entropy bonus encourages exploration.

    Compared with the plain REINFORCE agent this approach:

    * Replaces the scalar mean-return baseline with a *learned*, state-
      dependent baseline, substantially reducing gradient variance.
    * Trains a value function that can be reused for downstream extensions
      (e.g. GAE, PPO).

    References:
        Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement
        Learning. ICML 2016.

    Config keys (all optional):
        learning_rate (float): SGD step size. Default: 1e-3.
        gamma         (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim    (int):   Shared hidden layer width. Default: 128.
        value_coef    (float): Critic-loss weight. Default: 0.5.
        entropy_coef  (float): Entropy-bonus weight (encourages exploration).
                               Default: 0.01.
        seed          (int):   RNG seed for weight initialisation. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialise the Actor-Critic agent.

        Args:
            observation_dim: Dimension of the observation vector.
            action_dim:      Number of discrete actions.
            config:          Hyperparameter dictionary (see class docstring).
        """
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 1e-3))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = int(config.get("hidden_dim", 128))
        self.value_coef: float = float(config.get("value_coef", 0.5))
        self.entropy_coef: float = float(config.get("entropy_coef", 0.01))

        # Validate hyperparameters early to avoid numerical issues later.
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {self.hidden_dim!r}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
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
            config = getattr(self, "config", None)
            if isinstance(config, dict):
                seed = config.get("seed")
            self._rng = np.random.default_rng(seed)

        if training:
            return int(self._rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update actor and critic parameters using the collected trajectory.

        The actor is updated with advantage-weighted policy gradients; the
        critic is updated via MSE regression towards Monte-Carlo returns.
        An optional entropy bonus (weighted by ``entropy_coef``) is added to
        encourage exploration by penalising overly-confident policies.

        Args:
            batch: Dictionary with keys:
                   - "observations": array of shape (T, observation_dim)
                   - "actions":      integer array of shape (T,)
                   - "rewards":      float array of shape (T,)

        Returns:
            Dictionary with keys:
            - "policy_loss": mean actor loss over the episode.
            - "value_loss":  mean critic loss over the episode.
        """
        observations: np.ndarray = np.asarray(batch["observations"], dtype=np.float64)
        actions: np.ndarray = np.asarray(batch["actions"], dtype=int)
        rewards: np.ndarray = np.asarray(batch["rewards"], dtype=np.float64)

        n_steps = len(rewards)
        if n_steps == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        if not (len(observations) == len(actions) == n_steps):
            raise ValueError(
                "Inconsistent trajectory lengths: "
                f"observations={len(observations)}, "
                f"actions={len(actions)}, "
                f"rewards={n_steps}"
            )

        # Monte-Carlo returns (same as REINFORCE, no normalisation here so the
        # value function retains scale information)
        returns = self._compute_returns(rewards)

        # Gradient accumulators
        grad_w1 = np.zeros_like(self._W1)
        grad_b1 = np.zeros_like(self._b1)
        grad_w_pi = np.zeros_like(self._W_pi)
        grad_b_pi = np.zeros_like(self._b_pi)
        grad_w_v = np.zeros_like(self._W_v)
        grad_b_v = 0.0

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for obs, action, g in zip(observations, actions, returns):
            probs, value, h = self._forward(obs)

            # Advantage estimate: how much better this action was than expected
            advantage = g - value

            # Actor loss: -log π(a|s) * advantage
            log_prob_a = np.log(np.clip(probs[action], 1e-8, 1.0))
            total_policy_loss += -log_prob_a * advantage

            # Critic loss: 0.5 * (V(s) - G)^2
            total_value_loss += 0.5 * (value - g) ** 2

            # Entropy: H = -Σ_a π(a) log π(a)
            log_probs = np.log(np.clip(probs, 1e-8, 1.0))
            entropy = -float(np.sum(probs * log_probs))

            # Gradient of total loss w.r.t. policy logits (gradient-descent direction):
            #   d(policy_loss)/d(z_j) = advantage * (π(j) - 1[j==a])
            #   d(-entropy_coef * H)/d(z_j) = entropy_coef * π(j) * (log π(j) + H)
            d_pi_logits = advantage * probs.copy()
            d_pi_logits[action] -= advantage
            d_pi_logits += self.entropy_coef * probs * (log_probs + entropy)

            # Gradient of total loss w.r.t. critic output:
            #   d(value_coef * 0.5*(V-G)^2)/d(V) = value_coef * (V - G)
            d_v_out = self.value_coef * (value - g)

            # Policy-head parameter gradients
            grad_w_pi += np.outer(h, d_pi_logits)
            grad_b_pi += d_pi_logits

            # Value-head parameter gradients
            grad_w_v += d_v_out * h
            grad_b_v += d_v_out

            # Backpropagate to shared trunk
            # z = h @ W_pi + b_pi  =>  d(L)/d(h) = W_pi @ d_pi_logits
            # v = h @ W_v  + b_v   =>  d(L)/d(h) += d_v_out * W_v
            d_h = self._W_pi @ d_pi_logits + d_v_out * self._W_v
            d_pre_h = d_h * (h > 0)  # ReLU derivative

            grad_w1 += np.outer(obs, d_pre_h)
            grad_b1 += d_pre_h

        # SGD update averaged over episode length
        self._W1 -= self.learning_rate * grad_w1 / n_steps
        self._b1 -= self.learning_rate * grad_b1 / n_steps
        self._W_pi -= self.learning_rate * grad_w_pi / n_steps
        self._b_pi -= self.learning_rate * grad_b_pi / n_steps
        self._W_v -= self.learning_rate * grad_w_v / n_steps
        self._b_v -= self.learning_rate * grad_b_v / n_steps

        return {
            "policy_loss": float(total_policy_loss / n_steps),
            "value_loss": float(total_value_loss / n_steps),
        }

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
