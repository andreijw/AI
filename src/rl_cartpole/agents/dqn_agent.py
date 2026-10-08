"""Deep Q-Network (DQN) agent for CartPole."""

from typing import Any

import numpy as np
import numpy.typing as npt

from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent with experience replay and target network.

    Approximates state-action values Q(s, a) using a multi-layer perceptron.
    Trained via off-policy Bellman temporal-difference error minimization:
        L = 0.5 * (Q_online(s, a) - (r + (1 - done) * γ * max_a' Q_target(s', a')))^2

    Features:
    * Separate online and target Q-networks for stable gradient updates.
    * Epsilon-greedy action selection with exponential decay.
    * Ring-buffered experience replay for uncorrelated mini-batch updates.
    * Hard or Polyak soft target network synchronization.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Initialise the DQN agent.

        Args:
            observation_dim: Dimension of observation vectors.
            action_dim: Number of discrete actions.
            config: Hyperparameter dictionary.
        """
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 1e-3))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = int(config.get("hidden_dim", 128))
        self.buffer_capacity: int = int(config.get("buffer_capacity", 10000))
        self.batch_size: int = int(config.get("batch_size", 64))
        self.min_buffer_size: int = int(config.get("min_buffer_size", self.batch_size))
        self.epsilon_start: float = float(config.get("epsilon_start", 1.0))
        self.epsilon_end: float = float(config.get("epsilon_end", 0.02))
        self.epsilon_decay: float = float(config.get("epsilon_decay", 0.995))
        self.target_update_frequency: int = int(config.get("target_update_frequency", 100))
        self.tau: float | None = float(config["tau"]) if config.get("tau") is not None else None
        self.max_grad_norm: float = float(config.get("max_grad_norm", 10.0))

        self._validate_dims(observation_dim, action_dim, self.hidden_dim)
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma!r}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start!r}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            raise ValueError(f"epsilon_end must be in [0, 1], got {self.epsilon_end!r}")
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay!r}")
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity!r}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size!r}")

        self.epsilon: float = self.epsilon_start
        self._update_counter: int = 0
        self._rng: np.random.Generator = np.random.default_rng(config.get("seed"))

        # He weight initialization
        scale1 = np.sqrt(2.0 / observation_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)

        # Online Q-network parameters
        self._W1: npt.NDArray[np.float64] = (
            self._rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        )
        self._b1: npt.NDArray[np.float64] = np.zeros(self.hidden_dim)
        self._W2: npt.NDArray[np.float64] = (
            self._rng.standard_normal((self.hidden_dim, action_dim)) * scale2
        )
        self._b2: npt.NDArray[np.float64] = np.zeros(action_dim)

        # Target Q-network parameters (identical copy)
        self._target_W1: npt.NDArray[np.float64] = np.copy(self._W1)
        self._target_b1: npt.NDArray[np.float64] = np.copy(self._b1)
        self._target_W2: npt.NDArray[np.float64] = np.copy(self._W2)
        self._target_b2: npt.NDArray[np.float64] = np.copy(self._b2)

        # Replay buffer
        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            observation_dim=self.observation_dim,
            seed=config.get("seed"),
        )

    def _forward_q(self, obs: np.ndarray, target: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward pass Q-values for an observation or batch of observations.

        Args:
            obs: Observation vector (observation_dim,) or batch (N, observation_dim).
            target: If True, uses the target network parameters; otherwise online network.

        Returns:
            Tuple of (q_values, hidden_activations).
        """
        w1 = self._target_W1 if target else self._W1
        b1 = self._target_b1 if target else self._b1
        w2 = self._target_W2 if target else self._W2
        b2 = self._target_b2 if target else self._b2

        h = np.maximum(0.0, obs @ w1 + b1)
        q = h @ w2 + b2
        return q, h

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        During training, applies epsilon-greedy action selection.
        During evaluation, acts greedily (argmax Q).

        Args:
            observation: Current observation vector.
            training: Whether in training mode.

        Returns:
            Selected action index.
        """
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_dim))

        q_vals, _ = self._forward_q(observation, target=False)
        return int(np.argmax(q_vals))

    def _compute_bellman_targets(
        self,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Bellman temporal-difference target values:
            y = r + (1 - done) * γ * max_a' Q_target(s', a')

        Args:
            rewards: Batch of scalar rewards.
            next_obs: Batch of next observation vectors.
            dones: Batch of termination booleans.

        Returns:
            Array of target values of shape (batch_size,).
        """
        next_q, _ = self._forward_q(next_obs, target=True)
        max_next_q = np.max(next_q, axis=-1)
        not_done = 1.0 - dones.astype(np.float64)
        return rewards + not_done * self.gamma * max_next_q

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Update the Q-network using collected transitions.

        Ingests transitions into the replay buffer, samples a mini-batch,
        computes the Bellman loss, and applies gradient descent.

        Args:
            batch: Trajectory dictionary with 'observations', 'actions', 'rewards',
                   and optionally 'next_observations' and 'dones'.

        Returns:
            Dictionary with training metrics: 'loss', 'epsilon', 'mean_q'.
        """
        if "next_observations" in batch and "dones" in batch:
            self.replay_buffer.add_batch(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )

        if not self.replay_buffer.can_sample(self.min_buffer_size):
            return {"loss": 0.0, "epsilon": self.epsilon, "mean_q": 0.0}

        # Sample mini-batch from experience replay
        samples = self.replay_buffer.sample(self.batch_size)
        obs = samples["observations"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        next_obs = samples["next_observations"]
        dones = samples["dones"]
        b_size = len(actions)

        # 1. Compute Bellman targets
        targets = self._compute_bellman_targets(rewards, next_obs, dones)

        # 2. Forward pass online network
        q_vals, h = self._forward_q(obs, target=False)
        pred_q = q_vals[np.arange(b_size), actions]

        # 3. TD error and MSE loss: 0.5 * (pred_q - targets)^2
        td_error = pred_q - targets
        loss = float(np.mean(0.5 * (td_error**2)))

        # 4. Backward pass
        d_q = np.zeros_like(q_vals)
        d_q[np.arange(b_size), actions] = td_error

        grad_w2 = (h.T @ d_q) / b_size
        grad_b2 = np.mean(d_q, axis=0)

        d_h = d_q @ self._W2.T
        d_pre_h = d_h * (h > 0.0)

        grad_w1 = (obs.T @ d_pre_h) / b_size
        grad_b1 = np.mean(d_pre_h, axis=0)

        # Clip gradients to avoid exploding gradients
        np.clip(grad_w1, -self.max_grad_norm, self.max_grad_norm, out=grad_w1)
        np.clip(grad_b1, -self.max_grad_norm, self.max_grad_norm, out=grad_b1)
        np.clip(grad_w2, -self.max_grad_norm, self.max_grad_norm, out=grad_w2)
        np.clip(grad_b2, -self.max_grad_norm, self.max_grad_norm, out=grad_b2)

        # 5. Gradient descent step
        self._W1 -= self.learning_rate * grad_w1
        self._b1 -= self.learning_rate * grad_b1
        self._W2 -= self.learning_rate * grad_w2
        self._b2 -= self.learning_rate * grad_b2

        # 6. Target network synchronization
        self._update_counter += 1
        if self.tau is not None and self.tau < 1.0:
            self._target_W1 = self.tau * self._W1 + (1.0 - self.tau) * self._target_W1
            self._target_b1 = self.tau * self._b1 + (1.0 - self.tau) * self._target_b1
            self._target_W2 = self.tau * self._W2 + (1.0 - self.tau) * self._target_W2
            self._target_b2 = self.tau * self._b2 + (1.0 - self.tau) * self._target_b2
        elif self._update_counter % self.target_update_frequency == 0:
            self._target_W1 = np.copy(self._W1)
            self._target_b1 = np.copy(self._b1)
            self._target_W2 = np.copy(self._W2)
            self._target_b2 = np.copy(self._b2)

        # 7. Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss,
            "epsilon": float(self.epsilon),
            "mean_q": float(np.mean(pred_q)),
        }

    def save(self, path: str) -> None:
        """
        Save network parameters and epsilon state to disk.

        Args:
            path: Destination checkpoint path (.pt or .npz).
        """
        save_path = self._resolve_save_path(path)
        with open(save_path, "wb") as f:
            np.savez(
                f,
                W1=self._W1,
                b1=self._b1,
                W2=self._W2,
                b2=self._b2,
                target_W1=self._target_W1,
                target_b1=self._target_b1,
                target_W2=self._target_W2,
                target_b2=self._target_b2,
                epsilon=np.array(self.epsilon),
            )

    def load(self, path: str) -> None:
        """
        Load network parameters and epsilon state from disk.

        Args:
            path: Source checkpoint path (.pt or .npz).
        """
        load_path = self._resolve_load_path(path)
        with np.load(load_path) as data:
            self._W1 = data["W1"]
            self._b1 = data["b1"]
            self._W2 = data["W2"]
            self._b2 = data["b2"]
            self._target_W1 = data["target_W1"]
            self._target_b1 = data["target_b1"]
            self._target_W2 = data["target_W2"]
            self._target_b2 = data["target_b2"]
            if "epsilon" in data:
                self.epsilon = float(data["epsilon"])
