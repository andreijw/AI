"""Proximal Policy Optimization (PPO) agent for CartPole."""

from typing import Any, Dict

import numpy as np

from .actor_critic_base import ActorCriticBase


class PPOAgent(ActorCriticBase):
    """
    PPO agent (clip objective) for CartPole.

    Uses an actor-critic architecture with a shared hidden layer and performs
    multiple optimization epochs over each on-policy trajectory.

    Config keys (all optional):
        learning_rate (float): SGD step size. Default: 3e-4.
        gamma (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim (int): Shared hidden layer width. Default: 128.
        value_coef (float): Critic-loss weight. Default: 0.5.
        entropy_coef (float): Entropy-bonus weight. Default: 0.01.
        clip_epsilon (float): PPO clip range. Default: 0.2.
        ppo_epochs (int): Number of PPO epochs per update. Default: 4.
        seed (int): RNG seed. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialise the PPO agent."""
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.clip_epsilon: float = float(config.get("clip_epsilon", 0.2))
        self.ppo_epochs: int = int(config.get("ppo_epochs", 4))

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if self.clip_epsilon <= 0.0:
            raise ValueError(f"clip_epsilon must be positive, got {self.clip_epsilon!r}")
        if self.ppo_epochs <= 0:
            raise ValueError(f"ppo_epochs must be a positive integer, got {self.ppo_epochs!r}")

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update policy/value parameters with PPO clipped objective.

        Args:
            batch: Dictionary with keys "observations", "actions", and "rewards".

        Returns:
            Dict with mean policy, value, entropy, and clipping statistics.
        """
        observations: np.ndarray = np.asarray(batch["observations"], dtype=np.float64)
        actions: np.ndarray = np.asarray(batch["actions"], dtype=int)
        rewards: np.ndarray = np.asarray(batch["rewards"], dtype=np.float64)

        n_steps = len(rewards)
        if n_steps == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_bonus": 0.0,
                "clip_fraction": 0.0,
            }
        if not (len(observations) == len(actions) == n_steps):
            raise ValueError(
                "Inconsistent trajectory lengths: "
                f"observations={len(observations)}, actions={len(actions)}, rewards={n_steps}"
            )

        returns = self._compute_returns(rewards)

        old_log_probs = np.empty(n_steps, dtype=np.float64)
        old_values = np.empty(n_steps, dtype=np.float64)
        for i, (obs, action) in enumerate(zip(observations, actions)):
            old_probs, old_value, _ = self._forward(obs)
            old_log_probs[i] = float(np.log(np.clip(old_probs[action], 1e-8, 1.0)))
            old_values[i] = old_value

        advantages = returns - old_values
        adv_std = float(advantages.std())
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()

        policy_loss = 0.0
        value_loss = 0.0
        entropy_bonus = 0.0
        clip_fraction = 0.0

        for _ in range(self.ppo_epochs):
            grad_w1 = np.zeros_like(self._W1)
            grad_b1 = np.zeros_like(self._b1)
            grad_w_pi = np.zeros_like(self._W_pi)
            grad_b_pi = np.zeros_like(self._b_pi)
            grad_w_v = np.zeros_like(self._W_v)
            grad_b_v = 0.0

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            clipped_count = 0

            for obs, action, ret, advantage, old_log_prob in zip(
                observations, actions, returns, advantages, old_log_probs
            ):
                probs, value, h = self._forward(obs)

                log_prob = float(np.log(np.clip(probs[action], 1e-8, 1.0)))
                ratio = float(np.exp(log_prob - old_log_prob))
                clipped_ratio = float(
                    np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                )

                surrogate_unclipped = ratio * float(advantage)
                surrogate_clipped = clipped_ratio * float(advantage)
                use_clipped = surrogate_clipped < surrogate_unclipped

                if use_clipped:
                    clipped_count += 1

                # Minimize negative PPO surrogate.
                epoch_policy_loss += -float(min(surrogate_unclipped, surrogate_clipped))
                epoch_value_loss += 0.5 * (value - ret) ** 2

                log_probs = np.log(np.clip(probs, 1e-8, 1.0))
                entropy = -float(np.sum(probs * log_probs))
                epoch_entropy += entropy

                d_pi_logits = np.zeros(self.action_dim, dtype=np.float64)
                if not use_clipped:
                    # d(log probs[action])/d(logits) = one_hot(action) - probs
                    one_hot = np.zeros(self.action_dim, dtype=np.float64)
                    one_hot[action] = 1.0
                    d_log_pi = one_hot - probs
                    d_pi_logits += -float(advantage) * ratio * d_log_pi

                # Entropy regularization gradient for loss term: -entropy_coef * H.
                d_pi_logits += self.entropy_coef * probs * (log_probs + entropy)

                d_v_out = self.value_coef * (value - ret)

                grad_w_pi += np.outer(h, d_pi_logits)
                grad_b_pi += d_pi_logits
                grad_w_v += d_v_out * h
                grad_b_v += d_v_out

                d_h = self._W_pi @ d_pi_logits + d_v_out * self._W_v
                d_pre_h = d_h * (h > 0)

                grad_w1 += np.outer(obs, d_pre_h)
                grad_b1 += d_pre_h

            scale = self.learning_rate / n_steps
            self._W1 -= scale * grad_w1
            self._b1 -= scale * grad_b1
            self._W_pi -= scale * grad_w_pi
            self._b_pi -= scale * grad_b_pi
            self._W_v -= scale * grad_w_v
            self._b_v -= scale * grad_b_v

            policy_loss = epoch_policy_loss / n_steps
            value_loss = epoch_value_loss / n_steps
            entropy_bonus = epoch_entropy / n_steps
            clip_fraction = clipped_count / n_steps

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_bonus": float(entropy_bonus),
            "clip_fraction": float(clip_fraction),
        }
