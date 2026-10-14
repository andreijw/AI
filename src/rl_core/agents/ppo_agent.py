"""Proximal Policy Optimization (PPO) agent for CartPole."""

from typing import Any

import numpy as np

from .actor_critic_base import ActorCriticBase
from .returns import discounted_returns


class PPOAgent(ActorCriticBase):
    """
    PPO agent (clip objective) for CartPole.

    Uses an actor-critic architecture with a shared hidden layer and performs
    multiple optimization epochs over each on-policy trajectory.

    Config keys (all optional):
        learning_rate (float): Step size for the configured optimizer. Default: 3e-4.
        optimizer (str): "sgd" or "adam"; Adam takes one step per epoch. Default: "sgd".
        max_grad_norm (float): Global gradient-norm clip; requires optimizer "adam". Default: none.
        gamma (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim (int): Shared hidden layer width. Default: 128.
        value_coef (float): Critic-loss weight. Default: 0.5.
        entropy_coef (float): Entropy-bonus weight. Default: 0.01.
        clip_epsilon (float): PPO clip range. Default: 0.2.
        ppo_epochs (int): Number of PPO epochs per update. Default: 4.
        seed (int): RNG seed. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
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

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Update policy/value parameters with PPO clipped objective.

        Args:
            batch: Dictionary with keys "observations", "actions", and "rewards".

        Returns:
            Dict with mean policy, value, entropy, and clipping statistics.
        """
        observations, actions, rewards, n_steps = self._parse_trajectory(batch)
        if n_steps == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_bonus": 0.0,
                "clip_fraction": 0.0,
            }

        returns = discounted_returns(rewards, self.gamma)

        old_log_probs = np.empty(n_steps, dtype=np.float64)
        old_values = np.empty(n_steps, dtype=np.float64)
        for i, (obs, action) in enumerate(zip(observations, actions, strict=True)):
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
            components, grads, clipped_count = self._epoch_loss_and_grads(
                observations, actions, returns, advantages, old_log_probs
            )

            self._step(grads, n_steps)

            policy_loss = components["policy"] / n_steps
            value_loss = components["value"] / n_steps
            entropy_bonus = components["entropy"] / n_steps
            clip_fraction = clipped_count / n_steps

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_bonus": float(entropy_bonus),
            "clip_fraction": float(clip_fraction),
        }

    def _epoch_loss_and_grads(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        """
        Compute the PPO loss and its gradients for one epoch over a batch, changing no state.

        The loss, summed over the steps, is
        ``Σ_t [ −min(r_t·A_t, clip(r_t, 1±ε)·A_t) + value_coef·½(V(s_t) − G_t)²
        − entropy_coef·H(π(·|s_t)) ]`` with ``r_t = π(a_t|s_t) / π_old(a_t|s_t)``.
        Returns, advantages and old log-probabilities are held constant when differentiating.
        No gradient flows through the policy term of a step whose clipped surrogate is the
        smaller one.

        Args:
            observations:  Array of shape (T, observation_dim).
            actions:       Integer array of shape (T,).
            returns:       Array of shape (T,): the return G_t the critic regresses towards.
            advantages:    Array of shape (T,): the (normalised) advantage A_t of each step.
            old_log_probs: Array of shape (T,): log π_old(a_t|s_t) recorded before the update.

        Returns:
            Tuple of (components, gradients, clipped_count). Components holds the summed
            ``policy``, ``value`` and ``entropy`` terms and their weighted ``total``. Gradients
            are those of ``total``, keyed as in :meth:`_zero_ac_gradients`; :meth:`update`
            divides by the batch length. ``clipped_count`` is the number of clipped steps.
        """
        grads = self._zero_ac_gradients()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        clipped_count = 0

        for obs, action, ret, advantage, old_log_prob in zip(
            observations, actions, returns, advantages, old_log_probs, strict=True
        ):
            probs, value, h = self._forward(obs)

            log_prob = float(np.log(np.clip(probs[action], 1e-8, 1.0)))
            ratio = float(np.exp(log_prob - old_log_prob))
            clipped_ratio = float(np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon))

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

            self._accumulate_ac_gradients(grads, obs, h, d_pi_logits, d_v_out)

        components = {
            "policy": float(epoch_policy_loss),
            "value": float(epoch_value_loss),
            "entropy": float(epoch_entropy),
            "total": float(
                epoch_policy_loss
                + self.value_coef * epoch_value_loss
                - self.entropy_coef * epoch_entropy
            ),
        }
        return components, grads, clipped_count
