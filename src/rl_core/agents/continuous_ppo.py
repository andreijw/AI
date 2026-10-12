"""Continuous action space Proximal Policy Optimization (PPO) agent implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_actor_critic import ContinuousActorCriticBase


class ContinuousPPOAgent(ContinuousActorCriticBase):
    """
    Proximal Policy Optimization (PPO) agent with clipped objective for continuous control.

    Uses Generalized Advantage Estimation (GAE) and multiple epochs of mini-batch
    updates with probability ratio clipping:
      r_t(theta) = exp(log_prob(a_t|s_t) - old_log_prob(a_t|s_t))
      L_clip = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)

        self.clip_ratio: float = float(config.get("clip_ratio", 0.2))
        self.ppo_epochs: int = int(config.get("ppo_epochs", 8))
        self.batch_size: int = int(config.get("batch_size", 64))
        self.gae_lambda: float = float(config.get("gae_lambda", 0.95))
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))

    def update(self, trajectory: Any) -> dict[str, float]:
        """
        Perform PPO clipped policy updates over collected trajectory rollout.

        Args:
            trajectory: List of transition dicts or batch dictionary.

        Returns:
            Dictionary containing metrics averaged over PPO update epochs.
        """
        last_next_obs: np.ndarray | None = None
        if isinstance(trajectory, dict):
            if len(trajectory.get("observations", [])) == 0:
                return {}
            states = np.asarray(trajectory["observations"], dtype=np.float64)
            actions = np.asarray(trajectory["actions"], dtype=np.float64)
            rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
            dones = np.asarray(
                trajectory.get("dones", np.zeros(len(rewards), dtype=bool)), dtype=bool
            )
            next_obs = trajectory.get("next_observations")
            if next_obs is not None and len(next_obs) > 0:
                last_next_obs = np.asarray(next_obs[-1], dtype=np.float64)
        elif isinstance(trajectory, list):
            if not trajectory:
                return {}
            states = np.array([t["state"] for t in trajectory], dtype=np.float64)
            actions = np.array([t["action"] for t in trajectory], dtype=np.float64)
            rewards = np.array([t["reward"] for t in trajectory], dtype=np.float64)
            dones = np.array([t["done"] for t in trajectory], dtype=bool)
            if "next_state" in trajectory[-1]:
                last_next_obs = np.asarray(trajectory[-1]["next_state"], dtype=np.float64)
        else:
            return {}

        n_steps = len(rewards)

        # Compute values for all states
        _, _, values_raw, _ = self._forward(states)
        values = np.asarray(values_raw, dtype=np.float64)

        # Value of final next state for bootstrap
        next_val = 0.0
        if not dones[-1] and last_next_obs is not None:
            _, _, last_v, _ = self._forward(last_next_obs)
            next_val = float(last_v)

        # Compute Generalized Advantage Estimation (GAE)
        advantages = np.zeros(n_steps, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            next_value = values[t + 1] if t + 1 < n_steps else next_val
            done_mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * done_mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Normalize advantages
        if n_steps > 1:
            adv_std = np.std(advantages)
            advantages = (advantages - np.mean(advantages)) / (adv_std + 1e-8)

        # Old log probs under current policy before updates
        old_mu, old_log_std, _, _ = self._forward(states)
        old_log_probs = self._log_prob(actions, old_mu, old_log_std)

        # Epoch training metrics accumulators
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropies: list[float] = []

        indices = np.arange(n_steps)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_steps, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_size = len(batch_idx)

                # Forward pass
                b_h = np.maximum(0.0, b_states @ self._W1 + self._b1)  # (b_size, hidden_dim)
                b_mu = b_h @ self._W_mu + self._b_mu  # (b_size, action_dim)
                b_log_std = np.clip(self._log_std, -20.0, 2.0)
                b_values = b_h @ self._W_v + self._b_v  # (b_size,)

                # Log probs and ratio
                new_log_probs = self._log_prob(b_actions, b_mu, b_log_std)
                ratio = np.exp(new_log_probs - b_old_log_probs)

                # Clipped surrogate
                surr1 = ratio * b_advantages
                clipped_ratio = np.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                surr2 = clipped_ratio * b_advantages

                # Active gradient mask: only compute policy gradient where surr1 <= surr2
                # or ratio was unclipped
                active_mask = (surr1 <= surr2).astype(np.float64)

                # Gradient of ratio w.r.t new_log_probs is ratio
                # dL_pi / d(log_prob) = -active_mask * ratio * advantages
                d_log_p = -active_mask * ratio * b_advantages

                variance = np.exp(2.0 * b_log_std)
                residual = b_actions - b_mu
                d_mu = (residual / (variance + 1e-8)) * d_log_p[:, np.newaxis]
                d_log_std = (((residual**2) / (variance + 1e-8)) - 1.0) * d_log_p[:, np.newaxis]

                grad_w_mu = (b_h.T @ d_mu) / b_size
                grad_b_mu = np.mean(d_mu, axis=0)
                grad_log_std = np.mean(d_log_std, axis=0) - self.entropy_coef

                # Critic loss: 0.5 * (V - R)^2
                v_diff = b_values - b_returns
                grad_w_v = (b_h.T @ v_diff) * self.value_coef / b_size
                grad_b_v = float(np.mean(v_diff) * self.value_coef)

                # Backprop to trunk
                d_h_actor = d_mu @ self._W_mu.T
                d_h_critic = (v_diff[:, np.newaxis] * self.value_coef) @ self._W_v[np.newaxis, :]
                d_h = (d_h_actor + d_h_critic) * (b_h > 0) / b_size

                grad_w1 = b_states.T @ d_h
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
                self._adam_step(grads, self.learning_rate)

                epoch_policy_losses.append(float(-np.mean(np.minimum(surr1, surr2))))
                epoch_value_losses.append(float(0.5 * np.mean(np.square(v_diff))))
                epoch_entropies.append(self._entropy(b_log_std))

        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
        }
