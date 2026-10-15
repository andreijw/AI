"""Continuous action space Proximal Policy Optimization (PPO) agent implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_actor_critic import ContinuousActorCriticBase
from .optim import anneal_fraction, anneal_steps_from_config
from .returns import gae


class ContinuousPPOAgent(ContinuousActorCriticBase):
    """
    Proximal Policy Optimization (PPO) agent with clipped objective for continuous control.

    Uses Generalized Advantage Estimation (GAE) and multiple epochs of mini-batch
    updates with probability ratio clipping:
      r_t(theta) = exp(log_prob(a_t|s_t) - old_log_prob(a_t|s_t))
      L_clip = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

    Each ``update`` call is one rollout (one episode from the Trainer). When the batch carries
    ``terminated``, a final step cut by the time limit (done but not terminated) bootstraps from
    V(next state); without it every ``done`` ends the episode.

    Config keys, beyond those of :class:`ContinuousActorCriticBase` (all optional):
        clip_epsilon (float): PPO clip range; ``clip_ratio`` is accepted as an alias. Default: 0.2.
        ppo_epochs (int): Passes over each rollout. Default: 8.
        batch_size (int): Mini-batch size. Default: 64.
        gae_lambda (float): GAE lambda. Default: 0.95.
        anneal_steps (int): Decay the learning rate and clip range linearly to zero over this many
            trained transitions; unset keeps both constant. Default: unset.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)

        # The clip range: ``clip_epsilon`` as in the discrete PPO, ``clip_ratio`` kept as an alias.
        epsilon, ratio = config.get("clip_epsilon"), config.get("clip_ratio")
        if epsilon is not None and ratio is not None and float(epsilon) != float(ratio):
            raise ValueError(f"clip_epsilon and clip_ratio disagree ({epsilon!r} vs {ratio!r})")
        self.clip_ratio: float = float(
            epsilon if epsilon is not None else ratio if ratio is not None else 0.2
        )
        if self.clip_ratio <= 0.0:
            raise ValueError(f"clip_epsilon must be positive, got {self.clip_ratio!r}")
        self.ppo_epochs: int = int(config.get("ppo_epochs", 8))
        self.batch_size: int = int(config.get("batch_size", 64))
        self.gae_lambda: float = float(config.get("gae_lambda", 0.95))
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        # Decay the learning rate and clip range linearly to zero over this many trained
        # transitions (counted per update); unset keeps both constant.
        self.anneal_steps: int | None = anneal_steps_from_config(config)
        self._steps_trained = 0

    def update(self, trajectory: Any) -> dict[str, float]:
        """
        Perform PPO clipped policy updates over collected trajectory rollout.

        Args:
            trajectory: Batch dictionary ("observations", "actions", "rewards", optionally "dones",
                "next_observations", "terminated") or a list of transition dicts ("state",
                "action", "reward", "done", optionally "next_state", "terminated"). Only the final
                step's ``terminated`` is used: if it is False, that step bootstraps from
                V(next state). Without a next state it is treated as terminal.

        Returns:
            Mean ``policy_loss``, ``value_loss`` and ``entropy`` over the mini-batch steps, and
            ``anneal_fraction`` (the share of the learning rate and clip range used).
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
            terminated = trajectory.get("terminated")
            if terminated is not None and len(terminated) > 0 and len(dones) > 0:
                # A final step cut by the time limit is done but not terminated: bootstrap it.
                dones = dones.copy()
                dones[-1] = bool(np.asarray(terminated, dtype=bool)[-1])
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
            if "terminated" in trajectory[-1]:
                dones[-1] = bool(trajectory[-1]["terminated"])
            if "next_state" in trajectory[-1]:
                last_next_obs = np.asarray(trajectory[-1]["next_state"], dtype=np.float64)
        else:
            return {}

        n_steps = len(rewards)
        rewards = rewards * self.reward_scale
        fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        learning_rate = self.learning_rate * fraction
        clip_epsilon = self.clip_ratio * fraction

        # Compute values for all states
        _, _, values_raw, _ = self._forward(states)
        values = np.asarray(values_raw, dtype=np.float64)

        # Value of final next state for bootstrap
        next_val = 0.0
        if not dones[-1] and last_next_obs is not None:
            _, _, last_v, _ = self._forward(last_next_obs)
            next_val = float(last_v)

        advantages = gae(rewards, values, dones, self.gamma, self.gae_lambda, last_value=next_val)

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
            self._rng.shuffle(indices)

            for start in range(0, n_steps, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]

                components, grads = self._minibatch_loss_and_grads(
                    b_states,
                    b_actions,
                    b_returns,
                    b_advantages,
                    b_old_log_probs,
                    clip_epsilon=clip_epsilon,
                )
                self._adam_step(grads, learning_rate)

                epoch_policy_losses.append(components["policy"])
                epoch_value_losses.append(components["value"])
                epoch_entropies.append(components["entropy"])

        self._steps_trained += n_steps
        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "anneal_fraction": float(fraction),
        }

    def _extra_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"ppo_steps_trained": np.array(self._steps_trained)}

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        self._steps_trained = int(data["ppo_steps_trained"]) if "ppo_steps_trained" in data else 0

    def _minibatch_loss_and_grads(
        self,
        b_states: np.ndarray,
        b_actions: np.ndarray,
        b_returns: np.ndarray,
        b_advantages: np.ndarray,
        b_old_log_probs: np.ndarray,
        clip_epsilon: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Compute the PPO loss and its gradients for one mini-batch, changing no state.

        The loss is ``−mean(min(r·A, clip(r, 1±ε)·A)) + value_coef·½·mean((V(s) − R)²)
        − entropy_coef·H`` with ``r = exp(log π(a|s) − log π_old(a|s))`` and ``H`` the entropy of
        the Gaussian policy summed over action dimensions. Returns, advantages and old
        log-probabilities are held constant when differentiating. No gradient flows through the
        policy term of a step whose clipped surrogate is the smaller one.

        Args:
            b_states:        Array of shape (B, observation_dim).
            b_actions:       Array of shape (B, action_dim).
            b_returns:       Array of shape (B,): the return R the critic regresses towards.
            b_advantages:    Array of shape (B,): the (normalised) advantage of each step.
            b_old_log_probs: Array of shape (B,): log π_old(a|s) recorded before the update.

        Returns:
            Tuple of (components, gradients). Components holds the ``policy``, ``value`` and
            ``entropy`` terms and their weighted ``total``. Gradients are those of ``total``,
            keyed ``W1``, ``b1``, ``W_mu``, ``b_mu``, ``log_std``, ``W_v``, ``b_v``.
        """
        b_size = len(b_returns)

        # Forward pass
        b_h = np.maximum(0.0, b_states @ self._W1 + self._b1)  # (b_size, hidden_dim)
        b_mu = b_h @ self._W_mu + self._b_mu  # (b_size, action_dim)
        b_log_std = np.clip(self._log_std, -20.0, 2.0)
        b_values = b_h @ self._W_v + self._b_v  # (b_size,)

        # Log probs and ratio
        new_log_probs = self._log_prob(b_actions, b_mu, b_log_std)
        # A log-ratio beyond ±20 means the old log-probability is stale: clamp it so exp cannot
        # overflow, and give that step no policy gradient (with a negative advantage the
        # unclipped term would otherwise win and its gradient would be about e^20 times |A|).
        log_ratio = new_log_probs - b_old_log_probs
        stale = np.abs(log_ratio) > 20.0
        ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))

        # Clipped surrogate
        surr1 = ratio * b_advantages
        epsilon = self.clip_ratio if clip_epsilon is None else clip_epsilon
        clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surr2 = clipped_ratio * b_advantages

        # Active gradient mask: only compute policy gradient where surr1 <= surr2
        # or ratio was unclipped
        active_mask = ((surr1 <= surr2) & ~stale).astype(np.float64)

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

        policy_loss = float(-np.mean(np.minimum(surr1, surr2)))
        value_loss = float(0.5 * np.mean(np.square(v_diff)))
        entropy = self._entropy(b_log_std)
        components = {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
            "total": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
        }
        return components, grads
