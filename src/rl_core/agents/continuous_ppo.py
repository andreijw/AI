"""Continuous action space Proximal Policy Optimization (PPO) agent implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config_values import positive_int
from .continuous_actor_critic import ContinuousActorCriticBase
from .optim import anneal_fraction, anneal_steps_from_config
from .returns import episode_end, gae
from .rollout import EpisodeRollout, RolloutColumns


class ContinuousPPOAgent(ContinuousActorCriticBase):
    """
    Proximal Policy Optimization (PPO) agent with clipped objective for continuous control.

    Uses Generalized Advantage Estimation (GAE) and multiple epochs of mini-batch
    updates with probability ratio clipping:
      r_t(theta) = exp(log_prob(a_t|s_t) - old_log_prob(a_t|s_t))
      L_clip = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

    Each ``update`` call brings one episode from the Trainer. The agent buffers whole episodes
    until it holds at least ``rollout_steps`` transitions (1: train on every episode), then trains
    once on all of them. Training waits for whole episodes, so a rollout can exceed
    ``rollout_steps`` by up to one episode, and the transitions of a last unfilled rollout are not
    trained on. When an episode was cut by the time limit (done but not ``terminated``),
    ``gamma * V(next state)`` is added to its last scaled reward; a terminated episode does not
    bootstrap. A partly filled rollout is saved in checkpoints and restored on load (see
    :class:`EpisodeRollout`).

    Config keys, beyond those of :class:`ContinuousActorCriticBase` (all optional):
        clip_epsilon (float): PPO clip range; ``clip_ratio`` is accepted as an alias. Default: 0.2.
        ppo_epochs (int): Passes over each rollout. Default: 8.
        batch_size (int): Mini-batch size. Default: 64.
        gae_lambda (float): GAE lambda. Default: 0.95.
        anneal_steps (int): Decay the learning rate and clip range linearly to zero over this many
            trained transitions; unset keeps both constant. Default: unset.
        rollout_steps (int): Transitions to buffer, in whole episodes, before training. Default: 1.
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
        self.ppo_epochs: int = positive_int(config, "ppo_epochs", 8)
        self.batch_size: int = positive_int(config, "batch_size", 64)
        self.gae_lambda: float = float(config.get("gae_lambda", 0.95))
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        # Decay the learning rate and clip range linearly to zero over this many trained
        # transitions (counted per update); unset keeps both constant.
        self.anneal_steps: int | None = anneal_steps_from_config(config)
        self._steps_trained = 0
        self.rollout_steps: int = positive_int(config, "rollout_steps", 1)
        self._rollout = EpisodeRollout(self.rollout_steps)

    def update(self, trajectory: Any) -> dict[str, float]:
        """
        Buffer one episode, and train once the rollout holds ``rollout_steps`` transitions.

        Args:
            trajectory: One episode, as a batch dictionary ("observations", "actions", "rewards",
                optionally "dones", "next_observations", "terminated") or a list of transition
                dicts ("state", "action", "reward", "done", optionally "next_state",
                "terminated"). See :meth:`_rollout_episode` for how its end is treated.

        Returns:
            ``{}`` while buffering (or for an empty episode). After training: mean
            ``policy_loss``, ``value_loss`` and ``entropy`` over the mini-batch steps,
            ``anneal_fraction`` (the share of the learning rate and clip range used) and
            ``rollout_steps`` (transitions trained on).
        """
        part = self._rollout_episode(trajectory)
        if part is None:
            return {}
        rollout = self._rollout.add(part)
        if rollout is None:
            return {}
        return self._train_on_rollout(*rollout)

    def _rollout_episode(self, trajectory: Any) -> RolloutColumns | None:
        """
        One episode as ``(states, actions, rewards, dones)`` ready to join a rollout.

        Rewards are scaled by ``reward_scale``. The last step always ends the advantage estimate
        (``dones[-1]`` is True), so that no episode borrows from the next. If the episode was cut
        rather than terminated (its final ``terminated``, or else ``done``, is False) and the next
        state is known, ``gamma * V(next state)`` is added to its last scaled reward, as
        Stable-Baselines3 does for timeouts. Without a next state the cut is treated as terminal.
        The caller's arrays are not modified.

        Returns:
            The four arrays, or None for an empty or unrecognised episode.
        """
        if isinstance(trajectory, dict):
            if len(trajectory.get("observations", [])) == 0:
                return None
            states = np.asarray(trajectory["observations"], dtype=np.float64)
            actions = np.asarray(trajectory["actions"], dtype=np.float64)
            rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
            dones, bootstrap_obs = episode_end(
                len(rewards),
                trajectory.get("dones"),
                trajectory.get("terminated"),
                trajectory.get("next_observations"),
            )
        elif isinstance(trajectory, list):
            if not trajectory:
                return None
            states = np.array([t["state"] for t in trajectory], dtype=np.float64)
            actions = np.array([t["action"] for t in trajectory], dtype=np.float64)
            rewards = np.array([t["reward"] for t in trajectory], dtype=np.float64)
            last = trajectory[-1]
            dones, bootstrap_obs = episode_end(
                len(rewards),
                [t["done"] for t in trajectory],
                [last["terminated"]] if "terminated" in last else None,
                [last["next_state"]] if "next_state" in last else None,
            )
        else:
            return None

        rewards = rewards * self.reward_scale
        if bootstrap_obs is not None:
            _, _, last_v, _ = self._forward(bootstrap_obs)
            rewards[-1] += self.gamma * float(last_v)
        return RolloutColumns(states, actions, rewards, dones)

    def _rollout_targets(
        self, states: np.ndarray, rewards: np.ndarray, dones: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantages (not yet normalised) and the critic's return targets for a rollout."""
        _, _, values_raw, _ = self._forward(states)
        values = np.asarray(values_raw, dtype=np.float64)
        advantages = gae(rewards, values, dones, self.gamma, self.gae_lambda, last_value=0.0)
        return advantages, advantages + values

    def _train_on_rollout(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray
    ) -> dict[str, float]:
        """Run the PPO epochs over one rollout of whole episodes."""
        n_steps = len(rewards)
        fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        learning_rate = self.learning_rate * fraction
        clip_epsilon = self.clip_ratio * fraction

        advantages, returns = self._rollout_targets(states, rewards, dones)

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
            self._shuffle_rng.shuffle(indices)

            for start in range(0, n_steps, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                components, grads = self._minibatch_loss_and_grads(
                    states[batch_idx],
                    actions[batch_idx],
                    returns[batch_idx],
                    advantages[batch_idx],
                    old_log_probs[batch_idx],
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
            "rollout_steps": float(n_steps),
        }

    def _extra_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {
            "ppo_steps_trained": np.array(self._steps_trained),
            **self._rollout.checkpoint_state(),
        }

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        self._steps_trained = int(data["ppo_steps_trained"]) if "ppo_steps_trained" in data else 0
        self._rollout.restore(data, self.observation_dim, (self.action_dim,))

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
        fwd = self._batch_forward(b_states)

        # Log probs and ratio
        new_log_probs = self._log_prob(b_actions, fwd.mu, fwd.log_std)
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

        grads, v_diff = self._policy_value_grads(b_actions, b_returns, fwd, d_log_p)

        policy_loss = float(-np.mean(np.minimum(surr1, surr2)))
        value_loss = float(0.5 * np.mean(np.square(v_diff)))
        entropy = self._entropy(fwd.log_std)
        components = {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
            "total": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
        }
        return components, grads
