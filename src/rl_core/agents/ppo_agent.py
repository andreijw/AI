"""Proximal Policy Optimization (PPO) agent for discrete actions."""

from typing import Any

import numpy as np

from rl_core.seeding import AGENT_REPLAY_STREAM, derive_rng

from .actor_critic_base import ActorCriticBase
from .config_values import optional_positive_int, positive_int
from .optim import anneal_fraction, anneal_steps_from_config
from .returns import episode_end, gae
from .rollout import EpisodeRollout, RolloutColumns


class PPOAgent(ActorCriticBase):
    """
    PPO with the clipped objective (Schulman et al., 2017) for discrete actions.

    The agent buffers whole episodes until it holds at least ``rollout_steps`` transitions, then
    trains on all of them: generalised advantage estimates over the rollout, old log-probabilities
    and values recorded once, then ``ppo_epochs`` passes of shuffled mini-batch steps on the
    mean-reduced loss
    ``−mean(min(r·A, clip(r, 1±ε)·A)) + value_coef·½·mean((V − R)²) − entropy_coef·mean(H)``.
    The critic's target is ``R = A + V_old``; advantages are normalised once per rollout.

    ``update`` is called once per episode. It returns ``{}`` while it only buffers. Every episode
    it receives ends its GAE trace at its last step, so that no episode borrows from the next. An
    episode cut rather than terminated (a time limit, or the Trainer's ``max_steps_per_episode``)
    adds ``gamma * V(next state)`` to its last reward when the next state is known, as continuous
    PPO and Stable-Baselines3 do; a terminated episode does not bootstrap. Steps still buffered
    when training stops are never trained on. Checkpoints hold the weights, the optimizer state,
    the annealing counter and the buffered steps (see :class:`EpisodeRollout`).
    Rollouts are whole episodes, so the counter can step past ``anneal_steps`` mid-rollout; the
    fraction is then held at zero.

    Config keys (all optional):
        learning_rate (float): Step size for the configured optimizer. Default: 3e-4.
        optimizer (str): "sgd" or "adam"; one step per mini-batch. Default: "sgd".
        max_grad_norm (float): Global gradient-norm clip; requires optimizer "adam". Default: none.
        gamma (float): Discount factor ∈ (0, 1]. Default: 0.99.
        gae_lambda (float): GAE λ ∈ [0, 1]; 1 gives the episode return minus V(s). Default: 1.0.
        rollout_steps (int): Transitions to collect before training; 1 trains on every episode.
            Default: 1.
        minibatch_size (int): Transitions per optimizer step; unset means the whole rollout.
            Default: unset.
        ppo_epochs (int): Passes over each rollout. Default: 4.
        clip_epsilon (float): PPO clip range. Default: 0.2.
        anneal_steps (int): Decay the learning rate and clip range linearly to zero over this many
            trained transitions (counted per rollout); unset keeps both constant. Default: unset.
        hidden_dim (int): Shared hidden layer width. Default: 128.
        value_coef (float): Critic-loss weight. Default: 0.5.
        entropy_coef (float): Entropy-bonus weight. Default: 0.01.
        seed (int): Seed of the weights, action-sampling and mini-batch-order streams. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
        """Initialise the PPO agent."""
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.clip_epsilon: float = float(config.get("clip_epsilon", 0.2))
        self.ppo_epochs: int = positive_int(config, "ppo_epochs", 4)
        self.gae_lambda: float = float(config.get("gae_lambda", 1.0))
        self.rollout_steps: int = positive_int(config, "rollout_steps", 1)
        self.anneal_steps: int | None = anneal_steps_from_config(config)
        self.minibatch_size: int | None = optional_positive_int(config, "minibatch_size")

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if self.clip_epsilon <= 0.0:
            raise ValueError(f"clip_epsilon must be positive, got {self.clip_epsilon!r}")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda!r}")

        # One generator for action sampling and mini-batch shuffling (the same one
        # BaseAgent would create lazily for sampling).
        # Minibatch order on its own stream; action sampling uses the base class's exploration stream.
        self._shuffle_rng = derive_rng(config.get("seed"), AGENT_REPLAY_STREAM)
        self._rollout = EpisodeRollout(self.rollout_steps)
        self._steps_trained = 0

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Buffer one episode, and train once the rollout holds ``rollout_steps`` transitions.

        Args:
            batch: Dictionary with "observations", "actions" and "rewards", and optionally
                "dones", "terminated" and "next_observations". The batch is taken to be whole
                episodes: its last step always ends an episode. See :meth:`_rollout_episode` for
                how a cut episode is bootstrapped.

        Returns:
            ``{}`` while buffering. After training: mean ``policy_loss``, ``value_loss`` and
            ``entropy_bonus`` over the mini-batch steps, ``clip_fraction`` (share of clipped
            steps over all epochs), ``rollout_steps`` (transitions trained on) and
            ``anneal_fraction`` (the share of the learning rate and clip range used; 1 without
            ``anneal_steps``).
        """
        episode = self._rollout_episode(batch)
        if episode is None:
            return {}

        rollout = self._rollout.add(episode)
        if rollout is None:
            return {}
        return self._train_on_rollout(*rollout)

    def _rollout_episode(self, batch: dict[str, Any]) -> RolloutColumns | None:
        """
        One episode as ``(observations, actions, rewards, dones)`` ready to join a rollout.

        The last step always ends the advantage estimate (see :func:`episode_end`). If the episode
        was cut rather than terminated and its next state is known, ``gamma * V(next state)`` is
        added to a copy of its last reward. The caller's arrays are not modified.

        Returns:
            The four arrays, or None for an empty episode.
        """
        observations, actions, rewards, n_steps = self._parse_trajectory(batch)
        if n_steps == 0:
            return None
        dones, bootstrap_obs = episode_end(
            n_steps, batch.get("dones"), batch.get("terminated"), batch.get("next_observations")
        )
        if bootstrap_obs is not None:
            _, next_value, _ = self._forward(bootstrap_obs)
            rewards = rewards.copy()
            rewards[-1] += self.gamma * next_value
        return RolloutColumns(observations, actions, rewards, dones)

    def _train_on_rollout(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Run the PPO epochs over one rollout of whole episodes."""
        n_steps = len(rewards)
        probs, values, _ = self._forward_batch(observations)
        old_log_probs = np.log(np.clip(probs[np.arange(n_steps), actions], 1e-8, 1.0))

        advantages = gae(rewards, values, dones, self.gamma, self.gae_lambda)
        returns = advantages + values
        advantages = advantages - advantages.mean()
        if n_steps > 1:
            advantages = advantages / (advantages.std() + 1e-8)

        minibatch_size = self.minibatch_size or n_steps
        fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        learning_rate = self.learning_rate * fraction
        clip_epsilon = self.clip_epsilon * fraction
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clipped_total = 0

        for _ in range(self.ppo_epochs):
            order = self._shuffle_rng.permutation(n_steps)
            for start in range(0, n_steps, minibatch_size):
                idx = order[start : start + minibatch_size]
                components, grads, clipped = self._minibatch_loss_and_grads(
                    observations[idx],
                    actions[idx],
                    returns[idx],
                    advantages[idx],
                    old_log_probs[idx],
                    clip_epsilon=clip_epsilon,
                )
                self._step(grads, 1, learning_rate)
                policy_losses.append(components["policy"])
                value_losses.append(components["value"])
                entropies.append(components["entropy"])
                clipped_total += clipped

        self._steps_trained += n_steps
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_bonus": float(np.mean(entropies)),
            "clip_fraction": float(clipped_total / (n_steps * self.ppo_epochs)),
            "rollout_steps": float(n_steps),
            "anneal_fraction": float(fraction),
        }

    def _minibatch_loss_and_grads(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        clip_epsilon: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        """
        Compute the PPO loss and its gradients for one mini-batch, changing no state.

        The loss is mean-reduced over the B steps:
        ``−mean(min(r·A, clip(r, 1±ε)·A)) + value_coef·½·mean((V(s) − R)²) − entropy_coef·mean(H)``
        with ``r = π(a|s) / π_old(a|s)``. Returns, advantages and old log-probabilities are held
        constant when differentiating. No gradient flows through the policy term of a step whose
        clipped surrogate is the smaller one.

        Args:
            observations:  Array of shape (B, observation_dim).
            actions:       Integer array of shape (B,).
            returns:       Array of shape (B,): the target R the critic regresses towards.
            advantages:    Array of shape (B,): the (normalised) advantage of each step.
            old_log_probs: Array of shape (B,): log π_old(a|s) recorded before the update.
            clip_epsilon:  Clip range for this step; defaults to ``self.clip_epsilon``.

        Returns:
            Tuple of (components, gradients, clipped_count). Components holds the mean
            ``policy``, ``value`` and ``entropy`` terms and their weighted ``total``. Gradients
            are those of ``total``, keyed as in :meth:`_zero_ac_gradients`. ``clipped_count`` is
            the number of steps whose clipped surrogate was the smaller one.
        """
        b_size = len(actions)
        rows = np.arange(b_size)
        probs, values, h = self._forward_batch(observations)
        log_probs_all = np.log(np.clip(probs, 1e-8, 1.0))
        log_probs = log_probs_all[rows, actions]

        ratio = np.exp(log_probs - old_log_probs)
        epsilon = self.clip_epsilon if clip_epsilon is None else clip_epsilon
        clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surrogate = ratio * advantages
        surrogate_clipped = clipped_ratio * advantages
        use_clipped = surrogate_clipped < surrogate
        entropy = -np.sum(probs * log_probs_all, axis=1)

        # Policy: d(−r·A)/d logits = −A·r·(onehot(a) − π), only where the unclipped term is used.
        one_hot = np.zeros_like(probs)
        one_hot[rows, actions] = 1.0
        policy_weight = np.where(use_clipped, 0.0, -advantages * ratio)
        d_logits = policy_weight[:, np.newaxis] * (one_hot - probs)
        # Entropy bonus: d(−entropy_coef·H)/d logits = entropy_coef·π·(log π + H).
        d_logits += self.entropy_coef * probs * (log_probs_all + entropy[:, np.newaxis])
        d_logits /= b_size
        d_values = self.value_coef * (values - returns) / b_size

        d_hidden = (d_logits @ self._W_pi.T + np.outer(d_values, self._W_v)) * (h > 0.0)
        grads = {
            "W1": observations.T @ d_hidden,
            "b1": d_hidden.sum(axis=0),
            "W_pi": h.T @ d_logits,
            "b_pi": d_logits.sum(axis=0),
            "W_v": h.T @ d_values,
            "b_v": float(d_values.sum()),
        }

        policy_loss = float(-np.mean(np.minimum(surrogate, surrogate_clipped)))
        value_loss = float(0.5 * np.mean(np.square(values - returns)))
        mean_entropy = float(np.mean(entropy))
        components = {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": mean_entropy,
            "total": policy_loss + self.value_coef * value_loss - self.entropy_coef * mean_entropy,
        }
        return components, grads, int(np.sum(use_clipped))

    def _extra_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {
            "ppo_steps_trained": np.array(self._steps_trained),
            **self._rollout.checkpoint_state(),
        }

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        self._steps_trained = int(data["ppo_steps_trained"]) if "ppo_steps_trained" in data else 0
        self._rollout.restore(data, self.observation_dim, ())
