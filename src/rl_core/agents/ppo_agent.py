"""Proximal Policy Optimization (PPO) agent for discrete actions."""

from typing import Any

import numpy as np

from .actor_critic_base import ActorCriticBase
from .returns import gae


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
    it receives ends its GAE trace at its last step, whether or not that step is marked done. Steps
    still buffered when training stops are never trained on. Checkpoints hold the weights, the
    optimizer state and the annealing counter, but not the buffer; loading a checkpoint empties it.
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
        seed (int): Seed for weights, action sampling and mini-batch shuffling. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
        """Initialise the PPO agent."""
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.clip_epsilon: float = float(config.get("clip_epsilon", 0.2))
        self.ppo_epochs: int = int(config.get("ppo_epochs", 4))
        self.gae_lambda: float = float(config.get("gae_lambda", 1.0))
        self.rollout_steps: int = int(config.get("rollout_steps", 1))
        raw_anneal = config.get("anneal_steps")
        self.anneal_steps: int | None = None if raw_anneal is None else int(raw_anneal)
        raw_minibatch = config.get("minibatch_size")
        self.minibatch_size: int | None = None if raw_minibatch is None else int(raw_minibatch)

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if self.clip_epsilon <= 0.0:
            raise ValueError(f"clip_epsilon must be positive, got {self.clip_epsilon!r}")
        if self.ppo_epochs <= 0:
            raise ValueError(f"ppo_epochs must be a positive integer, got {self.ppo_epochs!r}")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda!r}")
        if self.rollout_steps <= 0:
            raise ValueError(
                f"rollout_steps must be a positive integer, got {self.rollout_steps!r}"
            )
        if self.anneal_steps is not None and self.anneal_steps <= 0:
            raise ValueError(f"anneal_steps must be a positive integer, got {self.anneal_steps!r}")
        if self.minibatch_size is not None and self.minibatch_size <= 0:
            raise ValueError(
                f"minibatch_size must be a positive integer, got {self.minibatch_size!r}"
            )

        # One generator for action sampling and mini-batch shuffling (the same one
        # BaseAgent would create lazily for sampling).
        self._rng = np.random.default_rng(config.get("seed"))
        self._rollout: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._rollout_length = 0
        self._steps_trained = 0

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Buffer one episode, and train once the rollout holds ``rollout_steps`` transitions.

        Args:
            batch: Dictionary with "observations", "actions" and "rewards", and optionally
                "dones". The batch is taken to be whole episodes: its last step always ends an
                episode.

        Returns:
            ``{}`` while buffering. After training: mean ``policy_loss``, ``value_loss`` and
            ``entropy_bonus`` over the mini-batch steps, ``clip_fraction`` (share of clipped
            steps over all epochs), ``rollout_steps`` (transitions trained on) and
            ``anneal_fraction`` (the share of the learning rate and clip range used; 1 without
            ``anneal_steps``).
        """
        observations, actions, rewards, n_steps = self._parse_trajectory(batch)
        if n_steps == 0:
            return {}

        dones = np.zeros(n_steps, dtype=bool)
        if batch.get("dones") is not None:
            dones |= np.asarray(batch["dones"], dtype=bool)
        dones[-1] = True

        self._rollout.append((observations, actions, rewards, dones))
        self._rollout_length += n_steps
        if self._rollout_length < self.rollout_steps:
            return {}

        parts = [np.concatenate(columns) for columns in zip(*self._rollout, strict=True)]
        self._rollout, self._rollout_length = [], 0
        return self._train_on_rollout(*parts)

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
        fraction = 1.0
        if self.anneal_steps is not None:
            fraction = max(0.0, 1.0 - self._steps_trained / self.anneal_steps)
        learning_rate = self.learning_rate * fraction
        clip_epsilon = self.clip_epsilon * fraction
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clipped_total = 0

        for _ in range(self.ppo_epochs):
            order = self._rng.permutation(n_steps)
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
        return {"ppo_steps_trained": np.array(self._steps_trained)}

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        self._steps_trained = int(data["ppo_steps_trained"]) if "ppo_steps_trained" in data else 0

    def load(self, path: str) -> None:
        """Load weights and optimizer state, and drop any partly collected rollout."""
        super().load(path)
        self._rollout, self._rollout_length = [], 0
