"""Continuous action space Actor-Critic base and agent implementations (Gaussian policy)."""

from __future__ import annotations

import os
from typing import Any, ClassVar, NamedTuple

import numpy as np

from rl_core.seeding import (
    AGENT_EXPLORATION_STREAM,
    AGENT_REPLAY_STREAM,
    AGENT_WEIGHTS_STREAM,
    derive_rng,
)

from .base_agent import BaseAgent
from .bounds import action_bounds
from .config_values import non_negative_int, positive_int
from .optim import Adam, anneal_fraction, anneal_steps_from_config
from .returns import discounted_returns, gae


class _BatchForward(NamedTuple):
    """One forward pass over a batch: what both Gaussian agents' backward passes need."""

    states: np.ndarray  # (n, observation_dim) the batch the pass was made on
    h: np.ndarray  # (n, hidden_dim) post-ReLU activations
    mu: np.ndarray  # (n, action_dim) policy means
    dmu_dz: np.ndarray | float  # d(mu)/d(z) of the mean head; 1.0 when unbounded
    log_std: np.ndarray  # (action_dim,) clipped log standard deviation
    values: np.ndarray  # (n,) state values


class ContinuousActorCriticBase(BaseAgent):
    """
    Base class for continuous action space actor-critic agents (Gaussian policy).

    Uses a shared neural trunk with two output heads:
      1. Gaussian Actor Head: outputs state-dependent mean mu(s) and trainable log_std
      2. Critic Head: outputs state-value estimate V(s)

    In training, actions are samples from N(mu, sigma^2) returned unclipped, so the batch holds the
    action whose log-probability the update uses; the environment clips the torque it applies. In
    evaluation the action is mu clipped to [action_low, action_high].

    Config keys shared by both continuous agents (all optional):
        learning_rate (float): Adam step size. Default: 3e-4.
        gamma (float): Discount factor. Default: 0.99.
        hidden_dim (int): Shared hidden layer width. Default: 128.
        value_coef (float): Critic-loss weight. Default: 0.5.
        entropy_coef (float): Entropy-bonus weight. Default: 0.001.
        max_grad_norm (float): Global gradient-norm clip. Default: 0.5.
        mean_init_scale (float): Scale of the mean head's initial weights relative to He scaling;
            small keeps the initial mean inside the bounds. Default: 0.01.
        reward_scale (float): Multiplier on rewards before returns are computed. Default: 1.0.
        init_log_std (float): Initial log standard deviation. Default: -0.5.
        action_low, action_high: Action bounds. Default: -1.0, 1.0.
        bounded_mean (bool): Policy mean bias + scale * tanh(z), inside the action bounds; needs
            finite bounds with action_low < action_high. Default: False.
        seed (int): Seed of the weights, action-noise and shuffle streams. Default: None.
    """

    # The trainable parameters, stored as ``_<name>``: the order of the optimiser state, the
    # gradients and the checkpoint. ``b_v`` is a Python float, the rest are arrays. _adam_step
    # updates arrays in place and special-cases b_v by name: a new scalar parameter needs the
    # same, or ``param -= step`` would rebind a local name and silently lose its update.
    _PARAM_NAMES: ClassVar[tuple[str, ...]] = ("W1", "b1", "W_mu", "b_mu", "log_std", "W_v", "b_v")

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)

        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = positive_int(config, "hidden_dim", 128)
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.value_coef: float = float(config.get("value_coef", 0.5))
        self.entropy_coef: float = float(config.get("entropy_coef", 0.001))
        # Scale of the mean head's initial weights relative to He scaling. Small (0.01, as in
        # CleanRL) keeps the initial mean near zero, inside the action bounds.
        self.mean_init_scale: float = float(config.get("mean_init_scale", 0.01))
        # Rewards are multiplied by this before returns and advantages are computed.
        self.reward_scale: float = float(config.get("reward_scale", 1.0))
        for name in ("mean_init_scale", "reward_scale"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        self.max_grad_norm: float = float(config.get("max_grad_norm", 0.5))

        # bounded_mean: mu = bias + scale * tanh(z) keeps the policy mean inside the bounds. Without it
        # the mean is the linear head and can drift past a bound, where every executed action is the same.
        self.bounded_mean: bool = bool(config.get("bounded_mean", False))
        self.action_low, self.action_high = action_bounds(
            config, action_dim, require_finite=self.bounded_mean
        )
        low, high = self.action_low.astype(np.float64), self.action_high.astype(np.float64)
        self._mean_scale: np.ndarray = (high - low) / 2.0
        self._mean_bias: np.ndarray = (high + low) / 2.0

        self._validate_dims(observation_dim, action_dim)

        # Initialize network parameters using He initialization
        weights_rng = derive_rng(config.get("seed"), AGENT_WEIGHTS_STREAM)
        # Action noise on its own stream; the shuffle generator is used only by continuous PPO.
        self._explore_rng = derive_rng(config.get("seed"), AGENT_EXPLORATION_STREAM)
        self._shuffle_rng = derive_rng(config.get("seed"), AGENT_REPLAY_STREAM)
        scale1 = np.sqrt(2.0 / observation_dim)
        scale_head = np.sqrt(2.0 / self.hidden_dim)

        # Trunk: obs -> hidden
        self._W1: np.ndarray = (
            weights_rng.standard_normal((observation_dim, self.hidden_dim)) * scale1
        )
        self._b1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        # Actor head: hidden -> mu
        self._W_mu: np.ndarray = (
            weights_rng.standard_normal((self.hidden_dim, action_dim)) * scale_head
        ) * self.mean_init_scale
        self._b_mu: np.ndarray = np.zeros(action_dim, dtype=np.float64)

        # Trainable log standard deviation: initialized to -0.5 (std ~ 0.6)
        init_log_std = float(config.get("init_log_std", -0.5))
        self._log_std: np.ndarray = np.full(action_dim, init_log_std, dtype=np.float64)

        # Critic head: hidden -> value
        self._W_v: np.ndarray = weights_rng.standard_normal(self.hidden_dim) * scale_head
        self._b_v: float = 0.0

        # Adam with global gradient-norm clipping
        self._optimizer = Adam(self._params(), max_grad_norm=self.max_grad_norm)

    def _params(self) -> dict[str, Any]:
        """The trainable parameters by name, as the agent holds them (not copies)."""
        return {name: getattr(self, f"_{name}") for name in self._PARAM_NAMES}

    @staticmethod
    def _validate_dims(obs_dim: int, act_dim: int) -> None:
        if obs_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {obs_dim}")
        if act_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {act_dim}")

    def _mean(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | None]:
        """
        The policy mean from hidden activations, its derivative with respect to the mean head's
        output z, and tanh(z) when bounded.

        Unbounded, mu = z and the derivative is 1. With ``bounded_mean``, mu = bias + scale *
        tanh(z) with scale and bias from the action bounds, and the derivative is
        scale * (1 - tanh(z)^2). Both backward passes multiply d(loss)/d(mu) by it.
        """
        z = h @ self._W_mu + self._b_mu
        if not self.bounded_mean:
            return z, 1.0, None
        u = np.tanh(z)
        return self._mean_bias + self._mean_scale * u, self._mean_scale * (1.0 - u**2), u

    def _forward(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float | np.ndarray, np.ndarray]:
        """
        Forward pass through trunk and heads.

        Returns:
            mu: Action mean vector(s)
            log_std: Action log standard deviation
            v: State value estimate(s)
            h: Hidden layer activation(s) (post-ReLU)
        """
        s = np.asarray(state, dtype=np.float64)
        h = np.maximum(0.0, s @ self._W1 + self._b1)  # ReLU
        mu, _, _ = self._mean(h)
        log_std = np.clip(self._log_std, -20.0, 2.0)
        v = (h @ self._W_v + self._b_v).item() if s.ndim == 1 else h @ self._W_v + self._b_v
        return mu, log_std, v, h

    def select_action(
        self, state: np.ndarray, training: bool = True, evaluate: bool | None = None
    ) -> np.ndarray:
        """
        Select continuous action for a single environment state.

        Args:
            state: Observation vector of shape (observation_dim,).
            training: If True, samples stochastically; if False, acts deterministically.
            evaluate: Legacy alias; if True, acts deterministically. Overrides training if passed.

        Returns:
            Action vector of shape (action_dim,). In training: the unclipped sample, which may lie
            outside [action_low, action_high] (clipping it here would make the stored action differ
            from the sampled one, and its log-probability meaningless when mu is out of bounds).
            In evaluation: mu clipped to [action_low, action_high].
        """
        mu, log_std, _, _ = self._forward(state)
        is_eval = evaluate if evaluate is not None else not training
        if is_eval:
            return np.clip(mu, self.action_low, self.action_high)
        std = np.exp(log_std)
        noise = self._explore_rng.standard_normal(self.action_dim)
        return mu + std * noise

    def _log_prob(self, action: np.ndarray, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        """Compute Gaussian log-likelihood log N(action; mu, sigma^2)."""
        std = np.exp(log_std)
        variance = std**2
        log_2pi = np.log(2.0 * np.pi)
        residual = action - mu
        log_probs = -0.5 * ((residual**2) / (variance + 1e-8) + 2.0 * log_std + log_2pi)
        return np.sum(log_probs, axis=-1)

    def _entropy(self, log_std: np.ndarray) -> float:
        """Differential entropy of multivariate Gaussian with diagonal covariance."""
        log_2pi_e = np.log(2.0 * np.pi * np.e)
        return float(np.sum(log_std + 0.5 * log_2pi_e))

    def _adam_step(self, grads: dict[str, np.ndarray | float], lr: float) -> None:
        """Apply one step of Adam optimization with gradient clipping, updating arrays in place."""
        for name, step in self._optimizer.step(grads, lr).items():
            if name == "b_v":
                self._b_v -= float(step)
            else:
                param = getattr(self, f"_{name}")
                param -= step

    def _batch_forward(self, states: np.ndarray) -> _BatchForward:
        """The forward pass over a batch of states, (n, observation_dim), for a backward pass."""
        h = np.maximum(0.0, states @ self._W1 + self._b1)  # ReLU
        mu, dmu_dz, _ = self._mean(h)
        log_std = np.clip(self._log_std, -20.0, 2.0)
        values = h @ self._W_v + self._b_v
        return _BatchForward(states, h, mu, dmu_dz, log_std, values)

    def _policy_value_grads(
        self,
        actions: np.ndarray,
        returns: np.ndarray,
        fwd: _BatchForward,
        d_log_p: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """
        Gradients of ``policy + value_coef·½·mean((V − R)²) − entropy_coef·H`` over a batch.

        The policy term enters only through ``d_log_p``, the derivative of the policy loss with
        respect to each step's log-probability before the batch mean, which is taken here. The
        actor-critic passes ``−A``; PPO passes ``−mask·ratio·A``. Returns are held constant.

        Returns:
            The gradients keyed as :attr:`_PARAM_NAMES`, and ``V − R`` per step.
        """
        n = len(returns)
        variance = np.exp(2.0 * fwd.log_std)
        residual = actions - fwd.mu  # (n, action_dim)

        # d(log_prob)/d(mu) = residual / variance; d(log_prob)/d(log_std) = residual²/variance − 1
        d_mu = (residual / (variance + 1e-8)) * d_log_p[:, np.newaxis]
        d_z = d_mu * fwd.dmu_dz  # through tanh when the mean is bounded
        d_log_std = (((residual**2) / (variance + 1e-8)) - 1.0) * d_log_p[:, np.newaxis]

        grad_w_mu = (fwd.h.T @ d_z) / n
        grad_b_mu = np.mean(d_z, axis=0)
        grad_log_std = np.mean(d_log_std, axis=0) - self.entropy_coef

        # Critic loss: 0.5 * (V - R)^2
        v_diff = fwd.values - returns  # (n,)
        grad_w_v = (fwd.h.T @ v_diff) * self.value_coef / n
        grad_b_v = float(np.mean(v_diff) * self.value_coef)

        # Backpropagation into the shared hidden layer, through the ReLU
        d_h_actor = d_z @ self._W_mu.T  # (n, hidden_dim)
        d_h_critic = (v_diff[:, np.newaxis] * self.value_coef) @ self._W_v[np.newaxis, :]
        d_h = (d_h_actor + d_h_critic) * (fwd.h > 0) / n

        grads = {
            "W1": fwd.states.T @ d_h,
            "b1": np.sum(d_h, axis=0),
            "W_mu": grad_w_mu,
            "b_mu": grad_b_mu,
            "log_std": grad_log_std,
            "W_v": grad_w_v,
            "b_v": grad_b_v,
        }
        return grads, v_diff

    def save(self, path: str) -> None:
        """
        Save network weights to disk in NumPy ``.npz`` format.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has one of
        these extensions, parameters are written exactly to that filename. If
        *path* has no extension, ``.npz`` is appended.
        """
        save_path = self._resolve_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            arrays: dict[str, Any] = {
                **self._params(),
                "b_v": np.array(self._b_v),
                "bounded_mean": np.array(self.bounded_mean),
                **self._optimizer.state(),
                **self._extra_checkpoint_state(),
            }
            np.savez_compressed(f, **arrays)

    def load(self, path: str) -> None:
        """
        Load network weights from disk.

        Supported extensions are ``.pt`` and ``.npz``. If *path* has no
        extension, ``.npz`` is preferred.

        Raises:
            ValueError: If the checkpoint's ``bounded_mean`` differs from this agent's (a
                checkpoint without the key was saved unbounded): the same weights would act
                differently.
        """
        resolved_path = self._resolve_load_path(path)
        with np.load(resolved_path) as data:
            stored = bool(data["bounded_mean"]) if "bounded_mean" in data else False
            if stored != self.bounded_mean:
                raise ValueError(
                    f"Checkpoint was saved with bounded_mean={stored}, this agent has "
                    f"bounded_mean={self.bounded_mean}"
                )
            for name in self._PARAM_NAMES:
                setattr(self, f"_{name}", data[name])
            self._b_v = float(data["b_v"])
            self._optimizer.load_state(data)
            self._load_extra_checkpoint_state(data)


class ContinuousActorCriticAgent(ContinuousActorCriticBase):
    """
    Advantage Actor-Critic (A2C) agent for continuous control.

    By default each episode makes one update from Monte Carlo returns:
      A(s, a) = G_t - V(s)   (normalised over the episode)
      L_policy = -log_prob(a|s) * A(s, a) - entropy_coef * H
      L_critic = 0.5 * (G_t - V(s))^2

    With ``n_steps`` > 0 the episode is replayed in consecutive chunks of that many steps (the last
    may be shorter). Each chunk takes GAE targets (``gamma``, ``gae_lambda``) from the current critic,
    bootstrapped from the value of the chunk's next state unless the episode truly terminated there
    (the batch's ``terminated`` flag, so a time-limit cut bootstraps), and makes one optimizer step.
    The chunks of one episode were all collected before its first step, so later chunks are slightly
    off-policy; true A2C would need the trainer to call ``update`` every ``n_steps``.

    Config keys, beyond those of :class:`ContinuousActorCriticBase` (all optional):
        n_steps (int): Steps per update within an episode; 0 for one Monte Carlo update per
            episode. Default: 0.
        gae_lambda (float): GAE lambda for ``n_steps`` updates, in [0, 1]. Default: 1.0.
        normalize_advantages (bool): Normalise advantages over each update. Default: True.
        anneal_steps (int): Decay the learning rate linearly to zero over this many trained
            transitions; unset keeps it constant. Default: unset.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(observation_dim, action_dim, config)
        self.n_steps: int = non_negative_int(config, "n_steps", 0)
        self.gae_lambda: float = float(config.get("gae_lambda", 1.0))
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda!r}")
        self.normalize_advantages: bool = bool(config.get("normalize_advantages", True))
        self.anneal_steps: int | None = anneal_steps_from_config(config)
        self._steps_trained = 0

    def update(self, trajectory: Any) -> dict[str, float]:
        """
        Update policy and value networks on one episode.

        Whether the episode truly ended (no bootstrap after its last step) is read from its final
        ``terminated`` flag, or, when the batch has none, from its final ``done``: a batch with
        only ``dones`` treats a time-limit cut as terminal, as before.

        Args:
            trajectory: Batch dictionary ("observations", "actions", "rewards", optionally
                "next_observations", "dones", "terminated") or a list of transition dicts ("state",
                "action", "reward", optionally "next_state", "done", "terminated").

        Returns:
            Means over the updates of ``loss``, ``policy_loss``, ``value_loss`` and ``entropy``;
            with ``bounded_mean`` also ``saturation``, the share of steps whose mean was at a
            bound (|tanh| > 0.99) before training; with ``anneal_steps`` also
            ``anneal_fraction``, the share of the learning rate used at the episode's first update.
        """
        parsed = self._parse_episode(trajectory)
        if parsed is None:
            return {}
        states, actions, rewards, next_obs, terminated_last = parsed
        n = len(rewards)
        rewards = rewards * self.reward_scale
        first_fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        saturation: float | None = None
        if self.bounded_mean:
            _, _, _, hidden = self._forward(states)
            _, _, u = self._mean(hidden)
            saturation = None if u is None else float(np.mean(np.abs(u) > 0.99))

        if self.n_steps == 0:
            returns = discounted_returns(rewards, self.gamma)
            advantages = None
            if not self.normalize_advantages:
                _, _, values, _ = self._forward(states)
                advantages = returns - np.asarray(values, dtype=np.float64)
            chunks = [self._train_step(states, actions, returns, advantages)]
        else:
            chunks = []
            for start in range(0, n, self.n_steps):
                end = min(start + self.n_steps, n)
                if next_obs is not None:
                    bootstrap_obs: np.ndarray | None = next_obs[end - 1]
                else:
                    bootstrap_obs = states[end] if end < n else None
                terminal = end == n and (terminated_last or bootstrap_obs is None)
                advantages, returns = self._chunk_targets(
                    states[start:end], rewards[start:end], bootstrap_obs, terminal
                )
                if self.normalize_advantages and end - start > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                chunks.append(
                    self._train_step(states[start:end], actions[start:end], returns, advantages)
                )

        metrics = {
            "loss": float(np.mean([c["total"] for c in chunks])),
            "policy_loss": float(np.mean([c["policy"] for c in chunks])),
            "value_loss": float(np.mean([c["value"] for c in chunks])),
            "entropy": float(np.mean([c["entropy"] for c in chunks])),
        }
        if saturation is not None:
            metrics["saturation"] = saturation
        if self.anneal_steps is not None:
            metrics["anneal_fraction"] = first_fraction
        return metrics

    def _parse_episode(
        self, trajectory: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool] | None:
        """``(states, actions, raw rewards, next observations or None, terminated at the end)``."""
        if isinstance(trajectory, dict):
            if len(trajectory.get("observations", [])) == 0:
                return None
            next_obs = trajectory.get("next_observations")
            ends = trajectory.get("terminated", trajectory.get("dones"))
            return (
                np.asarray(trajectory["observations"], dtype=np.float64),
                np.asarray(trajectory["actions"], dtype=np.float64),
                np.asarray(trajectory["rewards"], dtype=np.float64),
                None
                if next_obs is None or len(next_obs) == 0
                else np.asarray(next_obs, dtype=np.float64),
                bool(np.asarray(ends)[-1]) if ends is not None and len(ends) > 0 else False,
            )
        if isinstance(trajectory, list) and trajectory:
            last = trajectory[-1]
            has_next = all("next_state" in t for t in trajectory)
            return (
                np.array([t["state"] for t in trajectory], dtype=np.float64),
                np.array([t["action"] for t in trajectory], dtype=np.float64),
                np.array([t["reward"] for t in trajectory], dtype=np.float64),
                np.array([t["next_state"] for t in trajectory], dtype=np.float64)
                if has_next
                else None,
                bool(last.get("terminated", last.get("done", False))),
            )
        return None

    def _chunk_targets(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        bootstrap_obs: np.ndarray | None,
        terminal: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GAE advantages (not normalised) and return targets for one chunk of (scaled) rewards.

        The last step bootstraps from V(``bootstrap_obs``) unless ``terminal`` (or no next state).
        """
        _, _, values_raw, _ = self._forward(states)
        values = np.asarray(values_raw, dtype=np.float64)
        last_value = 0.0
        if not terminal and bootstrap_obs is not None:
            _, _, v_next, _ = self._forward(bootstrap_obs)
            last_value = float(v_next)
        dones = np.zeros(len(rewards), dtype=bool)
        advantages = gae(rewards, values, dones, self.gamma, self.gae_lambda, last_value=last_value)
        return advantages, advantages + values

    def _train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray | None,
    ) -> dict[str, float]:
        """One optimizer step on one update's loss, at the annealed learning rate."""
        fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        components, grads = self._loss_and_grads(states, actions, returns, advantages)
        self._adam_step(grads, self.learning_rate * fraction)
        self._steps_trained += len(returns)
        return components

    def _extra_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"ac_steps_trained": np.array(self._steps_trained)}

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        self._steps_trained = int(data["ac_steps_trained"]) if "ac_steps_trained" in data else 0

    def _loss_and_grads(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Compute the actor-critic loss and its gradients for one episode, changing no state.

        The loss is ``−mean(log π(a_t|s_t)·A_t) + value_coef·½·mean((V(s_t) − G_t)²)
        − entropy_coef·H``, where ``H`` is the entropy of the Gaussian policy summed over action
        dimensions. Returns and advantages are held constant when differentiating.

        Args:
            states:     Array of shape (T, observation_dim).
            actions:    Array of shape (T, action_dim).
            returns:    Array of shape (T,): the Monte-Carlo return G_t of each step.
            advantages: Array of shape (T,) to use as A_t, exactly as given: it is not
                        normalised again. If omitted, ``G_t − V(s_t)`` from the current weights,
                        normalised over the episode, is used, which lets :meth:`update` make a
                        single forward pass. Passing that same array explicitly gives identical
                        results.

        Returns:
            Tuple of (components, gradients). Components holds the ``policy``, ``value`` and
            ``entropy`` terms and their weighted ``total``. Gradients are those of ``total``,
            keyed ``W1``, ``b1``, ``W_mu``, ``b_mu``, ``log_std``, ``W_v``, ``b_v``.

        Raises:
            ValueError: If ``advantages`` is given with a different length from ``returns``.
        """
        n_steps = len(returns)
        if advantages is not None and len(advantages) != n_steps:
            raise ValueError(
                f"advantages has {len(advantages)} entries but the batch has {n_steps} steps"
            )

        fwd = self._batch_forward(states)

        if advantages is None:
            advantages = returns - fwd.values
            # Normalize advantages for training stability
            if n_steps > 1:
                adv_std = np.std(advantages)
                advantages = (advantages - np.mean(advantages)) / (adv_std + 1e-8)

        # The policy loss −mean(log π · A) has d/d(log π) = −A per step.
        grads, v_diff = self._policy_value_grads(actions, returns, fwd, -advantages)

        policy_loss = float(-np.mean(self._log_prob(actions, fwd.mu, fwd.log_std) * advantages))
        value_loss = float(0.5 * np.mean(np.square(v_diff)))
        entropy = self._entropy(fwd.log_std)

        components = {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
            "total": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
        }
        return components, grads
