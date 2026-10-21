"""Deep Deterministic Policy Gradient (DDPG) agent for continuous actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from rl_core.seeding import (
    AGENT_EXPLORATION_STREAM,
    AGENT_WEIGHTS_STREAM,
    derive_rng,
)

from .base_agent import BaseAgent
from .bounds import action_bounds
from .config_values import (
    check_positive_int,
    non_negative_int,
    optional_positive_int,
    positive_int,
)
from .mlp import Params, init_mlp, mlp_backward, mlp_forward
from .optim import Adam
from .replay_buffer import ReplayBuffer

Array = npt.NDArray[np.float64]


def _global_norm(grads: Params) -> float:
    """The L2 norm of all gradients together."""
    return float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))


class DDPGAgent(BaseAgent):
    """
    DDPG (Lillicrap et al., 2016) as Stable-Baselines3 implements it and rl-baselines3-zoo
    configures it for Pendulum-v1.

    * Actor ``mu(s) = bias + scale * tanh(mlp(s))``, scaled to the action bounds, with
      ``scale = (high - low) / 2`` and ``bias = (high + low) / 2``.
    * Critic ``Q(s, a) = mlp([s, a])``: the action joins the observation at the input.
    * Target ``y = r + gamma * (1 - terminated) * Q'(s', mu'(s'))`` from Polyak-averaged target
      networks; a time-limit cut is not terminal, so Pendulum always bootstraps.
    * Per gradient step: the critic minimises ``mean ½ (Q(s, a) - y)²``, then the actor minimises
      ``-mean Q(s, mu(s))`` against the updated critic, then both targets move a fraction ``tau``.
    * Exploration: uniform random actions for the first ``learning_starts`` training actions,
      then ``mu(s)`` plus Gaussian noise of ``noise_std`` in the scaled [-1, 1] space, clipped to
      the bounds. The clipped action is stored: the critic learns the value of what was executed.

    Defaults are rl-zoo's Pendulum-v1 settings. Weights, exploration and replay sampling each
    draw from their own stream of the seed (``rl_core.seeding``).
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Args:
            observation_dim: Length of an observation vector.
            action_dim: Length of an action vector.
            config: Hyperparameters; see the class docstring and ``configs/pendulum_ddpg.yaml``.

        Raises:
            ValueError: For a non-positive dimension or an invalid hyperparameter.
        """
        super().__init__(observation_dim, action_dim, config)
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"observation_dim and action_dim must be positive, got {observation_dim!r} "
                f"and {action_dim!r}"
            )

        raw_sizes = config.get("hidden_sizes", [400, 300])
        if np.ndim(raw_sizes) != 1:
            raise ValueError(f"hidden_sizes must be a list of widths, got {raw_sizes!r}")
        if len(raw_sizes) == 0:
            raise ValueError(f"hidden_sizes must hold at least one width, got {raw_sizes!r}")
        self.hidden_sizes: list[int] = [
            check_positive_int(f"hidden_sizes[{i}]", width) for i, width in enumerate(raw_sizes)
        ]
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.gamma = float(config.get("gamma", 0.98))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = positive_int(config, "batch_size", 256)
        self.buffer_capacity = positive_int(config, "buffer_capacity", 200_000)
        self.learning_starts = non_negative_int(config, "learning_starts", 10_000)
        self.noise_std = float(config.get("noise_std", 0.1))
        self.updates_per_step = float(config.get("updates_per_step", 1.0))
        self.max_updates_per_call: int | None = optional_positive_int(
            config, "max_updates_per_call"
        )
        raw_clip = config.get("max_grad_norm")
        self.max_grad_norm: float | None = None if raw_clip is None else float(raw_clip)
        self._validate(str(config.get("optimizer", "adam")).lower())

        self.action_low, self.action_high = action_bounds(config, action_dim, require_finite=True)
        low, high = self.action_low.astype(np.float64), self.action_high.astype(np.float64)
        self.scale: Array = (high - low) / 2.0
        self.bias: Array = (high + low) / 2.0

        seed = config.get("seed")
        weights_rng = derive_rng(seed, AGENT_WEIGHTS_STREAM)
        self._actor: Params = init_mlp(
            [observation_dim, *self.hidden_sizes, action_dim], weights_rng
        )
        self._critic: Params = init_mlp(
            [observation_dim + action_dim, *self.hidden_sizes, 1], weights_rng
        )
        self._target_actor: Params = {k: v.copy() for k, v in self._actor.items()}
        self._target_critic: Params = {k: v.copy() for k, v in self._critic.items()}
        self._actor_optimizer = Adam(self._actor, self.max_grad_norm)
        self._critic_optimizer = Adam(self._critic, self.max_grad_norm)

        self._explore_rng = derive_rng(seed, AGENT_EXPLORATION_STREAM)
        # The buffer samples on the replay stream of the seed itself.
        self.replay_buffer = ReplayBuffer(
            self.buffer_capacity, observation_dim, seed=seed, action_dim=action_dim
        )
        self._training_actions = 0
        self._gradient_steps = 0

    def _validate(self, optimizer: str) -> None:
        """Raise ValueError for any hyperparameter outside its range."""
        if optimizer != "adam":
            raise ValueError(f"DDPG trains with Adam; optimizer must be 'adam', got {optimizer!r}")
        checks = [
            ("learning_rate", self.learning_rate > 0.0),
            ("gamma", 0.0 < self.gamma <= 1.0),
            ("tau", 0.0 < self.tau <= 1.0),
            ("noise_std", self.noise_std >= 0.0),
            ("updates_per_step", self.updates_per_step > 0.0),
            ("max_grad_norm", self.max_grad_norm is None or self.max_grad_norm > 0.0),
        ]
        for name, ok in checks:
            if not ok:
                raise ValueError(f"{name} is out of range: {getattr(self, name)!r}")

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------

    def _actor_forward(
        self, obs: Array, params: Params
    ) -> tuple[Array, Array, dict[str, list[Array]]]:
        """``(mu, u, cache)`` for observations (B, O): u = tanh(mlp(s)) and mu = bias + scale·u."""
        z, cache = mlp_forward(params, obs)
        u = np.tanh(z)
        return self.bias + self.scale * u, u, cache

    def _critic_forward(
        self, obs: Array, actions: Array, params: Params
    ) -> tuple[Array, dict[str, list[Array]]]:
        """``(Q, cache)``: Q of shape (B,) for observations (B, O) and actions (B, A)."""
        q, cache = mlp_forward(params, np.concatenate([obs, actions], axis=1))
        return q[:, 0], cache

    def _bellman_targets(self, rewards: Array, next_obs: Array, terminated: Array) -> Array:
        """``r + gamma (1 - terminated) Q'(s', mu'(s'))`` from the target networks, shape (B,)."""
        next_actions, _, _ = self._actor_forward(next_obs, self._target_actor)
        next_q, _ = self._critic_forward(next_obs, next_actions, self._target_critic)
        not_done = 1.0 - np.asarray(terminated, dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64) + self.gamma * not_done * next_q

    def _critic_loss_and_grads(
        self, obs: Array, actions: Array, targets: Array
    ) -> tuple[dict[str, float], Params]:
        """
        Critic loss ``mean ½ (Q(s, a) - y)²`` and its gradients, with ``targets`` constant.

        Changes no state.
        """
        q, cache = self._critic_forward(obs, actions, self._critic)
        td_error = q - targets
        d_q = (td_error / len(q))[:, None]
        grads, _ = mlp_backward(self._critic, cache, d_q)
        components = {
            "critic_loss": float(np.mean(0.5 * td_error**2)),
            "mean_q": float(np.mean(q)),
            "critic_grad_norm": _global_norm(grads),
        }
        return components, grads

    def _actor_loss_and_grads(self, obs: Array) -> tuple[dict[str, float], Params]:
        """
        Actor loss ``-mean Q(s, mu(s))`` and its gradients for the actor, critic held constant.

        The gradient reaches the actor through the critic's action input:
        ``dL/da = -(1/B) dQ/da``, then ``dz = dL/da · scale · (1 - u²)``. Changes no state.
        """
        mu, u, actor_cache = self._actor_forward(obs, self._actor)
        q, critic_cache = self._critic_forward(obs, mu, self._critic)
        d_q = np.full((len(q), 1), -1.0 / len(q))
        _, d_input = mlp_backward(self._critic, critic_cache, d_q)
        d_mu = d_input[:, self.observation_dim :]
        grads, _ = mlp_backward(self._actor, actor_cache, d_mu * self.scale * (1.0 - u**2))
        components = {
            "actor_loss": float(-np.mean(q)),
            "saturation": float(np.mean(np.abs(u) > 0.99)),
            "actor_grad_norm": _global_norm(grads),
        }
        return components, grads

    def _apply(self, params: Params, optimizer: Adam, grads: Params) -> None:
        """One Adam step on ``params`` in place."""
        for name, step in optimizer.step(grads, self.learning_rate).items():
            params[name] -= step

    def _polyak(self) -> None:
        """Move each target parameter a fraction ``tau`` towards its online parameter."""
        for online, target in (
            (self._actor, self._target_actor),
            (self._critic, self._target_critic),
        ):
            for name in online:
                target[name] = self.tau * online[name] + (1.0 - self.tau) * target[name]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        A float32 action of shape (action_dim,) inside the bounds.

        Evaluation (``training=False``) returns ``mu(s)`` and changes nothing. A training action
        counts towards ``learning_starts``: uniform in the bounds during warm-up, then ``mu(s)``
        plus Gaussian noise (``noise_std`` times ``scale``), clipped to the bounds.
        """
        obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        if not training:
            return self._actor_forward(obs, self._actor)[0][0].astype(np.float32)

        self._training_actions += 1
        if self._training_actions <= self.learning_starts:
            action = self._explore_rng.uniform(self.action_low, self.action_high)
        else:
            mu = self._actor_forward(obs, self._actor)[0][0]
            noise = self._explore_rng.normal(0.0, self.noise_std, self.action_dim) * self.scale
            action = np.clip(mu + noise, self.action_low, self.action_high)
        return np.asarray(action, dtype=np.float32)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Store the episode's transitions, then train once warm-up is over.

        Runs ``updates_per_step`` gradient steps per transition (at least one, at most
        ``max_updates_per_call``), each: critic step, actor step, Polyak.

        Returns:
            Means over the gradient steps of ``critic_loss``, ``actor_loss``, ``mean_q``,
            ``saturation`` (share of actor outputs with |tanh| > 0.99) and the global gradient
            norms ``critic_grad_norm`` and ``actor_grad_norm``; empty during warm-up or when the
            batch stored nothing.
        """
        n = self.replay_buffer.add_transitions(batch)
        if (
            n == 0
            or self._training_actions < self.learning_starts
            or not self.replay_buffer.can_sample(self.batch_size)
        ):
            return {}

        num_updates = max(1, int(n * self.updates_per_step))
        if self.max_updates_per_call is not None:
            num_updates = min(num_updates, self.max_updates_per_call)

        totals: dict[str, float] = {}
        for _ in range(num_updates):
            sample = self.replay_buffer.sample(self.batch_size)
            targets = self._bellman_targets(
                sample["rewards"], sample["next_observations"], sample["dones"]
            )
            critic, critic_grads = self._critic_loss_and_grads(
                sample["observations"], sample["actions"], targets
            )
            self._apply(self._critic, self._critic_optimizer, critic_grads)
            actor, actor_grads = self._actor_loss_and_grads(sample["observations"])
            self._apply(self._actor, self._actor_optimizer, actor_grads)
            self._polyak()
            self._gradient_steps += 1
            for name, value in {**critic, **actor}.items():
                totals[name] = totals.get(name, 0.0) + value
        return {name: total / num_updates for name, total in totals.items()}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _networks(self) -> dict[str, Params]:
        return {
            "actor": self._actor,
            "critic": self._critic,
            "target_actor": self._target_actor,
            "target_critic": self._target_critic,
        }

    def _checkpoint_arrays(self) -> dict[str, Any]:
        """Everything training needs to resume beyond the generators: the four networks, both
        Adam states, the action and gradient-step counters and the replay buffer."""
        arrays: dict[str, Any] = {
            f"{net}_{name}": value
            for net, params in self._networks().items()
            for name, value in params.items()
        }
        arrays.update(self._actor_optimizer.state(prefix="actor_adam_"))
        arrays.update(self._critic_optimizer.state(prefix="critic_adam_"))
        arrays["training_actions"] = np.array(self._training_actions)
        arrays["gradient_steps"] = np.array(self._gradient_steps)
        arrays.update(self.replay_buffer.checkpoint_state())
        return arrays

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """
        Restore a checkpoint; one saved before the replay buffer was saved empties the buffer.

        Raises:
            ValueError: If a stored network has a different shape (other ``hidden_sizes``).
        """
        for net, params in self._networks().items():
            for name in params:
                stored = data[f"{net}_{name}"]
                if stored.shape != params[name].shape:
                    raise ValueError(
                        f"{net}_{name} has shape {stored.shape}, expected "
                        f"{params[name].shape}: the checkpoint has other hidden_sizes, "
                        "observation_dim or action_dim"
                    )
        self.replay_buffer.restore(data)
        for net, params in self._networks().items():
            for name in params:
                params[name] = np.array(data[f"{net}_{name}"], dtype=np.float64)
        self._actor_optimizer.load_state(data, prefix="actor_adam_")
        self._critic_optimizer.load_state(data, prefix="critic_adam_")
        self._training_actions = int(data["training_actions"])
        self._gradient_steps = int(data["gradient_steps"])
