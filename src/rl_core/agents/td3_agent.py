"""Twin Delayed DDPG (TD3) agent for continuous actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rl_core.seeding import AGENT_TARGET_NOISE_STREAM, derive_rng

from .config_values import positive_int
from .ddpg_agent import Array, DDPGAgent
from .mlp import Params, init_mlp
from .optim import Adam

TD3_NETWORKS = ("critic_2", "target_critic_2")


class TD3Agent(DDPGAgent):
    """
    TD3 (Fujimoto et al., 2018) as Stable-Baselines3 implements it: DDPG with three changes.

    * Twin critics: ``y = r + gamma * (1 - terminated) * min(Q1'(s', a'), Q2'(s', a'))``, and each
      critic minimises its own ``mean ½ (Q_i(s, a) - y)²``.
    * Target-policy smoothing: ``a' = clip(mu'(s') + scale * clip(eps, -c, c), low, high)`` with
      ``eps ~ N(0, target_policy_noise²)`` per action element, in the scaled [-1, 1] space as SB3
      draws it, from its own stream of the seed (so exploration and replay draw as DDPG's do).
    * Delayed updates: gradient steps are counted before each is taken; on every
      ``policy_delay``-th the actor minimises ``-mean Q1(s, mu(s))`` and the target actor and both
      target critics move a fraction ``tau``. The count is checkpointed, so a resumed run keeps
      the rhythm.

    Everything else is DDPGAgent's: exploration, warm-up, the replay buffer, when :meth:`update`
    trains, :meth:`end_episode`. For the same seed the actor and the first critic start as
    DDPG's; the second critic is drawn after them. Each critic has its own Adam, as one Adam over
    both would be without clipping; ``max_grad_norm`` (SB3's TD3 has none) clips each critic on
    its own. ``critic_loss`` is the mean of the two critics' losses, comparable with DDPG's.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Args:
            config: DDPGAgent's keys, plus ``policy_delay`` (default 2), ``target_policy_noise``
                (0.2) and ``target_noise_clip`` (0.5), SB3's defaults; see
                ``configs/pendulum_td3.yaml``.

        Raises:
            ValueError: For an invalid hyperparameter.
        """
        super().__init__(observation_dim, action_dim, config)
        self.policy_delay = positive_int(config, "policy_delay", 2)
        self.target_policy_noise = float(config.get("target_policy_noise", 0.2))
        self.target_noise_clip = float(config.get("target_noise_clip", 0.5))
        for name in ("target_policy_noise", "target_noise_clip"):
            if not 0.0 <= getattr(self, name) < float("inf"):  # NaN fails this too
                raise ValueError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
        self._target_noise_rng = derive_rng(config.get("seed"), AGENT_TARGET_NOISE_STREAM)

    def _build_networks(self, weights_rng: np.random.Generator) -> None:
        super()._build_networks(weights_rng)
        sizes = [self.observation_dim + self.action_dim, *self.hidden_sizes, 1]
        self._critic_2: Params = init_mlp(sizes, weights_rng)
        self._target_critic_2: Params = {k: v.copy() for k, v in self._critic_2.items()}
        self._critic_2_optimizer = Adam(self._critic_2, self.max_grad_norm)

    def _networks(self) -> dict[str, Params]:
        return {
            **super()._networks(),
            "critic_2": self._critic_2,
            "target_critic_2": self._target_critic_2,
        }

    def _target_pairs(self) -> list[tuple[Params, Params]]:
        return [*super()._target_pairs(), (self._critic_2, self._target_critic_2)]

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _draw_target_noise(self, batch_size: int) -> Array:
        """One step's target-policy noise, (batch_size, action_dim), in the scaled space."""
        eps = self._target_noise_rng.normal(
            0.0, self.target_policy_noise, (batch_size, self.action_dim)
        )
        return np.clip(eps, -self.target_noise_clip, self.target_noise_clip)

    def _target_actions(self, next_obs: Array, noise: Array) -> Array:
        """The target actor's actions plus ``scale * noise``, clipped to the bounds."""
        mu, _, _ = self._actor_forward(next_obs, self._target_actor)
        return np.clip(mu + self.scale * noise, self.action_low, self.action_high)

    def _td3_targets(
        self, rewards: Array, next_obs: Array, terminated: Array, noise: Array
    ) -> Array:
        """``r + gamma (1 - terminated) min(Q1', Q2')`` at the smoothed target actions, (B,)."""
        next_actions = self._target_actions(next_obs, noise)
        q1, _ = self._critic_forward(next_obs, next_actions, self._target_critic)
        q2, _ = self._critic_forward(next_obs, next_actions, self._target_critic_2)
        not_done = 1.0 - np.asarray(terminated, dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64) + self.gamma * not_done * np.minimum(q1, q2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """
        ``num_updates`` TD3 steps. Returns means over the steps of ``critic_loss`` (the mean of
        both critics'), ``mean_q`` (the first critic's) and ``critic_grad_norm`` (both critics
        together), and over the actor steps of ``actor_loss``, ``saturation`` and
        ``actor_grad_norm``, which are absent from a call without one; ``actor_updates`` counts
        the actor steps.
        """
        critic_totals: dict[str, float] = {}
        actor_totals: dict[str, float] = {}
        actor_updates = 0
        for _ in range(num_updates):
            self._gradient_steps += 1
            sample = self.replay_buffer.sample(self.batch_size)
            obs, actions = sample["observations"], sample["actions"]
            noise = self._draw_target_noise(self.batch_size)
            targets = self._td3_targets(
                sample["rewards"], sample["next_observations"], sample["dones"], noise
            )
            first, grads_1 = self._critic_loss_and_grads(obs, actions, targets, self._critic)
            second, grads_2 = self._critic_loss_and_grads(obs, actions, targets, self._critic_2)
            self._apply(self._critic, self._critic_optimizer, grads_1)
            self._apply(self._critic_2, self._critic_2_optimizer, grads_2)
            step = {
                "critic_loss": (first["critic_loss"] + second["critic_loss"]) / 2.0,
                "mean_q": first["mean_q"],
                "critic_grad_norm": float(
                    np.hypot(first["critic_grad_norm"], second["critic_grad_norm"])
                ),
            }
            for name, value in step.items():
                critic_totals[name] = critic_totals.get(name, 0.0) + value
            if self._gradient_steps % self.policy_delay == 0:
                actor, actor_grads = self._actor_loss_and_grads(obs)
                self._apply(self._actor, self._actor_optimizer, actor_grads)
                self._polyak()
                actor_updates += 1
                for name, value in actor.items():
                    actor_totals[name] = actor_totals.get(name, 0.0) + value
        metrics = {name: total / num_updates for name, total in critic_totals.items()}
        if actor_updates:
            metrics.update({name: total / actor_updates for name, total in actor_totals.items()})
        metrics["actor_updates"] = float(actor_updates)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _checkpoint_arrays(self) -> dict[str, Any]:
        """DDPGAgent's arrays (all networks, through :meth:`_networks`) plus the second critic's
        Adam state; the target-noise generator is saved with the others."""
        arrays = super()._checkpoint_arrays()
        arrays.update(self._critic_2_optimizer.state(prefix="critic_2_adam_"))
        return arrays

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """
        Raises:
            ValueError: If the checkpoint has no second critic (a DDPG checkpoint), or a stored
                network has a different shape.
        """
        missing = [
            f"{net}_{name}"
            for net in TD3_NETWORKS
            for name in self._networks()[net]
            if f"{net}_{name}" not in data
        ]
        if missing:
            more = ", ..." if len(missing) > 2 else ""
            raise ValueError(f"not a TD3 checkpoint: it has no {', '.join(missing[:2])}{more}")
        super()._load_checkpoint_arrays(data)
        self._critic_2_optimizer.load_state(data, prefix="critic_2_adam_")
