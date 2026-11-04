"""Twin Delayed DDPG (TD3) agent for continuous actions."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from rl_core.seeding import AGENT_TARGET_NOISE_STREAM, derive_rng
from rl_core.utils.config_values import positive_int

from .ddpg_agent import DDPGAgent
from .mlp import Params, init_mlp
from .off_policy import Array, add_metrics, mean_metrics
from .optim import Adam


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
    trains, :meth:`end_episode`, checkpoints (a DDPG checkpoint, without the second critic, is
    refused). For the same seed the actor and the first critic start as
    DDPG's; the second critic is drawn after them. Each critic has its own Adam, as one Adam over
    both would be without clipping; ``max_grad_norm`` (SB3's TD3 has none) clips each critic on
    its own. ``critic_loss`` is the mean of the two critics' losses, comparable with DDPG's.
    """

    BACKENDS: ClassVar[dict[str, str]] = {"torch": "rl_core.agents.torch_td3_agent:TorchTD3Agent"}
    SHORT_NAME: ClassVar[str] = "TD3"

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

    def _optimizers_by_net(self) -> dict[str, Adam]:
        return {**super()._optimizers_by_net(), "critic_2": self._critic_2_optimizer}

    def _critic_names(self) -> tuple[str, ...]:
        return ("critic", "critic_2")

    def _target_names(self) -> list[tuple[str, str]]:
        return [*super()._target_names(), ("critic_2", "target_critic_2")]

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

    def _is_actor_step(self, gradient_step: int) -> bool:
        """Whether gradient step ``gradient_step`` (counted from 1 over the agent's life) also
        updates the actor and the targets; every backend's TD3 asks this."""
        return gradient_step % self.policy_delay == 0

    @staticmethod
    def _step_means(
        critic_totals: dict[str, float],
        actor_totals: dict[str, float],
        num_updates: int,
        actor_updates: int,
    ) -> dict[str, float]:
        """What :meth:`update` returns from the summed per-step metrics: critic metrics over
        every step, actor metrics over the actor steps (absent without one), and
        ``actor_updates``. Every backend's TD3 reduces its sums with this."""
        metrics = mean_metrics(critic_totals, num_updates)
        if actor_updates:
            metrics.update(mean_metrics(actor_totals, actor_updates))
        metrics["actor_updates"] = float(actor_updates)
        return metrics

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
            add_metrics(critic_totals, self._critic_step(obs, actions, targets))
            if self._is_actor_step(self._gradient_steps):
                actor, actor_grads = self._actor_loss_and_grads(obs)
                self._apply(self._actor, self._actor_optimizer, actor_grads)
                self._polyak()
                actor_updates += 1
                add_metrics(actor_totals, actor)
        return self._step_means(critic_totals, actor_totals, num_updates, actor_updates)
