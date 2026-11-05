"""Deep Deterministic Policy Gradient (DDPG) agent for continuous actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from .exploration_noise import make_exploration_noise
from .inference import DenseLayer, InferencePolicy
from .mlp import Params, init_mlp, layer_count, mlp_backward, mlp_forward
from .off_policy import Array, OffPolicyAgent, add_metrics, global_norm, mean_metrics
from .optim import Adam


class DDPGAgent(OffPolicyAgent):
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
      then ``mu(s)`` plus noise in the scaled [-1, 1] space, clipped to the bounds: Gaussian of
      ``noise_std`` by default, or with ``noise_type: ornstein_uhlenbeck`` SB3's OU process of
      sigma ``noise_std`` (``ou_theta``, ``ou_dt``; see ``rl_core.agents.exploration_noise``).
      :meth:`end_episode` resets the noise; the Trainer calls it after every training episode.
      The clipped action is stored: the critic learns the value of what was executed.

    Defaults are rl-zoo's Pendulum-v1 settings. The config keys, warm-up, the replay buffer, when
    :meth:`update` trains, the critic loss, Polyak averaging and checkpoints are
    ``OffPolicyAgent``'s; this class adds the deterministic actor, its target network and the
    exploration noise.
    """

    BACKENDS: ClassVar[dict[str, str]] = {"torch": "rl_core.agents.torch_ddpg_agent:TorchDDPGAgent"}
    DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "hidden_sizes": [400, 300],
        "learning_rate": 1e-3,
        "gamma": 0.98,
        "tau": 0.005,
        "batch_size": 256,
        "buffer_capacity": 200_000,
        "learning_starts": 10_000,
        "updates_per_step": 1.0,
    }
    SHORT_NAME: ClassVar[str] = "DDPG"

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
        self.noise_std = float(config.get("noise_std", 0.1))
        self._noise = make_exploration_noise(config, action_dim, default_std=self.noise_std)

    def _build_networks(self, weights_rng: np.random.Generator) -> None:
        """The actor and critic drawn from the weights stream, in that order, their targets as
        copies and an Adam for each. A subclass with more networks draws them after these."""
        obs_dim, act_dim = self.observation_dim, self.action_dim
        self._actor: Params = init_mlp([obs_dim, *self.hidden_sizes, act_dim], weights_rng)
        self._critic: Params = init_mlp([obs_dim + act_dim, *self.hidden_sizes, 1], weights_rng)
        self._target_actor: Params = {k: v.copy() for k, v in self._actor.items()}
        self._target_critic: Params = {k: v.copy() for k, v in self._critic.items()}
        self._actor_optimizer = Adam(self._actor, self.max_grad_norm)
        self._critic_optimizer = Adam(self._critic, self.max_grad_norm)

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

    def _bellman_targets(self, rewards: Array, next_obs: Array, terminated: Array) -> Array:
        """``r + gamma (1 - terminated) Q'(s', mu'(s'))`` from the target networks, shape (B,)."""
        next_actions, _, _ = self._actor_forward(next_obs, self._target_actor)
        next_q, _ = self._critic_forward(next_obs, next_actions, self._target_critic)
        not_done = 1.0 - np.asarray(terminated, dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64) + self.gamma * not_done * next_q

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
            "actor_grad_norm": global_norm(grads),
        }
        return components, grads

    def _critic_names(self) -> tuple[str, ...]:
        return ("critic",)

    def _target_names(self) -> list[tuple[str, str]]:
        return [("actor", "target_actor"), ("critic", "target_critic")]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _greedy_action(self, obs: Array) -> Array:
        """``mu(s)``."""
        return self._actor_forward(obs, self._actor)[0][0]

    def _training_action(self, obs: Array) -> Array:
        """``mu(s)`` plus the exploration noise times ``scale``, clipped to the bounds."""
        mu = self._actor_forward(obs, self._actor)[0][0]
        noise = self._noise.sample(self._explore_rng) * self.scale
        return np.clip(mu + noise, self.action_low, self.action_high)

    def end_episode(self) -> None:
        """Restart the exploration noise: OU noise starts the next episode from zero."""
        self._noise.reset()

    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """
        ``num_updates`` DDPG steps, each: critic step, actor step, Polyak.

        Returns:
            Means over the steps of ``critic_loss``, ``actor_loss``, ``mean_q``, ``saturation``
            (share of actor outputs with |tanh| > 0.99) and the global gradient norms
            ``critic_grad_norm`` and ``actor_grad_norm``.
        """
        totals: dict[str, float] = {}
        for _ in range(num_updates):
            sample = self.replay_buffer.sample(self.batch_size)
            targets = self._bellman_targets(
                sample["rewards"], sample["next_observations"], sample["dones"]
            )
            critic = self._critic_step(sample["observations"], sample["actions"], targets)
            actor, actor_grads = self._actor_loss_and_grads(sample["observations"])
            self._apply(self._actor, self._actor_optimizer, actor_grads)
            self._polyak()
            self._gradient_steps += 1
            add_metrics(totals, {**critic, **actor})
        return mean_metrics(totals, num_updates)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _networks(self) -> dict[str, Params]:
        """Every network by name, targets included, in checkpoint order."""
        return {
            "actor": self._actor,
            "critic": self._critic,
            "target_actor": self._target_actor,
            "target_critic": self._target_critic,
        }

    def _optimizers_by_net(self) -> dict[str, Adam]:
        """The Adam of every trained network, by the network's name in :meth:`_networks`. Another
        backend trains these same networks and mirrors its optimizer state into these."""
        return {"actor": self._actor_optimizer, "critic": self._critic_optimizer}

    def _inference_network(self) -> InferencePolicy:
        """The actor: ReLU on every layer but the last, then ``bias + scale · tanh``."""
        actor = self._networks()["actor"]
        count = layer_count(actor)
        return InferencePolicy(
            layers=tuple(
                DenseLayer(actor[f"W{i}"], actor[f"b{i}"], relu=i < count - 1) for i in range(count)
            ),
            head="tanh_scale",
            action_low=self.action_low,
            action_high=self.action_high,
        )
