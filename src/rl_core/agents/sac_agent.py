"""Soft Actor-Critic (SAC) agent for continuous actions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

import numpy as np

from rl_core.seeding import AGENT_POLICY_NOISE_STREAM, derive_rng

from .continuous_actor_critic import LOG_STD_MAX, LOG_STD_MIN
from .inference import InferencePolicy
from .mlp import Params, init_mlp, mlp_backward, mlp_forward
from .off_policy import Array, OffPolicyAgent, add_metrics, global_norm, mean_metrics
from .optim import Adam

HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
LOG_2 = math.log(2.0)


def log_one_minus_tanh_squared(u: Array) -> Array:
    """
    ``log(1 - tanh(u)²)``, elementwise, without forming ``1 - tanh(u)²``.

    Uses ``1 - tanh² u = 4 / (e^u + e^-u)²``, so the log is ``2 (log 2 - u - softplus(-2u))``,
    finite for every finite u where the direct form reaches ``log 0 = -inf`` once tanh rounds
    to ±1 (|u| above about 19 in float64). SB3 writes ``log(1 - tanh² u + 1e-6)`` instead.
    """
    return 2.0 * (LOG_2 - u - np.logaddexp(0.0, -2.0 * u))


class PolicySample(NamedTuple):
    """One pass of the squashed Gaussian actor for a batch (B rows, A action elements)."""

    mean: Array  # m, (B, A): the first A outputs of the actor
    log_std_raw: Array  # the last A outputs, (B, A), before the clip
    log_std: Array  # l = clip(log_std_raw, LOG_STD_MIN, LOG_STD_MAX), (B, A)
    eps: Array  # the standard normal noise, (B, A)
    pre_tanh: Array  # u = m + e^l · eps, (B, A)
    squashed: Array  # t = tanh(u), (B, A)
    action: Array  # a = bias + scale · t, (B, A), inside the bounds
    log_prob: Array  # log pi(t) of the squashed value in [-1, 1], (B,)
    cache: dict[str, list[Array]]  # the actor's forward cache for mlp_backward


def _parse_ent_coef(raw: Any) -> tuple[bool, float]:
    """``(learned, initial alpha)`` for the ``ent_coef`` setting: ``"auto"`` learns alpha from 1,
    ``"auto_<x>"`` learns it from x, and a number (or a numeric string) fixes it."""
    if isinstance(raw, bool):
        raise ValueError(f"ent_coef must be 'auto', 'auto_<x>' or a positive number, got {raw!r}")
    learned, text = False, raw
    if isinstance(raw, str) and raw.startswith("auto"):
        learned, text = True, raw[len("auto") :]
        if text == "":
            return True, 1.0
        if not text.startswith("_"):
            raise ValueError(f"ent_coef must be 'auto' or 'auto_<x>', got {raw!r}")
        text = text[1:]
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"ent_coef must be 'auto', 'auto_<x>' or a positive number, got {raw!r}"
        ) from None
    if not 0.0 < value < math.inf:  # NaN fails this too
        raise ValueError(f"ent_coef must be positive and finite, got {raw!r}")
    return learned, value


def _parse_target_entropy(raw: Any, action_dim: int) -> float:
    """The target entropy: ``"auto"`` is ``-action_dim``; otherwise a finite number."""
    if isinstance(raw, str) and raw == "auto":
        return -float(action_dim)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
        raise ValueError(f"target_entropy must be 'auto' or a number, got {raw!r}")
    if not math.isfinite(float(raw)):
        raise ValueError(f"target_entropy must be finite, got {raw!r}")
    return float(raw)


class SACAgent(OffPolicyAgent):
    """
    SAC (Haarnoja et al., 2018, with the learned temperature of their follow-up paper) as
    Stable-Baselines3 implements it.

    * Actor: one MLP with 2A outputs, the mean ``m`` and the log standard deviation ``l`` (clipped
      to [LOG_STD_MIN, LOG_STD_MAX], zero gradient where clipped or on a bound). A sample is
      ``u = m + e^l · eps`` with ``eps ~ N(0, I)``, ``t = tanh(u)``, ``a = bias + scale · t``.
      ``log pi`` is the density of ``t`` in [-1, 1] (SB3's), so the target entropy ``-A`` means
      what it means there: ``sum(-eps²/2 - l - log(2 pi)/2) - sum log(1 - t²)``, the last term in
      the stable form of :func:`log_one_minus_tanh_squared` (SB3 adds 1e-6 inside the log).
    * Twin critics ``Q_i(s, a) = mlp([s, a])`` with Polyak-averaged targets; no actor target.
      ``y = r + gamma (1 - terminated) (min_i Q'_i(s', a') - alpha log pi(a'|s'))``, with ``a'``
      drawn from the online actor; each critic minimises ``mean ½ (Q_i - y)²``.
    * Actor loss ``mean(alpha log pi(a|s) - min(Q1, Q2)(s, a))`` on the reparameterised sample,
      against the critics just updated.
    * Temperature: with ``ent_coef: auto`` (or ``auto_<x>``, starting from x rather than 1),
      ``log alpha`` minimises ``-log alpha · mean(log pi + target_entropy)`` by Adam at the
      learning rate, unclipped; ``target_entropy: auto`` is ``-A``. A number fixes alpha.
    * Per gradient step, SB3's order: sample, the actor's noise then the next state's (both from
      their own stream of the seed), log pi at the current actor, the temperature step (alpha
      from before it is used for the rest of the step), the soft targets, both critics, the actor,
      then Polyak on the target critics.
    * Acting: uniform random actions for the first ``learning_starts`` training actions, then a
      sample of the policy (noise from the exploration stream); it lies in the bounds and is what
      is stored. Evaluation acts ``bias + scale · tanh(m)``.

    ``log alpha`` is a parameter group of shape (1,), ``log_ent_coef``, saved in every checkpoint
    with its Adam even when alpha is fixed (then never trained), so a TD3 or DDPG file is refused
    for lacking it and a SAC file is refused by them for lacking the actor target. Loading restores
    the saved alpha. ``max_grad_norm`` clips the actor and each critic, never the temperature.

    Defaults are SB3's SAC defaults; ``configs/pendulum_sac.yaml`` is rl-zoo's Pendulum-v1 run.
    """

    DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "hidden_sizes": [256, 256],
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "buffer_capacity": 1_000_000,
        "learning_starts": 100,
        "updates_per_step": 1.0,
    }
    SHORT_NAME: ClassVar[str] = "SAC"

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Args:
            observation_dim: Length of an observation vector.
            action_dim: Length of an action vector.
            config: ``OffPolicyAgent``'s keys, plus ``ent_coef`` (``"auto"``, ``"auto_<x>"`` or a
                positive number; default ``"auto"``) and ``target_entropy`` (``"auto"`` or a
                number; default ``"auto"``); see ``configs/pendulum_sac.yaml``.

        Raises:
            ValueError: For a non-positive dimension or an invalid hyperparameter.
        """
        super().__init__(observation_dim, action_dim, config)
        self._policy_noise_rng = derive_rng(config.get("seed"), AGENT_POLICY_NOISE_STREAM)

    def _validate(self, config: Mapping[str, Any]) -> None:
        super()._validate(config)
        self.learns_ent_coef, self._initial_ent_coef = _parse_ent_coef(
            config.get("ent_coef", "auto")
        )
        self.target_entropy = _parse_target_entropy(
            config.get("target_entropy", "auto"), self.action_dim
        )

    def _build_networks(self, weights_rng: np.random.Generator) -> None:
        """The actor, critic 1 and critic 2 from the weights stream in that order, the target
        critics as copies, ``log alpha`` from ``ent_coef``, and an Adam for each trained group."""
        obs_dim, act_dim = self.observation_dim, self.action_dim
        critic_sizes = [obs_dim + act_dim, *self.hidden_sizes, 1]
        self._actor: Params = init_mlp([obs_dim, *self.hidden_sizes, 2 * act_dim], weights_rng)
        self._critic: Params = init_mlp(critic_sizes, weights_rng)
        self._critic_2: Params = init_mlp(critic_sizes, weights_rng)
        self._target_critic: Params = {k: v.copy() for k, v in self._critic.items()}
        self._target_critic_2: Params = {k: v.copy() for k, v in self._critic_2.items()}
        self._log_ent_coef: Params = {"value": np.array([math.log(self._initial_ent_coef)])}
        self._actor_optimizer = Adam(self._actor, self.max_grad_norm)
        self._critic_optimizer = Adam(self._critic, self.max_grad_norm)
        self._critic_2_optimizer = Adam(self._critic_2, self.max_grad_norm)
        self._log_ent_coef_optimizer = Adam(self._log_ent_coef)  # SB3 clips no temperature

    @property
    def ent_coef(self) -> float:
        """The current temperature alpha."""
        return math.exp(float(self._log_ent_coef["value"][0]))

    # ------------------------------------------------------------------
    # The squashed Gaussian actor
    # ------------------------------------------------------------------

    def _policy(self, obs: Array, eps: Array) -> PolicySample:
        """The actor's sample for observations (B, O) and standard normal noise ``eps`` (B, A).
        Changes no state."""
        z, cache = mlp_forward(self._actor, obs)
        act_dim = self.action_dim
        mean, log_std_raw = z[:, :act_dim], z[:, act_dim:]
        log_std = np.clip(log_std_raw, LOG_STD_MIN, LOG_STD_MAX)
        pre_tanh = mean + np.exp(log_std) * eps
        squashed = np.tanh(pre_tanh)
        gaussian = -0.5 * eps**2 - log_std - HALF_LOG_2PI
        log_prob = np.sum(gaussian - log_one_minus_tanh_squared(pre_tanh), axis=1)
        return PolicySample(
            mean=mean,
            log_std_raw=log_std_raw,
            log_std=log_std,
            eps=eps,
            pre_tanh=pre_tanh,
            squashed=squashed,
            action=self.bias + self.scale * squashed,
            log_prob=log_prob,
            cache=cache,
        )

    def _draw_policy_noise(self, batch_size: int) -> Array:
        """Standard normal noise (batch_size, action_dim) for an update, from its own stream."""
        return self._policy_noise_rng.standard_normal((batch_size, self.action_dim))

    # ------------------------------------------------------------------
    # Losses: each computes, none applies
    # ------------------------------------------------------------------

    def _soft_targets(
        self, rewards: Array, next_obs: Array, terminated: Array, eps_next: Array, alpha: float
    ) -> Array:
        """``r + gamma (1 - terminated) (min(Q1', Q2')(s', a') - alpha log pi(a'|s'))``, (B,),
        with ``a'`` the online actor's sample at ``s'`` for the noise ``eps_next``."""
        nxt = self._policy(next_obs, eps_next)
        q1, _ = self._critic_forward(next_obs, nxt.action, self._target_critic)
        q2, _ = self._critic_forward(next_obs, nxt.action, self._target_critic_2)
        soft_q = np.minimum(q1, q2) - alpha * nxt.log_prob
        not_done = 1.0 - np.asarray(terminated, dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64) + self.gamma * not_done * soft_q

    def _actor_loss_and_grads(
        self, obs: Array, eps: Array, alpha: float, sample: PolicySample | None = None
    ) -> tuple[dict[str, float], Params]:
        """
        Actor loss ``mean(alpha log pi - min(Q1, Q2))`` and its gradients for the actor, with
        ``eps``, ``alpha`` and both critics constant; ``sample`` is the actor's pass for these
        inputs if already computed. Changes no state.

        With ``g_a = dL/da`` through the smaller critic's action input and ``dlog pi/du = 2t``:
        ``g_u = (alpha/B) 2t + g_a · scale · (1 - t²)``, ``dm = g_u`` and
        ``dl = (-(alpha/B) + g_u · e^l · eps)``, zero where ``l`` was clipped or sits on a bound.
        """
        p = self._policy(obs, eps) if sample is None else sample
        batch = len(obs)
        q1, cache_1 = self._critic_forward(obs, p.action, self._critic)
        q2, cache_2 = self._critic_forward(obs, p.action, self._critic_2)
        first_smaller = q1 <= q2
        d_q1 = np.where(first_smaller, -1.0 / batch, 0.0)[:, None]
        d_q2 = np.where(first_smaller, 0.0, -1.0 / batch)[:, None]
        _, d_in_1 = mlp_backward(self._critic, cache_1, d_q1)
        _, d_in_2 = mlp_backward(self._critic_2, cache_2, d_q2)
        g_a = d_in_1[:, self.observation_dim :] + d_in_2[:, self.observation_dim :]
        coef = alpha / batch
        t = p.squashed
        g_u = coef * 2.0 * t + g_a * self.scale * (1.0 - t**2)
        # Strict, as SB3's clamp with scalar bounds: no gradient on a bound either.
        inside = (p.log_std_raw > LOG_STD_MIN) & (p.log_std_raw < LOG_STD_MAX)
        d_log_std = np.where(inside, -coef + g_u * np.exp(p.log_std) * p.eps, 0.0)
        grads, _ = mlp_backward(self._actor, p.cache, np.concatenate([g_u, d_log_std], axis=1))
        components = {
            "actor_loss": float(np.mean(alpha * p.log_prob - np.minimum(q1, q2))),
            "saturation": float(np.mean(np.abs(t) > 0.99)),
            "actor_grad_norm": global_norm(grads),
        }
        return components, grads

    def _ent_coef_loss_and_grads(self, log_prob: Array) -> tuple[dict[str, float], Params]:
        """
        Temperature loss ``-log alpha · mean(log pi + target_entropy)`` and its gradient for
        ``log alpha``, ``-mean(log pi) - target_entropy``, with ``log_prob`` constant.
        """
        log_alpha = float(self._log_ent_coef["value"][0])
        gap = float(np.mean(log_prob)) + self.target_entropy
        return {"ent_coef_loss": -log_alpha * gap}, {"value": np.array([-gap])}

    # ------------------------------------------------------------------
    # OffPolicyAgent hooks
    # ------------------------------------------------------------------

    def _networks(self) -> dict[str, Params]:
        """Every trained parameter group and target by name, in checkpoint order; ``log_ent_coef``
        holds one value and is not an MLP."""
        return {
            "actor": self._actor,
            "critic": self._critic,
            "critic_2": self._critic_2,
            "target_critic": self._target_critic,
            "target_critic_2": self._target_critic_2,
            "log_ent_coef": self._log_ent_coef,
        }

    def _optimizers_by_net(self) -> dict[str, Adam]:
        return {
            "actor": self._actor_optimizer,
            "critic": self._critic_optimizer,
            "critic_2": self._critic_2_optimizer,
            "log_ent_coef": self._log_ent_coef_optimizer,
        }

    def _critic_names(self) -> tuple[str, ...]:
        return ("critic", "critic_2")

    def _target_names(self) -> list[tuple[str, str]]:
        return [("critic", "target_critic"), ("critic_2", "target_critic_2")]

    def _greedy_action(self, obs: Array) -> Array:
        """``bias + scale · tanh(m)``."""
        mean = mlp_forward(self._actor, obs)[0][0, : self.action_dim]
        return self.bias + self.scale * np.tanh(mean)

    def _training_action(self, obs: Array) -> Array:
        """A sample of the policy, its noise from the exploration stream."""
        eps = self._explore_rng.standard_normal((1, self.action_dim))
        return self._policy(obs, eps).action[0]

    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """
        ``num_updates`` SAC steps (see the class docstring for the order).

        Returns:
            Means over the steps of ``critic_loss`` (the mean of both critics'), ``mean_q`` (the
            first critic's), ``critic_grad_norm`` (both critics together), ``actor_loss``,
            ``actor_grad_norm``, ``saturation`` (share of sampled action elements with
            |tanh| > 0.99), ``entropy`` (``-mean log pi``, the entropy estimate the temperature
            steers towards ``target_entropy``), ``ent_coef`` (the alpha each step used) and,
            when alpha is learned, ``ent_coef_loss``.
        """
        totals: dict[str, float] = {}
        for _ in range(num_updates):
            self._gradient_steps += 1
            s = self.replay_buffer.sample(self.batch_size)
            obs = s["observations"]
            eps = self._draw_policy_noise(self.batch_size)
            eps_next = self._draw_policy_noise(self.batch_size)
            sample = self._policy(obs, eps)
            alpha = self.ent_coef
            step: dict[str, float] = {
                "ent_coef": alpha,
                "entropy": -float(np.mean(sample.log_prob)),
            }
            if self.learns_ent_coef:
                temperature, grads = self._ent_coef_loss_and_grads(sample.log_prob)
                self._apply(self._log_ent_coef, self._log_ent_coef_optimizer, grads)
                step.update(temperature)
            targets = self._soft_targets(
                s["rewards"], s["next_observations"], s["dones"], eps_next, alpha
            )
            step.update(self._critic_step(obs, s["actions"], targets))
            actor, actor_grads = self._actor_loss_and_grads(obs, eps, alpha, sample)
            self._apply(self._actor, self._actor_optimizer, actor_grads)
            self._polyak()
            add_metrics(totals, {**step, **actor})
        return mean_metrics(totals, num_updates)

    def _inference_network(self) -> InferencePolicy:
        """The actor's mean, its first A outputs, then ``bias + scale · tanh``."""
        return self._tanh_scale_policy(self._actor)
