"""SAC with its gradient steps in PyTorch, on the GPU when there is one.

Imported only when a config sets ``backend: torch`` for SAC (``BaseAgent.class_for_config``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import torch

from .continuous_actor_critic import LOG_STD_MAX, LOG_STD_MIN
from .sac_agent import HALF_LOG_2PI, LOG_2, SACAgent
from .torch_off_policy import TorchOffPolicyAgent, forward


class TorchSACAgent(TorchOffPolicyAgent, SACAgent):
    """
    SACAgent with each gradient step computed in PyTorch: the same algorithm, defaults, config and
    checkpoints, and everything TorchOffPolicyAgent describes (the device copies, the NumPy
    mirror, staging, CUDA graphs, the config keys ``device``, ``dtype`` and ``cuda_graph``).

    A step is SACAgent's, in its order: the squashed Gaussian sample and log pi at the current
    actor, the temperature step (alpha from before it serves the rest of the step), the soft
    targets, both critics, the actor against the critics just updated, then Polyak on the target
    critics. Both noises, ``eps`` for the batch and ``eps'`` for the next states, are drawn on the
    host from SACAgent's own stream, with the batch, so float64 matches SACAgent to rounding
    (tests/test_torch_ddpg_agent.py). ``log_ent_coef`` is a one-value parameter group on the
    device with its own Adam, never clipped, as SACAgent's; with a fixed ``ent_coef`` it is never
    stepped. :attr:`ent_coef` reads it from the device, so it is current between updates.
    """

    STEP_METRICS: ClassVar[tuple[str, ...]] = (
        "ent_coef",
        "entropy",
        "ent_coef_loss",  # dropped from what update returns when alpha is fixed
        "critic_loss",
        "mean_q",
        "critic_grad_norm",
        "actor_loss",
        "saturation",
        "actor_grad_norm",
    )

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Raises:
            ValueError: For an invalid SAC hyperparameter or torch setting, or ``device: cuda``
                without CUDA.
        """
        super().__init__(observation_dim, action_dim, config)
        self._zero_t = torch.zeros((), dtype=self.dtype, device=self.device)

    @property
    def ent_coef(self) -> float:
        """The current temperature alpha, read from the device."""
        return math.exp(float(self._nets["log_ent_coef"]["value"].detach()[0]))

    def _input_shapes(self) -> dict[str, tuple[int, ...]]:
        noise = (self.batch_size, self.action_dim)
        return {**super()._input_shapes(), "eps": noise, "eps_next": noise}

    def _sample_inputs(self) -> Mapping[str, np.ndarray]:
        """The batch, then the actor's noise and the next state's, as SACAgent draws them."""
        batch = super()._sample_inputs()
        eps = self._draw_policy_noise(self.batch_size)
        return {**batch, "eps": eps, "eps_next": self._draw_policy_noise(self.batch_size)}

    def _reduce_metrics(self, sums: torch.Tensor, kinds: Counter[str]) -> dict[str, float]:
        metrics = super()._reduce_metrics(sums, kinds)
        if not self.learns_ent_coef:
            del metrics["ent_coef_loss"]
        return metrics

    def _sample(
        self, obs: torch.Tensor, eps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``SACAgent._policy`` on the device: ``(a, tanh(u), log pi)`` of the actor for
        observations (B, O) and standard normal noise (B, A). The clamp passes no gradient on a
        bound, as SACAgent's strict mask, because its bounds are Python floats: with tensor bounds
        torch 2.14 splits the gradient on a bound, passing half of it."""
        z = forward(self._nets["actor"], obs)
        act_dim = self.action_dim
        mean, log_std = z[:, :act_dim], torch.clamp(z[:, act_dim:], LOG_STD_MIN, LOG_STD_MAX)
        pre_tanh = mean + torch.exp(log_std) * eps
        squashed = torch.tanh(pre_tanh)
        gaussian = -0.5 * eps**2 - log_std - HALF_LOG_2PI
        # log(1 - tanh² u) in sac_agent.log_one_minus_tanh_squared's stable form
        log_jacobian = 2.0 * (LOG_2 - pre_tanh - torch.logaddexp(self._zero_t, -2.0 * pre_tanh))
        log_prob = torch.sum(gaussian - log_jacobian, dim=1)
        return self._bias_t + self._scale_t * squashed, squashed, log_prob

    def _step(self, kind: str) -> None:
        """One SAC step on the inputs in ``self._inputs`` (see the class docstring)."""
        x, nets = self._inputs, self._nets
        obs, next_obs = x["observations"], x["next_observations"]
        actions, squashed, log_prob = self._sample(obs, x["eps"])
        log_alpha = nets["log_ent_coef"]["value"]
        alpha = torch.exp(log_alpha.detach())  # a new tensor: the temperature step leaves it
        mean_log_prob = torch.mean(log_prob.detach())
        if self.learns_ent_coef:
            gap = mean_log_prob + self.target_entropy
            ent_coef_loss = -log_alpha[0] * gap
            self._adam_step("log_ent_coef", ent_coef_loss)
        else:
            ent_coef_loss = self._zero_t
        with torch.no_grad():
            next_actions, _, next_log_prob = self._sample(next_obs, x["eps_next"])
            targets = self._bootstrap(self._next_q(next_obs, next_actions) - alpha * next_log_prob)
        critic_metrics = self._critics_update(targets)
        q1 = self._q(nets["critic"], obs, actions)
        q2 = self._q(nets["critic_2"], obs, actions)
        # The first critic where it is the smaller or equal, as SACAgent's gradient takes it.
        min_q = torch.where(q1 <= q2, q1, q2)
        actor_loss = torch.mean(alpha * log_prob - min_q)
        actor_norm = self._adam_step("actor", actor_loss)
        saturation = torch.mean((torch.abs(squashed.detach()) > 0.99).to(self.dtype))
        self._move_targets()
        with torch.no_grad():
            self._metric_sums += torch.stack(
                [
                    alpha[0],
                    -mean_log_prob,
                    ent_coef_loss.detach(),
                    *critic_metrics,
                    actor_loss,
                    saturation,
                    actor_norm,
                ]
            ).detach()

    # SACAgent's own NumPy training maths (OffPolicyAgent's is stubbed by the base). Its
    # ``_policy`` stays: acting samples the NumPy actor.

    def _soft_targets(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()

    def _actor_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()

    def _ent_coef_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()
