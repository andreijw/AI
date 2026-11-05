"""DDPG with its gradient steps in PyTorch, on the GPU when there is one.

Imported only when a config sets ``backend: torch`` (``BaseAgent.class_for_config``), so nothing
else in the framework needs PyTorch. The measurements behind it are in
docs/pytorch_jetson_timing.md.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from .ddpg_agent import DDPGAgent
from .torch_off_policy import TensorParams, TorchOffPolicyAgent, forward


class TorchDDPGAgent(TorchOffPolicyAgent, DDPGAgent):
    """
    DDPGAgent with each gradient step computed in PyTorch: the same algorithm, defaults, config
    and checkpoints. The device copies, the NumPy mirror, staging, CUDA graphs and the config
    keys ``device``, ``dtype`` and ``cuda_graph`` are TorchOffPolicyAgent's; this class supplies
    DDPG's step: the critic towards the target networks' Bellman target, the actor on
    ``-mean Q(s, mu(s))``, then Polyak on both targets.

    A backend for a subclass of DDPGAgent subclasses both (``class TorchX(TorchDDPGAgent, X)``)
    and supplies its own step, as TorchTD3Agent does; networks, optimizers and target pairs
    follow the NumPy agent's hooks.
    """

    STEP_METRICS: ClassVar[tuple[str, ...]] = (
        "critic_loss",
        "mean_q",
        "critic_grad_norm",
        "actor_loss",
        "saturation",
        "actor_grad_norm",
    )

    def _policy(self, params: TensorParams, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mu, u)`` of the actor ``params``: u = tanh(mlp(s)), mu = bias + scale · u."""
        u = torch.tanh(forward(params, obs))
        return self._bias_t + self._scale_t * u, u

    def _step(self, kind: str) -> None:
        """One DDPG step on the batch in ``self._inputs``: critic, then actor, then Polyak."""
        x = self._inputs
        with torch.no_grad():
            next_actions, _ = self._policy(self._nets["target_actor"], x["next_observations"])
            targets = self._bootstrap(self._next_q(x["next_observations"], next_actions))
        metrics = self._critics_update(targets)
        metrics += self._actor_update()
        self._move_targets()
        with torch.no_grad():
            self._metric_sums += torch.stack(metrics).detach()

    def _actor_update(self) -> list[torch.Tensor]:
        """One Adam step of the actor on ``-mean Q1(s, mu(s))``. Returns the loss, the fraction of
        saturated tanh outputs and the gradient norm before clipping."""
        x, nets = self._inputs, self._nets
        mu, u = self._policy(nets["actor"], x["observations"])
        loss = -torch.mean(self._q(nets["critic"], x["observations"], mu))
        # Gradients for the actor only: the critic's stay those of its own loss.
        norm = self._adam_step("actor", loss)
        saturation = torch.mean((torch.abs(u.detach()) > 0.99).to(self.dtype))
        return [loss, saturation, norm]

    # DDPGAgent's own NumPy training maths (OffPolicyAgent's is stubbed by the base).

    def _bellman_targets(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()

    def _actor_loss_and_grads(self, *args: Any, **kwargs: Any) -> Any:
        raise self._on_device()
