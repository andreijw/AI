"""TD3 with its gradient steps in PyTorch, on the GPU when there is one.

Imported only when a config sets ``backend: torch`` for TD3 (``BaseAgent.class_for_config``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .td3_agent import TD3Agent
from .torch_ddpg_agent import TorchDDPGAgent

# The first three of STEP_METRICS are the critics', summed every step; the rest the actor's.
CRITIC_METRICS = 3


class TorchTD3Agent(TorchDDPGAgent, TD3Agent):
    """
    TD3Agent with each gradient step computed in PyTorch: the same algorithm, defaults, config and
    checkpoints, and everything TorchDDPGAgent describes (the device copies, the NumPy mirror,
    staging, the config keys ``device``, ``dtype`` and ``cuda_graph``).

    A step is one of two kinds, as TD3Agent's ``_is_actor_step`` decides from its number: a
    ``critic`` step trains both critics; a ``full`` step also trains the actor and moves the three
    target networks. On CUDA each kind is captured as its own graph. The target-policy noise is
    drawn on the host from TD3Agent's own generator, with the batch, so float64 matches TD3Agent
    to rounding (tests/test_torch_ddpg_agent.py).
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Raises:
            ValueError: For an invalid TD3 hyperparameter or torch setting, or ``device: cuda``
                without CUDA.
        """
        super().__init__(observation_dim, action_dim, config)
        self._low_t = torch.tensor(self.action_low, dtype=self.dtype, device=self.device)
        self._high_t = torch.tensor(self.action_high, dtype=self.dtype, device=self.device)

    def _input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {**super()._input_shapes(), "noise": (self.batch_size, self.action_dim)}

    def _sample_inputs(self) -> Mapping[str, np.ndarray]:
        """The batch, then its target noise: each from its own generator, as TD3Agent draws them."""
        return {**super()._sample_inputs(), "noise": self._draw_target_noise(self.batch_size)}

    def _step_kind(self, gradient_step: int) -> str:
        return "full" if self._is_actor_step(gradient_step) else "critic"

    def _reduce_metrics(self, sums: torch.Tensor, kinds: Counter[str]) -> dict[str, float]:
        names, totals = self.STEP_METRICS, sums.cpu().tolist()
        return self._step_means(
            dict(zip(names[:CRITIC_METRICS], totals[:CRITIC_METRICS], strict=True)),
            dict(zip(names[CRITIC_METRICS:], totals[CRITIC_METRICS:], strict=True)),
            sum(kinds.values()),
            kinds["full"],
        )

    def _step(self, kind: str) -> None:
        """One TD3 step on the batch in ``self._inputs``: both critics towards the smoothed twin
        target, then, on a ``full`` step, the actor and the targets."""
        x, nets = self._inputs, self._nets
        with torch.no_grad():
            mu, _ = self._policy(nets["target_actor"], x["next_observations"])
            next_actions = torch.clamp(mu + self._scale_t * x["noise"], self._low_t, self._high_t)
            q1 = self._q(nets["target_critic"], x["next_observations"], next_actions)
            q2 = self._q(nets["target_critic_2"], x["next_observations"], next_actions)
            targets = x["rewards"] + self.gamma * (1.0 - x["dones"]) * torch.minimum(q1, q2)
        loss_1, q, norm_1 = self._critic_update("critic", targets)
        loss_2, _, norm_2 = self._critic_update("critic_2", targets)
        metrics = [(loss_1 + loss_2) / 2.0, q.mean(), torch.hypot(norm_1, norm_2)]
        if kind == "full":
            metrics += self._actor_update()
            self._move_targets()
        with torch.no_grad():
            self._metric_sums[: len(metrics)] += torch.stack(metrics).detach()
