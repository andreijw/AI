"""Optimisers shared by the NumPy agents.

Agents keep their parameters as named arrays (and the odd Python float). An optimiser here holds
only its own state; :meth:`Adam.step` returns the amount to subtract from each parameter and the
agent applies it, so each agent stays in charge of its own weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

OPTIMIZERS = ("sgd", "adam")


class Adam:
    """
    Adam (Kingma & Ba, 2015) with optional clipping of the global gradient norm.

    Uses beta1 0.9, beta2 0.999 and epsilon 1e-8, with bias correction. When ``max_grad_norm`` is
    set, all gradients are first scaled together so that their joint L2 norm is at most that
    value.

    Attributes:
        m: First-moment estimate per parameter name.
        v: Second-moment estimate per parameter name.
        t: Number of steps taken.
    """

    BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

    def __init__(
        self, params: Mapping[str, np.ndarray | float], max_grad_norm: float | None = None
    ) -> None:
        """
        Args:
            params: The parameters to optimise, by name; only their shapes are used.
            max_grad_norm: Clip the global gradient norm to this value, or None for no clip.
        """
        self.max_grad_norm = max_grad_norm
        self.m: dict[str, Any] = {name: _zeros_like(value) for name, value in params.items()}
        self.v: dict[str, Any] = {name: _zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(self, grads: Mapping[str, Any], lr: float) -> dict[str, Any]:
        """
        Advance the moments by one step and return each parameter's update.

        Args:
            grads: Gradient per parameter name (not modified).
            lr: Learning rate.

        Returns:
            Amount to subtract from each parameter, keyed like ``grads``.
        """
        self.t += 1
        beta1, beta2 = self.BETA1, self.BETA2

        clip_coef = 1.0
        if self.max_grad_norm is not None:
            total_norm_sq = 0.0
            for g in grads.values():
                total_norm_sq += np.sum(np.square(g))
            total_norm = np.sqrt(total_norm_sq)
            clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-6))

        steps: dict[str, Any] = {}
        for k, grad in grads.items():
            g_clipped = grad * clip_coef
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g_clipped
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * np.square(g_clipped)

            m_hat = self.m[k] / (1.0 - beta1**self.t)
            v_hat = self.v[k] / (1.0 - beta2**self.t)
            steps[k] = lr * m_hat / (np.sqrt(v_hat) + self.EPS)
        return steps

    def state(self) -> dict[str, np.ndarray]:
        """The optimiser state as arrays to store in a checkpoint, under ``adam_`` keys."""
        state = {"adam_t": np.array(self.t)}
        for name in self.m:
            state[f"adam_m_{name}"] = np.asarray(self.m[name])
            state[f"adam_v_{name}"] = np.asarray(self.v[name])
        return state

    def load_state(self, data: Mapping[str, Any]) -> None:
        """
        Restore state written by :meth:`state`.

        A checkpoint without optimiser state (one saved with plain SGD) resets the moments to
        zero and the step count to 0. A checkpoint with only ``adam_t`` (continuous agents saved
        before the moments were stored) keeps that step count with zero moments, as it loaded
        before.

        Raises:
            ValueError: If a stored moment's shape differs from its parameter's.
        """
        self.t = int(data["adam_t"]) if "adam_t" in data else 0
        for name in self.m:
            m_key, v_key = f"adam_m_{name}", f"adam_v_{name}"
            if m_key in data and v_key in data:
                expected = np.shape(self.m[name])
                for key in (m_key, v_key):
                    if np.shape(data[key]) != expected:
                        raise ValueError(
                            f"{key} has shape {np.shape(data[key])}, expected {expected}"
                        )
                self.m[name] = np.array(data[m_key], dtype=np.float64)
                self.v[name] = np.array(data[v_key], dtype=np.float64)
            else:
                self.m[name] = _zeros_like(self.m[name])
                self.v[name] = _zeros_like(self.v[name])


def optimizer_from_config(config: Mapping[str, Any], params: Mapping[str, Any]) -> Adam | None:
    """
    Build the optimiser an agent's config asks for.

    Args:
        config: Agent config. ``optimizer`` is ``"sgd"`` (default) or ``"adam"``;
            ``max_grad_norm`` (optional, positive) clips the global gradient norm for Adam.
        params: The agent's parameters by name.

    Returns:
        An :class:`Adam`, or None for plain SGD, which the agent applies itself.

    Raises:
        ValueError: For an unknown optimizer, a non-positive ``max_grad_norm``, or a
            ``max_grad_norm`` with plain SGD.
    """
    name = str(config.get("optimizer", "sgd")).lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {config['optimizer']!r}")
    max_grad_norm = config.get("max_grad_norm")
    if name == "sgd":
        if max_grad_norm is not None:
            raise ValueError('max_grad_norm requires optimizer "adam"; plain SGD does not clip')
        return None
    if max_grad_norm is not None and float(max_grad_norm) <= 0.0:
        raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm!r}")
    return Adam(params, None if max_grad_norm is None else float(max_grad_norm))


def _zeros_like(value: np.ndarray | float) -> Any:
    """Zeros shaped like a parameter: an array for an array, a float for a float."""
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return np.zeros_like(value, dtype=np.float64)
    return 0.0


def anneal_steps_from_config(config: Mapping[str, Any]) -> int | None:
    """
    Read ``anneal_steps``: decay the learning rate and clip range linearly to zero over this many
    trained transitions. Unset (None) means no decay.

    Raises:
        ValueError: If it is set and not a positive integer.
    """
    raw = config.get("anneal_steps")
    if raw is None:
        return None
    steps = int(raw)
    if steps <= 0:
        raise ValueError(f"anneal_steps must be a positive integer, got {steps!r}")
    return steps


def anneal_fraction(steps_trained: int, anneal_steps: int | None) -> float:
    """The share of the learning rate and clip range still in use: 1 without annealing, else
    ``max(0, 1 − steps_trained / anneal_steps)``, computed before the transitions of this update."""
    if anneal_steps is None:
        return 1.0
    return max(0.0, 1.0 - steps_trained / anneal_steps)
