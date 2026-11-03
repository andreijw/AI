"""Optimisers shared by the NumPy agents.

Agents keep their parameters as named arrays (and the odd Python float). An optimiser here holds
only its own state; its ``step`` returns the amount to subtract from each parameter and the agent
applies it, so each agent stays in charge of its own weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from rl_core.utils.config_values import optional_positive_int

OPTIMIZERS = ("sgd", "adam")


class Optimizer(Protocol):
    """What an agent needs of an optimiser: a step, and its state for a checkpoint."""

    max_grad_norm: float | None

    def step(self, grads: Mapping[str, Any], lr: float) -> dict[str, Any]: ...

    def state(self, prefix: str = "adam_") -> dict[str, np.ndarray]: ...

    def load_state(self, data: Mapping[str, Any], prefix: str = "adam_") -> None: ...


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

        clip_coef = _clip_coefficient(grads, self.max_grad_norm)
        steps: dict[str, Any] = {}
        for k, grad in grads.items():
            g_clipped = grad * clip_coef
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g_clipped
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * np.square(g_clipped)

            m_hat = self.m[k] / (1.0 - beta1**self.t)
            v_hat = self.v[k] / (1.0 - beta2**self.t)
            steps[k] = lr * m_hat / (np.sqrt(v_hat) + self.EPS)
        return steps

    def state(self, prefix: str = "adam_") -> dict[str, np.ndarray]:
        """
        The optimiser state as arrays to store in a checkpoint, under keys starting ``prefix``.

        An agent with more than one optimiser gives each its own prefix, so that their step
        counts and moments do not overwrite each other in one checkpoint.
        """
        state = {f"{prefix}t": np.array(self.t)}
        for name in self.m:
            state[f"{prefix}m_{name}"] = np.asarray(self.m[name])
            state[f"{prefix}v_{name}"] = np.asarray(self.v[name])
        return state

    def load_state(self, data: Mapping[str, Any], prefix: str = "adam_") -> None:
        """
        Restore state written by :meth:`state` with the same ``prefix``.

        A checkpoint without optimiser state (one saved with plain SGD) resets the moments to
        zero and the step count to 0. A checkpoint with only ``adam_t`` (continuous agents saved
        before the moments were stored) keeps that step count with zero moments, as it loaded
        before.

        Raises:
            ValueError: If a stored moment's shape differs from its parameter's.
        """
        self.t = int(data[f"{prefix}t"]) if f"{prefix}t" in data else 0
        for name in self.m:
            m_key, v_key = f"{prefix}m_{name}", f"{prefix}v_{name}"
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


class ClippedSGD:
    """
    Plain SGD with the global gradient-norm clip :class:`Adam` uses.

    Only a config that sets ``max_grad_norm`` with plain SGD gets one; without a clip the agents
    take their own SGD step. It keeps no state, so a checkpoint holds nothing for it.
    """

    def __init__(self, max_grad_norm: float) -> None:
        self.max_grad_norm: float | None = max_grad_norm

    def step(self, grads: Mapping[str, Any], lr: float) -> dict[str, Any]:
        """``lr`` times each gradient, all scaled together so their joint L2 norm is at most
        ``max_grad_norm``. ``grads`` is not modified."""
        scale = lr * _clip_coefficient(grads, self.max_grad_norm)
        return {name: scale * grad for name, grad in grads.items()}

    def state(self, prefix: str = "adam_") -> dict[str, np.ndarray]:
        """Nothing: plain SGD has no state to resume."""
        return {}

    def load_state(self, data: Mapping[str, Any], prefix: str = "adam_") -> None:
        """Nothing to restore; an Adam checkpoint's moments are ignored."""


def optimizer_from_config(config: Mapping[str, Any], params: Mapping[str, Any]) -> Optimizer | None:
    """
    Build the optimiser an agent's config asks for.

    Args:
        config: Agent config. ``optimizer`` is ``"sgd"`` (default) or ``"adam"``;
            ``max_grad_norm`` (optional, positive) clips the global norm of the gradient the
            agent steps with, for either.
        params: The agent's parameters by name.

    Returns:
        An :class:`Adam`; a :class:`ClippedSGD` for plain SGD with ``max_grad_norm``; or None for
        plain SGD without one, which the agent applies itself.

    Raises:
        ValueError: For an unknown optimizer or a non-positive ``max_grad_norm``.
    """
    name = str(config.get("optimizer", "sgd")).lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {config['optimizer']!r}")
    raw_clip = config.get("max_grad_norm")
    max_grad_norm = None if raw_clip is None else float(raw_clip)
    if max_grad_norm is not None and not max_grad_norm > 0.0:  # refuses NaN too
        raise ValueError(f"max_grad_norm must be positive, got {raw_clip!r}")
    if name == "sgd":
        return None if max_grad_norm is None else ClippedSGD(max_grad_norm)
    return Adam(params, max_grad_norm)


def _clip_coefficient(grads: Mapping[str, Any], max_grad_norm: float | None) -> float:
    """The factor that scales ``grads`` to a joint L2 norm of at most ``max_grad_norm`` (1 if
    unset)."""
    if max_grad_norm is None:
        return 1.0
    total_norm_sq = 0.0
    for g in grads.values():
        total_norm_sq += np.sum(np.square(g))
    total_norm = np.sqrt(total_norm_sq)
    return min(1.0, max_grad_norm / (total_norm + 1e-6))


def _zeros_like(value: np.ndarray | float) -> Any:
    """Zeros shaped like a parameter: an array for an array, a float for a float."""
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return np.zeros_like(value, dtype=np.float64)
    return 0.0


def anneal_steps_from_config(config: Mapping[str, Any]) -> int | None:
    """
    Read ``anneal_steps``: decay the learning rate (and PPO's clip range) linearly to zero over
    this many trained transitions. Unset (None) means no decay.

    Raises:
        ValueError: If it is set and not a positive integer.
    """
    return optional_positive_int(config, "anneal_steps")


def anneal_fraction(steps_trained: int, anneal_steps: int | None) -> float:
    """The share of a linearly annealed quantity still in use: 1 without annealing, else
    ``max(0, 1 − steps_trained / anneal_steps)``. PPO anneals its learning rate and clip range with
    it and DQN its learning rate (both counting trained transitions); DQN also anneals epsilon with
    it (counting training actions)."""
    if anneal_steps is None:
        return 1.0
    return max(0.0, 1.0 - steps_trained / anneal_steps)
