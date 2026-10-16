"""A plain multi-layer perceptron with a hand-written backward pass, shared by agents that need
more than one hidden layer.

Parameters are a dict of named arrays (``W0``, ``b0``, ``W1``, ...); the functions hold no state.
Every layer but the last is followed by a ReLU; the caller applies any output activation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Params = dict[str, npt.NDArray[np.float64]]


def init_mlp(sizes: Sequence[int], rng: np.random.Generator) -> Params:
    """
    Initialise the layers ``sizes[0] -> sizes[1] -> ... -> sizes[-1]``.

    Weights and biases are drawn uniformly from ±1/sqrt(fan_in), PyTorch's ``nn.Linear`` default
    (and so Stable-Baselines3's).

    Raises:
        ValueError: If there is not at least one layer, or a size is not positive.
    """
    if len(sizes) < 2 or any(int(size) <= 0 for size in sizes):
        raise ValueError(f"sizes must hold at least two positive layer widths, got {list(sizes)!r}")
    params: Params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        bound = 1.0 / np.sqrt(fan_in)
        params[f"W{i}"] = rng.uniform(-bound, bound, size=(int(fan_in), int(fan_out)))
        params[f"b{i}"] = rng.uniform(-bound, bound, size=int(fan_out))
    return params


def _layers(params: Mapping[str, Any]) -> int:
    return len(params) // 2


def mlp_forward(
    params: Mapping[str, npt.NDArray[np.float64]], x: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], dict[str, list[npt.NDArray[np.float64]]]]:
    """
    Forward pass for a batch ``x`` of shape (B, sizes[0]).

    Returns:
        The output (B, sizes[-1]) before any output activation, and a cache for
        :func:`mlp_backward`: ``inputs`` (each layer's input) and ``pre`` (each layer's
        pre-activation).
    """
    n = _layers(params)
    inputs, pre = [], []
    h = x
    for i in range(n):
        inputs.append(h)
        z = h @ params[f"W{i}"] + params[f"b{i}"]
        pre.append(z)
        h = np.maximum(z, 0.0) if i < n - 1 else z
    return h, {"inputs": inputs, "pre": pre}


def mlp_backward(
    params: Mapping[str, npt.NDArray[np.float64]],
    cache: Mapping[str, list[npt.NDArray[np.float64]]],
    d_out: npt.NDArray[np.float64],
) -> tuple[Params, npt.NDArray[np.float64]]:
    """
    Backward pass from ``d_out``, the gradient of a scalar loss with respect to the output.

    Returns:
        The gradient for every parameter (summed over the batch: divide in the loss) and the
        gradient with respect to the input ``x``. Neither ``params`` nor ``d_out`` is modified.
    """
    grads: Params = {}
    d = np.asarray(d_out, dtype=np.float64)
    for i in reversed(range(_layers(params))):
        if i < _layers(params) - 1:
            d = d * (cache["pre"][i] > 0.0)
        grads[f"W{i}"] = cache["inputs"][i].T @ d
        grads[f"b{i}"] = d.sum(axis=0)
        d = d @ params[f"W{i}"].T
    return grads, d
