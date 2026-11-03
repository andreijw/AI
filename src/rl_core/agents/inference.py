"""
An agent's greedy policy as plain data, for exporting it to run without the agent.

Every agent with a network acts greedily the same way: the observation, normalised if the agent
normalises observations, goes through dense layers (ReLU on every layer but the last) and then one
of three heads. :class:`InferencePolicy` holds that chain as read-only float64 arrays, checks it
fits together, and computes it in NumPy as the reference an exported copy is checked against.
Agents build it (``BaseAgent.inference_policy``); exporters (``rl_core.export``) only read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from rl_core.agents.observation_normalizer import standardize

Array = npt.NDArray[np.float64]

# ``argmax``: the index of the first largest output (a discrete action).
# ``clip``: the outputs clipped to the action bounds (a bound may be infinite).
# ``tanh_scale``: centre + half-width · tanh(output), inside finite bounds, then clipped to them.
Head = Literal["argmax", "clip", "tanh_scale"]
HEADS: tuple[str, ...] = get_args(Head)


def _frozen(values: npt.ArrayLike, name: str, ndim: int) -> Array:
    """A read-only float64 copy of ``values``, which must have ``ndim`` dimensions."""
    array = np.array(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s), got shape {array.shape}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class DenseLayer:
    """``x @ weight + bias``, then ReLU if ``relu``; ``weight`` is (inputs, outputs)."""

    weight: Array
    bias: Array
    relu: bool

    def __post_init__(self) -> None:
        weight = _frozen(self.weight, "weight", 2)
        bias = _frozen(self.bias, "bias", 1)
        if bias.shape[0] != weight.shape[1]:
            raise ValueError(
                f"bias holds {bias.shape[0]} values for a weight with {weight.shape[1]} outputs"
            )
        if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
            raise ValueError("weights and biases must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    @property
    def inputs(self) -> int:
        return int(self.weight.shape[0])

    @property
    def outputs(self) -> int:
        return int(self.weight.shape[1])


@dataclass(frozen=True)
class InferencePolicy:
    """
    A greedy policy: optional observation normalisation, dense layers, a head.

    Attributes:
        layers: The dense layers in order; the last has no ReLU.
        head: How the last layer's outputs become an action (see :data:`Head`).
        action_low, action_high: The action bounds, one per output; None for ``argmax``.
        obs_mean, obs_scale: With observation normalisation, the input is
            ``(x - obs_mean) / obs_scale`` clipped to ±10 (``obs_scale`` is ``sqrt(var + 1e-8)``);
            both None without it.
        observation_settings: The environment's observation settings the agent was built for
            (``frame_stack``, ``mask_velocity``, ...); the policy's input is the observation after
            them.
        agent_name: The agent's class name.
    """

    layers: tuple[DenseLayer, ...]
    head: Head
    action_low: Array | None = None
    action_high: Array | None = None
    obs_mean: Array | None = None
    obs_scale: Array | None = None
    observation_settings: Mapping[str, bool | int] = field(default_factory=dict)
    agent_name: str = ""

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("an inference policy needs at least one layer")
        for i in range(1, len(layers)):
            if layers[i].inputs != layers[i - 1].outputs:
                raise ValueError(
                    f"layer {i} takes {layers[i].inputs} inputs but layer {i - 1} gives "
                    f"{layers[i - 1].outputs}"
                )
        if layers[-1].relu:
            raise ValueError("the last layer gives the outputs and takes no ReLU")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "observation_settings", dict(self.observation_settings))
        self._check_head()
        self._check_normalisation()

    def _check_head(self) -> None:
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {list(HEADS)}, got {self.head!r}")
        if self.head == "argmax":
            if self.action_low is not None or self.action_high is not None:
                raise ValueError("argmax takes no action bounds")
            return
        if self.action_low is None or self.action_high is None:
            raise ValueError(f"{self.head} needs action bounds")
        low = _frozen(self.action_low, "action_low", 1)
        high = _frozen(self.action_high, "action_high", 1)
        for bound in (low, high):
            if bound.shape != (self.output_dim,):
                raise ValueError(
                    f"action bounds must hold one value for each of the {self.output_dim} actions, "
                    f"got shape {bound.shape}"
                )
        if self.head == "tanh_scale" and not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("tanh_scale needs finite action bounds")
        if (low > high).any() or np.isnan(low).any() or np.isnan(high).any():
            raise ValueError(f"action_low must not exceed action_high, got {low} and {high}")
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)

    def _check_normalisation(self) -> None:
        if (self.obs_mean is None) != (self.obs_scale is None):
            raise ValueError("obs_mean and obs_scale are given together or not at all")
        if self.obs_mean is None or self.obs_scale is None:
            return
        mean = _frozen(self.obs_mean, "obs_mean", 1)
        scale = _frozen(self.obs_scale, "obs_scale", 1)
        for name, array in (("obs_mean", mean), ("obs_scale", scale)):
            if array.shape != (self.input_dim,):
                raise ValueError(
                    f"{name} must hold {self.input_dim} values, got shape {array.shape}"
                )
        if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("obs_mean must be finite and obs_scale finite and positive")
        object.__setattr__(self, "obs_mean", mean)
        object.__setattr__(self, "obs_scale", scale)

    @property
    def input_dim(self) -> int:
        return self.layers[0].inputs

    @property
    def output_dim(self) -> int:
        return self.layers[-1].outputs

    @property
    def discrete(self) -> bool:
        return self.head == "argmax"

    @property
    def normalizes(self) -> bool:
        return self.obs_mean is not None

    def outputs(self, observation: npt.ArrayLike) -> Array:
        """The last layer's outputs, before the head, for one observation or a batch (B, input)."""
        x = np.asarray(observation, dtype=np.float64)
        if self.obs_mean is not None and self.obs_scale is not None:
            x = standardize(x, self.obs_mean, self.obs_scale)
        for layer in self.layers:
            x = x @ layer.weight + layer.bias
            if layer.relu:
                x = np.maximum(x, 0.0)
        return x

    def act(self, observation: npt.ArrayLike) -> int | npt.NDArray[np.int64] | Array:
        """The greedy action: an int for one observation of a discrete policy, an index array for a
        batch, and float64 actions (A,) or (B, A) for a continuous one."""
        z = self.outputs(observation)
        if self.head == "argmax":
            choice = np.argmax(z, axis=-1).astype(np.int64)
            return int(choice) if choice.ndim == 0 else choice
        assert self.action_low is not None and self.action_high is not None  # _check_head
        if self.head == "tanh_scale":
            centre = (self.action_high + self.action_low) / 2.0
            half_width = (self.action_high - self.action_low) / 2.0
            z = centre + half_width * np.tanh(z)
        return np.clip(z, self.action_low, self.action_high)
