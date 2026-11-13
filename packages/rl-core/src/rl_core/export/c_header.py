"""
Render an :class:`~rl_core.agents.inference.InferencePolicy` as one self-contained C99 header.

The header holds every weight as a ``float`` written as a hexadecimal literal, exact for the
float32 value it stands for, and two functions: ``policy_outputs`` (the last layer's outputs) and
``policy_act`` (the greedy action: an ``int`` for a discrete policy, ``POLICY_ACTION_DIM`` floats
for a continuous one). It computes in float32 and includes only ``<math.h>`` (``tanhf``); it holds
no date, so the same policy gives the same bytes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from rl_core.agents.inference import InferencePolicy
from rl_core.agents.observation_normalizer import CLIP

__all__ = ["render_c_header"]

GUARD = "POLICY_WEIGHTS_H"
_PER_LINE = 4


def c_float(value: float) -> str:
    """``value`` rounded to float32 as an exact C hexadecimal float literal (``0x1.8p+1f``)."""
    number = float(np.float32(value))
    if not np.isfinite(number):
        raise ValueError(f"cannot write the non-finite value {number} as a C literal")
    mantissa, exponent = number.hex().split("p")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".") or "0"
        if "." not in mantissa:
            mantissa += ".0"
    return f"{mantissa}p{exponent}f"


def _array(name: str, values: Iterable[float]) -> list[str]:
    literals = [c_float(v) for v in values]
    rows = [
        "    " + ", ".join(literals[i : i + _PER_LINE]) + ","
        for i in range(0, len(literals), _PER_LINE)
    ]
    return [f"static const float {name}[{len(literals)}] = {{", *rows, "};"]


def _comment(policy: InferencePolicy, source: str | None) -> list[str]:
    settings = policy.observation_settings
    described = ", ".join(f"{k}: {v}" for k, v in sorted(settings.items())) or "none recorded"
    widths = " -> ".join([str(policy.input_dim), *(str(layer.outputs) for layer in policy.layers)])
    lines = [
        f"Greedy policy of {policy.agent_name or 'an agent'}, exported by rl_core.export.",
        *([f"Source: {source}"] if source else []),
        f"Network: {widths} (ReLU between layers), head: {policy.head}.",
        "Input: the observation after the environment's observation settings "
        f"({described}), as float.",
    ]
    if policy.normalizes:
        lines.append(f"The input is normalised in here: (x - mean) / scale, clipped to +-{CLIP:g}.")
    lines += [
        "Weights are stored [outputs][inputs] per layer. Generated: do not edit.",
        "Its names (POLICY_*, policy_*) are fixed: one policy per program, and include this header",
        "in one source file only, or each copy holds its own weights.",
    ]
    return ["/*", *(f" * {line}" for line in lines), " */"]


def _act(policy: InferencePolicy) -> list[str]:
    if policy.discrete:
        return [
            "/* The greedy action: the index of the first largest output. */",
            "static inline int policy_act(const float *obs) {",
            "    float out[POLICY_OUTPUT_DIM];",
            "    int best = 0;",
            "    policy_outputs(obs, out);",
            "    for (int j = 1; j < POLICY_OUTPUT_DIM; ++j) {",
            "        if (out[j] > out[best]) best = j;",
            "    }",
            "    return best;",
            "}",
        ]
    assert policy.action_low is not None and policy.action_high is not None  # continuous head
    body = []
    for j, (low, high) in enumerate(zip(policy.action_low, policy.action_high, strict=True)):
        if policy.head == "tanh_scale":
            centre = c_float(policy.squash_centre[j])
            half = c_float(policy.squash_half_width[j])
            body.append(f"    action[{j}] = {centre} + {half} * tanhf(action[{j}]);")
        if np.isfinite(low):
            body.append(f"    if (action[{j}] < {c_float(low)}) action[{j}] = {c_float(low)};")
        if np.isfinite(high):
            body.append(f"    if (action[{j}] > {c_float(high)}) action[{j}] = {c_float(high)};")
    squash = "squashed into" if policy.head == "tanh_scale" else "clipped to"
    return [
        f"/* The greedy action: the outputs {squash} the action bounds. */",
        "static inline void policy_act(const float *obs, float *action) {",
        "    policy_outputs(obs, action);",
        *body,
        "}",
    ]


def _outputs(policy: InferencePolicy) -> list[str]:
    lines: list[str] = []
    source = "obs"
    used: set[str] = set()  # the scratch buffers written, declared only if used (-Wunused)
    if policy.normalizes:
        limit = c_float(CLIP)
        lines += [
            "    for (int i = 0; i < POLICY_INPUT_DIM; ++i) {",
            "        float z = (obs[i] - policy_obs_mean[i]) / policy_obs_scale[i];",
            f"        a[i] = z > {limit} ? {limit} : (z < -{limit} ? -{limit} : z);",
            "    }",
        ]
        source = "a"
        used.add("a")
    count = len(policy.layers)
    for i, layer in enumerate(policy.layers):
        target = "out" if i == count - 1 else ("b" if source == "a" else "a")
        relu = 1 if layer.relu else 0
        lines.append(
            f"    policy_dense(policy_w{i}, policy_b{i}, {source}, {target}, "
            f"{layer.inputs}, {layer.outputs}, {relu});"
        )
        source = target
        used.add(target)
    scratch = ", ".join(f"{name}[POLICY_MAX_WIDTH]" for name in ("a", "b") if name in used)
    return [
        "/* The last layer's outputs, before the head. */",
        "static inline void policy_outputs(const float *obs, float *out) {",
        *([f"    float {scratch};"] if scratch else []),
        *lines,
        "}",
    ]


_DENSE = [
    "/* y = x W + b, then ReLU if relu; w is [n_out][n_in]. */",
    "static inline void policy_dense(const float *w, const float *bias, const float *x, float *y,",
    "                                int n_in, int n_out, int relu) {",
    "    for (int j = 0; j < n_out; ++j) {",
    "        const float *row = w + j * n_in;",
    "        float s = bias[j];",
    "        for (int i = 0; i < n_in; ++i) s += row[i] * x[i];",
    "        y[j] = (relu && s < 0.0f) ? 0.0f : s;",
    "    }",
    "}",
]


def render_c_header(policy: InferencePolicy, source: str | None = None) -> str:
    """
    The C99 header for ``policy``.

    Args:
        policy: The policy to export.
        source: A line naming where the policy came from (a checkpoint and its hash), written
            into the header's comment.
    """
    widths = [policy.input_dim, *(layer.outputs for layer in policy.layers)]
    lines = [
        *_comment(policy, source),
        f"#ifndef {GUARD}",
        f"#define {GUARD}",
        "",
        "#include <math.h>",
        "",
        f"#define POLICY_INPUT_DIM {policy.input_dim}",
        f"#define POLICY_OUTPUT_DIM {policy.output_dim}",
    ]
    if not policy.discrete:
        lines.append(f"#define POLICY_ACTION_DIM {policy.output_dim}")
    lines += [f"#define POLICY_MAX_WIDTH {max(widths)}", ""]
    if policy.obs_mean is not None and policy.obs_scale is not None:
        lines += _array("policy_obs_mean", policy.obs_mean)
        lines += _array("policy_obs_scale", policy.obs_scale)
    for i, layer in enumerate(policy.layers):
        lines += _array(f"policy_w{i}", layer.weight.T.ravel())
        lines += _array(f"policy_b{i}", layer.bias)
    lines += ["", *_DENSE, "", *_outputs(policy), "", *_act(policy), "", f"#endif /* {GUARD} */"]
    return "\n".join(lines) + "\n"
