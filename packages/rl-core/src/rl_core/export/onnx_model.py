"""
Export a greedy policy (:class:`~rl_core.agents.inference.InferencePolicy`) as an ONNX model, and
check it in onnxruntime against the NumPy reference.

The model takes ``observation`` (float32, ``[batch, input]``; one observation is a batch of one) and
gives ``outputs`` (float32 ``[batch, O]``, the last layer's outputs) and ``action``: int64 ``[batch]``,
the index of the first largest output, for a discrete policy; float32 ``[batch, A]`` for a continuous
one. It computes what the C header computes: the observation normalisation clipped to ±10, the dense
layers (``Gemm``, then ``Relu`` on hidden layers), and the head. Weights are float32 initializers;
the agent, the checkpoint and the observation settings are in the model's metadata.

onnx and onnxruntime are rl-core's optional ``onnx`` extra: this is the only module that imports them,
and only inside its functions.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import numpy.typing as npt

from rl_core.agents.inference import InferencePolicy
from rl_core.agents.observation_normalizer import CLIP
from rl_core.export.verify import VerificationReport, compare_with_policy

__all__ = ["EXTRA", "missing_packages", "render_onnx_model", "verify_onnx_model"]

# Pinned rather than onnx's newest, so a model built with a recent onnx still loads in an older
# onnxruntime (IR 8 is opset 17's, onnx 1.12; onnxruntime has loaded both since 1.14).
OPSET = 17
IR_VERSION = 8
#: What a missing onnx or onnxruntime needs, for an installed rl-core as for a checkout.
EXTRA = "rl-core's onnx extra (onnx and onnxruntime)"


def missing_packages(verify: bool) -> list[str]:
    """The packages of the onnx extra that do not import: onnx to build a model, and onnxruntime
    too when it is to be ``verify``-ed."""
    missing = []
    for name in ("onnx", "onnxruntime") if verify else ("onnx",):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def render_onnx_model(policy: InferencePolicy, source: str | None = None) -> bytes:
    """
    The serialized ONNX model of ``policy``; ``source`` (the checkpoint) goes in its metadata.

    Raises:
        onnx.checker.ValidationError: If the model is not valid ONNX (a bug here, not in the input).
    """
    from onnx import TensorProto, checker, helper, numpy_helper

    nodes: list[Any] = []
    initializers: list[Any] = []

    def constant(name: str, values: npt.ArrayLike) -> str:
        array = np.asarray(values, dtype=np.float32)
        initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def node(op: str, inputs: list[str], output: str, **attributes: Any) -> str:
        nodes.append(helper.make_node(op, inputs, [output], **attributes))
        return output

    x = "observation"
    if policy.obs_mean is not None and policy.obs_scale is not None:
        x = node("Sub", [x, constant("obs_mean", policy.obs_mean)], "obs_centred")
        x = node("Div", [x, constant("obs_scale", policy.obs_scale)], "obs_standardized")
        bounds = [constant("obs_clip_low", -CLIP), constant("obs_clip_high", CLIP)]
        x = node("Clip", [x, *bounds], "obs_normalized")
    last = len(policy.layers) - 1
    for i, layer in enumerate(policy.layers):
        weights = [
            constant(f"layer{i}_weight", layer.weight),
            constant(f"layer{i}_bias", layer.bias),
        ]
        x = node("Gemm", [x, *weights], "outputs" if i == last else f"layer{i}_linear")
        if layer.relu:
            x = node("Relu", [x], f"layer{i}_relu")

    if policy.discrete:
        node("ArgMax", [x], "action", axis=1, keepdims=0, select_last_index=0)
        action = helper.make_tensor_value_info("action", TensorProto.INT64, ["batch"])
    else:
        assert policy.action_low is not None and policy.action_high is not None  # continuous head
        if policy.head == "tanh_scale":
            x = node("Tanh", [x], "squashed")
            x = node("Mul", [x, constant("action_half_width", policy.squash_half_width)], "scaled")
            x = node("Add", [x, constant("action_centre", policy.squash_centre)], "unclipped")
        # Clip takes one scalar bound; these are per action, and ±inf leaves that side open.
        x = node("Max", [x, constant("action_low", policy.action_low)], "above_low")
        node("Min", [x, constant("action_high", policy.action_high)], "action")
        action = helper.make_tensor_value_info(
            "action", TensorProto.FLOAT, ["batch", policy.output_dim]
        )

    graph = helper.make_graph(
        nodes,
        "greedy_policy",
        [
            helper.make_tensor_value_info(
                "observation", TensorProto.FLOAT, ["batch", policy.input_dim]
            )
        ],
        [
            helper.make_tensor_value_info(
                "outputs", TensorProto.FLOAT, ["batch", policy.output_dim]
            ),
            action,
        ],
        initializers,
        doc_string=_describe(policy),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="rl_core.export"
    )
    model.ir_version = IR_VERSION
    helper.set_model_props(
        model,
        {
            "agent": policy.agent_name,
            "source": source or "",
            "head": policy.head,
            "normalizes": "true" if policy.normalizes else "false",
            "observation_settings": json.dumps(dict(policy.observation_settings), sort_keys=True),
        },
    )
    checker.check_model(model, full_check=True)
    return bytes(model.SerializeToString())


def _describe(policy: InferencePolicy) -> str:
    action = (
        "action: int64 [batch], the index of the first largest output"
        if policy.discrete
        else f"action: float32 [batch, {policy.output_dim}], the outputs "
        + ("squashed into" if policy.head == "tanh_scale" else "clipped to")
        + " the action bounds"
    )
    settings = ", ".join(f"{k}: {v}" for k, v in sorted(policy.observation_settings.items()))
    return (
        f"{policy.agent_name or 'An agent'}'s greedy policy. "
        f"observation: float32 [batch, {policy.input_dim}], the environment's observation after "
        f"its settings ({settings or 'none'}), which this model does not apply. "
        f"outputs: float32 [batch, {policy.output_dim}], the network's outputs. {action}."
    )


def run_onnx_model(
    model: bytes, policy: InferencePolicy, observations: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
    """Run ``model`` in onnxruntime (CPU, default optimisation) on ``observations`` (B, input),
    rounded to float32. Returns its outputs (B, O) and actions ((B,) or (B, A)).

    One thread: a policy this size gains nothing from more, the result does not depend on how the
    batch was split between threads, and no idle worker threads are left in the process (a later
    ``fork``, as in an async vector environment, could deadlock on one)."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    obs = np.asarray(observations, dtype=np.float32).reshape(-1, policy.input_dim)
    outputs, actions = session.run(["outputs", "action"], {"observation": obs})
    return np.asarray(outputs, dtype=np.float64), np.asarray(actions)


def verify_onnx_model(
    model: bytes, policy: InferencePolicy, observations: npt.ArrayLike
) -> VerificationReport:
    """Run ``model`` on ``observations`` (see :func:`run_onnx_model`) and compare it with ``policy``
    computed in float64 on the same float32-rounded observations."""
    obs = np.asarray(observations, dtype=np.float32).reshape(-1, policy.input_dim)
    outputs, actions = run_onnx_model(model, policy, obs)
    return compare_with_policy(policy, obs, outputs, actions, target="ONNX")
