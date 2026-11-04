"""The agents and exporters the export tests share: ``tests/test_export.py`` (the NumPy reference, the C
header and the checks common to every format) and ``tests/test_export_onnx.py`` (the ONNX model)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import (  # noqa: E402
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    ReinforceAgent,
    TD3Agent,
)
from rl_core.agents.inference import DenseLayer  # noqa: E402
from rl_core.export import c_header, onnx_model, verify  # noqa: E402
from tests.helpers import minimal_config, needs_cc, needs_onnx, run_main  # noqa: E402

OBS_DIM = 5
BOUNDS = {"action_low": [-2.0, -0.5], "action_high": [2.0, 1.5]}
DISCRETE = [
    (ReinforceAgent, {"seed": 3, "hidden_dim": 6}),
    (ActorCriticAgent, {"seed": 3, "hidden_dim": 6}),
    (PPOAgent, {"seed": 3, "hidden_dim": 6}),
    (PPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True}),
    (DQNAgent, {"seed": 3, "hidden_dim": 6}),
    (DQNAgent, {"seed": 3, "hidden_dim": 6, "dueling": True}),
    (DQNAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True}),
]
CONTINUOUS = [
    (ContinuousActorCriticAgent, {"seed": 3, "hidden_dim": 6}),
    (ContinuousActorCriticAgent, {"seed": 3, "hidden_dim": 6, "bounded_mean": True, **BOUNDS}),
    (ContinuousPPOAgent, {"seed": 3, "hidden_dim": 6, **BOUNDS}),
    (ContinuousPPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True, **BOUNDS}),
    (DDPGAgent, {"seed": 3, "hidden_sizes": [7, 5], **BOUNDS}),
    (DDPGAgent, {"seed": 3, "hidden_sizes": [7, 5], "normalize_observations": True, **BOUNDS}),
    (TD3Agent, {"seed": 3, "hidden_sizes": [6], **BOUNDS}),
]
EXPORTABLE = DISCRETE + CONTINUOUS
# Cases whose action passes through tanh or whose input is normalised: huge inputs must saturate.
SQUASHING = [
    c for c in CONTINUOUS if c[1].get("bounded_mean") or c[1].get("normalize_observations")
]
SQUASHING += [c for c in CONTINUOUS if c[0] is DDPGAgent]
NORMALISING = [c for c in EXPORTABLE if c[1].get("normalize_observations")]


def case_id(case):
    cls, config = case
    options = [k for k in ("dueling", "bounded_mean", "normalize_observations") if config.get(k)]
    return "-".join([cls.__name__, *options])


def action_dim(cls):
    return 3 if any(cls is c for c, _ in DISCRETE) else 2


def random_observations(n=200, seed=0, scale=1.5):
    return np.random.default_rng(seed).normal(0.0, scale, size=(n, OBS_DIM)).astype(np.float32)


def make_agent(cls, config):
    """An agent with its initial weights; a normalising one has seen 300 shifted observations, so
    its statistics are far from the identity."""
    agent = cls(observation_dim=OBS_DIM, action_dim=action_dim(cls), config=dict(config))
    if config.get("normalize_observations"):
        for obs in random_observations(300, seed=9, scale=3.0) + 2.0:
            agent.normalize_observation(obs, training=True)
    return agent


def dense_layer(n_in, n_out, relu=True, value=0.1):
    return DenseLayer(weight=np.full((n_in, n_out), value), bias=np.zeros(n_out), relu=relu)


def make_checkpoint(tmp_path, cls=PPOAgent, config=None):
    """A checkpoint for the 4-observation, 2-action environment of ``minimal_config``."""
    path = str(tmp_path / "agent.pt")
    cls(observation_dim=4, action_dim=2, config=config or {}).save(path)
    return path


OUTPUT_NAMES = {"c": "policy_weights.h", "onnx": "policy.onnx"}


def run_export(
    tmp_path, monkeypatch, export_format="c", agent_type="ppo", extra=(), checkpoint=None
):
    """Run export.py on ``minimal_config``; returns the --output path (tmp_path/out/...). C passes
    no --format, so the C tests run the default."""
    output = tmp_path / "out" / OUTPUT_NAMES[export_format]
    chosen = [] if export_format == "c" else ["--format", export_format]
    run_main(
        ["--config", minimal_config(tmp_path), "--agent-type", agent_type, *chosen]
        + ["--checkpoint", checkpoint or make_checkpoint(tmp_path), "--output", str(output)]
        + list(extra),
        monkeypatch,
        module="export",
    )
    return output


def check_c(policy, observations, workdir):
    """Compile the policy's C header (warnings are errors) and compare it with NumPy."""
    header = c_header.render_c_header(policy)
    return verify.verify_c_header(header, policy, observations, workdir=workdir, werror=True)


def check_onnx(policy, observations, workdir):
    """Run the policy's ONNX model in onnxruntime and compare it with NumPy."""
    return onnx_model.verify_onnx_model(onnx_model.render_onnx_model(policy), policy, observations)


# Each exporter as (policy, observations, workdir) -> VerificationReport, for the checks every
# format must pass.
EXPORTERS = [
    pytest.param(check_c, marks=needs_cc, id="c"),
    pytest.param(check_onnx, marks=needs_onnx, id="onnx"),
]
