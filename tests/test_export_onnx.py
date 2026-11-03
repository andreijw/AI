"""Tests for the ONNX export: the model's structure, what is particular to running it in onnxruntime,
and ``export.py --format onnx``. The checks every exporter must pass (agreement with NumPy for every
agent, huge observations, ties, infinite bounds) run for ONNX in ``tests/test_export.py``."""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import DDPGAgent, PPOAgent  # noqa: E402
from rl_core.export import onnx_model, verify  # noqa: E402
from tests.export_cases import (  # noqa: E402
    BOUNDS,
    OBS_DIM,
    make_agent,
    random_observations,
    run_export,
)
from tests.helpers import needs_onnx  # noqa: E402

INSTALL = 'pip install -e ".[onnx]"'


def _discrete_policy():
    return make_agent(
        PPOAgent, {"seed": 3, "hidden_dim": 6, "normalize_observations": True}
    ).inference_policy()


def _continuous_policy():
    return make_agent(DDPGAgent, {"seed": 3, "hidden_sizes": [7, 5], **BOUNDS}).inference_policy()


def _model(policy, source=None):
    import onnx

    return onnx.load_from_string(onnx_model.render_onnx_model(policy, source=source))


def _dims(value_info):
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


@needs_onnx
@pytest.mark.parametrize("make_policy", [_discrete_policy, _continuous_policy])
def test_model_passes_the_onnx_checker(make_policy):
    import onnx

    onnx.checker.check_model(_model(make_policy()), full_check=True)


@needs_onnx
def test_model_pins_its_opset_and_ir_version():
    """Pinned, so a model built with a newer onnx still loads in an older onnxruntime."""
    model = _model(_discrete_policy())
    assert model.ir_version == onnx_model.IR_VERSION == 8
    assert [(o.domain, o.version) for o in model.opset_import] == [("", onnx_model.OPSET)]
    assert onnx_model.OPSET == 17


@needs_onnx
def test_discrete_model_takes_a_batch_and_gives_outputs_and_an_index():
    from onnx import TensorProto

    graph = _model(_discrete_policy()).graph
    assert [i.name for i in graph.input] == ["observation"]
    assert graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert _dims(graph.input[0]) == ["batch", OBS_DIM]
    outputs = {o.name: o for o in graph.output}
    assert list(outputs) == ["outputs", "action"]
    assert _dims(outputs["outputs"]) == ["batch", 3]
    assert outputs["action"].type.tensor_type.elem_type == TensorProto.INT64
    assert _dims(outputs["action"]) == ["batch"]


@needs_onnx
def test_continuous_model_gives_an_action_vector():
    from onnx import TensorProto

    action = [o for o in _model(_continuous_policy()).graph.output if o.name == "action"][0]
    assert action.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _dims(action) == ["batch", 2]


@needs_onnx
def test_weights_are_float32_initializers_and_not_inputs():
    """Listed as graph inputs, weights could be overridden at run time and would not be constants."""
    from onnx import numpy_helper

    policy = _continuous_policy()
    graph = _model(policy).graph
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    assert not set(initializers) & {i.name for i in graph.input}
    for i, layer in enumerate(policy.layers):
        for name, values in (("weight", layer.weight), ("bias", layer.bias)):
            stored = initializers[f"layer{i}_{name}"]
            assert stored.dtype == np.float32
            np.testing.assert_array_equal(stored, values.astype(np.float32))


@needs_onnx
def test_model_names_the_agent_head_settings_and_source():
    agent = make_agent(PPOAgent, {"seed": 3, "hidden_dim": 4, "normalize_observations": True})
    agent.record_observation_settings({"frame_stack": 4}, {"frame_stack": 1})
    model = _model(agent.inference_policy(), source="agent.pt (sha256 abc123)")
    metadata = {p.key: p.value for p in model.metadata_props}
    assert metadata["agent"] == "PPOAgent"
    assert metadata["source"] == "agent.pt (sha256 abc123)"
    assert metadata["head"] == "argmax"
    assert metadata["normalizes"] == "true"
    assert json.loads(metadata["observation_settings"])["frame_stack"] == 4


@needs_onnx
def test_model_is_reproducible():
    policy = _continuous_policy()
    assert onnx_model.render_onnx_model(policy) == onnx_model.render_onnx_model(policy)


@needs_onnx
def test_one_observation_runs_as_a_batch_of_one():
    """The batch dimension is free: one row gives what the same row gives in a larger batch."""
    policy = _discrete_policy()
    model = onnx_model.render_onnx_model(policy)
    observations = random_observations(20)
    outputs, actions = onnx_model.run_onnx_model(model, policy, observations)
    one_output, one_action = onnx_model.run_onnx_model(model, policy, observations[3:4])
    assert one_output.shape == (1, 3) and one_action.shape == (1,)
    np.testing.assert_allclose(one_output[0], outputs[3], rtol=1e-5, atol=1e-6)
    assert one_action[0] == actions[3]


@needs_onnx
def test_verification_reports_onnx():
    policy = _discrete_policy()
    report = onnx_model.verify_onnx_model(
        onnx_model.render_onnx_model(policy), policy, random_observations(10)
    )
    assert report.summary().startswith("ONNX verification passed"), report.summary()


# ---------------------------------------------------------------------------------------------
# Which packages are missing
# ---------------------------------------------------------------------------------------------


def test_onnx_is_needed_even_without_verification(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    assert "onnx" in onnx_model.missing_packages(verify=False)


@needs_onnx
def test_onnxruntime_is_needed_only_to_verify(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert onnx_model.missing_packages(verify=True) == ["onnxruntime"]
    assert onnx_model.missing_packages(verify=False) == []


# ---------------------------------------------------------------------------------------------
# export.py --format onnx
# ---------------------------------------------------------------------------------------------


@needs_onnx
def test_export_writes_a_verified_onnx_model(tmp_path, monkeypatch, capsys):
    import onnx
    import onnxruntime

    output = run_export(tmp_path, monkeypatch, "onnx", extra=["--verify-episodes", "1"])
    onnx.checker.check_model(onnx.load(str(output)), full_check=True)
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    outputs, action = session.run(None, {"observation": np.zeros((2, 4), dtype=np.float32)})
    assert outputs.shape == (2, 2) and action.shape == (2,)
    assert "ONNX verification passed" in capsys.readouterr().out


@needs_onnx
def test_export_writes_an_unverified_onnx_model_naming_its_checkpoint(tmp_path, monkeypatch):
    import onnx

    output = run_export(tmp_path, monkeypatch, "onnx", extra=["--verify-episodes", "0"])
    metadata = {p.key: p.value for p in onnx.load(str(output)).metadata_props}
    digest = hashlib.sha256((tmp_path / "agent.pt").read_bytes()).hexdigest()
    assert digest in metadata["source"]
    assert sorted(os.listdir(tmp_path)) == ["agent.pt", "config.yaml", "out"]


def test_export_without_onnx_stops_before_loading_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """A checkpoint that does not exist: the missing package is reported before anything loads."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(SystemExit) as stop:
        run_export(tmp_path, monkeypatch, "onnx", checkpoint=str(tmp_path / "missing.pt"))
    assert stop.value.code == 1
    assert INSTALL in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@needs_onnx
def test_export_without_onnxruntime_needs_verification_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(SystemExit) as stop:
        run_export(tmp_path, monkeypatch, "onnx", extra=["--verify-episodes", "1"])
    assert stop.value.code == 1
    err = capsys.readouterr().err
    assert INSTALL in err and "--verify-episodes 0" in err
    assert not (tmp_path / "out").exists(), "an unverified model must not be left at --output"
    assert run_export(tmp_path, monkeypatch, "onnx", extra=["--verify-episodes", "0"]).exists()


@needs_onnx
def test_export_keeps_no_onnx_model_that_failed_verification(tmp_path, monkeypatch):
    def failing(model, policy, observations):
        n = len(observations)
        return verify.compare(
            np.zeros((n, 2)), np.zeros(n), np.ones((n, 2)), np.zeros(n), True, target="ONNX"
        )

    monkeypatch.setattr(onnx_model, "verify_onnx_model", failing)
    with pytest.raises(SystemExit) as stop:
        run_export(tmp_path, monkeypatch, "onnx", extra=["--verify-episodes", "1"])
    assert stop.value.code == 1
    assert not (tmp_path / "out" / "policy.onnx").exists()
