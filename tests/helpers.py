"""Shared helpers for tests: running the train.py / record.py scripts, gradient checks, and a
reference Adam optimiser."""

import importlib
import importlib.util
import math
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np


def run_main(args, monkeypatch, module="train"):
    """Import a script module fresh and run its main() with the given sys.argv."""
    monkeypatch.setattr(sys, "argv", [f"{module}.py"] + args)
    # Force re-import so module-level imports pick up anything the test patched.
    sys.modules.pop(module, None)
    importlib.import_module(module).main()


def load_script(name):
    """Import ``scripts/<name>.py`` by path; ``scripts/`` is not a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_config(tmp_path, environment=None, training=None):
    """
    Write a minimal YAML config to tmp_path and return its path.

    ``environment`` and ``training`` update those sections' keys.
    """
    import yaml

    config = {
        "environment": {
            "name": "CartPole-v1",
            "render_mode": None,
            "max_episode_steps": 20,
            "seed": 42,
        },
        "agent": {
            "type": "random",
            "config": {"learning_rate": 0.0003, "gamma": 0.99},
        },
        "training": {
            "num_episodes": 4,
            "max_steps_per_episode": 20,
            "eval_frequency": 2,
            "save_frequency": 10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
        "logging": {"level": "INFO", "log_metrics": True},
    }
    config["environment"].update(environment or {})
    config["training"].update(training or {})
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def numerical_grad(
    loss_fn: Callable[[], float], param: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    Central-difference gradient of a scalar loss with respect to one parameter array.

    Args:
        loss_fn: Recomputes the loss from the current contents of ``param`` (and anything else
            it closes over). It must be deterministic: fixed batch, no sampling.
        param: The float64 array to differentiate with respect to. It is perturbed in place,
            one element at a time, and restored before returning.
        eps: Perturbation size.

    Returns:
        Array shaped like ``param`` holding d(loss)/d(param).

    Raises:
        ValueError: If ``param`` is not float64; float32 is too coarse for a 1e-5 perturbation.
    """
    if param.dtype != np.float64:
        raise ValueError(f"numerical_grad needs a float64 parameter, got {param.dtype}.")
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps
        loss_plus = loss_fn()
        param[idx] = original - eps
        loss_minus = loss_fn()
        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def assert_gradients_match(
    loss_fn: Callable[[], float],
    params: Mapping[str, np.ndarray],
    analytic_grads: Mapping[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """
    Assert that hand-derived gradients agree with finite differences for every parameter.

    Args:
        loss_fn: Recomputes the scalar loss from the current parameter values.
        params: Name -> the live float64 parameter array the loss reads.
        analytic_grads: Name -> the gradient the code under test computed for that parameter.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: Naming the first parameter whose gradient is missing or disagrees.
    """
    for name, param in params.items():
        assert name in analytic_grads, f"Parameter {name!r} has no analytic gradient to check."
        np.testing.assert_allclose(
            analytic_grads[name],
            numerical_grad(loss_fn, param),
            rtol=rtol,
            atol=atol,
            err_msg=f"Analytic gradient for parameter {name!r} disagrees with finite differences.",
        )


class ReferenceAdam:
    """Adam with optional global gradient-norm clipping, written out independently of the agents.

    ``step`` returns the amount to subtract from each parameter; the caller applies it.
    """

    def __init__(self, learning_rate: float, max_grad_norm: float | None = None):
        self.learning_rate, self.max_grad_norm = learning_rate, max_grad_norm
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, grads: Mapping[str, np.ndarray | float]) -> dict[str, np.ndarray]:
        self.t += 1
        clip = 1.0
        if self.max_grad_norm is not None:
            norm = math.sqrt(sum(float(np.sum(np.square(g))) for g in grads.values()))
            clip = min(1.0, self.max_grad_norm / (norm + 1e-6))
        steps = {}
        for name, raw in grads.items():
            gradient = np.asarray(raw, dtype=np.float64) * clip
            self.m[name] = 0.9 * self.m.get(name, 0.0) + 0.1 * gradient
            self.v[name] = 0.999 * self.v.get(name, 0.0) + 0.001 * gradient**2
            m_hat = self.m[name] / (1.0 - 0.9**self.t)
            v_hat = self.v[name] / (1.0 - 0.999**self.t)
            steps[name] = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return steps


def assert_same_checkpoint(expected, actual, tmp_path) -> None:
    """Assert two agents hold the same state by saving both and comparing every stored array.

    Goes through ``save`` only, so it holds for any agent and survives renamed attributes.
    """
    expected_path, actual_path = tmp_path / "expected.npz", tmp_path / "actual.npz"
    expected.save(str(expected_path))
    actual.save(str(actual_path))
    with np.load(expected_path) as want, np.load(actual_path) as got:
        assert sorted(got.files) == sorted(want.files), "The checkpoints hold different keys."
        for key in want.files:
            np.testing.assert_array_equal(got[key], want[key], err_msg=f"{key} differs.")


def write_package(root, name, benchmark, init=True):
    """Create package ``name`` under ``root``, with ``benchmark.py`` holding ``benchmark`` if given."""
    package = root / name
    package.mkdir()
    if init:
        (package / "__init__.py").write_text("", encoding="utf-8")
    if benchmark is not None:
        (package / "benchmark.py").write_text(benchmark, encoding="utf-8")


def unique_name(prefix="rl_fake"):
    """A package or key name no other test uses, so imports never collide."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def torch_backend_cases(ddpg_config: Mapping[str, object]) -> list[tuple[type, dict[str, object]]]:
    """The PyTorch DDPG backend as an agent-contract case, float64 on the CPU, where PyTorch is
    installed; none elsewhere (.venv and CI). Append it to a list of DDPG cases."""
    if importlib.util.find_spec("torch") is None:
        return []
    from rl_core.agents.torch_ddpg_agent import TorchDDPGAgent

    return [
        (TorchDDPGAgent, {**ddpg_config, "backend": "torch", "device": "cpu", "dtype": "float64"})
    ]
