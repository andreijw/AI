"""Shared helpers for tests: running the train.py / record.py scripts, and gradient checks."""

import importlib
import sys
from collections.abc import Callable, Mapping

import numpy as np


def run_main(args, monkeypatch, module="train"):
    """Import a script module fresh and run its main() with the given sys.argv."""
    monkeypatch.setattr(sys, "argv", [f"{module}.py"] + args)
    # Force re-import so module-level imports pick up anything the test patched.
    sys.modules.pop(module, None)
    importlib.import_module(module).main()


def minimal_config(tmp_path):
    """Write a minimal YAML config to tmp_path and return its path."""
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
