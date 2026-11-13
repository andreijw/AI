"""Shared helpers for tests: running the train.py / record.py scripts, gradient checks, and a
reference Adam optimiser."""

import importlib
import importlib.util
import math
import os
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest


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


def minimal_config(tmp_path, environment=None, training=None, agent=None):
    """
    Write a minimal YAML config to tmp_path and return its path.

    ``environment``, ``training`` and ``agent`` update those sections' keys; ``agent`` updates
    the agent section shallowly, so a ``config`` given there replaces the whole agent config.
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
    config["agent"].update(agent or {})
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


Ends = bool | np.ndarray | None


def _ends(value: Ends, dones: np.ndarray, n: int) -> np.ndarray | None:
    """``terminated`` as given: ``True`` copies ``dones``, ``False`` is all clear."""
    if isinstance(value, bool):
        return dones.copy() if value else np.zeros(n, dtype=bool)
    return value


def _transition_batch(
    rng: np.random.Generator,
    n: int,
    obs_dim: int,
    draw_actions: Callable[[], np.ndarray],
    reward_range: tuple[float, float] | None,
    rewards: np.ndarray | None,
    dones: np.ndarray | None,
    terminated: Ends,
) -> dict[str, np.ndarray]:
    # Drawn in this order: observations, actions, rewards, next observations.
    observations = rng.standard_normal((n, obs_dim))
    actions = draw_actions()
    if rewards is None:
        rewards = (
            rng.standard_normal(n) if reward_range is None else rng.uniform(*reward_range, size=n)
        )
    batch = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": rng.standard_normal((n, obs_dim)),
        "dones": np.array([False] * (n - 1) + [True]) if dones is None else dones,
    }
    ends = _ends(terminated, batch["dones"], n)
    if ends is not None:
        batch["terminated"] = ends
    return batch


def continuous_batch(
    n: int,
    obs_dim: int,
    action_dim: int,
    *,
    seed: int = 0,
    low: float | np.ndarray = -1.0,
    high: float | np.ndarray = 1.0,
    reward_range: tuple[float, float] | None = None,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
    terminated: Ends = None,
) -> dict[str, np.ndarray]:
    """A trainer batch of ``n`` random transitions with float32 actions in ``[low, high]``.

    Rewards are standard normal, uniform over ``reward_range``, or ``rewards`` as given (no draw).
    ``dones`` defaults to the last step only; ``terminated`` is left out when ``None``, copies
    ``dones`` when ``True`` and is all clear when ``False``.
    """
    rng = np.random.default_rng(seed)
    return _transition_batch(
        rng,
        n,
        obs_dim,
        lambda: rng.uniform(low, high, size=(n, action_dim)).astype(np.float32),
        reward_range,
        rewards,
        dones,
        terminated,
    )


def discrete_batch(
    n: int,
    obs_dim: int,
    n_actions: int,
    *,
    seed: int = 0,
    reward_range: tuple[float, float] | None = None,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
    terminated: Ends = None,
) -> dict[str, np.ndarray]:
    """``continuous_batch`` with integer actions in ``[0, n_actions)``."""
    rng = np.random.default_rng(seed)
    return _transition_batch(
        rng,
        n,
        obs_dim,
        lambda: rng.integers(0, n_actions, size=n),
        reward_range,
        rewards,
        dones,
        terminated,
    )


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


# Prefix of every fake challenge package a test writes; conftest.py drops them after each test.
FAKE_PACKAGE_PREFIX = "rl_fake"


def unique_name(prefix=FAKE_PACKAGE_PREFIX):
    """A package or key name no other test uses, so imports never collide."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def write_dist(root, dist, entry_points):
    """Write the installed metadata of distribution ``dist`` under ``root``: a ``.dist-info``
    directory declaring ``entry_points`` (name to module) in the challenge entry-point group."""
    from rl_core.challenges import registry

    info = root / f"{dist.replace('-', '_')}-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1.0\n", encoding="utf-8"
    )
    lines = "".join(f"{name} = {module}\n" for name, module in entry_points.items())
    (info / "entry_points.txt").write_text(
        f"[{registry.ENTRY_POINT_GROUP}]\n{lines}", encoding="utf-8"
    )


def entry_points_under(*roots):
    """The challenge entry points of the distributions written under ``roots``, in that order."""
    from importlib.metadata import distributions

    from rl_core.challenges import registry

    return [
        ep
        for root in roots
        for dist in distributions(path=[str(root)])
        for ep in dist.entry_points.select(group=registry.ENTRY_POINT_GROUP)
    ]


def first_party_entry_points(source):
    """``source``'s challenge entry points restricted to this repository's distributions, so a
    plugin installed in the same environment (the maintainer's game) never enters the tests."""
    from rl_core.challenges import registry

    def first_party():
        return [ep for ep in source() if registry.dist_name(ep) in registry.FIRST_PARTY_DISTS]

    return first_party


# Prefix for the code of a child Python process that runs discovery: the same first-party restriction
# conftest.py applies to this process, without importing this module (which a footprint test must not).
FIRST_PARTY_DISCOVERY = (
    "from rl_core.challenges import registry as _registry; "
    "_all_entry_points = _registry.challenge_entry_points; "
    "_registry.challenge_entry_points = lambda: [ep for ep in _all_entry_points() "
    "if _registry.dist_name(ep) in _registry.FIRST_PARTY_DISTS]; "
)


def foreign_checkout_message(rootpath, module_file):
    """Why tests under ``rootpath`` must not run when rl_core comes from ``module_file``, or None.

    A git worktree whose tools run from the main checkout's environment imports the main
    checkout's code: its tests would pass or fail on code they were not written against. An
    installed (not editable) copy is refused as well: CI's Packages job tests the built
    rl-navigation wheel with ``--noconftest``, without this guard and without rl_core."""
    root, module = Path(rootpath).resolve(), Path(module_file).resolve()
    if root in module.parents:
        return None
    return (
        f"tests under {root} are importing another checkout's code (rl_core from {module}): "
        "use this worktree's own environment, made by scripts/worktree_venv.sh"
    )


# A committed config that sets ``backend: torch`` builds only where PyTorch is installed (.venv-gpu),
# never in .venv or CI.
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="the config sets backend: torch"
)

# A Box2D environment (LunarLander, BipedalWalker) builds only where box2d-py is installed
# (./setup.sh), never in CI.
needs_box2d = pytest.mark.skipif(
    importlib.util.find_spec("Box2D") is None, reason="Box2D is not installed (./setup.sh)"
)

# The C export's compiled tests need a C compiler: $CC, cc, gcc or clang (CI's Linux runner has gcc;
# test_export.test_ci_has_a_c_compiler holds it to that). MSVC's cl is not supported.
needs_cc = pytest.mark.skipif(
    sys.platform == "win32"
    or not (os.environ.get("CC") or any(shutil.which(cc) for cc in ("cc", "gcc", "clang"))),
    reason="no C compiler (set CC, or install gcc or clang)",
)

# The ONNX export needs the onnx extra (pip install -e "./packages/rl-core[onnx]"): onnx builds the model, onnxruntime
# checks it. CI installs it. Importing onnxruntime starts a thread that never ends, and a fork after it
# (an async vector environment) may deadlock, so tests/conftest.py runs these tests last.
NEEDS_ONNX_REASON = 'the onnx extra is not installed (pip install -e "./packages/rl-core[onnx]")'
needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason=NEEDS_ONNX_REASON,
)


# Writing an MP4 needs imageio-ffmpeg, from the video extra (pip install -e "./packages/rl-core[video]"); CI does not
# install it.
needs_video = pytest.mark.skipif(
    importlib.util.find_spec("imageio_ffmpeg") is None,
    reason='the video extra is not installed (pip install -e "./packages/rl-core[video]")',
)


def uses_onnx(item: pytest.Item) -> bool:
    """Whether a collected test is marked ``needs_onnx``."""
    return any(m.kwargs.get("reason") == NEEDS_ONNX_REASON for m in item.iter_markers("skipif"))


def env_marks(environment_id: str) -> list[pytest.MarkDecorator]:
    """The marks a test that builds ``environment_id`` needs: ``needs_box2d`` for Gymnasium's Box2D
    environments (the rule the wrappers' install hint uses), else none."""
    from rl_core.environments import uses_box2d

    return [needs_box2d] if uses_box2d(environment_id) else []


def _build_challenge_env(challenge_cls: type, **settings: object) -> object:
    """``challenge_cls``'s wrapper from environment-config ``settings``, without rendering."""
    return challenge_cls().make_env(settings, render_mode=None)


def challenge_env_factories() -> list[object]:
    """One ``pytest.param`` per registered challenge, with the id of its key and the marks its
    environment needs (:func:`env_marks`): a factory taking environment-config settings as keywords
    (``seed``, ``obs_noise_std``, ``frame_stack`` ...) and returning the challenge's wrapper.

    A test parametrized over it covers a new challenge package without being edited."""
    from functools import partial

    from rl_core.challenges import discover_challenges

    return [
        pytest.param(partial(_build_challenge_env, cls), marks=env_marks(cls().name), id=key)
        for key, cls in sorted(discover_challenges().items())
    ]


def torch_td3_agent() -> type:
    """The PyTorch TD3 class, imported only when called: torch is optional."""
    from rl_core.agents.torch_td3_agent import TorchTD3Agent

    return TorchTD3Agent


def torch_sac_agent() -> type:
    """The PyTorch SAC class, imported only when called: torch is optional."""
    from rl_core.agents.torch_sac_agent import TorchSACAgent

    return TorchSACAgent


def torch_backend_cases(ddpg_config: Mapping[str, object]) -> list[tuple[type, dict[str, object]]]:
    """The PyTorch DDPG, TD3 and SAC backends as agent-contract cases, float64 on the CPU, where
    PyTorch is installed; none elsewhere (.venv and CI). Append them to a list of DDPG cases."""
    if importlib.util.find_spec("torch") is None:
        return []
    from rl_core.agents.torch_ddpg_agent import TorchDDPGAgent
    from rl_core.agents.torch_sac_agent import TorchSACAgent
    from rl_core.agents.torch_td3_agent import TorchTD3Agent

    config = {**ddpg_config, "backend": "torch", "device": "cpu", "dtype": "float64"}
    return [(TorchDDPGAgent, config), (TorchTD3Agent, dict(config)), (TorchSACAgent, dict(config))]
