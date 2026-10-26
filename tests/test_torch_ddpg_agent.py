"""The PyTorch DDPG backend trains exactly as the NumPy DDPGAgent does, and shares its checkpoints.

Skipped unless PyTorch is installed (it is not in .venv or CI); the CUDA tests also need a GPU.
Run them from the gpu profile's environment (./setup.sh --profile gpu):

    .venv-gpu/bin/python -m pytest tests/test_torch_ddpg_agent.py --no-cov
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import DDPGAgent  # noqa: E402
from rl_core.agents.torch_ddpg_agent import TorchDDPGAgent  # noqa: E402
from rl_core.challenges import discover_challenges  # noqa: E402
from tests.agent_internals import weights  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

OBS, ACT, EPISODE = 3, 2, 20
CONFIG = {
    "seed": 7,
    "hidden_sizes": [8, 6],
    "learning_rate": 0.01,
    "gamma": 0.9,
    "tau": 0.05,
    "batch_size": 16,
    "buffer_capacity": 200,
    "learning_starts": 25,  # more than one episode, so the first update only warms up
    "noise_std": 0.3,
    "updates_per_step": 4,  # 80 gradient steps per update: more than one staging chunk (64)
    "action_low": [-2.0, -1.0],
    "action_high": [2.0, 1.0],
}
CPU64 = {"backend": "torch", "device": "cpu", "dtype": "float64"}


def _build(**overrides):
    config = {**CONFIG, **overrides}
    return DDPGAgent.class_for_config(config)(OBS, ACT, config)


def _episode(rng, agent_list, ends_terminal):
    """Play one synthetic episode with every agent in lockstep; return its batch and the actions."""
    obs = rng.normal(size=(EPISODE + 1, OBS))
    actions = [np.array([a.select_action(o, training=True) for o in obs[:-1]]) for a in agent_list]
    terminated = np.zeros(EPISODE, dtype=bool)
    terminated[-1] = ends_terminal
    batch = {
        "observations": obs[:-1],
        "rewards": rng.normal(size=EPISODE),
        "next_observations": obs[1:],
        "terminated": terminated,
    }
    return batch, actions


def _train_in_lockstep(agents, episodes, seed=0):
    """Same episodes for every agent; each stores its own actions. Returns each update's metrics."""
    rng = np.random.default_rng(seed)
    history = []
    for episode in range(episodes):
        batch, actions = _episode(rng, agents, ends_terminal=episode % 2 == 1)
        history.append(
            [
                agent.update({**batch, "actions": act})
                for agent, act in zip(agents, actions, strict=True)
            ]
        )
    return history


def _assert_same(expected, actual, rtol=1e-9, atol=1e-12, what=""):
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name], value, rtol=rtol, atol=atol, err_msg=f"{what}{name} differs"
        )


def _assert_same_checkpoint(expected_agent, actual_agent, tmp_path, rtol=1e-9, atol=1e-12):
    """Every array the two agents would save, Adam moments and step counts included."""
    expected_agent.save(str(tmp_path / "expected.pt"))
    actual_agent.save(str(tmp_path / "actual.pt"))
    expected, actual = np.load(tmp_path / "expected.pt"), np.load(tmp_path / "actual.pt")
    assert sorted(actual.files) == sorted(expected.files)
    for key in expected.files:
        if expected[key].dtype.kind in "fc":
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=rtol, atol=atol, err_msg=key
            )
        else:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("max_grad_norm", [None, 0.05])  # 0.05 clips every step here
def test_float64_cpu_backend_trains_exactly_as_the_numpy_agent(max_grad_norm, tmp_path):
    """Same seed, same episodes: same metrics, then the same checkpoint (weights, Adam moments and
    step counts, counters, buffer, generators) and the same greedy action."""
    numpy_agent = _build(max_grad_norm=max_grad_norm)
    torch_agent = _build(max_grad_norm=max_grad_norm, **CPU64)
    assert type(torch_agent) is TorchDDPGAgent

    history = _train_in_lockstep([numpy_agent, torch_agent], episodes=4)

    assert history[0] == [{}, {}], "Both should still be warming up after the first episode."
    for step, (expected, actual) in enumerate(history[1:], start=1):
        assert set(actual) == set(expected) and expected, f"episode {step} trained both"
        _assert_same(expected, actual, what=f"episode {step} metric ")
    _assert_same_checkpoint(numpy_agent, torch_agent, tmp_path)
    obs = np.array([0.4, -0.3, 0.2])
    np.testing.assert_allclose(
        torch_agent.select_action(obs, training=False),
        numpy_agent.select_action(obs, training=False),
        rtol=1e-6,
    )


def test_a_torch_checkpoint_loads_into_the_numpy_agent_and_back(tmp_path):
    """Checkpoints are one format: saved from either backend, every array is the same."""
    torch_agent = _build(**CPU64)
    _train_in_lockstep([torch_agent], episodes=3)
    torch_agent.save(str(tmp_path / "torch.pt"))

    numpy_agent = _build(seed=99)
    numpy_agent.load(str(tmp_path / "torch.pt"))
    numpy_agent.save(str(tmp_path / "numpy.pt"))
    back = _build(seed=98, **CPU64)
    back.load(str(tmp_path / "numpy.pt"))
    back.save(str(tmp_path / "back.pt"))

    first, second = np.load(tmp_path / "torch.pt"), np.load(tmp_path / "back.pt")
    assert (
        sorted(first.files) == sorted(np.load(tmp_path / "numpy.pt").files) == sorted(second.files)
    )
    for key in first.files:
        np.testing.assert_array_equal(second[key], first[key], err_msg=f"{key} changed on the way")


def test_a_loaded_torch_agent_continues_as_its_numpy_twin(tmp_path):
    """After loading, the device holds what the file holds: training continues identically."""
    source = _build()
    _train_in_lockstep([source], episodes=3)
    source.save(str(tmp_path / "start.pt"))
    numpy_agent, torch_agent = _build(seed=1), _build(seed=2, **CPU64)
    numpy_agent.load(str(tmp_path / "start.pt"))
    torch_agent.load(str(tmp_path / "start.pt"))

    _train_in_lockstep([numpy_agent, torch_agent], episodes=2, seed=5)

    _assert_same(weights(numpy_agent), weights(torch_agent), what="after loading, weight ")


def test_challenges_build_the_torch_agent_with_their_action_bounds():
    challenge = discover_challenges()["pendulum"]()
    env = challenge.make_env({"seed": 1}, render_mode=None)
    try:
        agent = challenge.make_agent("ddpg", {"seed": 1, "hidden_sizes": [4], **CPU64}, env)
    finally:
        env.close()
    assert type(agent) is TorchDDPGAgent
    action = agent.select_action(np.zeros(3), training=False)
    assert action.shape == (1,) and -2.0 <= action[0] <= 2.0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"device": "tpu"}, "device must be one of"),
        ({"dtype": "float16"}, "dtype must be one of"),
        ({"cuda_graph": "yes"}, "cuda_graph must be true, false or null"),
        ({"device": "cpu", "cuda_graph": True}, "cuda_graph needs device cuda"),
    ],
)
def test_invalid_torch_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _build(backend="torch", **overrides)


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has CUDA")
def test_device_cuda_without_a_gpu_is_rejected():
    with pytest.raises(ValueError, match="device cuda, but CUDA is not available"):
        _build(backend="torch", device="cuda")


@needs_cuda
def test_the_cuda_graph_trains_as_the_same_step_run_eagerly():
    """Replaying the captured step equals running it eagerly; capture itself trains nothing."""
    graphed = _build(backend="torch", device="cuda", cuda_graph=True)
    eager = _build(backend="torch", device="cuda", cuda_graph=False)
    numpy_agent = _build()

    _train_in_lockstep([graphed, eager, numpy_agent], episodes=4)

    # Capture runs warm-up steps; had they trained, the graphed agent would be ahead of the eager one.
    _assert_same(
        weights(eager), weights(graphed), rtol=1e-5, atol=1e-6, what="graph against eager, "
    )
    _assert_same(
        weights(numpy_agent), weights(graphed), rtol=1e-3, atol=1e-4, what="float32 against NumPy, "
    )


@needs_cuda
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_float64_on_cuda_trains_as_the_numpy_agent(cuda_graph, tmp_path):
    """float64 on the GPU is a reference run too: Adam's step count must not be float32."""
    numpy_agent = _build()
    torch_agent = _build(backend="torch", device="cuda", dtype="float64", cuda_graph=cuda_graph)
    _train_in_lockstep([numpy_agent, torch_agent], episodes=4)
    _assert_same_checkpoint(numpy_agent, torch_agent, tmp_path)


@needs_cuda
def test_loading_into_a_captured_agent_retargets_its_graph(tmp_path):
    """A checkpoint loaded after capture is what the graph trains from, not the old tensors."""
    source = _build()
    _train_in_lockstep([source], episodes=3)
    source.save(str(tmp_path / "start.pt"))

    captured = _build(seed=3, backend="torch", device="cuda", cuda_graph=True)
    _train_in_lockstep([captured], episodes=3, seed=9)  # captures its graph
    captured.load(str(tmp_path / "start.pt"))
    fresh = _build(seed=4, backend="torch", device="cuda", cuda_graph=True)
    fresh.load(str(tmp_path / "start.pt"))

    _train_in_lockstep([captured, fresh], episodes=2, seed=5)

    _assert_same(weights(fresh), weights(captured), rtol=1e-6, atol=1e-7, what="after loading, ")
