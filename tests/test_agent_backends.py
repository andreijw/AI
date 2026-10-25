"""Choosing an agent's backend with the ``backend`` config key (BaseAgent.class_for_config).

These run with or without PyTorch: the torch backend's own tests are in
tests/test_torch_ddpg_agent.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_core.agents import DDPGAgent, PPOAgent  # noqa: E402
from rl_core.challenges import discover_challenges  # noqa: E402
from tests.test_agent_harness import ALL_AGENTS, CONTINUOUS_AGENTS  # noqa: E402

NUMPY_AGENTS = sorted(
    {cls for cls, _ in ALL_AGENTS + CONTINUOUS_AGENTS if cls.BACKEND == "numpy"},
    key=lambda cls: cls.__name__,
)

DDPG_CONFIG = {"seed": 1, "hidden_sizes": [4], "action_low": [-2.0], "action_high": [2.0]}


@pytest.mark.parametrize("agent_cls", NUMPY_AGENTS)
@pytest.mark.parametrize("config", [{}, {"backend": "numpy"}])
def test_the_numpy_backend_is_every_agents_default(agent_cls, config):
    assert agent_cls.class_for_config(config) is agent_cls, (
        f"{agent_cls.__name__} should build itself when the config asks for no backend or numpy."
    )


def test_an_agent_without_a_torch_version_rejects_backend_torch():
    """backend: torch on an agent with no PyTorch version is an error, not silently NumPy."""
    with pytest.raises(
        ValueError, match=r"backend must be one of \['numpy'\] for PPOAgent, got 'torch'"
    ):
        PPOAgent.class_for_config({"backend": "torch"})


def test_an_unknown_backend_is_rejected_with_the_allowed_ones():
    with pytest.raises(
        ValueError, match=r"backend must be one of \['numpy', 'torch'\] for DDPGAgent, got 'jax'"
    ):
        DDPGAgent.class_for_config({"backend": "jax"})


def test_a_subclass_does_not_inherit_its_parents_backends():
    """A future TD3Agent(DDPGAgent) must not build TorchDDPGAgent for backend: torch."""

    class SubDDPG(DDPGAgent):
        pass

    with pytest.raises(ValueError, match=r"backend must be one of \['numpy'\] for SubDDPG"):
        SubDDPG.class_for_config({"backend": "torch"})


def test_building_a_class_against_its_configs_backend_is_rejected():
    """DDPGAgent built directly with backend: torch would silently train in NumPy."""
    with pytest.raises(
        ValueError, match="DDPGAgent is the numpy backend; the config asks for 'torch'"
    ):
        DDPGAgent(3, 1, {**DDPG_CONFIG, "backend": "torch"})


def test_backend_torch_without_pytorch_names_the_install_command(monkeypatch):
    """Without PyTorch, choosing it fails at once with the command that installs it."""
    # import torch now raises, whether PyTorch is installed or not
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "rl_core.agents.torch_ddpg_agent", raising=False)
    with pytest.raises(
        ImportError, match=r"backend: torch needs PyTorch.*pip install -e '\.\[torch\]'"
    ):
        DDPGAgent.class_for_config({"backend": "torch"})


def test_challenges_build_the_class_the_config_chooses():
    """make_agent goes through class_for_config, so a challenge config's backend key is honoured."""
    challenge = discover_challenges()["pendulum"]()
    env = challenge.make_env({"seed": 1}, render_mode=None)
    try:
        with pytest.raises(ValueError, match="got 'jax'"):
            challenge.make_agent("ddpg", {**DDPG_CONFIG, "backend": "jax"}, env)
        assert type(challenge.make_agent("ddpg", DDPG_CONFIG, env)) is DDPGAgent
    finally:
        env.close()


def test_a_backend_that_is_not_the_agent_is_refused(monkeypatch):
    """BACKENDS must name a subclass of the agent serving that backend, or nothing is built."""
    monkeypatch.setattr(DDPGAgent, "BACKENDS", {"torch": "rl_core.agents.ppo_agent:PPOAgent"})
    with pytest.raises(TypeError, match="is not DDPGAgent in the torch backend"):
        DDPGAgent.class_for_config({"backend": "torch"})


def test_a_missing_module_other_than_the_library_is_not_disguised(monkeypatch):
    """Only the backend's own library missing gets the install hint; any other import error shows."""
    monkeypatch.setattr(DDPGAgent, "BACKENDS", {"torch": "rl_core.agents.no_such_module:X"})
    with pytest.raises(ModuleNotFoundError, match="rl_core.agents.no_such_module"):
        DDPGAgent.class_for_config({"backend": "torch"})
