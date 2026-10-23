"""train.py and record.py choose the challenge through the registry and name none themselves."""

import importlib
import os
import re
from pathlib import Path

import pytest
import yaml

from rl_core.agents import RandomAgent
from rl_core.challenges import discover_challenges
from tests.helpers import minimal_config, run_main, unique_name, write_package

REPO_ROOT = Path(__file__).resolve().parent.parent

# A challenge neither script has heard of: CartPole under another id, with its own agent name.
FAKE_CHALLENGE = """
from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import RandomAgent
from rl_core.challenges import BenchmarkChallenge, register_challenge

CALLS = []


@register_challenge
class FakeChallenge(BenchmarkChallenge):
    name = "{env_name}"
    key = "{key}"
    solve_threshold = None
    default_episodes = 1
    supported_agents = ["scripted"]

    def make_env(self, env_config, render_mode):
        CALLS.append(("env", env_config["name"], render_mode))
        return CartPoleEnv(render_mode=render_mode, max_episode_steps=5, seed=env_config.get("seed"))

    def make_agent(self, agent_type, agent_config, env):
        CALLS.append(("agent", agent_type))
        return RandomAgent(observation_dim=4, action_dim=2, config=dict(agent_config))
"""


pytestmark = pytest.mark.usefixtures("clear_sdl")


@pytest.fixture
def fake_challenge(isolated_registry):
    """Register a challenge package under tmp_path; return (module, env_name, key)."""
    package, key = unique_name(), unique_name("key")
    env_name = f"Fake{key}-v0"
    write_package(isolated_registry, package, FAKE_CHALLENGE.format(env_name=env_name, key=key))
    discover_challenges(isolated_registry)
    return importlib.import_module(f"{package}.benchmark"), env_name, key


def _config(tmp_path, **environment):
    """minimal_config with the environment section changed; ``None`` removes a key."""
    path = minimal_config(tmp_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for name, value in environment.items():
        if value is None:
            data["environment"].pop(name, None)
        else:
            data["environment"][name] = value
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def test_train_runs_a_challenge_it_does_not_name(fake_challenge, tmp_path, monkeypatch):
    module, env_name, key = fake_challenge
    path = _config(tmp_path, name=env_name)

    run_main(
        ["--config", path, "--agent-type", "scripted", "--challenge", key, "--num-episodes", "1"],
        monkeypatch,
    )

    calls = module.CALLS
    # the training environment, then the separate evaluation environment, then the agent
    assert calls == [("env", env_name, None), ("env", env_name, None), ("agent", "scripted")]
    assert os.path.isdir(tmp_path / "logs" / key / "scripted"), "outputs go under the challenge key"


def test_train_picks_the_challenge_from_the_environment_name(fake_challenge, tmp_path, monkeypatch):
    module, env_name, _ = fake_challenge
    path = _config(tmp_path, name=env_name)

    run_main(["--config", path, "--agent-type", "scripted", "--num-episodes", "1"], monkeypatch)

    assert ("agent", "scripted") in module.CALLS


def test_record_plays_back_a_challenge_it_does_not_name(fake_challenge, tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    module, env_name, key = fake_challenge
    checkpoint = str(tmp_path / "agent.pt")
    RandomAgent(observation_dim=4, action_dim=2, config={}).save(checkpoint)
    video_dir = tmp_path / "videos"

    run_main(
        ["--config", _config(tmp_path, name=env_name), "--agent-type", "scripted"]
        + ["--checkpoint", checkpoint, "--episodes", "1", "--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )

    calls = module.CALLS
    assert calls[0] == ("env", env_name, "rgb_array")
    assert os.listdir(video_dir) == [f"{key}-playback-full.mp4"]


@pytest.mark.parametrize("module", ["train", "record"])
def test_unknown_environment_name_is_an_error(tmp_path, monkeypatch, module):
    """The bug fixed then: an unknown id (Acrobot-v1 at the time) silently trained CartPole."""
    args = ["--config", _config(tmp_path, name="Taxi-v3")]
    if module == "record":
        pytest.importorskip("moviepy")
        args += ["--checkpoint", str(tmp_path / "missing.pt")]
    with pytest.raises(ValueError, match="Unknown environment 'Taxi-v3'.*CartPole-v1"):
        run_main(args, monkeypatch, module=module)


def test_challenge_that_disagrees_with_the_config_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="CartPole-v1.*pendulum|pendulum.*CartPole-v1"):
        run_main(["--config", minimal_config(tmp_path), "--challenge", "pendulum"], monkeypatch)


def test_config_without_environment_name_needs_a_challenge(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="environment.name"):
        run_main(["--config", _config(tmp_path, name=None)], monkeypatch)


def test_config_without_environment_name_uses_the_challenge_given(tmp_path, monkeypatch):
    path = _config(tmp_path, name=None)
    run_main(["--config", path, "--challenge", "cartpole", "--num-episodes", "1"], monkeypatch)
    assert os.path.isdir(tmp_path / "logs" / "cartpole" / "random")


def test_agent_the_challenge_does_not_support_is_an_error(tmp_path, monkeypatch):
    path = _config(tmp_path, name="Pendulum-v1")
    with pytest.raises(ValueError, match="'dqn'.*pendulum") as error:
        run_main(["--config", path, "--agent-type", "dqn"], monkeypatch)
    assert "['actor_critic', 'ddpg', 'ppo', 'random']" in str(error.value)
    assert not (tmp_path / "logs").exists(), "the run should stop before creating any output"


def test_cli_choices_come_from_the_registry(fake_challenge, monkeypatch, capsys):
    _, _, key = fake_challenge
    with pytest.raises(SystemExit):
        run_main(["--help"], monkeypatch)
    help_text = " ".join(capsys.readouterr().out.split())
    assert key in help_text, "--challenge should offer every registered challenge"
    assert "scripted" in help_text, "--agent-type should offer every agent any challenge supports"


@pytest.mark.parametrize("script", ["train.py", "record.py"])
def test_scripts_name_no_challenge_package_or_environment(script):
    """Adding a challenge needs no edit to the scripts: they import and match none by name."""
    source = (REPO_ROOT / script).read_text(encoding="utf-8")
    for pattern in [
        r"rl_cartpole",
        r"rl_pendulum",
        r"CartPoleEnv",
        r"PendulumEnv",
        r"CartPole-v",
        r"Pendulum-v",
        r"in env_name",
        r"\.lower\(\)",
    ]:
        assert not re.search(pattern, source), f"{script} still names {pattern!r}"
