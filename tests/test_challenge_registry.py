"""The challenge registry in rl_core and how challenges are discovered without being named."""

import subprocess
import sys
from pathlib import Path

import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import DQNAgent, RandomAgent
from rl_core.challenges import (
    BenchmarkChallenge,
    discover_challenges,
    register_challenge,
    registry,
    unknown_challenge_message,
)
from tests.helpers import unique_name, write_package

REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_CHALLENGE = """
from rl_core.challenges import BenchmarkChallenge, register_challenge


@register_challenge
class FakeChallenge(BenchmarkChallenge):
    name = "Fake-v0"
    key = "{key}"
    solve_threshold = None
    default_episodes = 1
    supported_agents = ["random"]

    def make_env(self, env_config, render_mode):
        raise NotImplementedError

    def make_agent(self, agent_type, agent_config, env):
        raise NotImplementedError
"""


def test_discovery_finds_a_challenge_package_without_naming_it(isolated_registry):
    name, key = unique_name(), unique_name("key")
    write_package(isolated_registry, name, FAKE_CHALLENGE.format(key=key))

    found = discover_challenges(isolated_registry)

    assert key in found
    assert found[key].__module__ == f"{name}.benchmark"


def test_discovery_skips_what_is_not_a_challenge_package(isolated_registry):
    keys = [unique_name("key") for _ in range(4)]
    write_package(isolated_registry, "other_" + unique_name(), FAKE_CHALLENGE.format(key=keys[0]))
    write_package(
        isolated_registry, unique_name() + ".egg-info", FAKE_CHALLENGE.format(key=keys[1])
    )
    write_package(isolated_registry, unique_name(), FAKE_CHALLENGE.format(key=keys[2]), init=False)
    write_package(isolated_registry, unique_name(), None)

    found = discover_challenges(isolated_registry)

    assert not any(key in found for key in keys)
    assert registry.IMPORT_FAILURES == {}


def test_a_package_that_fails_to_import_is_reported_and_does_not_stop_the_others(isolated_registry):
    broken, good, key = unique_name(), unique_name(), unique_name("key")
    write_package(isolated_registry, broken, "raise RuntimeError('boom')\n")
    write_package(isolated_registry, good, FAKE_CHALLENGE.format(key=key))

    with pytest.warns(UserWarning, match=f"{broken}.*RuntimeError: boom"):
        found = discover_challenges(isolated_registry)

    assert key in found
    assert "RuntimeError: boom" in registry.IMPORT_FAILURES[broken]
    message = unknown_challenge_message("missing")
    assert "Unknown challenge 'missing'" in message
    assert f"{broken} (RuntimeError: boom)" in message


def test_discovering_twice_imports_each_package_once(isolated_registry):
    name, key = unique_name(), unique_name("key")
    marker = isolated_registry / "imports.txt"
    body = f"open({str(marker)!r}, 'a').write('x')\n" + FAKE_CHALLENGE.format(key=key)
    write_package(isolated_registry, name, body)

    discover_challenges(isolated_registry)
    discover_challenges(isolated_registry)

    assert marker.read_text() == "x"


def test_default_discovery_finds_the_repository_challenges():
    found = discover_challenges()
    assert {"acrobot", "cartpole", "pendulum"} <= set(found)
    assert found["cartpole"].__module__ == "rl_cartpole.benchmark"


def test_every_challenge_accepts_a_config_dir():
    class Minimal(BenchmarkChallenge):
        name, key, solve_threshold, default_episodes, supported_agents = "M", "m", None, 1, []

        def make_env(self, env_config, render_mode):
            raise NotImplementedError

        def make_agent(self, agent_type, agent_config, env):
            raise NotImplementedError

    assert Minimal(config_dir="elsewhere").config_dir == "elsewhere"
    assert Minimal().config_dir == "configs"


def test_running_the_benchmark_script_directly_lists_every_challenge():
    """src/benchmark.py run as __main__ still sees the challenges (the old sys.modules alias case)."""
    result = subprocess.run(
        [sys.executable, "src/benchmark.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "cartpole" in result.stdout and "pendulum" in result.stdout


def test_running_the_benchmark_script_directly_trains(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "src/benchmark.py",
            "--challenge",
            "cartpole",
            "--agents",
            "random",
            "--episodes",
            "2",
            "--seeds",
            "1",
            "--no-plot",
            "--eval-episodes",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert (tmp_path / "report.md").is_file(), result.stdout


def test_run_benchmark_discovers_challenges_in_a_fresh_process(tmp_path):
    """Called from Python with nothing imported first, run_benchmark still finds the challenges."""
    code = (
        "import sys; sys.path.insert(0, 'src'); import benchmark; "
        "benchmark.run_benchmark(challenge_name='pendulum', agents=['random'], episodes=1, "
        f"seeds=[1], output_dir={str(tmp_path / 'out')!r}, report_path=None, save_plot=False, "
        "eval_episodes=1); print('ran')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "ran" in result.stdout


def test_a_second_challenge_under_the_same_key_is_rejected(isolated_registry):
    first, second = sorted([unique_name(), unique_name()])  # discovery imports in sorted order
    key = unique_name("key")
    write_package(isolated_registry, first, FAKE_CHALLENGE.format(key=key))
    write_package(isolated_registry, second, FAKE_CHALLENGE.format(key=key))

    with pytest.warns(UserWarning, match=f"{second}.*ValueError: .*{key}.* already registered"):
        found = discover_challenges(isolated_registry)

    assert found[key].__module__ == f"{first}.benchmark"


def test_registering_the_same_class_again_is_harmless(isolated_registry):
    name, key = unique_name(), unique_name("key")
    write_package(isolated_registry, name, FAKE_CHALLENGE.format(key=key))
    cls = discover_challenges(isolated_registry)[key]
    assert register_challenge(cls) is cls


def test_cli_help_names_a_challenge_package_that_failed_to_import(
    isolated_registry, monkeypatch, capsys
):
    import benchmark

    broken = unique_name()
    write_package(isolated_registry, broken, "raise RuntimeError('boom')\n")
    monkeypatch.setattr(registry, "_DEFAULT_SEARCH_DIR", isolated_registry)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--help"])
    with pytest.warns(UserWarning), pytest.raises(SystemExit):
        benchmark.main()
    assert f"{broken} (RuntimeError: boom)" in capsys.readouterr().out


class _TableOnly(BenchmarkChallenge):
    """A challenge that declares only its agent table."""

    name, key, solve_threshold, default_episodes = "CartPole-v1", "table_only", None, 1
    AGENT_CLASSES = {"random": RandomAgent, "dqn": DQNAgent}

    def make_env(self, env_config, render_mode):
        return CartPoleEnv(seed=0, max_episode_steps=5)


def test_a_challenge_with_only_an_agent_table_lists_and_builds_its_agents():
    challenge = _TableOnly()
    env = challenge.make_env({}, None)
    try:
        agent = challenge.make_agent("dqn", {"seed": 0}, env)
    finally:
        env.close()
    assert challenge.supported_agents == ["dqn", "random"], "the table's keys, sorted"
    assert type(agent) is DQNAgent
    assert (agent.observation_dim, agent.action_dim) == (4, 2), "sized from the environment"


def test_the_agent_config_hook_reaches_the_agent():
    class WithHook(_TableOnly):
        def agent_config(self, agent_config, env):
            return {**agent_config, "hidden_dim": 7}

    env = WithHook().make_env({}, None)
    try:
        agent = WithHook().make_agent("dqn", {"seed": 0}, env)
    finally:
        env.close()
    assert agent.hidden_dim == 7, "make_agent should build the agent from agent_config's result"


class _NoTable(BenchmarkChallenge):
    """A challenge that declares no agent table: the base's empty one applies."""

    name, key, solve_threshold, default_episodes = "CartPole-v1", "no_table", None, 1

    def make_env(self, env_config, render_mode):
        return CartPoleEnv(seed=0, max_episode_steps=5)


@pytest.mark.parametrize("challenge_cls", [_TableOnly, _NoTable], ids=["not-in-table", "no-table"])
def test_an_agent_outside_the_table_is_the_standard_error(challenge_cls):
    challenge = challenge_cls()
    env = challenge.make_env({}, None)
    try:
        with pytest.raises(ValueError, match=f"not supported by challenge '{challenge.key}'"):
            challenge.make_agent("ppo", {}, env)
    finally:
        env.close()


def test_a_challenge_without_an_agent_table_supports_no_agents():
    assert _NoTable().supported_agents == []


def test_the_base_agent_table_cannot_be_changed():
    with pytest.raises(TypeError):
        BenchmarkChallenge.AGENT_CLASSES["random"] = RandomAgent
