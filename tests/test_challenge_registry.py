"""The challenge registry in rl_core and how challenges are discovered without being named."""

import subprocess
import sys
import tomllib
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
from tests.helpers import (
    entry_points_under,
    first_party_entry_points,
    foreign_checkout_message,
    unique_name,
    write_dist,
    write_package,
)

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
    expected = {"acrobot", "cartpole", "mountaincar", "mountaincar_continuous", "pendulum"}
    assert expected <= set(found)
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
    """benchmark.py run as __main__ still sees the challenges (the old sys.modules alias case)."""
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--help"],
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
            "benchmark.py",
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
        "import sys; import benchmark; "
        "benchmark.run_benchmark(challenge_name='pendulum', settings=benchmark.BenchmarkSettings("
        f"agents=('random',), episodes=1, seeds=(1,), output_dir={str(tmp_path / 'out')!r}, "
        "eval_episodes=1), report_path=None, save_plot=False); print('ran')"
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
    write_dist(isolated_registry, "broken-plugin", {broken: f"{broken}.benchmark"})
    monkeypatch.setattr(
        registry, "challenge_entry_points", lambda: entry_points_under(isolated_registry)
    )
    monkeypatch.setattr(sys, "argv", ["benchmark", "--help"])
    with pytest.warns(UserWarning), pytest.raises(SystemExit):
        benchmark.main()
    assert f"{broken} (RuntimeError: boom)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Entry points: what an installed distribution declares
# ---------------------------------------------------------------------------


def _plugin(root, dist, key, package=None, entry=None, body=FAKE_CHALLENGE):
    """A fake challenge package under ``root`` and the distribution metadata that declares it."""
    package = package or unique_name()
    write_package(root, package, body.format(key=key))
    write_dist(root, dist, {entry or package: f"{package}.benchmark"})
    return package


def _use_entry_points(monkeypatch, *roots):
    monkeypatch.setattr(registry, "challenge_entry_points", lambda: entry_points_under(*roots))


def test_an_installed_plugin_is_discovered_through_its_entry_point(isolated_registry, monkeypatch):
    """What another project (a game) does: declare one entry point, edit nothing here."""
    key = unique_name("key")
    package = _plugin(isolated_registry, "my-game", key)
    _use_entry_points(monkeypatch, isolated_registry)

    found = discover_challenges()

    assert found[key].__module__ == f"{package}.benchmark"
    assert registry.IMPORT_FAILURES == {}


def test_a_plugin_cannot_take_a_first_party_key(isolated_registry, monkeypatch):
    """This repository's distributions load first, so a plugin claiming one of their keys is the
    one recorded as failed, although its name sorts first."""
    key = unique_name("key")
    ours = _plugin(isolated_registry, "zz-first-party", key)
    theirs = _plugin(isolated_registry, "aa-plugin", key)
    _use_entry_points(monkeypatch, isolated_registry)
    monkeypatch.setattr(registry, "FIRST_PARTY_DISTS", frozenset({"zz-first-party"}))

    with pytest.warns(UserWarning, match=f"{theirs}.*already registered"):
        found = discover_challenges()

    assert found[key].__module__ == f"{ours}.benchmark"
    assert set(registry.IMPORT_FAILURES) == {theirs}


@pytest.mark.parametrize("reverse", [False, True], ids=["a-first-on-path", "b-first-on-path"])
def test_the_load_order_does_not_depend_on_sys_path(
    tmp_path, isolated_registry, monkeypatch, reverse
):
    key = unique_name("key")
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
    first = _plugin(roots[0], "plugin-a", key)
    _plugin(roots[1], "plugin-b", key)
    for root in roots:
        monkeypatch.syspath_prepend(str(root))
    _use_entry_points(monkeypatch, *(reversed(roots) if reverse else roots))

    with pytest.warns(UserWarning, match="already registered"):
        found = discover_challenges()

    assert found[key].__module__ == f"{first}.benchmark", "plugin-a sorts before plugin-b"


def test_a_module_declared_by_two_distributions_loads_once(
    tmp_path, isolated_registry, monkeypatch
):
    """Two distributions shipping one package (an old monolithic install beside the split ones):
    the module is imported once, not reported as clashing with itself."""
    key, package = unique_name("key"), unique_name()
    write_package(isolated_registry, package, FAKE_CHALLENGE.format(key=key))
    other = tmp_path / "other"
    other.mkdir()
    write_dist(isolated_registry, "old-monolith", {package: f"{package}.benchmark"})
    write_dist(other, "new-split", {package: f"{package}.benchmark"})
    _use_entry_points(monkeypatch, isolated_registry, other)

    found = discover_challenges()

    assert found[key].__module__ == f"{package}.benchmark"
    assert registry.IMPORT_FAILURES == {}


def test_a_failure_is_kept_when_another_module_of_its_package_loads(isolated_registry, monkeypatch):
    package, key = unique_name(), unique_name("key")
    write_package(isolated_registry, package, "raise RuntimeError('boom')\n")
    (isolated_registry / package / "zz_good.py").write_text(
        FAKE_CHALLENGE.format(key=key), encoding="utf-8"
    )
    write_dist(
        isolated_registry,
        "plugin",
        {"a_broken": f"{package}.benchmark", "b_good": f"{package}.zz_good"},
    )
    _use_entry_points(monkeypatch, isolated_registry)

    with pytest.warns(UserWarning, match="RuntimeError: boom"):
        found = discover_challenges()

    assert key in found
    assert list(registry.IMPORT_FAILURES.items()) == [(package, "RuntimeError: boom")]


def test_an_entry_name_reused_for_another_module_is_a_recorded_clash(
    isolated_registry, monkeypatch
):
    first_key, second_key, name = unique_name("key"), unique_name("key"), unique_name()
    first = _plugin(isolated_registry, "plugin-a", first_key, entry=name)
    second = _plugin(isolated_registry, "plugin-b", second_key, entry=name)
    _use_entry_points(monkeypatch, isolated_registry)

    with pytest.warns(UserWarning, match=f"entry point plugin-b:{name} was not loaded.*clashes"):
        found = discover_challenges()

    assert found[first_key].__module__ == f"{first}.benchmark"
    assert second_key not in found and f"{second}.benchmark" not in sys.modules
    assert set(registry.IMPORT_FAILURES) == {f"plugin-b:{name}"}


def test_a_plugin_that_exits_on_import_is_recorded_not_fatal(isolated_registry, monkeypatch):
    """A third-party module calling sys.exit at import must not end train.py or the benchmark."""
    exiting, key = unique_name(), unique_name("key")
    write_package(isolated_registry, exiting, "import sys\nsys.exit(3)\n")
    good = _plugin(isolated_registry, "zz-plugin", key)
    write_dist(isolated_registry, "aa-plugin", {exiting: f"{exiting}.benchmark"})
    _use_entry_points(monkeypatch, isolated_registry)

    with pytest.warns(UserWarning, match=f"{exiting} failed to import \\(SystemExit: 3\\)"):
        found = discover_challenges()

    assert found[key].__module__ == f"{good}.benchmark", "the challenges after it still load"
    assert list(registry.IMPORT_FAILURES) == [exiting]


def test_a_failure_is_recorded_under_the_module_package_not_the_entry_name(
    isolated_registry, monkeypatch
):
    package = unique_name()
    write_package(isolated_registry, package, "raise RuntimeError('boom')\n")
    write_dist(isolated_registry, "plugin", {"anything": f"{package}.benchmark"})
    _use_entry_points(monkeypatch, isolated_registry)

    with pytest.warns(UserWarning, match=f"Challenge package {package} failed to import"):
        discover_challenges()

    assert list(registry.IMPORT_FAILURES.items()) == [(package, "RuntimeError: boom")]


def test_with_no_challenge_installed_the_message_says_how_to_install(monkeypatch):
    monkeypatch.setattr(registry, "CHALLENGES", {})
    monkeypatch.setattr(registry, "IMPORT_FAILURES", {})

    assert registry.INSTALL_HINT in unknown_challenge_message("cartpole")
    with pytest.raises(ValueError, match="No challenge package is installed"):
        registry.select_challenge("CartPole-v1", None)


def test_the_hint_is_left_out_once_a_challenge_is_registered():
    assert discover_challenges()
    assert registry.INSTALL_HINT not in unknown_challenge_message("missing")


def _declared_entry_points():
    """Every distribution's challenge entry points, from the pyproject.toml files under packages/."""
    declared = {}
    for path in sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")):
        with open(path, "rb") as f:
            project = tomllib.load(f)["project"]
        declared.update(project.get("entry-points", {}).get(registry.ENTRY_POINT_GROUP, {}))
    return declared


def test_every_challenge_package_declares_an_entry_point():
    """The one edit a new challenge package needs: an entry point named after it."""
    packages = sorted(
        path.parent.name for path in (REPO_ROOT / "packages").glob("*/src/*/benchmark.py")
    )
    declared = _declared_entry_points()
    assert sorted(declared) == packages, (
        'Every packages/<dist>/src/<package>/benchmark.py needs `<package> = "<package>.benchmark"` '
        f'under [project.entry-points."{registry.ENTRY_POINT_GROUP}"] in its pyproject.toml, '
        "and nothing else"
    )
    assert all(module == f"{name}.benchmark" for name, module in declared.items())


def test_the_installed_metadata_has_the_declared_entry_points():
    """A stale editable install (made before an entry point was added) finds too few challenges."""
    installed = {ep.name: ep.value for ep in registry.challenge_entry_points()}
    assert installed == _declared_entry_points(), (
        "The installed metadata is stale: re-run the editable install "
        "(pip install -r requirements-dev.txt, or scripts/worktree_venv.sh in a worktree)"
    )


def test_the_tests_see_only_first_party_entry_points(isolated_registry, monkeypatch):
    """An installed game never enters the suite: conftest.py restricts the entry-point source."""
    write_dist(isolated_registry, "my-game", {"my_game": "my_game.benchmark"})
    source = first_party_entry_points(lambda: entry_points_under(isolated_registry))
    assert source() == []
    monkeypatch.setattr(registry, "FIRST_PARTY_DISTS", frozenset({"my-game"}))
    assert [ep.name for ep in source()] == ["my_game"]


def test_tests_refuse_another_checkouts_code(tmp_path):
    worktree, main = tmp_path / "worktree", tmp_path / "main"
    inside = worktree / "packages" / "rl-core" / "src" / "rl_core" / "__init__.py"
    outside = main / "packages" / "rl-core" / "src" / "rl_core" / "__init__.py"
    assert foreign_checkout_message(worktree, inside) is None
    message = foreign_checkout_message(worktree, outside)
    assert message is not None and "scripts/worktree_venv.sh" in message


def test_tests_refuse_an_installed_copy_too(tmp_path):
    """A non-editable install in site-packages is not this checkout's code either."""
    import sysconfig

    installed = Path(sysconfig.get_path("purelib")) / "rl_core" / "__init__.py"
    assert foreign_checkout_message(tmp_path / "checkout", installed) is not None


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
