"""The challenge registry in rl_core and how challenges are discovered without being named."""

import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from rl_core.challenges import (
    CHALLENGES,
    BenchmarkChallenge,
    discover_challenges,
    register_challenge,
    registry,
    unknown_challenge_message,
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

    def setup_run(self, agent_type, seed, episodes):
        raise NotImplementedError
"""


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Put tmp_path on the import path and undo every registration and import afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    challenges, failures = dict(CHALLENGES), dict(registry.IMPORT_FAILURES)
    modules = set(sys.modules)
    yield tmp_path
    CHALLENGES.clear()
    CHALLENGES.update(challenges)
    registry.IMPORT_FAILURES.clear()
    registry.IMPORT_FAILURES.update(failures)
    for name in set(sys.modules) - modules:
        if name.startswith("rl_fake"):
            del sys.modules[name]


def _package(root: Path, name: str, benchmark: str | None, init: bool = True) -> None:
    package = root / name
    package.mkdir()
    if init:
        (package / "__init__.py").write_text("", encoding="utf-8")
    if benchmark is not None:
        (package / "benchmark.py").write_text(benchmark, encoding="utf-8")


def _unique(prefix: str = "rl_fake") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_discovery_finds_a_challenge_package_without_naming_it(isolated_registry):
    name, key = _unique(), _unique("key")
    _package(isolated_registry, name, FAKE_CHALLENGE.format(key=key))

    found = discover_challenges(isolated_registry)

    assert key in found
    assert found[key].__module__ == f"{name}.benchmark"


def test_discovery_skips_what_is_not_a_challenge_package(isolated_registry):
    keys = [_unique("key") for _ in range(4)]
    _package(isolated_registry, "other_" + _unique(), FAKE_CHALLENGE.format(key=keys[0]))
    _package(isolated_registry, _unique() + ".egg-info", FAKE_CHALLENGE.format(key=keys[1]))
    _package(isolated_registry, _unique(), FAKE_CHALLENGE.format(key=keys[2]), init=False)
    _package(isolated_registry, _unique(), None)

    found = discover_challenges(isolated_registry)

    assert not any(key in found for key in keys)
    assert registry.IMPORT_FAILURES == {}


def test_a_package_that_fails_to_import_is_reported_and_does_not_stop_the_others(isolated_registry):
    broken, good, key = _unique(), _unique(), _unique("key")
    _package(isolated_registry, broken, "raise RuntimeError('boom')\n")
    _package(isolated_registry, good, FAKE_CHALLENGE.format(key=key))

    with pytest.warns(UserWarning, match=f"{broken}.*RuntimeError: boom"):
        found = discover_challenges(isolated_registry)

    assert key in found
    assert "RuntimeError: boom" in registry.IMPORT_FAILURES[broken]
    message = unknown_challenge_message("missing")
    assert "Unknown challenge 'missing'" in message
    assert f"{broken} (RuntimeError: boom)" in message


def test_discovering_twice_imports_each_package_once(isolated_registry):
    name, key = _unique(), _unique("key")
    marker = isolated_registry / "imports.txt"
    body = f"open({str(marker)!r}, 'a').write('x')\n" + FAKE_CHALLENGE.format(key=key)
    _package(isolated_registry, name, body)

    discover_challenges(isolated_registry)
    discover_challenges(isolated_registry)

    assert marker.read_text() == "x"


def test_default_discovery_finds_the_repository_challenges():
    found = discover_challenges()
    assert {"cartpole", "pendulum"} <= set(found)
    assert found["cartpole"].__module__ == "rl_cartpole.benchmark"


def test_every_challenge_accepts_a_config_dir():
    class Minimal(BenchmarkChallenge):
        name, key, solve_threshold, default_episodes, supported_agents = "M", "m", None, 1, []

        def setup_run(self, agent_type, seed, episodes):
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
    first, second = sorted([_unique(), _unique()])  # discovery imports in sorted order
    key = _unique("key")
    _package(isolated_registry, first, FAKE_CHALLENGE.format(key=key))
    _package(isolated_registry, second, FAKE_CHALLENGE.format(key=key))

    with pytest.warns(UserWarning, match=f"{second}.*ValueError: .*{key}.* already registered"):
        found = discover_challenges(isolated_registry)

    assert found[key].__module__ == f"{first}.benchmark"


def test_registering_the_same_class_again_is_harmless(isolated_registry):
    name, key = _unique(), _unique("key")
    _package(isolated_registry, name, FAKE_CHALLENGE.format(key=key))
    cls = discover_challenges(isolated_registry)[key]
    assert register_challenge(cls) is cls


def test_cli_help_names_a_challenge_package_that_failed_to_import(
    isolated_registry, monkeypatch, capsys
):
    import benchmark

    broken = _unique()
    _package(isolated_registry, broken, "raise RuntimeError('boom')\n")
    monkeypatch.setattr(registry, "_DEFAULT_SEARCH_DIR", isolated_registry)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--help"])
    with pytest.warns(UserWarning), pytest.raises(SystemExit):
        benchmark.main()
    assert f"{broken} (RuntimeError: boom)" in capsys.readouterr().out
