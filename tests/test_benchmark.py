"""Unit and integration tests for general benchmark interface (src/benchmark.py) and CartPole challenge."""

import json
import os
import sys
from typing import Any

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import (
    CHALLENGES,
    BenchmarkChallenge,
    BenchmarkRunner,
    find_solved_episode,
    main,
    run_benchmark,
)
from rl_cartpole.agents.random_agent import RandomAgent
from rl_cartpole.benchmark import CartPoleChallenge
from rl_cartpole.environments.cartpole_env import CartPoleEnv


def test_find_solved_episode_empty():
    """Empty reward list returns None."""
    assert find_solved_episode([]) is None


def test_find_solved_episode_never_reached():
    """Returns None when rewards never cross the solve threshold."""
    rewards = [10.0] * 50
    assert find_solved_episode(rewards, threshold=475.0, window=5) is None


def test_find_solved_episode_solved():
    """Returns 1-based episode index when rolling average crosses threshold."""
    rewards = [10.0] * 5 + [500.0] * 10
    # Rolling average window 5 reaches 500 on episode 10
    solved_ep = find_solved_episode(rewards, threshold=475.0, window=5)
    assert solved_ep is not None
    assert solved_ep == 10


def test_run_benchmark_invalid_challenge():
    """Unknown challenge raises ValueError."""
    with pytest.raises(ValueError, match="Unknown challenge 'invalid_challenge'"):
        run_benchmark(challenge_name="invalid_challenge")


def test_run_benchmark_invalid_agents():
    """Invalid agent lists or types raise ValueError."""
    with pytest.raises(ValueError, match="agents list must not be empty"):
        run_benchmark(agents=[])

    with pytest.raises(ValueError, match="is not supported by challenge"):
        run_benchmark(agents=["unsupported_agent"])


def test_run_benchmark_invalid_episodes():
    """Non-positive episodes raise ValueError."""
    with pytest.raises(ValueError, match="episodes must be positive"):
        run_benchmark(agents=["random"], episodes=0)


def test_run_benchmark_invalid_seeds():
    """Empty seeds list raises ValueError."""
    with pytest.raises(ValueError, match="seeds list must not be empty"):
        run_benchmark(agents=["random"], seeds=[])


def test_cartpole_challenge_registered():
    """CartPole challenge is properly registered in CHALLENGES."""
    assert "cartpole" in CHALLENGES
    assert CHALLENGES["cartpole"] is CartPoleChallenge


def test_run_benchmark_cartpole_minimal_execution(tmp_path):
    """Run minimal end-to-end benchmark on CartPole across 2 agents and verify outputs."""
    out_dir = str(tmp_path / "bench_out")
    report_file = str(tmp_path / "report.md")

    bench_res = run_benchmark(
        challenge_name="cartpole",
        agents=["random", "ppo"],
        episodes=3,
        seeds=[42],
        output_dir=out_dir,
        report_path=report_file,
        save_plot=True,
        eval_episodes=2,
    )

    assert bench_res["challenge"] == "CartPole-v1"
    assert "random" in bench_res["summary"]
    assert "ppo" in bench_res["summary"]
    assert bench_res["summary"]["random"]["runs"] == 1
    assert bench_res["plot_path"] is not None
    assert os.path.isfile(bench_res["plot_path"])
    assert os.path.isfile(bench_res["metrics_json"])
    assert os.path.isfile(report_file)

    # Verify JSON content
    with open(bench_res["metrics_json"]) as f:
        data = json.load(f)
    assert data["challenge"] == "CartPole-v1"
    assert "summary" in data
    assert "metadata" in data

    # Verify Markdown report content
    with open(report_file) as f:
        md = f.read()
    assert "# CartPole-v1 Multi-Agent Benchmark Results" in md
    assert "random" in md
    assert "ppo" in md


def test_run_benchmark_no_plot(tmp_path):
    """Running with save_plot=False skips plot generation."""
    out_dir = str(tmp_path / "bench_no_plot")
    bench_res = run_benchmark(
        agents=["random"],
        episodes=2,
        seeds=[42],
        output_dir=out_dir,
        report_path=None,
        save_plot=False,
        eval_episodes=1,
    )
    assert bench_res["plot_path"] is None
    assert not os.path.exists(os.path.join(out_dir, "benchmark_comparison.png"))


class CustomGenericChallenge(BenchmarkChallenge):
    """A minimal mock challenge demonstrating that BenchmarkRunner is completely decoupled."""

    @property
    def name(self) -> str:
        return "GenericWorld-v0"

    @property
    def key(self) -> str:
        return "generic"

    @property
    def solve_threshold(self) -> float | None:
        return 20.0

    @property
    def default_episodes(self) -> int:
        return 2

    @property
    def supported_agents(self) -> list[str]:
        return ["mock_agent"]

    def setup_run(
        self, agent_type: str, seed: int, episodes: int
    ) -> tuple[Any, Any, dict[str, Any]]:
        env = CartPoleEnv(seed=seed, max_episode_steps=25)
        agent = RandomAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config={"seed": seed},
        )
        train_cfg = {"num_episodes": episodes, "save_frequency": 999}
        return env, agent, train_cfg


def test_custom_challenge_runner(tmp_path):
    """BenchmarkRunner successfully evaluates arbitrary challenges implementing BenchmarkChallenge."""
    challenge = CustomGenericChallenge()
    runner = BenchmarkRunner(challenge)

    out_dir = str(tmp_path / "custom_out")
    res = runner.run(
        agents=["mock_agent"],
        episodes=2,
        seeds=[11],
        output_dir=out_dir,
        save_plot=False,
        eval_episodes=1,
    )

    assert res["challenge"] == "GenericWorld-v0"
    assert "mock_agent" in res["summary"]
    assert res["summary"]["mock_agent"]["runs"] == 1


def test_benchmark_cli_help(monkeypatch, capsys):
    """CLI prints help without error."""
    monkeypatch.setattr(sys, "argv", ["benchmark", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "General Reinforcement Learning Benchmark Suite" in captured.out


def test_benchmark_cli_execution(tmp_path, monkeypatch):
    """CLI runs and produces outputs when invoked via main."""
    out_dir = str(tmp_path / "cli_out")
    report_file = str(tmp_path / "cli_report.md")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--challenge",
            "cartpole",
            "--agents",
            "random",
            "--episodes",
            "2",
            "--seeds",
            "7",
            "--output-dir",
            out_dir,
            "--report",
            report_file,
            "--eval-episodes",
            "1",
        ],
    )
    main()

    assert os.path.isfile(os.path.join(out_dir, "benchmark_comparison.png"))
    assert os.path.isfile(os.path.join(out_dir, "benchmark_metrics.json"))
    assert os.path.isfile(report_file)


def test_benchmark_cli_default_report(monkeypatch):
    """CLI defaults report path to docs/benchmarks/<challenge>_results.md when omitted."""
    called_kwargs = {}

    def mock_run_benchmark(**kwargs):
        called_kwargs.update(kwargs)
        return {}

    import benchmark

    monkeypatch.setattr(benchmark, "run_benchmark", mock_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--challenge", "cartpole", "--agents", "random"],
    )
    main()

    assert called_kwargs.get("report_path") == "docs/benchmarks/cartpole_results.md"
