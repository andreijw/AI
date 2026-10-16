"""Unit and integration tests for general benchmark interface (src/benchmark.py) and CartPole challenge."""

import json
import os
import sys
from typing import Any

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import (
    BenchmarkRunner,
    classify_run,
    find_solved_episode,
    main,
    run_benchmark,
)
from rl_cartpole.benchmark import CartPoleChallenge
from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents.random_agent import RandomAgent
from rl_core.challenges import CHALLENGES, BenchmarkChallenge
from rl_core.utils.config import load_config
from rl_pendulum.benchmark import PendulumChallenge


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
        self, agent_type: str, seed: int, episodes: int | None
    ) -> tuple[Any, Any, dict[str, Any]]:
        env = CartPoleEnv(seed=seed, max_episode_steps=25)
        agent = RandomAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config={"seed": seed},
        )
        budget = self.default_episodes if episodes is None else episodes
        train_cfg = {"num_episodes": budget, "save_frequency": 999}
        return env, agent, train_cfg


class PerAgentBudgetChallenge(CustomGenericChallenge):
    """A challenge whose agents have different budgets when no override is given."""

    BUDGETS = {"short": 2, "long": 3}

    @property
    def supported_agents(self) -> list[str]:
        return sorted(self.BUDGETS)

    def setup_run(
        self, agent_type: str, seed: int, episodes: int | None
    ) -> tuple[Any, Any, dict[str, Any]]:
        own = self.BUDGETS[agent_type] if episodes is None else episodes
        return super().setup_run(agent_type, seed, own)


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


# ---------------------------------------------------------------------------
# Solve rule: only the rolling training average counts
# ---------------------------------------------------------------------------


def test_classify_run_training_average_reaching_threshold_is_a_solve():
    """A rolling training average at the threshold solves the run, at that episode."""
    rewards = [10.0] * 5 + [500.0] * 10
    result = classify_run(rewards, eval_mean=10.0, threshold=475.0, window=5)
    assert result == {
        "is_solved": True,
        "solved_episode": 10,
        "solved_step": None,
        "held": True,
        "eval_reached_threshold": False,
    }


def test_runner_rejects_a_non_positive_override_for_any_challenge(tmp_path):
    """The runner checks --episodes itself, even for a challenge that loads no config."""
    with pytest.raises(ValueError, match="^episodes must be positive, got 0$"):
        BenchmarkRunner(CustomGenericChallenge()).run(
            episodes=0, seeds=[1], output_dir=str(tmp_path / "zero"), save_plot=False
        )


def test_classify_run_needs_a_full_window_before_it_can_solve():
    """One high early episode is not an average over the window, so it is not a solve."""
    rewards = [500.0] + [9.0] * 99
    result = classify_run(rewards, eval_mean=9.0, threshold=475.0, window=10)
    assert result["is_solved"] is False
    assert result["solved_episode"] is None


def test_find_solved_episode_counts_the_first_full_window():
    """A run that is high from the start solves at the episode that completes the first window."""
    assert find_solved_episode([500.0] * 12, threshold=475.0, window=10) == 10


def test_find_solved_episode_with_fewer_episodes_than_the_window_never_solves():
    """A run shorter than the window has no full window to judge."""
    assert find_solved_episode([500.0] * 3, threshold=475.0, window=10) is None


def test_classify_run_evaluation_alone_is_not_a_solve():
    """A final evaluation at the threshold is recorded, but is not a solve and has no episode."""
    rewards = [10.0] * 20
    result = classify_run(rewards, eval_mean=500.0, threshold=475.0, window=5)
    assert result == {
        "is_solved": False,
        "solved_episode": None,
        "solved_step": None,
        "held": False,
        "eval_reached_threshold": True,
    }


def test_classify_run_evaluation_exactly_at_threshold_reaches_it():
    """The evaluation comparison is inclusive, like the training one."""
    result = classify_run([0.0], eval_mean=475.0, threshold=475.0, window=1)
    assert result["eval_reached_threshold"] is True


def test_classify_run_without_threshold_never_solves():
    """A challenge with no threshold has no solves and no evaluation successes."""
    result = classify_run([1e6] * 5, eval_mean=1e6, threshold=None, window=1)
    assert result == {
        "is_solved": False,
        "solved_episode": None,
        "solved_step": None,
        "held": None,
        "eval_reached_threshold": False,
    }


# ---------------------------------------------------------------------------
# Budget: each agent's own config unless --episodes is given
# ---------------------------------------------------------------------------

CHALLENGE_CLASSES = [CartPoleChallenge, PendulumChallenge]


def _write_random_config(challenge_cls, config_dir, training_yaml: str) -> None:
    path = config_dir / challenge_cls.CONFIG_MAP["random"]
    path.write_text(f"agent:\n  config: {{}}\n{training_yaml}", encoding="utf-8")


@pytest.mark.parametrize("challenge_cls", CHALLENGE_CLASSES)
def test_setup_run_uses_the_agents_config_budget_without_override(challenge_cls, tmp_path):
    """With no episodes given, the run trains for the config's training.num_episodes."""
    _write_random_config(challenge_cls, tmp_path, "training:\n  num_episodes: 7\n")
    env, _, train_cfg = challenge_cls(config_dir=str(tmp_path)).setup_run("random", 3, None)
    env.close()
    assert train_cfg["num_episodes"] == 7
    assert train_cfg["save_frequency"] > 7


@pytest.mark.parametrize("challenge_cls", CHALLENGE_CLASSES)
def test_setup_run_override_replaces_the_config_budget(challenge_cls, tmp_path):
    """An explicit episodes value wins over the config's budget."""
    _write_random_config(challenge_cls, tmp_path, "training:\n  num_episodes: 7\n")
    env, _, train_cfg = challenge_cls(config_dir=str(tmp_path)).setup_run("random", 3, 4)
    env.close()
    assert train_cfg["num_episodes"] == 4


@pytest.mark.parametrize("challenge_cls", CHALLENGE_CLASSES)
def test_setup_run_falls_back_to_challenge_default_without_config_budget(challenge_cls, tmp_path):
    """A config without training.num_episodes gets the challenge's default budget."""
    _write_random_config(challenge_cls, tmp_path, "")
    challenge = challenge_cls(config_dir=str(tmp_path))
    env, _, train_cfg = challenge.setup_run("random", 3, None)
    env.close()
    assert train_cfg["num_episodes"] == challenge.default_episodes


@pytest.mark.parametrize("challenge_cls", CHALLENGE_CLASSES)
def test_setup_run_rejects_a_non_positive_config_budget(challenge_cls, tmp_path):
    """A config asking for zero episodes is an error, not an empty run."""
    _write_random_config(challenge_cls, tmp_path, "training:\n  num_episodes: 0\n")
    challenge = challenge_cls(config_dir=str(tmp_path))
    with pytest.raises(ValueError, match="num_episodes must be positive, got 0"):
        challenge.load_run_config("random", 3, None)


def test_load_run_config_names_the_challenge_for_an_unmapped_agent(tmp_path):
    """An agent with no config file in CONFIG_MAP gets an error naming the challenge."""
    with pytest.raises(KeyError, match="CartPole-v1.*'nobody'"):
        CartPoleChallenge(config_dir=str(tmp_path)).load_run_config("nobody", 3, None)


@pytest.mark.parametrize("challenge_cls", CHALLENGE_CLASSES)
def test_setup_run_applies_the_seed(challenge_cls, tmp_path):
    """The run's seed reaches the agent config, whatever the file says."""
    _write_random_config(challenge_cls, tmp_path, "training:\n  num_episodes: 7\n")
    challenge = challenge_cls(config_dir=str(tmp_path))
    config = challenge.load_run_config("random", 11, None)
    assert config["environment"]["seed"] == 11
    assert config["agent"]["config"]["seed"] == 11


def test_committed_configs_give_each_cartpole_agent_its_own_budget():
    """Without an override, each CartPole agent's budget is the one in its committed config."""
    from rl_core.training.budget import budget_of

    challenge = CartPoleChallenge()
    for agent_type, config_file in CartPoleChallenge.CONFIG_MAP.items():
        expected = budget_of(load_config(os.path.join("configs", config_file))["training"])
        assert budget_of(challenge.load_run_config(agent_type, 1, None)["training"]) == expected


def test_runner_trains_each_agent_for_its_own_budget(tmp_path):
    """Without an override, the runner records and reports each agent's own budget."""
    out_dir = tmp_path / "budgets"
    report = tmp_path / "budgets.md"
    res = BenchmarkRunner(PerAgentBudgetChallenge()).run(
        seeds=[1],
        output_dir=str(out_dir),
        report_path=str(report),
        save_plot=False,
        eval_episodes=1,
    )

    assert res["results"]["short"][0]["total_episodes"] == 2
    assert res["results"]["long"][0]["total_episodes"] == 3
    assert res["summary"]["short"]["episodes"] == 2
    assert res["summary"]["long"]["episodes"] == 3

    with open(res["metrics_json"]) as f:
        metadata = json.load(f)["metadata"]
    assert metadata["episodes"] is None
    assert metadata["episodes_per_agent"] == {"long": 3, "short": 2}

    md = report.read_text(encoding="utf-8")
    assert "each agent's own config" in md
    assert "| **short** | 2 episodes | 2 |" in md
    assert "| **long** | 3 episodes | 3 |" in md


def test_runner_override_applies_to_every_agent(tmp_path):
    """An explicit episodes value trains every agent for that number and is recorded."""
    report = tmp_path / "override.md"
    res = BenchmarkRunner(PerAgentBudgetChallenge()).run(
        episodes=4,
        seeds=[1],
        output_dir=str(tmp_path / "override"),
        report_path=str(report),
        save_plot=False,
        eval_episodes=1,
    )
    assert [res["results"][a][0]["total_episodes"] for a in ("long", "short")] == [4, 4]
    with open(res["metrics_json"]) as f:
        metadata = json.load(f)["metadata"]
    assert metadata["episodes"] == 4
    assert metadata["episodes_per_agent"] == {"long": 4, "short": 4}
    assert "4 episodes for every agent" in report.read_text(encoding="utf-8")


class _ScriptedTrainer:
    """Stands in for Trainer: training rewards and evaluation mean are fixed by the seed."""

    # seed -> (training reward per episode, evaluation mean); the challenge's threshold is 20
    OUTCOMES = {1: (5.0, 30.0), 2: (25.0, 30.0), 3: (5.0, 5.0)}

    def __init__(self, agent, env, config, logger):
        self.seed = agent.config["seed"]
        self.num_episodes = config["num_episodes"]
        self.episode_rewards: list[float] = []

    def train(self) -> None:
        self.episode_rewards = [self.OUTCOMES[self.seed][0]] * self.num_episodes
        self.episode_lengths = [1] * self.num_episodes

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        return {"mean_reward": self.OUTCOMES[self.seed][1]}


def test_runner_reports_evaluation_successes_separately(tmp_path, monkeypatch):
    """An evaluation-only success is recorded apart from the solves, and the report explains the rule."""
    import benchmark

    monkeypatch.setattr(benchmark, "Trainer", _ScriptedTrainer)
    report = tmp_path / "rule.md"
    res = BenchmarkRunner(PerAgentBudgetChallenge()).run(
        agents=["long"],
        seeds=[1, 2, 3],
        output_dir=str(tmp_path / "rule"),
        report_path=str(report),
        save_plot=False,
        eval_episodes=1,
        window=1,
    )
    runs = {r["seed"]: r for r in res["results"]["long"]}
    stats = res["summary"]["long"]

    # Seed 1 reaches the threshold in evaluation only: recorded, but not a solve.
    assert runs[1]["is_solved"] is False
    assert runs[1]["solved_episode"] is None
    assert runs[1]["eval_reached_threshold"] is True
    # Seed 2 solves in training from the first episode; seed 3 does neither.
    assert runs[2]["is_solved"] is True
    assert runs[2]["solved_episode"] == 1
    assert runs[3]["eval_reached_threshold"] is False

    assert stats["solve_rate"] == pytest.approx(1 / 3)
    assert stats["avg_solved_episode"] == 1.0
    assert stats["eval_threshold_rate"] == pytest.approx(2 / 3)

    md = report.read_text(encoding="utf-8")
    assert "| **long** | 3 episodes | 3 | 21.7 | ±11.8 | 33% | 1.0 | 1 | 33% | 67% |" in md
    assert "## How a Run Counts as Solved" in md
    assert "does not count as a solve" in md
    assert "(1 greedy episode)" in md


def test_report_without_threshold_says_nothing_can_solve(tmp_path):
    """A challenge with no threshold says so instead of describing a rule it cannot apply."""

    class NoThresholdChallenge(PerAgentBudgetChallenge):
        @property
        def solve_threshold(self) -> float | None:
            return None

    report = tmp_path / "none.md"
    BenchmarkRunner(NoThresholdChallenge()).run(
        agents=["short"],
        seeds=[1],
        output_dir=str(tmp_path / "none"),
        report_path=str(report),
        save_plot=False,
        eval_episodes=3,
    )
    md = report.read_text(encoding="utf-8")
    assert "no solve threshold, so no run counts as solved" in md
    assert "greedy episodes" not in md


# ---------------------------------------------------------------------------
# Step budgets, steps to solve and held in the runner, report and CLI
# ---------------------------------------------------------------------------


def test_runner_trains_for_a_step_budget_and_records_steps(tmp_path):
    res = BenchmarkRunner(CartPoleChallenge()).run(
        agents=["random"],
        steps=60,
        seeds=[1],
        output_dir=str(tmp_path / "steps"),
        report_path=str(tmp_path / "steps.md"),
        save_plot=True,
        eval_episodes=1,
    )
    run = res["results"]["random"][0]
    assert run["total_steps"] >= 60
    assert run["window"] == 100
    assert {"solved_step", "held"} <= set(run)
    stats = res["summary"]["random"]
    assert (stats["budget"], stats["budget_unit"]) == (60, "steps")
    assert stats["episodes"] == run["total_episodes"]
    assert {"avg_solved_step", "held_rate"} <= set(stats)
    with open(res["metrics_json"]) as f:
        metadata = json.load(f)["metadata"]
    assert (metadata["episodes"], metadata["steps"]) == (None, 60)
    assert metadata["budgets_per_agent"] == {"random": {"value": 60, "unit": "steps"}}
    md = (tmp_path / "steps.md").read_text(encoding="utf-8")
    assert "60 steps for every agent (`--steps`)" in md
    assert "| **random** | 60 steps |" in md
    assert "Avg Steps to Solve" in md and "| Held |" in md


def test_runner_rejects_an_episode_and_a_step_budget_together(tmp_path):
    with pytest.raises(ValueError, match="either an episode or a step budget"):
        BenchmarkRunner(CartPoleChallenge()).run(
            agents=["random"],
            episodes=2,
            steps=60,
            seeds=[1],
            output_dir=str(tmp_path / "x"),
            save_plot=False,
        )


def test_report_states_the_window_used(tmp_path):
    for window, text in ((100, "100 consecutive"), (7, "7 consecutive")):
        report = tmp_path / f"w{window}.md"
        BenchmarkRunner(PerAgentBudgetChallenge()).run(
            agents=["short"],
            seeds=[1],
            output_dir=str(tmp_path / f"w{window}"),
            report_path=str(report),
            save_plot=False,
            eval_episodes=1,
            window=window,
        )
        md = report.read_text(encoding="utf-8")
        assert text in md
        assert "held" in md.lower()


def test_cli_rejects_episodes_and_steps_together(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--episodes", "2", "--steps", "60"])
    with pytest.raises(SystemExit):
        main()
    assert "not allowed with argument" in capsys.readouterr().err


def test_cli_default_window_is_100(monkeypatch):
    called = {}
    import benchmark

    monkeypatch.setattr(benchmark, "run_benchmark", lambda **kwargs: called.update(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "--agents", "random", "--steps", "60"])
    main()
    assert (called["window"], called["steps"], called["episodes"]) == (100, 60, None)


def test_load_run_config_applies_a_step_budget(tmp_path):
    (tmp_path / CartPoleChallenge.CONFIG_MAP["random"]).write_text(
        "agent:\n  config: {}\ntraining:\n  num_episodes: 7\n  total_steps: 900\n", encoding="utf-8"
    )
    challenge = CartPoleChallenge(config_dir=str(tmp_path))
    assert challenge.load_run_config("random", 1, None)["training"]["total_steps"] == 900
    assert challenge.load_run_config("random", 1, None, steps=50)["training"]["total_steps"] == 50
    by_episodes = challenge.load_run_config("random", 1, 3)["training"]
    assert by_episodes["num_episodes"] == 3 and "total_steps" not in by_episodes
