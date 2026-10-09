"""
General Reinforcement Learning Benchmark Interface & Runner.

Provides an extensible, challenge-agnostic benchmarking framework supporting
arbitrary reinforcement learning challenges (environments) and agent algorithms.
"""

from __future__ import annotations

import abc
import argparse
import contextlib
import json
import logging
import os
import sys
import time
from typing import Any

# Ensure src is in Python path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from rl_cartpole.training.trainer import Trainer
from rl_cartpole.utils.visualization import _rolling_average, plot_benchmark_comparison

# Global registry of available benchmark challenges
CHALLENGES: dict[str, type[BenchmarkChallenge]] = {}


def register_challenge(cls: type[BenchmarkChallenge]) -> type[BenchmarkChallenge]:
    """Decorator or function to register a BenchmarkChallenge implementation."""
    # Instantiate or inspect to get key
    temp_inst = cls()
    CHALLENGES[temp_inst.key] = cls
    return cls


# ---------------------------------------------------------------------------
# Benchmark Challenge Interface (ABC)
# ---------------------------------------------------------------------------


class BenchmarkChallenge(abc.ABC):
    """
    Abstract interface defining an AI / RL benchmark challenge.

    Allows benchmarking arbitrary environments and algorithms under identical conditions.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable challenge name (e.g. 'CartPole-v1', 'Pendulum-v1')."""
        ...

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """CLI / lookup identifier key (e.g. 'cartpole', 'pendulum')."""
        ...

    @property
    @abc.abstractmethod
    def solve_threshold(self) -> float | None:
        """Reward score threshold defining task completion, or None if unbounded."""
        ...

    @property
    @abc.abstractmethod
    def default_episodes(self) -> int:
        """Default number of training episodes for this challenge."""
        ...

    @property
    @abc.abstractmethod
    def supported_agents(self) -> list[str]:
        """List of supported agent types for this challenge."""
        ...

    @abc.abstractmethod
    def setup_run(
        self, agent_type: str, seed: int, episodes: int
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Setup and return (env, agent, training_config) for a single benchmark run.

        Args:
            agent_type: Identifier of the agent to instantiate.
            seed: Random seed for environment and agent reproducibility.
            episodes: Number of training episodes configured.

        Returns:
            Tuple of (initialized_env, initialized_agent, training_config_dict).
        """
        ...


# Alias current module in sys.modules so imported challenge modules
# (e.g. rl_cartpole.benchmark) register into the active CHALLENGES dictionary
# even when src/benchmark.py is executed directly as __main__.
sys.modules.setdefault("benchmark", sys.modules[__name__])


# ---------------------------------------------------------------------------
# General Benchmark Metrics & Runner
# ---------------------------------------------------------------------------


def find_solved_episode(
    rewards: list[float],
    threshold: float | None = 475.0,
    window: int = 10,
) -> int | None:
    """
    Find the first episode where the rolling average reward meets or exceeds threshold.

    Args:
        rewards: List of episode rewards.
        threshold: Score threshold defining task completion (None if unbounded).
        window: Rolling-average window size (default: 10).

    Returns:
        1-based episode index of first solve, or None if never reached.
    """
    if not rewards or threshold is None:
        return None
    rolling = _rolling_average(rewards, window)
    for idx, avg in enumerate(rolling):
        if avg >= threshold:
            return idx + 1
    return None


class BenchmarkRunner:
    """
    Reusable multi-agent benchmark orchestration engine.

    Evaluates arbitrary RL agents across arbitrary challenge implementations and seeds,
    recording sample efficiency, convergence speed, evaluation stability, and wall-clock time.
    """

    def __init__(
        self,
        challenge: BenchmarkChallenge,
        logger: logging.Logger | None = None,
    ) -> None:
        self.challenge = challenge
        self.logger = logger or logging.getLogger("rl_benchmark")
        self.logger.setLevel(logging.WARNING)

    def run(
        self,
        agents: list[str] | None = None,
        episodes: int | None = None,
        seeds: list[int] | None = None,
        output_dir: str | None = None,
        report_path: str | None = None,
        save_plot: bool = True,
        window: int = 10,
        eval_episodes: int = 10,
    ) -> dict[str, Any]:
        """
        Execute benchmark evaluation across agents and seeds.

        Args:
            agents: List of agent types to benchmark. Defaults to challenge's supported agents.
            episodes: Training episodes per run. Defaults to challenge's default_episodes.
            seeds: List of random seeds (default: [42, 123]).
            output_dir: Output folder for plots and metrics.
            report_path: Path for Markdown summary report.
            save_plot: Whether to generate and save comparative learning curve plots.
            window: Rolling average window for smoothing learning curves.
            eval_episodes: Number of evaluation episodes at end of training.

        Returns:
            Dictionary containing raw runs, aggregated statistics, and output artifact paths.
        """
        agents = self.challenge.supported_agents if agents is None else agents
        episodes = self.challenge.default_episodes if episodes is None else episodes
        seeds = [42, 123] if seeds is None else seeds
        output_dir = output_dir or f"plots/benchmarks/{self.challenge.key}"

        if not agents:
            raise ValueError("agents list must not be empty.")
        for agent_type in agents:
            if agent_type not in self.challenge.supported_agents:
                raise ValueError(
                    f"Agent '{agent_type}' is not supported by challenge '{self.challenge.name}'. "
                    f"Supported: {self.challenge.supported_agents}"
                )

        if episodes <= 0:
            raise ValueError(f"episodes must be positive, got {episodes}")
        if not seeds:
            raise ValueError("seeds list must not be empty.")

        os.makedirs(output_dir, exist_ok=True)

        results: dict[str, list[dict[str, Any]]] = {}
        curve_data: dict[str, list[list[float]]] = {}

        print("=" * 70)
        print(f"Starting Benchmark Suite for Challenge: {self.challenge.name}")
        print(f"Agents: {agents}")
        print(f"Seeds:  {seeds}")
        print(f"Episodes per run: {episodes}")
        if self.challenge.solve_threshold is not None:
            print(f"Solve Threshold:  {self.challenge.solve_threshold}")
        print("=" * 70)

        for agent_type in agents:
            results[agent_type] = []
            curve_data[agent_type] = []

            print(f"\n[Benchmarking] Agent: {agent_type.upper()} ({len(seeds)} seeds)")

            for seed in seeds:
                env, agent, train_cfg = self.challenge.setup_run(agent_type, seed, episodes)

                trainer = Trainer(
                    agent=agent,
                    env=env,
                    config=train_cfg,
                    logger=self.logger,
                )

                start_time = time.perf_counter()
                trainer.train()
                duration = time.perf_counter() - start_time

                eval_stats = trainer.evaluate(num_episodes=eval_episodes)
                eval_mean = float(eval_stats.get("mean_reward", 0.0))

                episode_rewards = [float(r) for r in trainer.episode_rewards]
                solved_ep = find_solved_episode(
                    episode_rewards,
                    threshold=self.challenge.solve_threshold,
                    window=window,
                )

                # An agent has solved the challenge if rolling average met threshold
                # OR its final greedy evaluation met the threshold
                is_solved = (solved_ep is not None) or (
                    self.challenge.solve_threshold is not None
                    and eval_mean >= self.challenge.solve_threshold
                )
                if solved_ep is None and is_solved:
                    solved_ep = len(episode_rewards)

                run_result = {
                    "seed": seed,
                    "duration": duration,
                    "eval_mean_reward": eval_mean,
                    "final_rolling_avg": float(_rolling_average(episode_rewards, window)[-1])
                    if episode_rewards
                    else 0.0,
                    "solved_episode": solved_ep,
                    "is_solved": is_solved,
                    "total_episodes": len(episode_rewards),
                }
                results[agent_type].append(run_result)
                curve_data[agent_type].append(episode_rewards)

                env.close()

                status = f"Solved @ ep {solved_ep}" if is_solved else "Did not solve"
                print(
                    f"  -> Seed {seed:4d} | Eval Reward: {eval_mean:6.1f} | "
                    f"Time: {duration:5.2f}s | {status}"
                )

        summary: dict[str, dict[str, Any]] = {}
        for agent_type, runs in results.items():
            eval_rewards = [r["eval_mean_reward"] for r in runs]
            durations = [r["duration"] for r in runs]
            solved_runs = [r for r in runs if r["is_solved"]]
            solved_eps = [
                r["solved_episode"] for r in solved_runs if r["solved_episode"] is not None
            ]

            summary[agent_type] = {
                "mean_eval_reward": float(np.mean(eval_rewards)),
                "std_eval_reward": float(np.std(eval_rewards)),
                "mean_duration": float(np.mean(durations)),
                "solve_rate": float(len(solved_runs) / len(runs)),
                "avg_solved_episode": float(np.mean(solved_eps)) if solved_eps else None,
                "runs": len(runs),
            }

        plot_file = None
        if save_plot:
            plot_path = os.path.join(output_dir, "benchmark_comparison.png")
            plot_file = plot_benchmark_comparison(
                curve_data,
                window=window,
                save_path=plot_path,
                target_score=self.challenge.solve_threshold,
                title=f"{self.challenge.name} Multi-Agent Benchmark Comparison",
            )
            print(f"\nBenchmark comparison plot saved to: {plot_file}")

        metrics_json_path = os.path.join(output_dir, "benchmark_metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "challenge": self.challenge.name,
                    "summary": summary,
                    "detailed_runs": results,
                    "metadata": {
                        "agents": agents,
                        "seeds": seeds,
                        "episodes": episodes,
                        "window": window,
                    },
                },
                f,
                indent=2,
            )

        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            report_md = self._generate_markdown_report(
                summary, agents, seeds, episodes, plot_file, report_path
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            print(f"Benchmark markdown report saved to: {report_path}")

        return {
            "challenge": self.challenge.name,
            "summary": summary,
            "results": results,
            "plot_path": plot_file,
            "metrics_json": metrics_json_path,
            "report_path": report_path,
        }

    def _generate_markdown_report(
        self,
        summary: dict[str, dict[str, Any]],
        agents: list[str],
        seeds: list[int],
        episodes: int,
        plot_file: str | None,
        report_path: str | None = None,
    ) -> str:
        """Generate formatted Markdown report text for this challenge."""
        thresh_str = (
            f">= {self.challenge.solve_threshold:.0f}"
            if self.challenge.solve_threshold is not None
            else "N/A"
        )
        lines = [
            f"# {self.challenge.name} Multi-Agent Benchmark Results",
            "",
            "Comparative benchmark evaluation across reinforcement learning algorithms.",
            "",
            "## Benchmark Configuration",
            "",
            f"- **Challenge / Environment**: `{self.challenge.name}` (Solved criteria: average reward {thresh_str})",
            f"- **Episodes per Run**: {episodes}",
            f"- **Seeds Evaluated**: `{seeds}`",
            f"- **Algorithms**: `{agents}`",
            "",
            "## Performance Summary",
            "",
            "| Algorithm | Mean Eval Reward | Reward Std | Solve Rate | Avg Solve Episode | Mean Time (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for agent_type in agents:
            stats = summary[agent_type]
            solve_pct = f"{stats['solve_rate'] * 100:.0f}%"
            avg_solve = (
                f"{stats['avg_solved_episode']:.1f}"
                if stats["avg_solved_episode"] is not None
                else "N/A"
            )
            lines.append(
                f"| **{agent_type}** | {stats['mean_eval_reward']:.1f} | "
                f"±{stats['std_eval_reward']:.1f} | {solve_pct} | "
                f"{avg_solve} | {stats['mean_duration']:.2f}s |"
            )

        if plot_file:
            rel_plot = (
                os.path.relpath(
                    plot_file, start=os.path.dirname(os.path.abspath(report_path))
                ).replace("\\", "/")
                if report_path
                else os.path.basename(plot_file)
            )
            lines.extend(
                [
                    "",
                    "## Learning Trajectories",
                    "",
                    f"![Benchmark Learning Curves]({rel_plot})",
                ]
            )

        lines.append("")
        return "\n".join(lines)


def run_benchmark(
    challenge_name: str = "cartpole",
    agents: list[str] | None = None,
    episodes: int | None = None,
    seeds: list[int] | None = None,
    config_dir: str = "configs",
    output_dir: str | None = None,
    report_path: str | None = None,
    save_plot: bool = True,
    window: int = 10,
    eval_episodes: int = 10,
) -> dict[str, Any]:
    """
    High-level API entrypoint to execute a benchmark on any registered challenge.

    Args:
        challenge_name: Name of challenge ('cartpole').
        agents: List of agent names (defaults to all supported).
        episodes: Number of episodes per run.
        seeds: List of random seeds.
        config_dir: Configuration directory.
        output_dir: Output directory for plots and metrics.
        report_path: Path to Markdown report file.
        save_plot: Whether to save comparison plot.
        window: Rolling window size.
        eval_episodes: Number of evaluation episodes.

    Returns:
        Dictionary with benchmark results and file paths.
    """
    # Lazy import to ensure cartpole challenge is registered
    if "cartpole" not in CHALLENGES:
        import rl_cartpole.benchmark  # noqa: F401

    if challenge_name not in CHALLENGES:
        raise ValueError(
            f"Unknown challenge '{challenge_name}'. Supported: {sorted(CHALLENGES.keys())}"
        )

    challenge_cls = CHALLENGES[challenge_name]
    challenge = (
        challenge_cls(config_dir=config_dir)  # type: ignore[call-arg]
        if hasattr(challenge_cls, "__init__")
        and "config_dir" in challenge_cls.__init__.__code__.co_varnames
        else challenge_cls()
    )
    runner = BenchmarkRunner(challenge)

    return runner.run(
        agents=agents,
        episodes=episodes,
        seeds=seeds,
        output_dir=output_dir,
        report_path=report_path,
        save_plot=save_plot,
        window=window,
        eval_episodes=eval_episodes,
    )


def main() -> None:
    """CLI entrypoint for benchmark runner."""
    # Ensure standard challenges are loaded
    with contextlib.suppress(ImportError):
        import rl_cartpole.benchmark  # noqa: F401

    parser = argparse.ArgumentParser(description="General Reinforcement Learning Benchmark Suite")
    parser.add_argument(
        "--challenge",
        type=str,
        default="cartpole",
        choices=sorted(CHALLENGES.keys()) if CHALLENGES else ["cartpole"],
        help="Challenge / Environment to benchmark (default: cartpole)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="List of agent types to benchmark (defaults to all supported for challenge)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes per run (default: challenge default)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123],
        help="List of random seeds (default: 42 123)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to directory containing configuration YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated plots and metrics JSON",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="File path to save the Markdown benchmark report",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling average window size for curves (default: 10)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating comparison plot",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per run (default: 10)",
    )

    args = parser.parse_args()
    report_file = (
        args.report if args.report is not None else f"docs/benchmarks/{args.challenge}_results.md"
    )

    run_benchmark(
        challenge_name=args.challenge,
        agents=args.agents,
        episodes=args.episodes,
        seeds=args.seeds,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        report_path=report_file,
        save_plot=not args.no_plot,
        window=args.window,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
