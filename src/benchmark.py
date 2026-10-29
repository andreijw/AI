"""
General Reinforcement Learning Benchmark Interface & Runner.

Provides an extensible, challenge-agnostic benchmarking framework supporting
arbitrary reinforcement learning challenges (environments) and agent algorithms.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# Ensure src is in Python path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from rl_core.challenges import (
    CHALLENGES,
    BenchmarkChallenge,
    discover_challenges,
    import_failures_text,
    unknown_challenge_message,
)
from rl_core.seeding import eval_seed_for
from rl_core.training.budget import apply_budget_override, budget_of
from rl_core.training.trainer import Trainer
from rl_core.utils.smoothing import rolling_average

# ---------------------------------------------------------------------------
# General Benchmark Metrics & Runner
# ---------------------------------------------------------------------------


def find_solved_episode(
    rewards: list[float],
    threshold: float | None = 475.0,
    window: int = 100,
) -> int | None:
    """
    Find the first episode where the average of the last ``window`` rewards meets the threshold.

    Only full windows count: the first candidate is episode ``window``, so one high early
    episode is not a solve.

    Args:
        rewards: List of episode rewards.
        threshold: Score threshold defining task completion (None if unbounded).
        window: Rolling-average window size (default: 100).

    Returns:
        1-based episode index of first solve, or None if never reached.
    """
    if not rewards or threshold is None:
        return None
    rolling = rolling_average(rewards, window)
    for idx in range(window - 1, len(rolling)):
        if rolling[idx] >= threshold:
            return idx + 1
    return None


def _describe_budget(episodes: int | None, steps: int | None = None) -> str:
    """Say how long each run trains for, in the words the console and report use."""
    if steps is not None:
        return f"{steps} steps for every agent (`--steps`)"
    if episodes is not None:
        return f"{episodes} episodes for every agent (`--episodes`)"
    return "each agent's own config (`training.total_steps` or `num_episodes`), shown per agent"


def classify_run(
    rewards: list[float],
    eval_mean: float,
    threshold: float | None,
    window: int,
    episode_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """
    Decide whether one run solved the challenge, and whether it held it.

    A run is solved when the average of ``window`` consecutive training rewards first reaches
    the threshold; the solve episode is the last episode of that window, and ``solved_step`` the
    training steps up to and including it. It is held when the run has at least ``window``
    episodes and the average of its last ``window`` is at or above the threshold. A final
    evaluation at or above the threshold is recorded as ``eval_reached_threshold`` and is not a
    solve.

    Args:
        rewards: Training episode rewards.
        eval_mean: Mean reward of the final evaluation episodes.
        threshold: Solve threshold, or None if the challenge has none.
        window: Episodes averaged.
        episode_lengths: Steps of each training episode, for ``solved_step``.

    Returns:
        Dict with ``is_solved``, ``solved_episode`` (1-based, or None), ``solved_step`` (or None),
        ``held`` (None without a threshold) and ``eval_reached_threshold``.
    """
    solved_ep = find_solved_episode(rewards, threshold=threshold, window=window)
    solved_step = None
    if solved_ep is not None and episode_lengths is not None:
        solved_step = int(sum(episode_lengths[:solved_ep]))
    held = None
    if threshold is not None:
        held = len(rewards) >= window and float(np.mean(rewards[-window:])) >= threshold
    return {
        "is_solved": solved_ep is not None,
        "solved_episode": solved_ep,
        "solved_step": solved_step,
        "held": held,
        "eval_reached_threshold": threshold is not None and eval_mean >= threshold,
    }


def summarise_runs(
    results: dict[str, list[dict[str, Any]]], budgets: dict[str, tuple[int, str]]
) -> dict[str, dict[str, Any]]:
    """
    Aggregate each agent's runs into the statistics the metrics file and report show.

    Args:
        results: Per agent, the run records :meth:`BenchmarkRunner.run` collects.
        budgets: Per agent, the ``(value, unit)`` training budget of its runs.

    Returns:
        Per agent: budget, mean episodes, solve, held and evaluation rates, the mean and spread of
        the evaluation reward, mean duration, and the average solve episode and step (None when
        no run solved).
    """
    summary: dict[str, dict[str, Any]] = {}
    for agent_type, runs in results.items():
        eval_rewards = [r["eval_mean_reward"] for r in runs]
        durations = [r["duration"] for r in runs]
        solved_eps = [r["solved_episode"] for r in runs if r["is_solved"]]
        solved_steps = [r["solved_step"] for r in runs if r["solved_step"] is not None]
        eval_reached = [r for r in runs if r["eval_reached_threshold"]]
        held = [r for r in runs if r["held"]]

        summary[agent_type] = {
            "budget": budgets[agent_type][0],
            "budget_unit": budgets[agent_type][1],
            "episodes": float(np.mean([r["total_episodes"] for r in runs])),
            "avg_solved_step": float(np.mean(solved_steps)) if solved_steps else None,
            "held_rate": float(len(held) / len(runs)),
            "mean_eval_reward": float(np.mean(eval_rewards)),
            "std_eval_reward": float(np.std(eval_rewards)),
            "mean_duration": float(np.mean(durations)),
            "solve_rate": float(len(solved_eps) / len(runs)),
            "avg_solved_episode": float(np.mean(solved_eps)) if solved_eps else None,
            "eval_threshold_rate": float(len(eval_reached) / len(runs)),
            "runs": len(runs),
        }
    return summary


@dataclass(frozen=True)
class BenchmarkSettings:
    """
    What one benchmark runs, for how long, and how it is scored.

    Args:
        agents: Agent types to benchmark. None runs every agent the challenge supports.
        seeds: Random seeds; every agent runs once per seed.
        episodes: Training episodes for every agent. With neither this nor ``steps``, each agent
            trains for its own config's budget (see :meth:`BenchmarkChallenge.load_run_config`).
        steps: Training environment steps for every agent; at most one of the two budgets.
        window: Episodes averaged for the solve rule, held and the learning curves.
        eval_episodes: Greedy evaluation episodes at the end of each run.
        output_dir: Folder for the plot and metrics JSON. None means plots/benchmarks/<key>.
    """

    agents: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = (42, 123)
    episodes: int | None = None
    steps: int | None = None
    window: int = 100
    eval_episodes: int = 10
    output_dir: str | None = None


@dataclass(frozen=True)
class _ResolvedSettings:
    """BenchmarkSettings checked against one challenge, with its defaults filled in.

    Lists, not tuples: the banner and the report print them, and published reports show `[42, 123]`.
    """

    agents: list[str]
    seeds: list[int]
    episodes: int | None
    steps: int | None
    window: int
    eval_episodes: int
    output_dir: str


@dataclass(frozen=True)
class SeedRun:
    """One agent's run on one seed: its record, learning curve and training budget."""

    result: dict[str, Any]
    rewards: list[float]
    lengths: list[int]
    budget: tuple[int, str]
    separate_eval: bool


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
        settings: BenchmarkSettings | None = None,
        *,
        report_path: str | None = None,
        save_plot: bool = True,
    ) -> dict[str, Any]:
        """
        Execute benchmark evaluation across agents and seeds.

        Args:
            settings: What to run and how to score it (default: ``BenchmarkSettings()``).
            report_path: Path for Markdown summary report.
            save_plot: Whether to generate and save comparative learning curve plots.

        Returns:
            Dictionary containing raw runs, aggregated statistics, and output artifact paths.

        Raises:
            ValueError: For invalid settings, before anything is written.
            ImportError: If an agent's config needs a backend that is not installed, before any
                agent trains.
        """
        resolved = self._resolve(BenchmarkSettings() if settings is None else settings)
        os.makedirs(resolved.output_dir, exist_ok=True)
        self._print_banner(resolved)

        runs = {agent_type: self._run_agent(agent_type, resolved) for agent_type in resolved.agents}
        results = {agent: [run.result for run in agent_runs] for agent, agent_runs in runs.items()}
        # Each agent's budget as its last seed set it; every seed of an agent shares one config.
        budgets = {agent: agent_runs[-1].budget for agent, agent_runs in runs.items()}
        # Whether every run evaluated on a separate environment (challenges without configs do not).
        separate_eval = all(run.separate_eval for agent_runs in runs.values() for run in agent_runs)
        summary = summarise_runs(results, budgets)

        plot_file = self._save_plot(runs, budgets, resolved) if save_plot else None
        metrics_json_path = self._write_metrics_json(resolved, summary, results, budgets)
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            report_md = self._generate_markdown_report(
                summary, resolved, plot_file, report_path, separate_eval
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

    def _resolve(self, requested: BenchmarkSettings) -> _ResolvedSettings:
        """Fill in the defaults and check the settings of :meth:`run`, before anything is written."""
        agents = list(
            self.challenge.supported_agents if requested.agents is None else requested.agents
        )
        seeds = list(requested.seeds)
        episodes, steps = requested.episodes, requested.steps
        output_dir = requested.output_dir or f"plots/benchmarks/{self.challenge.key}"

        if not agents:
            raise ValueError("agents list must not be empty.")
        for agent_type in agents:
            if agent_type not in self.challenge.supported_agents:
                raise ValueError(
                    f"Agent '{agent_type}' is not supported by challenge '{self.challenge.name}'. "
                    f"Supported: {self.challenge.supported_agents}"
                )
            self.challenge.check_agent_buildable(agent_type)

        if episodes is not None and episodes <= 0:
            raise ValueError(f"episodes must be positive, got {episodes}")
        if steps is not None and steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        apply_budget_override({}, episodes, steps)  # rejects both at once
        if requested.window < 1:
            raise ValueError(f"window must be >= 1, got {requested.window}")
        if requested.eval_episodes < 1:
            raise ValueError(f"eval_episodes must be >= 1, got {requested.eval_episodes}")
        if not seeds:
            raise ValueError("seeds list must not be empty.")
        return _ResolvedSettings(
            agents,
            seeds,
            episodes,
            steps,
            requested.window,
            requested.eval_episodes,
            output_dir,
        )

    def _print_banner(self, settings: _ResolvedSettings) -> None:
        print("=" * 70)
        print(f"Starting Benchmark Suite for Challenge: {self.challenge.name}")
        print(f"Agents: {settings.agents}")
        print(f"Seeds:  {settings.seeds}")
        print(f"Budget per run: {_describe_budget(settings.episodes, settings.steps)}")
        if self.challenge.solve_threshold is not None:
            print(f"Solve Threshold:  {self.challenge.solve_threshold}")
        print("=" * 70)

    def _run_agent(self, agent_type: str, settings: _ResolvedSettings) -> list[SeedRun]:
        """Run one agent on every seed in turn, printing a line per seed."""
        print(f"\n[Benchmarking] Agent: {agent_type.upper()} ({len(settings.seeds)} seeds)")
        runs = []
        for seed in settings.seeds:
            run = self._run_one_seed(agent_type, seed, settings)
            runs.append(run)
            result = run.result
            status = (
                f"Solved @ ep {result['solved_episode']}"
                if result["is_solved"]
                else "Did not solve"
            )
            print(
                f"  -> Seed {seed:4d} | Eval Reward: {result['eval_mean_reward']:6.1f} | "
                f"Time: {result['duration']:5.2f}s | {status}"
            )
        return runs

    def _run_one_seed(self, agent_type: str, seed: int, settings: _ResolvedSettings) -> SeedRun:
        """
        Train and evaluate one agent on one seed, and classify the run.

        The training run is set up before the evaluation environment, and both are closed even
        if training fails.
        """
        if settings.steps is None:
            env, agent, train_cfg = self.challenge.setup_run(agent_type, seed, settings.episodes)
        else:
            env, agent, train_cfg = self.challenge.setup_run(
                agent_type, seed, settings.episodes, steps=settings.steps
            )
        budget = budget_of(train_cfg)
        eval_env = self.challenge.setup_eval_env(agent_type, seed)
        try:
            trainer = Trainer(
                agent=agent,
                env=env,
                config=train_cfg,
                logger=self.logger,
                eval_env=eval_env,
                eval_seed=None if eval_env is None else eval_seed_for(seed),
            )

            start_time = time.perf_counter()
            trainer.train()
            duration = time.perf_counter() - start_time

            eval_stats = trainer.evaluate(num_episodes=settings.eval_episodes)
            eval_mean = float(eval_stats.get("mean_reward", 0.0))

            episode_rewards = [float(r) for r in trainer.episode_rewards]
            episode_lengths = [int(n) for n in trainer.episode_lengths]
        finally:
            env.close()
            if eval_env is not None:
                eval_env.close()

        result = self._run_record(
            seed, duration, eval_mean, episode_rewards, episode_lengths, settings.window
        )
        return SeedRun(result, episode_rewards, episode_lengths, budget, eval_env is not None)

    def _run_record(
        self,
        seed: int,
        duration: float,
        eval_mean: float,
        episode_rewards: list[float],
        episode_lengths: list[int],
        window: int,
    ) -> dict[str, Any]:
        """The record of one finished run, as the metrics file stores it."""
        outcome = classify_run(
            episode_rewards,
            eval_mean,
            threshold=self.challenge.solve_threshold,
            window=window,
            episode_lengths=episode_lengths,
        )
        return {
            "seed": seed,
            "duration": duration,
            "eval_mean_reward": eval_mean,
            "final_rolling_avg": float(rolling_average(episode_rewards, window)[-1])
            if episode_rewards
            else 0.0,
            "solved_episode": outcome["solved_episode"],
            "solved_step": outcome["solved_step"],
            "is_solved": outcome["is_solved"],
            "held": outcome["held"],
            "eval_reached_threshold": outcome["eval_reached_threshold"],
            "total_episodes": len(episode_rewards),
            "total_steps": int(sum(episode_lengths)),
            "window": window,
        }

    def _save_plot(
        self,
        runs: dict[str, list[SeedRun]],
        budgets: dict[str, tuple[int, str]],
        settings: _ResolvedSettings,
    ) -> str | None:
        """Plot every agent's learning curves into the output folder; return the image path."""
        # Imported here: matplotlib adds about 0.8 s to every start-up, plotted or not.
        from rl_core.utils.visualization import plot_benchmark_comparison

        curve_data = {
            agent: [run.rewards for run in agent_runs] for agent, agent_runs in runs.items()
        }
        length_data = {
            agent: [run.lengths for run in agent_runs] for agent, agent_runs in runs.items()
        }
        plot_file = plot_benchmark_comparison(
            curve_data,
            window=settings.window,
            save_path=os.path.join(settings.output_dir, "benchmark_comparison.png"),
            target_score=self.challenge.solve_threshold,
            title=f"{self.challenge.name} Multi-Agent Benchmark Comparison",
            # Against steps only when every agent has a step budget: under episode budgets
            # the shortest seed would cut every other seed's curve short.
            episode_lengths=(
                length_data if all(u == "steps" for _, u in budgets.values()) else None
            ),
        )
        print(f"\nBenchmark comparison plot saved to: {plot_file}")
        return plot_file

    def _write_metrics_json(
        self,
        settings: _ResolvedSettings,
        summary: dict[str, dict[str, Any]],
        results: dict[str, list[dict[str, Any]]],
        budgets: dict[str, tuple[int, str]],
    ) -> str:
        """Write the summary, every run and the settings to benchmark_metrics.json; return its path."""
        metrics_json_path = os.path.join(settings.output_dir, "benchmark_metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "challenge": self.challenge.name,
                    "summary": summary,
                    "detailed_runs": results,
                    "metadata": {
                        "agents": settings.agents,
                        "seeds": settings.seeds,
                        "episodes": settings.episodes,
                        "steps": settings.steps,
                        "budgets_per_agent": {
                            agent: {"value": value, "unit": unit}
                            for agent, (value, unit) in budgets.items()
                        },
                        "episodes_per_agent": {
                            agent: summary[agent]["episodes"] for agent in summary
                        },
                        "window": settings.window,
                        "eval_episodes": settings.eval_episodes,
                    },
                },
                f,
                indent=2,
            )
        return metrics_json_path

    def _generate_markdown_report(
        self,
        summary: dict[str, dict[str, Any]],
        settings: _ResolvedSettings,
        plot_file: str | None,
        report_path: str,
        separate_eval: bool,
    ) -> str:
        """Generate formatted Markdown report text for this challenge."""
        has_step_budget = any(stats["budget_unit"] == "steps" for stats in summary.values())
        lines = [
            *self._report_header(),
            *self._report_configuration(settings),
            *self._report_solve_rule(settings, separate_eval, has_step_budget),
            *_report_table(settings.agents, summary),
            *_report_plot_link(plot_file, report_path),
            "",
        ]
        return "\n".join(lines)

    def _report_header(self) -> list[str]:
        return [
            f"# {self.challenge.name} Multi-Agent Benchmark Results",
            "",
            "Comparative benchmark evaluation across reinforcement learning algorithms.",
            "",
        ]

    def _report_configuration(self, settings: _ResolvedSettings) -> list[str]:
        threshold = self.challenge.solve_threshold
        thresh_str = f">= {threshold:.0f}" if threshold is not None else "N/A"
        return [
            "## Benchmark Configuration",
            "",
            f"- **Challenge / Environment**: `{self.challenge.name}` (Solved criteria: average reward {thresh_str})",
            f"- **Budget per Run**: {_describe_budget(settings.episodes, settings.steps)}",
            f"- **Seeds Evaluated**: `{settings.seeds}`",
            f"- **Algorithms**: `{settings.agents}`",
            "",
        ]

    def _report_solve_rule(
        self, settings: _ResolvedSettings, separate_eval: bool, has_step_budget: bool
    ) -> list[str]:
        """The solve rule in words, with the evaluation and step-budget notes that apply."""
        lines = ["## How a Run Counts as Solved", ""]
        threshold = self.challenge.solve_threshold
        if threshold is None:
            lines.append("This challenge has no solve threshold, so no run counts as solved.")
            return lines
        window, eval_episodes = settings.window, settings.eval_episodes
        eval_desc = f"{eval_episodes} greedy episode{'' if eval_episodes == 1 else 's'}"
        lines.extend(
            [
                f"A run is solved when its average training reward over {window} consecutive",
                f"episodes reaches {threshold:.0f}. **Avg Solve Episode** is the last episode of",
                "the first such window, and **Avg Steps to Solve** the training steps up to it,",
                "averaged over the solved runs. **Held** is the share of runs with at least",
                f"{window} episodes whose last {window} average at least {threshold:.0f}.",
                "**Eval ≥ Threshold** is the share of runs whose final evaluation",
                f"({eval_desc}) averaged at least {threshold:.0f}; it does not count as a solve.",
            ]
        )
        if separate_eval:
            lines.extend(
                [
                    "Evaluation runs on a separate environment and plays the same start states",
                    "every time for a seed, so it does not change the training episodes.",
                ]
            )
        if has_step_budget:
            lines.append(
                "A step budget finishes the episode that crosses it, so a run can overshoot "
                "its budget by up to one episode."
            )
        return lines


def _report_table(agents: list[str], summary: dict[str, dict[str, Any]]) -> list[str]:
    """The Performance Summary section: one row per agent, in the order they ran."""
    return [
        "",
        "## Performance Summary",
        "",
        "| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate "
        "| Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        *(_report_row(agent_type, summary[agent_type]) for agent_type in agents),
    ]


def _report_row(agent_type: str, stats: dict[str, Any]) -> str:
    solve_pct = f"{stats['solve_rate'] * 100:.0f}%"
    eval_pct = f"{stats['eval_threshold_rate'] * 100:.0f}%"
    held_pct = f"{stats['held_rate'] * 100:.0f}%"
    avg_solve = (
        f"{stats['avg_solved_episode']:.1f}" if stats["avg_solved_episode"] is not None else "N/A"
    )
    avg_steps = (
        f"{stats['avg_solved_step']:,.0f}" if stats["avg_solved_step"] is not None else "N/A"
    )
    return (
        f"| **{agent_type}** | {stats['budget']} {stats['budget_unit']} | "
        f"{stats['episodes']:.0f} | {stats['mean_eval_reward']:.1f} | "
        f"±{stats['std_eval_reward']:.1f} | {solve_pct} | {avg_solve} | {avg_steps} | "
        f"{held_pct} | {eval_pct} | {stats['mean_duration']:.2f}s |"
    )


def _report_plot_link(plot_file: str | None, report_path: str) -> list[str]:
    """The Learning Trajectories section, linking the plot relative to the report; none without one."""
    if not plot_file:
        return []
    rel_plot = os.path.relpath(
        plot_file, start=os.path.dirname(os.path.abspath(report_path))
    ).replace("\\", "/")
    return ["", "## Learning Trajectories", "", f"![Benchmark Learning Curves]({rel_plot})"]


def run_benchmark(
    challenge_name: str = "cartpole",
    *,
    settings: BenchmarkSettings | None = None,
    config_dir: str = "configs",
    report_path: str | None = None,
    save_plot: bool = True,
) -> dict[str, Any]:
    """
    High-level API entrypoint to execute a benchmark on any registered challenge.

    Args:
        challenge_name: Key of a registered challenge ('cartpole').
        settings: What to run and how to score it (default: ``BenchmarkSettings()``).
        config_dir: Configuration directory.
        report_path: Path to Markdown report file.
        save_plot: Whether to save comparison plot.

    Returns:
        Dictionary with benchmark results and file paths.
    """
    discover_challenges()
    if challenge_name not in CHALLENGES:
        raise ValueError(unknown_challenge_message(challenge_name))

    challenge = CHALLENGES[challenge_name](config_dir=config_dir)
    runner = BenchmarkRunner(challenge)

    return runner.run(settings, report_path=report_path, save_plot=save_plot)


def main() -> None:
    """CLI entrypoint for benchmark runner."""
    discover_challenges()
    parser = argparse.ArgumentParser(
        description="General Reinforcement Learning Benchmark Suite",
        epilog=import_failures_text() or None,
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default="cartpole",
        choices=sorted(CHALLENGES),
        help="Challenge / Environment to benchmark (default: cartpole)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="List of agent types to benchmark (defaults to all supported for challenge)",
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Training episodes for every agent (default: each agent's config budget)",
    )
    budget.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training environment steps for every agent (default: each agent's config budget)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(BenchmarkSettings.seeds),
        help=f"List of random seeds (default: {' '.join(map(str, BenchmarkSettings.seeds))})",
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
        default=BenchmarkSettings.window,
        help=(
            "Episodes averaged for the solve rule, held and the curves "
            f"(default: {BenchmarkSettings.window})"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating comparison plot",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=BenchmarkSettings.eval_episodes,
        help=f"Number of evaluation episodes per run (default: {BenchmarkSettings.eval_episodes})",
    )

    args = parser.parse_args()
    report_file = (
        args.report if args.report is not None else f"docs/benchmarks/{args.challenge}_results.md"
    )

    settings = BenchmarkSettings(
        agents=None if args.agents is None else tuple(args.agents),
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        steps=args.steps,
        window=args.window,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )
    run_benchmark(
        challenge_name=args.challenge,
        settings=settings,
        config_dir=args.config_dir,
        report_path=report_file,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
