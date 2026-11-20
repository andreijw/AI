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
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from rl_core.training.parallel import num_envs_of
from rl_core.training.trainer import Trainer, success_rate
from rl_core.utils import plot_benchmark_comparison
from rl_core.utils.smoothing import rolling_average
from rl_core.utils.visualization import MISSING_PLOT_EXTRA, plotting_available

# The report a full rerun commits, and the curve PNG it links, committed beside the other docs
# media and listed in its manifest. All under the repository root (this file's folder), whatever
# the current folder. Any other report keeps its plot in the output folder, so a smoke or scratch
# run never rewrites a committed image.
REPO_ROOT = Path(__file__).resolve().parent
TRACKED_REPORTS_DIR = "docs/benchmarks"
TRACKED_CURVES_DIR = "docs/images/benchmarks"
MEDIA_MANIFEST = "scripts/doc_media.json"
# A committed PNG must fit the manifest's budget (png_kib, 60 KiB). At 100 dpi with a 64-colour
# palette, seven agents by five seeds of 3,000 noisy episodes measure about 31 KiB (at 150 dpi in
# full colour, about 160 KiB). A tracked PNG over the budget is redrawn with fewer colours, then
# at lower resolutions, in this order.
CURVE_DPI = 100
CURVE_COLOURS = 64
CURVE_ATTEMPTS = ((CURVE_DPI, CURVE_COLOURS), (CURVE_DPI, 32), (80, 32), (60, 32))


class CurvesOverBudgetError(RuntimeError):
    """A tracked curve PNG is over the docs media budget at the smallest drawing."""


def tracked_report_path(challenge_key: str) -> str:
    """The committed report of a challenge, under the repository root: the CLI's default."""
    return os.path.join(REPO_ROOT, TRACKED_REPORTS_DIR, f"{challenge_key}_results.md")


def tracked_curves_path(challenge_key: str) -> str:
    """The committed learning-curve PNG that a challenge's tracked report links."""
    return os.path.join(REPO_ROOT, TRACKED_CURVES_DIR, f"{challenge_key}_learning_curves.png")


def _in_repo(path: str) -> str | None:
    """``path`` relative to the repository root, in ``/`` form; None when it is outside."""
    real, root = os.path.realpath(path), os.path.realpath(REPO_ROOT)
    if os.path.commonpath([real, root]) != root:
        return None
    return Path(os.path.relpath(real, root)).as_posix()


def write_curves_png(
    path: str,
    curves: dict[str, list[list[float]]],
    *,
    window: int,
    title: str,
    target_score: float | None,
    episode_lengths: dict[str, list[list[int]]] | None = None,
    dpi: int = CURVE_DPI,
    colours: int = CURVE_COLOURS,
) -> str:
    """Plot the agents' learning curves to ``path`` as a palette PNG; return its absolute path.

    Every benchmark plot is made this way, so a scratch run shows what a tracked run commits.
    Needs rl-core's plot extra (matplotlib, which brings Pillow).
    """
    plot_benchmark_comparison(
        curves,
        window=window,
        save_path=path,
        target_score=target_score,
        title=title,
        episode_lengths=episode_lengths,
        dpi=dpi,
    )
    from PIL import Image  # matplotlib's own dependency, installed wherever plotting is

    saved = os.path.abspath(path)
    with Image.open(saved) as image:
        palette = image.convert("RGB").quantize(colors=colours)
    palette.save(saved, optimize=True)
    return saved


def _media_manifest() -> dict[str, Any] | None:
    """The docs media manifest under the repository root; None, with a note, when there is none."""
    path = os.path.join(REPO_ROOT, MEDIA_MANIFEST)
    if not os.path.isfile(path):
        print(f"{MEDIA_MANIFEST} not found: its PNG budget is not checked, and the curves must be")
        print("listed in it before they are committed.")
        return None
    with open(path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)
    return manifest


def _png_budget_bytes(manifest: dict[str, Any] | None) -> int | None:
    """The manifest's PNG budget in bytes, or None without one."""
    budget = (manifest or {}).get("budget")
    kib = budget.get("png_kib") if isinstance(budget, dict) else None
    return kib * 1024 if isinstance(kib, int) and kib > 0 else None


def _record_in_media_manifest(manifest: dict[str, Any], png: str, made_by: str) -> None:
    """List ``png`` (repository-relative) in the manifest with the command that made it.

    A rerun updates the entry's ``made_by`` in place and keeps its other keys and position; a new
    PNG is listed last. The file is replaced whole: written beside it, then moved over it.
    """
    media = manifest.setdefault("media", [])
    for entry in media:
        if entry.get("path") == png:
            entry["made_by"] = made_by
            break
    else:
        media.append({"path": png, "made_by": made_by})
    path = os.path.join(REPO_ROOT, MEDIA_MANIFEST)
    partial = f"{path}.partial"
    try:
        with open(partial, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    print(f"Listed {png} in {MEDIA_MANIFEST}")


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


def success_fields(
    eval_success: float | None, episode_successes: list[bool | None], window: int
) -> dict[str, float]:
    """
    A run record's success rates, for an environment that reports success: ``eval_success_rate``
    over the final evaluation and ``final_success_rate`` over the last ``window`` training
    episodes, the window of the solve rule and of held. Each is left out when unknown.
    """
    fields = {"eval_success_rate": eval_success}
    fields["final_success_rate"] = success_rate(episode_successes[-window:])
    return {name: float(rate) for name, rate in fields.items() if rate is not None}


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
        for field in ("eval_success_rate", "final_success_rate"):
            if all(field in r for r in runs):
                summary[agent_type][f"mean_{field}"] = float(np.mean([r[field] for r in runs]))
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
        output_dir: Folder for the plot and metrics JSON. None means plots/benchmarks/<key>. A
            run writing the tracked report puts its plot in docs/images/benchmarks/ instead.
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
    # The run's curriculum mode, or None for a run without one.
    curriculum: str | None = None
    # Training environments stepped in lockstep (training.num_envs).
    num_envs: int = 1


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
            report_path: Path for Markdown summary report. The tracked report
                (:func:`tracked_report_path`) links a plot at :func:`tracked_curves_path`, listed
                in the docs media manifest; any other report links the output folder's plot.
            save_plot: Whether to generate and save comparative learning curve plots.

        Returns:
            Dictionary containing raw runs, aggregated statistics, and output artifact paths.

        Raises:
            ValueError: For invalid settings, before anything is written.
            ImportError: If an agent's config needs a backend that is not installed, before any
                agent trains.
            ModuleNotFoundError: If ``save_plot`` and matplotlib (rl-core's plot extra) is not
                installed, before anything is written.
        """
        resolved = self._resolve(BenchmarkSettings() if settings is None else settings)
        if save_plot and not plotting_available():
            raise ModuleNotFoundError(MISSING_PLOT_EXTRA, name="matplotlib")
        os.makedirs(resolved.output_dir, exist_ok=True)
        self._print_banner(resolved)

        runs = {agent_type: self._run_agent(agent_type, resolved) for agent_type in resolved.agents}
        results = {agent: [run.result for run in agent_runs] for agent, agent_runs in runs.items()}
        # Each agent's budget as its last seed set it; every seed of an agent shares one config.
        budgets = {agent: agent_runs[-1].budget for agent, agent_runs in runs.items()}
        # Whether every run evaluated on a separate environment (challenges without configs do not).
        separate_eval = all(run.separate_eval for agent_runs in runs.values() for run in agent_runs)
        summary = summarise_runs(results, budgets)
        # Each agent's curriculum mode as its last seed set it; every seed shares one config.
        curricula = {
            agent: agent_runs[-1].curriculum
            for agent, agent_runs in runs.items()
            if agent_runs[-1].curriculum is not None
        }
        # Each agent's training.num_envs above 1, as its last seed set it.
        parallel = {
            agent: agent_runs[-1].num_envs
            for agent, agent_runs in runs.items()
            if agent_runs[-1].num_envs > 1
        }

        key = self.challenge.key
        tracked = report_path is not None and _same_path(report_path, tracked_report_path(key))
        if tracked:
            report_path = tracked_report_path(key)
        # Before the plot: a tracked plot over the docs budget stops the run, and the numbers stay.
        metrics_json_path = self._write_metrics_json(resolved, summary, results, budgets)
        manifest = _media_manifest() if tracked and save_plot else None
        earlier_curves = False
        if save_plot and tracked:
            plot_file: str | None = self._save_tracked_plot(
                runs, budgets, resolved, _png_budget_bytes(manifest), metrics_json_path
            )
        elif save_plot:
            plot_file = self._save_plot(
                runs,
                budgets,
                resolved,
                os.path.join(resolved.output_dir, "benchmark_comparison.png"),
            )
        else:
            # A tracked --no-plot rerun keeps linking the curves an earlier run committed.
            earlier_curves = tracked and os.path.isfile(tracked_curves_path(key))
            plot_file = tracked_curves_path(key) if earlier_curves else None
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            report_md = self._generate_markdown_report(
                summary,
                resolved,
                plot_file,
                report_path,
                separate_eval,
                curricula,
                parallel,
                earlier_curves,
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            print(f"Benchmark markdown report saved to: {report_path}")
        if manifest is not None:
            report_name = f"{TRACKED_REPORTS_DIR}/{key}_results.md"
            _record_in_media_manifest(
                manifest,
                f"{TRACKED_CURVES_DIR}/{key}_learning_curves.png",
                f"{self._command(resolved)} (the curves of {report_name})",
            )

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
        extra_envs = []
        try:
            extra_envs = self.challenge.setup_extra_envs(agent_type, seed, num_envs_of(train_cfg))
            curriculum = self.challenge.setup_curriculum(agent_type, seed, env)
            trainer = Trainer(
                agent=agent,
                env=env,
                config=train_cfg,
                logger=self.logger,
                eval_env=eval_env,
                eval_seed=None if eval_env is None else eval_seed_for(seed),
                curriculum=curriculum,
                extra_envs=extra_envs,
            )

            start_time = time.perf_counter()
            trainer.train()
            duration = time.perf_counter() - start_time

            eval_stats = trainer.evaluate(num_episodes=settings.eval_episodes)
            eval_mean = float(eval_stats.get("mean_reward", 0.0))
            eval_success = eval_stats.get("success_rate")

            episode_rewards = [float(r) for r in trainer.episode_rewards]
            episode_lengths = [int(n) for n in trainer.episode_lengths]
            episode_successes = list(trainer.episode_successes)
        finally:
            for opened in (env, *extra_envs):
                opened.close()
            if eval_env is not None:
                eval_env.close()

        result = self._run_record(
            seed, duration, eval_mean, episode_rewards, episode_lengths, settings.window
        )
        result.update(success_fields(eval_success, episode_successes, settings.window))
        return SeedRun(
            result,
            episode_rewards,
            episode_lengths,
            budget,
            eval_env is not None,
            None if curriculum is None else curriculum.mode,
            num_envs_of(train_cfg),
        )

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

    def _command(self, settings: _ResolvedSettings) -> str:
        """The benchmark.py command line that runs ``settings`` on this challenge."""
        parts = ["python benchmark.py", "--challenge", self.challenge.key]
        parts += ["--agents", *settings.agents, "--seeds", *map(str, settings.seeds)]
        if settings.episodes is not None:
            parts += ["--episodes", str(settings.episodes)]
        if settings.steps is not None:
            parts += ["--steps", str(settings.steps)]
        parts += ["--window", str(settings.window), "--eval-episodes", str(settings.eval_episodes)]
        config_dir = getattr(self.challenge, "config_dir", "configs")
        relative = _in_repo(config_dir)
        if relative is None:
            real = os.path.realpath(config_dir)
            print(
                f"Note: --config-dir {real} is outside the repository, so {MEDIA_MANIFEST} will "
                "hold this machine's path."
            )
            parts += ["--config-dir", real]
        elif relative != "configs":
            parts += ["--config-dir", relative]
        return " ".join(parts)

    def _save_tracked_plot(
        self,
        runs: dict[str, list[SeedRun]],
        budgets: dict[str, tuple[int, str]],
        settings: _ResolvedSettings,
        budget: int | None,
        metrics_json_path: str,
    ) -> str:
        """Plot the tracked curves within ``budget`` bytes, smaller in turn (``CURVE_ATTEMPTS``).

        Raises:
            CurvesOverBudgetError: If even the smallest drawing is over the budget.
        """
        path = tracked_curves_path(self.challenge.key)
        if budget is None:
            return self._save_plot(runs, budgets, settings, path)
        for dpi, colours in CURVE_ATTEMPTS:
            plot_file = self._save_plot(runs, budgets, settings, path, dpi=dpi, colours=colours)
            size = os.path.getsize(plot_file)
            if size <= budget:
                return plot_file
            print(f"{size / 1024:.1f} KiB at {dpi} dpi and {colours} colours is over the budget.")
        raise CurvesOverBudgetError(
            f"{TRACKED_CURVES_DIR}/{self.challenge.key}_learning_curves.png is {size / 1024:.1f} "
            f"KiB at {dpi} dpi and {colours} colours, over the {budget // 1024} KiB budget "
            f"(png_kib in {MEDIA_MANIFEST}); the report and the manifest were not written. The "
            f"run's numbers are in {metrics_json_path}."
        )

    def _save_plot(
        self,
        runs: dict[str, list[SeedRun]],
        budgets: dict[str, tuple[int, str]],
        settings: _ResolvedSettings,
        plot_path: str,
        **drawing: int,
    ) -> str:
        """Plot every agent's learning curves to ``plot_path``; return the image's absolute path."""
        curve_data = {
            agent: [run.rewards for run in agent_runs] for agent, agent_runs in runs.items()
        }
        length_data = {
            agent: [run.lengths for run in agent_runs] for agent, agent_runs in runs.items()
        }
        plot_file = write_curves_png(
            plot_path,
            curve_data,
            window=settings.window,
            target_score=self.challenge.solve_threshold,
            title=f"{self.challenge.name} Multi-Agent Benchmark Comparison",
            # Against steps only when every agent has a step budget: under episode budgets
            # the shortest seed would cut every other seed's curve short.
            episode_lengths=(
                length_data if all(u == "steps" for _, u in budgets.values()) else None
            ),
            **drawing,
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
        curricula: dict[str, str] | None = None,
        parallel: dict[str, int] | None = None,
        earlier_curves: bool = False,
    ) -> str:
        """Generate formatted Markdown report text for this challenge. ``parallel`` maps each agent
        that trained from several environments to its ``training.num_envs``; ``earlier_curves``
        says the linked plot is from an earlier run."""
        has_step_budget = any(stats["budget_unit"] == "steps" for stats in summary.values())
        lines = [
            *self._report_header(),
            *self._report_configuration(settings),
            *self._report_solve_rule(settings, separate_eval, has_step_budget, parallel or {}),
            *_report_curricula(curricula or {}),
            *_report_table(settings.agents, summary),
            *_report_plot_link(plot_file, report_path, earlier_curves),
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
        self,
        settings: _ResolvedSettings,
        separate_eval: bool,
        has_step_budget: bool,
        parallel: dict[str, int],
    ) -> list[str]:
        """The solve rule in words, with the evaluation and step-budget notes that apply.
        ``parallel`` maps each agent that trained from several environments to how many."""
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
            if parallel:
                trained = ", ".join(f"{agent}: {n}" for agent, n in sorted(parallel.items()))
                # One sentence whose agent list varies, wrapped to stay inside markdownlint's
                # 120-character MD013 limit (.markdownlint.json) like the fixed lines above.
                lines.extend(
                    textwrap.wrap(
                        f"Runs that trained from N environments (`training.num_envs`, {trained}) "
                        "also finish the episodes in flight in the other environments, so they "
                        "can overshoot by up to N - 1 more episodes.",
                        width=100,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
        return lines


def _report_table(agents: list[str], summary: dict[str, dict[str, Any]]) -> list[str]:
    """The Performance Summary section: one row per agent, in the order they ran, with an Eval
    Success column (the mean of the runs' evaluation success rates) when the runs report one."""
    with_success = any("mean_eval_success_rate" in summary[agent] for agent in agents)
    return [
        "",
        "## Performance Summary",
        "",
        "| Algorithm | Budget | Episodes | Mean Eval Reward | Reward Std | Solve Rate "
        "| Avg Solve Episode | Avg Steps to Solve | Held | Eval ≥ Threshold | Mean Time (s) |"
        + (" Eval Success |" if with_success else ""),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        + (" :---: |" if with_success else ""),
        *(_report_row(agent_type, summary[agent_type], with_success) for agent_type in agents),
    ]


def _report_row(agent_type: str, stats: dict[str, Any], with_success: bool) -> str:
    solve_pct = f"{stats['solve_rate'] * 100:.0f}%"
    eval_pct = f"{stats['eval_threshold_rate'] * 100:.0f}%"
    held_pct = f"{stats['held_rate'] * 100:.0f}%"
    avg_solve = (
        f"{stats['avg_solved_episode']:.1f}" if stats["avg_solved_episode"] is not None else "N/A"
    )
    avg_steps = (
        f"{stats['avg_solved_step']:,.0f}" if stats["avg_solved_step"] is not None else "N/A"
    )
    row = (
        f"| **{agent_type}** | {stats['budget']} {stats['budget_unit']} | "
        f"{stats['episodes']:.0f} | {stats['mean_eval_reward']:.1f} | "
        f"±{stats['std_eval_reward']:.1f} | {solve_pct} | {avg_solve} | {avg_steps} | "
        f"{held_pct} | {eval_pct} | {stats['mean_duration']:.2f}s |"
    )
    if with_success:
        rate = stats.get("mean_eval_success_rate")
        row += " N/A |" if rate is None else f" {rate * 100:.0f}% |"
    return row


def _report_curricula(curricula: dict[str, str]) -> list[str]:
    """A note for each agent that trained through a curriculum; none without one."""
    if not curricula:
        return []
    return [
        "",
        *(
            f"- **{agent}** trained through a curriculum ({mode}): its solve episode comes from "
            "training rewards at mixed difficulties; **Mean Eval Reward** is greedy evaluation "
            "at the target."
            for agent, mode in curricula.items()
        ),
    ]


def _same_path(first: str, second: str) -> bool:
    """Whether two paths, relative to the current folder, absolute or through a symlink, name the
    same file."""
    return os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second))


def _report_plot_link(plot_file: str | None, report_path: str, earlier: bool = False) -> list[str]:
    """The Learning Trajectories section, linking the plot relative to the report; none without
    one. ``earlier`` adds a line saying the plot is from an earlier run."""
    if not plot_file:
        return []
    rel_plot = os.path.relpath(
        plot_file, start=os.path.dirname(os.path.abspath(report_path))
    ).replace("\\", "/")
    note = ["The curves are from an earlier run: this one was run with `--no-plot`.", ""]
    return [
        "",
        "## Learning Trajectories",
        "",
        *(note if earlier else []),
        f"![Benchmark Learning Curves]({rel_plot})",
    ]


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
    if not args.no_plot and not plotting_available():
        parser.error(f"{MISSING_PLOT_EXTRA}; or pass --no-plot")
    report_file = args.report if args.report is not None else tracked_report_path(args.challenge)

    settings = BenchmarkSettings(
        agents=None if args.agents is None else tuple(args.agents),
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        steps=args.steps,
        window=args.window,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )
    try:
        run_benchmark(
            challenge_name=args.challenge,
            settings=settings,
            config_dir=args.config_dir,
            report_path=report_file,
            save_plot=not args.no_plot,
        )
    except CurvesOverBudgetError as error:
        parser.exit(1, f"benchmark.py: error: {error}\n")


if __name__ == "__main__":
    main()
