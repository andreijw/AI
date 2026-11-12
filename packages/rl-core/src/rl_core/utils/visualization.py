"""Visualization utilities for training metrics."""

import os

import matplotlib

# On headless machines (no display server and no explicit MPLBACKEND configured)
# fall back to the non-interactive Agg backend so that saving plots works
# without a display.  Users with a display or an explicit MPLBACKEND override
# this behaviour automatically.
_has_display = bool(
    os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
    or os.name == "nt"  # Windows always has display infrastructure
)
if not _has_display and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .smoothing import rolling_average  # noqa: E402

__all__ = ["plot_training_metrics", "plot_benchmark_comparison"]

_PLOT_FILENAME = "training_metrics.png"


def plot_training_metrics(
    episode_rewards: list[float],
    episode_lengths: list[int],
    window: int = 10,
    save_dir: str | None = None,
    show: bool = False,
    title: str = "CartPole Random Agent – Training Metrics",
) -> str | None:
    """
    Plot training metrics: episode rewards and episode lengths over time.

    Both raw values and a rolling average are displayed on each subplot.
    Configure the matplotlib backend (e.g. ``MPLBACKEND=Agg``) before calling
    this function when running on a headless machine.

    Args:
        episode_rewards: List of total reward per episode (must be non-empty
            and the same length as *episode_lengths*).
        episode_lengths: List of step count per episode (must be non-empty
            and the same length as *episode_rewards*).
        window: Rolling-average window size (must be >= 1; default: 10).
        save_dir: Directory to save the plot PNG.  If ``None`` the plot is
            not saved to disk.
        show: Whether to call ``plt.show()`` (requires an interactive display).
        title: Figure title.

    Returns:
        Absolute path to the saved PNG, or ``None`` if *save_dir* is not set.

    Raises:
        ValueError: If *episode_rewards* or *episode_lengths* is empty.
        ValueError: If *episode_rewards* and *episode_lengths* have different lengths.
        ValueError: If *window* is less than 1.
    """
    if not episode_rewards:
        raise ValueError("episode_rewards must not be empty.")
    if not episode_lengths:
        raise ValueError("episode_lengths must not be empty.")
    if len(episode_rewards) != len(episode_lengths):
        raise ValueError(
            f"episode_rewards and episode_lengths must have the same length; "
            f"got {len(episode_rewards)} and {len(episode_lengths)}."
        )
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}.")

    episodes = list(range(1, len(episode_rewards) + 1))
    rewards_avg = rolling_average(episode_rewards, window)
    lengths_avg = rolling_average([float(x) for x in episode_lengths], window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    # --- Episode rewards ---
    ax1.plot(episodes, episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax1.plot(
        episodes,
        rewards_avg,
        color="steelblue",
        linewidth=2,
        label=f"Rolling avg (w={window})",
    )
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Episode lengths ---
    ax2.plot(episodes, episode_lengths, alpha=0.3, color="darkorange", label="Episode length")
    ax2.plot(
        episodes,
        lengths_avg,
        color="darkorange",
        linewidth=2,
        label=f"Rolling avg (w={window})",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path: str | None = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.abspath(os.path.join(save_dir, _PLOT_FILENAME))
        fig.savefig(saved_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


class EpisodeLengthsMismatchError(ValueError):
    """``episode_lengths`` does not give one step count per episode of each run."""


def aligned_curves(
    runs: list[list[float]],
    window: int,
    episode_lengths: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth one agent's runs and align them on a shared axis: the x values, mean and spread drawn.

    Each run is smoothed with a trailing rolling average of ``window`` episodes. By episode, every
    run is cut to the shortest one and x is 1, 2, ... By environment steps (``episode_lengths``
    given, one list of episode step counts per run), each run is interpolated between its
    cumulative episode ends onto 200 points from the latest first episode end to the earliest last
    one, the stretch where every run has data, so runs of different episode counts are compared at
    equal training effort.

    Returns:
        ``(x, mean, std)`` arrays of equal length: the population standard deviation across runs,
        0 for a single run.

    Raises:
        ValueError: If ``window`` is below 1, there is no run or an empty one, or an episode length is
            not positive (the step grid needs strictly increasing episode ends).
        EpisodeLengthsMismatchError: If ``episode_lengths`` does not give one step count per episode
            of each run.
    """
    if window < 1 or not runs or not all(len(run) for run in runs):
        raise ValueError(
            f"aligned_curves needs window >= 1 and non-empty runs; got window {window}."
        )
    smoothed = [rolling_average(run, window) for run in runs]
    if episode_lengths is not None:
        if [len(lengths) for lengths in episode_lengths] != [len(r) for r in smoothed]:
            raise EpisodeLengthsMismatchError("episode_lengths do not match the runs.")
        if any(min(lengths) <= 0 for lengths in episode_lengths):
            raise ValueError("episode_lengths must all be positive.")
        ends = [np.cumsum(lengths) for lengths in episode_lengths]
        start = max(float(e[0]) for e in ends)
        stop = min(float(e[-1]) for e in ends)
        x = np.linspace(start, stop, 200)
        aligned = np.array([np.interp(x, e, r) for e, r in zip(ends, smoothed, strict=True)])
    else:
        shortest = min(len(r) for r in smoothed)
        aligned = np.array([r[:shortest] for r in smoothed], dtype=np.float64)
        x = np.arange(1, shortest + 1, dtype=np.float64)
    return x, np.mean(aligned, axis=0), np.std(aligned, axis=0)


def _check_benchmark_curves(agent_curves: dict[str, list[list[float]]], window: int) -> None:
    if not agent_curves:
        raise ValueError("agent_curves must not be empty.")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}.")
    for name, runs in agent_curves.items():
        if not runs:
            raise ValueError(f"Runs for agent '{name}' must not be empty.")
        for i, run in enumerate(runs):
            if not run:
                raise ValueError(f"Run {i} for agent '{name}' must not be empty.")


def plot_benchmark_comparison(
    agent_curves: dict[str, list[list[float]]],
    window: int = 10,
    save_path: str | None = None,
    show: bool = False,
    title: str = "CartPole Multi-Agent Benchmark Comparison",
    target_score: float | None = 475.0,
    episode_lengths: dict[str, list[list[int]]] | None = None,
) -> str | None:
    """
    Plot comparative reward trajectories for multiple agents across training episodes, or across
    environment steps when ``episode_lengths`` is given.

    For each agent, runs are smoothed using a rolling average, and the mean trajectory
    along with shaded +/- 1 standard deviation confidence bands are plotted.

    Args:
        agent_curves: Mapping of agent name to a list of runs, where each run is a list of
            episode reward floats.
        window: Rolling-average window size (must be >= 1; default: 10).
        save_path: File path to save the generated plot PNG. If None, the plot is not saved.
        show: Whether to display the plot interactively.
        title: Figure title.
        target_score: Optional reference score threshold (e.g. 475.0 for CartPole-v1 solved criterion).
        episode_lengths: Optional mapping like ``agent_curves`` holding each episode's step count.
            When given, the x-axis is cumulative environment steps: each run's smoothed curve is
            interpolated onto a common step grid up to the shortest run's total steps, so runs with
            different episode counts (step budgets) are compared at equal training effort.

    Returns:
        Absolute path to the saved PNG, or None if save_path is not set.

    Raises:
        ValueError: If agent_curves is empty or contains empty run data.
        ValueError: If window is less than 1.
    """
    _check_benchmark_curves(agent_curves, window)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title, fontsize=14)

    for idx, (name, runs) in enumerate(agent_curves.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        lengths = None if episode_lengths is None else episode_lengths[name]
        try:
            episodes, mean_curve, std_curve = aligned_curves(runs, window, lengths)
        except EpisodeLengthsMismatchError as err:
            raise ValueError(f"episode_lengths for agent '{name}' do not match its runs.") from err

        final_mean = float(mean_curve[-1])
        final_std = float(std_curve[-1]) if len(runs) > 1 else 0.0
        label = (
            f"{name} ({final_mean:.1f} ± {final_std:.1f})"
            if len(runs) > 1
            else f"{name} ({final_mean:.1f})"
        )

        ax.plot(episodes, mean_curve, label=label, color=color, linewidth=2)
        if len(runs) > 1:
            ax.fill_between(
                episodes,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.15,
            )

    if target_score is not None:
        ax.axhline(
            target_score,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label=f"Solved Threshold ({target_score:.0f})",
        )

    ax.set_xlabel("Environment steps" if episode_lengths is not None else "Episode", fontsize=11)
    ax.set_ylabel(f"Reward (Rolling Avg w={window})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    saved_path: str | None = None
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        saved_path = os.path.abspath(save_path)
        fig.savefig(saved_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path
