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

__all__ = ["plot_training_metrics", "plot_benchmark_comparison"]

_PLOT_FILENAME = "training_metrics.png"


def _rolling_average(data: list[float], window: int) -> list[float]:
    """
    Compute a trailing rolling average.

    Args:
        data: Sequence of float values.
        window: Look-back window size (must be >= 1).

    Returns:
        List of rolling-average values the same length as *data*.
    """
    result: list[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start : i + 1])))
    return result


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
    rewards_avg = _rolling_average(episode_rewards, window)
    lengths_avg = _rolling_average([float(x) for x in episode_lengths], window)

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


def plot_benchmark_comparison(
    agent_curves: dict[str, list[list[float]]],
    window: int = 10,
    save_path: str | None = None,
    show: bool = False,
    title: str = "CartPole Multi-Agent Benchmark Comparison",
    target_score: float | None = 475.0,
) -> str | None:
    """
    Plot comparative reward trajectories for multiple agents across training episodes.

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

    Returns:
        Absolute path to the saved PNG, or None if save_path is not set.

    Raises:
        ValueError: If agent_curves is empty or contains empty run data.
        ValueError: If window is less than 1.
    """
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

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title, fontsize=14)

    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]

    for idx, (name, runs) in enumerate(agent_curves.items()):
        color = palette[idx % len(palette)]
        smoothed_runs = [_rolling_average(run, window) for run in runs]
        min_len = min(len(r) for r in smoothed_runs)
        aligned = np.array([r[:min_len] for r in smoothed_runs], dtype=np.float64)

        mean_curve = np.mean(aligned, axis=0)
        std_curve = np.std(aligned, axis=0)
        episodes = np.arange(1, min_len + 1)

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

    ax.set_xlabel("Episode", fontsize=11)
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
