"""Visualization utilities for training metrics."""

import os
from typing import List, Optional

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

__all__ = ["plot_training_metrics"]

_PLOT_FILENAME = "training_metrics.png"


def _rolling_average(data: List[float], window: int) -> List[float]:
    """
    Compute a trailing rolling average.

    Args:
        data: Sequence of float values.
        window: Look-back window size (must be >= 1).

    Returns:
        List of rolling-average values the same length as *data*.
    """
    result: List[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start : i + 1])))
    return result


def plot_training_metrics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    window: int = 10,
    save_dir: Optional[str] = None,
    show: bool = False,
    title: str = "CartPole Random Agent – Training Metrics",
) -> Optional[str]:
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

    saved_path: Optional[str] = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.abspath(os.path.join(save_dir, _PLOT_FILENAME))
        fig.savefig(saved_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path
