"""Visualization utilities for training metrics."""

import os
from typing import List, Optional

import numpy as np


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

    Args:
        episode_rewards: List of total reward per episode.
        episode_lengths: List of step count per episode.
        window: Rolling-average window size (default: 10).
        save_dir: Directory to save the plot PNG.  If ``None`` the plot is
            not saved to disk.
        show: Whether to call ``plt.show()`` (requires a display).
        title: Title for the figure (default: CartPole Random Agent – Training Metrics).

    Returns:
        Absolute path to the saved PNG, or ``None`` if ``save_dir`` is not set.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe on headless machines
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "The 'matplotlib' package is required for visualization. "
            "Install it with: pip install matplotlib  "
            "or: pip install -e '.[viz]'"
        ) from exc

    episodes = list(range(1, len(episode_rewards) + 1))

    def _rolling_avg(data: List[float], w: int) -> List[float]:
        """Compute a trailing rolling average."""
        result = []
        for i, _ in enumerate(data):
            start = max(0, i - w + 1)
            result.append(float(np.mean(data[start : i + 1])))
        return result

    rewards_avg = _rolling_avg(episode_rewards, window)
    lengths_avg = _rolling_avg([float(x) for x in episode_lengths], window)

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
        saved_path = os.path.abspath(os.path.join(save_dir, "training_metrics.png"))
        fig.savefig(saved_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path
