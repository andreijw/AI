"""Tests for the visualization utilities."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_core.utils import visualization
from rl_core.utils.visualization import (
    plot_benchmark_comparison,
    plot_training_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data():
    """Return deterministic (rewards, lengths) lists of length 50."""
    rng = np.random.default_rng(0)
    rewards = rng.uniform(1, 100, size=50).tolist()
    lengths = rng.integers(1, 200, size=50).tolist()
    return rewards, lengths


# ---------------------------------------------------------------------------
# plot_training_metrics – input validation
# ---------------------------------------------------------------------------


def test_plot_raises_on_empty_rewards():
    """Empty episode_rewards should raise ValueError."""
    with pytest.raises(ValueError, match="episode_rewards must not be empty"):
        plot_training_metrics([], [1, 2])


def test_plot_raises_on_empty_lengths():
    """Empty episode_lengths should raise ValueError."""
    with pytest.raises(ValueError, match="episode_lengths must not be empty"):
        plot_training_metrics([1.0, 2.0], [])


def test_plot_raises_on_mismatched_lengths():
    """Mismatched list lengths should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        plot_training_metrics([1.0, 2.0], [1])


def test_plot_raises_on_invalid_window():
    """window < 1 should raise ValueError."""
    with pytest.raises(ValueError, match="window must be"):
        plot_training_metrics([1.0], [1], window=0)


# ---------------------------------------------------------------------------
# plot_training_metrics – normal behaviour
# ---------------------------------------------------------------------------


def test_plot_saves_png(tmp_path, sample_data):
    """plot_training_metrics should write a PNG file to save_dir."""
    rewards, lengths = sample_data
    result = plot_training_metrics(rewards, lengths, save_dir=str(tmp_path))

    assert result is not None
    assert os.path.isfile(result)
    assert result.endswith(".png")


def test_plot_returns_none_without_save_dir(sample_data):
    """plot_training_metrics returns None when save_dir is not provided."""
    rewards, lengths = sample_data
    result = plot_training_metrics(rewards, lengths, save_dir=None)
    assert result is None


def test_plot_creates_save_dir(tmp_path, sample_data):
    """plot_training_metrics should create the save_dir if it does not exist."""
    target = str(tmp_path / "new_dir" / "nested")
    assert not os.path.exists(target)

    rewards, lengths = sample_data
    plot_training_metrics(rewards, lengths, save_dir=target)

    assert os.path.isdir(target)


def test_plot_single_episode():
    """plot_training_metrics should work with a single data point."""
    result = plot_training_metrics([10.0], [5], save_dir=None)
    assert result is None  # no crash, nothing saved


def test_plot_window_larger_than_data(tmp_path, sample_data):
    """Rolling window larger than the data should not raise an error."""
    rewards = sample_data[0][:5]
    lengths = sample_data[1][:5]
    result = plot_training_metrics(rewards, lengths, window=100, save_dir=str(tmp_path))
    assert result is not None
    assert os.path.isfile(result)


def test_plot_filename(tmp_path, sample_data):
    """The saved file should be named 'training_metrics.png'."""
    rewards, lengths = sample_data
    result = plot_training_metrics(rewards, lengths, save_dir=str(tmp_path))
    assert os.path.basename(result) == "training_metrics.png"


def test_plot_custom_title(tmp_path, sample_data):
    """Custom title parameter should not raise an error."""
    rewards, lengths = sample_data
    result = plot_training_metrics(
        rewards, lengths, save_dir=str(tmp_path), title="Custom Agent – Metrics"
    )
    assert result is not None
    assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# plot_benchmark_comparison unit tests
# ---------------------------------------------------------------------------


def test_plot_benchmark_empty_curves():
    """Empty agent_curves dictionary should raise ValueError."""
    with pytest.raises(ValueError, match="agent_curves must not be empty"):
        plot_benchmark_comparison({})


def test_plot_benchmark_invalid_window():
    """Window less than 1 should raise ValueError."""
    with pytest.raises(ValueError, match="window must be >= 1"):
        plot_benchmark_comparison({"PPO": [[10.0, 20.0]]}, window=0)


def test_plot_benchmark_empty_runs():
    """Agent with empty list of runs or empty run should raise ValueError."""
    with pytest.raises(ValueError, match="Runs for agent 'PPO' must not be empty"):
        plot_benchmark_comparison({"PPO": []})

    with pytest.raises(ValueError, match="Run 0 for agent 'PPO' must not be empty"):
        plot_benchmark_comparison({"PPO": [[]]})


def test_plot_benchmark_returns_none_without_save_path():
    """Returns None when save_path is not specified."""
    curves = {
        "Random": [[10.0, 15.0, 12.0]],
        "DQN": [[20.0, 50.0, 100.0]],
    }
    result = plot_benchmark_comparison(curves, save_path=None)
    assert result is None


def test_plot_benchmark_saves_png(tmp_path):
    """Saves plot image to the requested path."""
    curves = {
        "PPO": [[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]],
        "DQN": [[5.0, 10.0, 25.0, 50.0]],
    }
    target = str(tmp_path / "plots" / "benchmark_test.png")
    result = plot_benchmark_comparison(curves, window=2, save_path=target, target_score=40.0)
    assert result == os.path.abspath(target)
    assert os.path.isfile(target)


def test_plot_interactive_show(monkeypatch, sample_data):
    """Calling with show=True invokes plt.show."""
    import matplotlib.pyplot as plt

    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

    rewards, lengths = sample_data
    plot_training_metrics(rewards, lengths, show=True)
    plot_benchmark_comparison({"PPO": [rewards]}, show=True)

    assert len(show_called) == 2


def test_plot_benchmark_unequal_run_lengths(tmp_path):
    """Unequal run lengths across seeds or agents are aligned to minimum length."""
    curves = {
        "A2C": [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0, 40.0, 50.0]],
        "REINFORCE": [[5.0, 15.0, 25.0, 35.0]],
    }
    target = str(tmp_path / "unequal.png")
    result = plot_benchmark_comparison(curves, save_path=target, target_score=None)
    assert result is not None
    assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# aligned_curves: the smoothing and alignment the comparison plot draws
# ---------------------------------------------------------------------------


def test_aligned_curves_by_episode_cut_every_run_to_the_shortest():
    """Window 2 smooths [10, 20, 30] to [10, 15, 25] and [30, 10] to [30, 20]; both keep 2 episodes."""
    x, mean, std = visualization.aligned_curves([[10.0, 20.0, 30.0], [30.0, 10.0]], window=2)
    np.testing.assert_array_equal(x, [1.0, 2.0])
    np.testing.assert_allclose(mean, [20.0, 17.5])
    np.testing.assert_allclose(std, [10.0, 2.5])


def test_aligned_curves_by_steps_share_a_grid_where_every_run_has_data():
    """Steps run from the latest first episode end to the earliest last one, on 200 points, each run
    interpolated between its episode ends."""
    runs = [[0.0, 10.0], [4.0, 4.0, 4.0]]
    lengths = [[10, 10], [5, 5, 30]]  # episode ends at 10, 20 and at 5, 10, 40
    x, mean, std = visualization.aligned_curves(runs, window=1, episode_lengths=lengths)
    assert len(x) == 200
    assert (x[0], x[-1]) == (10.0, 20.0)
    # At step 10 the first run is 0 and the second 4; at step 20, 10 and 4 + 0 (flat at 4).
    np.testing.assert_allclose([mean[0], mean[-1]], [2.0, 7.0])
    np.testing.assert_allclose([std[0], std[-1]], [2.0, 3.0])


def test_aligned_curves_of_one_run_have_no_spread():
    x, mean, std = visualization.aligned_curves([[1.0, 3.0]], window=1)
    np.testing.assert_array_equal(mean, [1.0, 3.0])
    np.testing.assert_array_equal(std, [0.0, 0.0])


def test_aligned_curves_reject_episode_lengths_that_do_not_match_the_runs():
    with pytest.raises(visualization.EpisodeLengthsMismatchError, match="do not match"):
        visualization.aligned_curves([[1.0, 2.0]], window=1, episode_lengths=[[5]])


@pytest.mark.parametrize(
    ("runs", "window", "lengths"),
    [([[1.0]], 0, None), ([], 2, None), ([[1.0], []], 2, None), ([[1.0, 2.0]], 1, [[5, 0]])],
    ids=["window-0", "no-runs", "empty-run", "zero-length-episode"],
)
def test_aligned_curves_reject_inputs_that_would_give_nan_or_a_wrong_grid(runs, window, lengths):
    """Without these checks a zero window gives NaN curves and a zero-length episode a step grid
    that interpolation silently misreads."""
    with pytest.raises(ValueError) as caught:
        visualization.aligned_curves(runs, window=window, episode_lengths=lengths)
    assert not isinstance(caught.value, visualization.EpisodeLengthsMismatchError)


def test_the_comparison_plot_does_not_call_other_errors_a_length_mismatch():
    with pytest.raises(ValueError, match="must all be positive"):
        plot_benchmark_comparison({"PPO": [[1.0, 2.0]]}, episode_lengths={"PPO": [[5, 0]]})


def test_the_comparison_plot_reports_which_agent_has_mismatched_lengths():
    with pytest.raises(ValueError, match="episode_lengths for agent 'PPO' do not match its runs"):
        plot_benchmark_comparison({"PPO": [[1.0, 2.0]]}, episode_lengths={"PPO": [[5]]})


@pytest.mark.parametrize("by_steps", [False, True])
def test_the_comparison_plot_draws_exactly_the_aligned_curves(by_steps, monkeypatch):
    import matplotlib.pyplot as plt

    figures = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    runs = [[10.0, 20.0, 30.0, 40.0], [5.0, 25.0, 15.0, 35.0, 45.0]]
    lengths = [[10, 20, 30, 40], [5, 5, 50, 40, 10]] if by_steps else None
    plot_benchmark_comparison(
        {"PPO": runs},
        window=2,
        target_score=None,
        episode_lengths={"PPO": lengths} if by_steps else None,
    )
    x, mean, std = visualization.aligned_curves(runs, window=2, episode_lengths=lengths)
    ax = figures[0].axes[0]
    (line,) = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), mean)
    assert line.get_label() == f"PPO ({mean[-1]:.1f} ± {std[-1]:.1f})"
    # The band spans one standard deviation either side of the mean.
    (band,) = ax.collections
    edges = band.get_paths()[0].vertices[:, 1]
    np.testing.assert_allclose([edges.min(), edges.max()], [(mean - std).min(), (mean + std).max()])
    plt.close(figures[0])
