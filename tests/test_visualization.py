"""Tests for the visualization utilities."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rl_cartpole.utils.visualization import _rolling_average, plot_training_metrics


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
# _rolling_average unit tests
# ---------------------------------------------------------------------------


def test_rolling_average_single_element():
    """Single-element list should return itself."""
    assert _rolling_average([5.0], 3) == pytest.approx([5.0])


def test_rolling_average_window_one():
    """Window of 1 should return the input unchanged."""
    data = [1.0, 2.0, 3.0]
    assert _rolling_average(data, 1) == pytest.approx(data)


def test_rolling_average_full_window():
    """Rolling average with window equal to list length."""
    data = [1.0, 2.0, 3.0, 4.0]
    result = _rolling_average(data, 4)
    assert result[-1] == pytest.approx(2.5)


def test_rolling_average_window_larger_than_data():
    """Window larger than data length should use all available values."""
    data = [2.0, 4.0]
    result = _rolling_average(data, 100)
    assert result[-1] == pytest.approx(3.0)


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
