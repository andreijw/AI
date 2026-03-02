"""Tests for the visualization utilities."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from rl_cartpole.utils.visualization import plot_training_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_data(n: int = 50):
    """Return simple deterministic reward/length lists."""
    import numpy as np

    rng = np.random.default_rng(0)
    rewards = rng.uniform(1, 100, size=n).tolist()
    lengths = rng.integers(1, 200, size=n).tolist()
    return rewards, lengths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plot_saves_png(tmp_path):
    """plot_training_metrics should write a PNG file to save_dir."""
    rewards, lengths = _sample_data()
    result = plot_training_metrics(rewards, lengths, save_dir=str(tmp_path))

    assert result is not None
    assert os.path.isfile(result)
    assert result.endswith(".png")


def test_plot_returns_none_without_save_dir():
    """plot_training_metrics returns None when save_dir is not provided."""
    rewards, lengths = _sample_data()
    result = plot_training_metrics(rewards, lengths, save_dir=None)
    assert result is None


def test_plot_creates_save_dir(tmp_path):
    """plot_training_metrics should create the save_dir if it does not exist."""
    target = str(tmp_path / "new_dir" / "nested")
    assert not os.path.exists(target)

    rewards, lengths = _sample_data()
    plot_training_metrics(rewards, lengths, save_dir=target)

    assert os.path.isdir(target)


def test_plot_single_episode():
    """plot_training_metrics should work with a single data point."""
    result = plot_training_metrics([10.0], [5], save_dir=None)
    assert result is None  # no crash, nothing saved


def test_plot_window_larger_than_data(tmp_path):
    """Rolling window larger than the data should not raise an error."""
    rewards, lengths = _sample_data(n=5)
    result = plot_training_metrics(rewards, lengths, window=100, save_dir=str(tmp_path))
    assert result is not None
    assert os.path.isfile(result)


def test_plot_filename(tmp_path):
    """The saved file should be named 'training_metrics.png'."""
    rewards, lengths = _sample_data()
    result = plot_training_metrics(rewards, lengths, save_dir=str(tmp_path))
    assert os.path.basename(result) == "training_metrics.png"


def test_plot_custom_title(tmp_path):
    """Custom title parameter should not raise an error."""
    rewards, lengths = _sample_data()
    result = plot_training_metrics(
        rewards, lengths, save_dir=str(tmp_path), title="Custom Agent – Metrics"
    )
    assert result is not None
    assert os.path.isfile(result)
