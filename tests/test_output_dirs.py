"""Tests for per-agent default output directories in train.py and record.py."""

import os
import sys

import pytest
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.agents import ReinforceAgent
from tests.helpers import minimal_config, run_main


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """Run each script from tmp_path so relative default directories land there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)


def _config_saving_at_episode_2(tmp_path, keep_checkpoint_dir=True):
    """Minimal config that saves a checkpoint at episode 2; optionally without checkpoint_dir."""
    config_path = minimal_config(tmp_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["training"]["save_frequency"] = 2
    if not keep_checkpoint_dir:
        del config["training"]["checkpoint_dir"]
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_checkpoints_saved_in_agent_subfolder(tmp_path, monkeypatch):
    run_main(["--config", _config_saving_at_episode_2(tmp_path)], monkeypatch)

    assert os.listdir(tmp_path / "checkpoints") == ["random"]
    assert "agent_episode_2.pt" in os.listdir(tmp_path / "checkpoints" / "random")


def test_checkpoint_subfolder_follows_agent_type_override(tmp_path, monkeypatch):
    config_path = _config_saving_at_episode_2(tmp_path)

    run_main(["--config", config_path, "--agent-type", "reinforce"], monkeypatch)

    assert os.listdir(tmp_path / "checkpoints") == ["reinforce"]


def test_checkpoint_subfolder_without_configured_checkpoint_dir(tmp_path, monkeypatch):
    """Without training.checkpoint_dir, checkpoints go to ./checkpoints/<agent_type>/."""
    config_path = _config_saving_at_episode_2(tmp_path, keep_checkpoint_dir=False)

    run_main(["--config", config_path], monkeypatch)

    assert "agent_episode_2.pt" in os.listdir(tmp_path / "checkpoints" / "random")


def test_train_defaults_videos_and_plots_to_agent_subfolders(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")

    run_main(["--config", minimal_config(tmp_path), "--record-video", "--plot"], monkeypatch)

    assert os.listdir(tmp_path / "videos") == ["random"]
    assert sorted(os.listdir(tmp_path / "videos" / "random")) == [
        "cartpole-training-episode-2.mp4",
        "cartpole-training-episode-4.mp4",
    ]
    assert os.listdir(tmp_path / "plots" / "random") == ["training_metrics.png"]


def test_train_uses_explicit_dirs_as_given(tmp_path, monkeypatch):
    """An explicit --video-dir / --plot-dir gets no agent subfolder appended."""
    pytest.importorskip("moviepy")

    run_main(
        ["--config", minimal_config(tmp_path), "--record-video", "--plot"]
        + ["--video-dir", "my_videos", "--plot-dir", "my_plots"],
        monkeypatch,
    )

    assert "cartpole-training-episode-2.mp4" in os.listdir(tmp_path / "my_videos")
    assert os.listdir(tmp_path / "my_plots") == ["training_metrics.png"]
    assert not (tmp_path / "videos").exists()
    assert not (tmp_path / "plots").exists()


def test_record_defaults_video_to_agent_subfolder(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    checkpoint = str(tmp_path / "agent.pt")
    ReinforceAgent(observation_dim=4, action_dim=2, config={}).save(checkpoint)

    run_main(
        ["--config", minimal_config(tmp_path), "--agent-type", "reinforce"]
        + ["--checkpoint", checkpoint, "--episodes", "1"],
        monkeypatch,
        module="record",
    )

    assert os.listdir(tmp_path / "videos" / "reinforce") == ["cartpole-playback-full.mp4"]


def test_checkpoint_subfolder_with_dqn(tmp_path, monkeypatch):
    """Verify DQN checkpoints land in checkpoints/dqn/ subfolder."""
    config_path = _config_saving_at_episode_2(tmp_path)

    run_main(["--config", config_path, "--agent-type", "dqn"], monkeypatch)

    assert os.listdir(tmp_path / "checkpoints") == ["dqn"]
    assert "agent_episode_2.pt" in os.listdir(tmp_path / "checkpoints" / "dqn")
