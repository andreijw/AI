"""Tests for record.py: recording a trained agent from a checkpoint into one video."""

import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.agents import ReinforceAgent
from tests.helpers import minimal_config, run_main


@pytest.fixture(autouse=True)
def _clear_sdl(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)


def _checkpoint(tmp_path):
    """Save an untrained REINFORCE agent and return the checkpoint path."""
    path = str(tmp_path / "agent.pt")
    ReinforceAgent(observation_dim=4, action_dim=2, config={}).save(path)
    return path


def _record(args, tmp_path, monkeypatch):
    """Run record.py with a minimal REINFORCE config; return the sorted files in the video dir."""
    video_dir = tmp_path / "videos"
    run_main(
        ["--config", minimal_config(tmp_path), "--agent-type", "reinforce"]
        + ["--video-dir", str(video_dir)]
        + args,
        monkeypatch,
        module="record",
    )
    return sorted(os.listdir(video_dir))


def test_record_writes_single_playback_video(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")

    names = _record(
        ["--checkpoint", _checkpoint(tmp_path), "--episodes", "2"], tmp_path, monkeypatch
    )

    assert names == ["cartpole-playback-full.mp4"]


def test_record_loads_checkpoint_and_records_each_episode(tmp_path, monkeypatch):
    """The checkpoint is loaded into the agent and every requested episode is merged."""
    pytest.importorskip("moviepy")
    import rl_cartpole.utils.video as video_mod

    loaded = []
    original_load = ReinforceAgent.load

    def spy_load(self, path):
        loaded.append(path)
        original_load(self, path)

    merged = []
    original_merge = video_mod.merge_episode_clips

    def spy_merge(video_dir, name_prefix):
        merged.append(sorted(os.listdir(video_dir)))
        return original_merge(video_dir, name_prefix=name_prefix)

    monkeypatch.setattr(ReinforceAgent, "load", spy_load)
    monkeypatch.setattr(video_mod, "merge_episode_clips", spy_merge)
    checkpoint = _checkpoint(tmp_path)

    _record(["--checkpoint", checkpoint, "--episodes", "3"], tmp_path, monkeypatch)

    assert loaded == [checkpoint]
    assert len(merged) == 1
    assert len(merged[0]) == 3
    assert all(name.startswith("cartpole-playback-episode-") for name in merged[0])


def test_record_missing_checkpoint_raises(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    with pytest.raises(FileNotFoundError):
        _record(["--checkpoint", str(tmp_path / "missing.pt")], tmp_path, monkeypatch)


def test_record_requires_checkpoint(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _record([], tmp_path, monkeypatch)
    assert "--checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-2"])
def test_record_episodes_must_be_positive(tmp_path, monkeypatch, capsys, value):
    with pytest.raises(SystemExit):
        _record(["--checkpoint", "x.pt", "--episodes", value], tmp_path, monkeypatch)
    assert "--episodes must be a positive integer" in capsys.readouterr().err
