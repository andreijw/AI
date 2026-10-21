"""Tests for training video recording and clip merging (rl_core.utils.video)."""

import gc
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.environments import CartPoleEnv
from rl_core.utils.video import TrainingVideoRecorder, merge_episode_clips

PREFIX = "cartpole-training"

pytest.importorskip("moviepy")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _headless_sdl(monkeypatch):
    """Render CartPole frames without a display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


@pytest.fixture
def unraisable(monkeypatch):
    """Collect exceptions raised in ``__del__`` during the test instead of letting pytest warn."""
    caught = []
    monkeypatch.setattr(sys, "unraisablehook", caught.append)
    return caught


def _play_episode(env):
    """Run one episode to termination or truncation with a fixed action."""
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated


def _make_recording_env(video_dir, progress, every):
    """A recording env; video_dir should not exist yet, or RecordVideo warns it overwrites."""
    env = CartPoleEnv(render_mode="rgb_array", seed=42, max_episode_steps=5)
    env.wrap_env(
        TrainingVideoRecorder(env.env, video_folder=str(video_dir), progress=progress, every=every)
    )
    return env


def _write_clip(path, color, num_frames, fps=10):
    """Write a solid-colour MP4 clip with num_frames frames."""
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[..., color] = 255
    ImageSequenceClip([frame] * num_frames, fps=fps).write_videofile(str(path), logger=None)


def _read_frames(path):
    """Return (frame_count, first_frame) for an MP4 file."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    width, height = meta["size"]
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3) for f in reader]
    return len(frames), frames[0]


def _mp4_names(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".mp4"))


# ---------------------------------------------------------------------------
# TrainingVideoRecorder
# ---------------------------------------------------------------------------


def test_recorder_records_every_nth_training_episode_and_skips_evaluation(tmp_path):
    """Only every Nth training episode is recorded, named by its training episode number."""
    progress = SimpleNamespace(current_episode=0, evaluating=False)
    env = _make_recording_env(tmp_path / "videos", progress, every=2)

    _play_episode(env)  # before training starts (current_episode == 0)
    for episode in range(1, 5):
        progress.current_episode = episode
        _play_episode(env)
        if episode == 2:
            progress.evaluating = True
            for _ in range(3):
                _play_episode(env)
            progress.evaluating = False
    env.close()

    assert _mp4_names(tmp_path / "videos") == [
        "cartpole-training-episode-2.mp4",
        "cartpole-training-episode-4.mp4",
    ]


def test_recorder_every_episode_ignores_evaluation_episodes(tmp_path):
    """every=1 records each training episode exactly once and no evaluation episode."""
    progress = SimpleNamespace(current_episode=0, evaluating=False)
    env = _make_recording_env(tmp_path / "videos", progress, every=1)

    for episode in range(1, 4):
        progress.current_episode = episode
        _play_episode(env)
        progress.evaluating = True
        _play_episode(env)
        progress.evaluating = False
    env.close()

    assert _mp4_names(tmp_path / "videos") == [
        f"cartpole-training-episode-{n}.mp4" for n in (1, 2, 3)
    ]


def test_recorder_accepts_progress_that_cannot_be_copied(tmp_path):
    """RecordVideo deep-copies its constructor args; the trainer (holding locks) must not be."""
    import threading

    progress = SimpleNamespace(current_episode=1, evaluating=False, lock=threading.RLock())
    env = _make_recording_env(tmp_path / "videos", progress, every=1)
    _play_episode(env)
    env.close()

    assert _mp4_names(tmp_path / "videos") == ["cartpole-training-episode-1.mp4"]


@pytest.mark.parametrize(
    "render_mode, every, match",
    [
        ("rgb_array", 0, "every"),
        ("rgb_array", -1, "every"),
        # Rejected by RecordVideo.__init__ itself, after part of its state is set.
        (None, 1, "render_mode"),
    ],
)
def test_rejected_recorder_raises_without_unraisable_error(
    tmp_path, unraisable, render_mode, every, match
):
    env = CartPoleEnv(render_mode=render_mode, seed=42)
    progress = SimpleNamespace(current_episode=0, evaluating=False)
    # Do not bind the exception (``as excinfo``): its traceback would keep the half-built
    # recorder alive past the assertion below, so the test could no longer fail.
    with pytest.raises(ValueError, match=match):
        TrainingVideoRecorder(env.env, video_folder=str(tmp_path), progress=progress, every=every)
    gc.collect()
    env.close()
    assert unraisable == []


def test_recorder_dropped_mid_recording_still_warns_about_unsaved_video(tmp_path):
    env = CartPoleEnv(render_mode="rgb_array", seed=42)
    progress = SimpleNamespace(current_episode=1, evaluating=False)
    recorder = TrainingVideoRecorder(
        env.env, video_folder=str(tmp_path / "videos"), progress=progress, every=1
    )
    recorder.reset()
    recorder.step(0)
    with pytest.warns(UserWarning, match="Unable to save last video"):
        del recorder
        gc.collect()
    env.close()


def test_recorder_collected_quietly_when_record_video_has_no_del(tmp_path, monkeypatch, unraisable):
    monkeypatch.delattr(RecordVideo, "__del__")
    env = CartPoleEnv(render_mode="rgb_array", seed=42)
    progress = SimpleNamespace(current_episode=0, evaluating=False)
    recorder = TrainingVideoRecorder(
        env.env, video_folder=str(tmp_path / "videos"), progress=progress, every=1
    )
    del recorder
    gc.collect()
    env.close()
    assert unraisable == []


# ---------------------------------------------------------------------------
# merge_episode_clips
# ---------------------------------------------------------------------------


def test_merge_concatenates_clips_in_episode_order_and_removes_them(tmp_path):
    """Clips are joined in numeric episode order (2 before 10) and then deleted."""
    _write_clip(tmp_path / "cartpole-training-episode-10.mp4", color=2, num_frames=4)  # blue
    _write_clip(tmp_path / "cartpole-training-episode-2.mp4", color=0, num_frames=3)  # red

    merged = merge_episode_clips(str(tmp_path), name_prefix=PREFIX)

    assert merged == str(tmp_path / "cartpole-training-full.mp4")
    assert _mp4_names(tmp_path) == ["cartpole-training-full.mp4"]
    frame_count, first_frame = _read_frames(merged)
    assert frame_count == 7
    red, _, blue = first_frame.reshape(-1, 3).mean(axis=0)
    assert red > 200 and blue < 50, "episode-2 (red) should come first"


def test_merge_single_clip(tmp_path):
    _write_clip(tmp_path / "cartpole-training-episode-1.mp4", color=1, num_frames=5)

    merged = merge_episode_clips(str(tmp_path), name_prefix=PREFIX)

    assert _mp4_names(tmp_path) == ["cartpole-training-full.mp4"]
    assert _read_frames(merged)[0] == 5


def test_merge_ignores_unrelated_files(tmp_path):
    """Files that are not episode clips with the given prefix are left untouched."""
    _write_clip(tmp_path / "cartpole-training-episode-1.mp4", color=0, num_frames=2)
    _write_clip(tmp_path / "other-episode-1.mp4", color=1, num_frames=2)

    merge_episode_clips(str(tmp_path), name_prefix=PREFIX)

    assert _mp4_names(tmp_path) == ["cartpole-training-full.mp4", "other-episode-1.mp4"]


def test_merge_without_clips_warns_and_returns_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert merge_episode_clips(str(tmp_path), name_prefix=PREFIX) is None

    assert _mp4_names(tmp_path) == []
    assert "No episode clips" in caplog.text


def test_merge_failure_raises_and_keeps_clips(tmp_path, monkeypatch):
    """If ffmpeg fails, a RuntimeError carries its error output and the clips are kept."""
    import subprocess

    _write_clip(tmp_path / "cartpole-training-episode-1.mp4", color=0, num_frames=2)

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"concat: invalid data")

    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="concat: invalid data"):
        merge_episode_clips(str(tmp_path), name_prefix=PREFIX)

    assert _mp4_names(tmp_path) == ["cartpole-training-episode-1.mp4"]
    assert [f for f in os.listdir(tmp_path) if f.endswith(".txt")] == []
