"""Tests for headless video recording support in train.py."""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.environments import CartPoleEnv


# ---------------------------------------------------------------------------
# CartPoleEnv.wrap_env
# ---------------------------------------------------------------------------


def test_wrap_env_replaces_inner_env():
    """wrap_env() should replace the inner gym env with the provided wrapper."""
    env = CartPoleEnv(seed=42)
    original_inner = env.env

    mock_wrapper = MagicMock()
    env.wrap_env(mock_wrapper)

    assert env.env is mock_wrapper
    assert env.env is not original_inner
    env.close()


def test_wrap_env_with_record_video(tmp_path):
    """wrap_env() should work correctly with the RecordVideo gymnasium wrapper."""
    from gymnasium.wrappers import RecordVideo

    env = CartPoleEnv(render_mode="rgb_array", seed=42)
    wrapper = RecordVideo(
        env.env,
        video_folder=str(tmp_path),
        episode_trigger=lambda ep: False,  # never record, just test wrapping
        disable_logger=True,
    )
    env.wrap_env(wrapper)
    assert env.env is wrapper
    env.close()


# ---------------------------------------------------------------------------
# train.py CLI behaviour
# ---------------------------------------------------------------------------


def _run_main_with_args(args, monkeypatch):
    """Import train.main() fresh and run it with the given sys.argv."""
    monkeypatch.setattr(sys, "argv", ["train.py"] + args)
    # Force re-import so the module-level code runs with the patched argv.
    if "train" in sys.modules:
        del sys.modules["train"]
    import train  # noqa: PLC0415

    train.main()


def _minimal_config(tmp_path):
    """Write a minimal YAML config to tmp_path and return its path."""
    import yaml

    config = {
        "environment": {
            "name": "CartPole-v1",
            "render_mode": None,
            "max_episode_steps": 20,
            "seed": 42,
        },
        "agent": {
            "type": "random",
            "config": {"learning_rate": 0.0003, "gamma": 0.99},
        },
        "training": {
            "num_episodes": 4,
            "max_steps_per_episode": 20,
            "eval_frequency": 2,
            "save_frequency": 10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
        "logging": {"level": "INFO", "log_metrics": True},
    }
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_mutual_exclusion_render_and_record_video(tmp_path, monkeypatch):
    """Passing both --render and --record-video should raise SystemExit."""
    config_path = _minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        _run_main_with_args(
            ["--render", "--record-video", "--config", config_path], monkeypatch
        )


def test_record_video_missing_moviepy_raises_import_error(tmp_path, monkeypatch):
    """--record-video should raise ImportError with install instructions when moviepy is absent."""
    config_path = _minimal_config(tmp_path)

    # Make moviepy unimportable
    with patch.dict(sys.modules, {"moviepy": None}):
        with pytest.raises(ImportError, match="moviepy"):
            _run_main_with_args(
                ["--record-video", "--config", config_path], monkeypatch
            )


def test_sdl_env_vars_set_for_headless(tmp_path, monkeypatch):
    """--record-video must set SDL_VIDEODRIVER=offscreen before CartPoleEnv creation."""
    config_path = _minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    # Clear any existing SDL vars so setdefault() picks them up
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    captured = {}

    class _StopAfterCapture(Exception):
        pass

    def patched_init(self, *a, **kw):
        # Capture the env vars at the moment CartPoleEnv is initialised
        captured["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER")
        captured["SDL_AUDIODRIVER"] = os.environ.get("SDL_AUDIODRIVER")
        raise _StopAfterCapture()

    import rl_cartpole.environments.cartpole_env as _ce_mod

    monkeypatch.setattr(_ce_mod.CartPoleEnv, "__init__", patched_init)

    # moviepy must appear importable so we get past the import check
    fake_moviepy = types.ModuleType("moviepy")

    with patch.dict(sys.modules, {"moviepy": fake_moviepy}):
        with pytest.raises(_StopAfterCapture):
            _run_main_with_args(
                ["--record-video", "--video-dir", video_dir, "--config", config_path],
                monkeypatch,
            )

    assert captured.get("SDL_VIDEODRIVER") == "offscreen", (
        "SDL_VIDEODRIVER was not set to 'offscreen' before CartPoleEnv creation"
    )
    assert captured.get("SDL_AUDIODRIVER") == "dummy", (
        "SDL_AUDIODRIVER was not set to 'dummy' before CartPoleEnv creation"
    )


def test_record_video_creates_videos(tmp_path, monkeypatch):
    """--record-video should produce MP4 files in the specified directory."""
    pytest.importorskip("moviepy")

    config_path = _minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    _run_main_with_args(
        ["--record-video", "--video-dir", video_dir, "--config", config_path],
        monkeypatch,
    )

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    assert len(mp4_files) > 0, "Expected at least one MP4 file to be recorded"


def test_episode_trigger_skips_episode_zero(tmp_path, monkeypatch):
    """episode_trigger must not record episode 0 (untrained initial state)."""
    pytest.importorskip("moviepy")

    config_path = _minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    captured_triggers = []
    original_RecordVideo = None

    from gymnasium.wrappers import RecordVideo as _RV

    original_RecordVideo = _RV

    def patching_RecordVideo(env, video_folder, episode_trigger, **kwargs):
        captured_triggers.append(episode_trigger)
        return original_RecordVideo(
            env, video_folder=video_folder, episode_trigger=episode_trigger, **kwargs
        )

    with patch("gymnasium.wrappers.RecordVideo", side_effect=patching_RecordVideo):
        _run_main_with_args(
            ["--record-video", "--video-dir", video_dir, "--config", config_path],
            monkeypatch,
        )

    assert captured_triggers, "RecordVideo was not constructed"
    trigger = captured_triggers[0]
    # Episode 0 should NOT be recorded
    assert trigger(0) is False
    # Episode equal to eval_frequency (2 in the test config) SHOULD be recorded
    assert trigger(2) is True
