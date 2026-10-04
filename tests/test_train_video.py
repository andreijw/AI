"""Tests for headless video recording support in train.py."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_cartpole.environments import CartPoleEnv
from tests.helpers import minimal_config, run_main

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
    pytest.importorskip("moviepy")
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


def test_mutual_exclusion_render_and_record_video(tmp_path, monkeypatch):
    """Passing both --render and --record-video should raise SystemExit."""
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--render", "--record-video", "--config", config_path], monkeypatch)


def test_num_episodes_override_requires_positive_int(tmp_path, monkeypatch):
    """--num-episodes must be a positive integer."""
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--num-episodes", "0", "--config", config_path], monkeypatch)


def test_agent_type_and_num_episodes_overrides_apply(tmp_path, monkeypatch):
    """CLI overrides should select the requested agent and training length."""
    config_path = minimal_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            config_path,
            "--agent-type",
            "reinforce",
            "--num-episodes",
            "1",
        ],
    )
    if "train" in sys.modules:
        del sys.modules["train"]
    import train  # noqa: PLC0415

    captured = {}

    class _StubReinforce:
        def __init__(self, observation_dim, action_dim, config):
            captured["agent_observation_dim"] = observation_dim
            captured["agent_action_dim"] = action_dim
            captured["agent_config"] = dict(config)

    class _StubTrainer:
        def __init__(self, env, agent, config, logger=None):
            captured["trainer_agent_type"] = type(agent).__name__
            captured["trainer_num_episodes"] = config.get("num_episodes")
            self.episode_rewards = []
            self.episode_lengths = []

        def train(self):
            return {"ok": True}

    monkeypatch.setitem(train.AGENT_CLASSES, "reinforce", _StubReinforce)
    monkeypatch.setattr(train, "Trainer", _StubTrainer)

    train.main()

    assert captured["trainer_agent_type"] == "_StubReinforce"
    assert captured["trainer_num_episodes"] == 1
    assert captured["agent_observation_dim"] == 4
    assert captured["agent_action_dim"] == 2


def test_record_video_missing_moviepy_raises_import_error(tmp_path, monkeypatch):
    """--record-video should raise ImportError with install instructions when moviepy is absent."""
    config_path = minimal_config(tmp_path)

    # Make moviepy unimportable
    with patch.dict(sys.modules, {"moviepy": None}), pytest.raises(ImportError, match="moviepy"):
        run_main(["--record-video", "--config", config_path], monkeypatch)


def test_sdl_env_vars_set_for_headless(tmp_path, monkeypatch):
    """--record-video must set SDL_VIDEODRIVER=offscreen before CartPoleEnv creation."""
    config_path = minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    # Clear any existing SDL vars so setdefault() picks them up
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    captured = {}

    class _StopAfterCaptureError(Exception):
        pass

    def patched_init(self, *a, **kw):
        # Capture the env vars at the moment CartPoleEnv is initialised
        captured["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER")
        captured["SDL_AUDIODRIVER"] = os.environ.get("SDL_AUDIODRIVER")
        raise _StopAfterCaptureError()

    import rl_cartpole.environments.cartpole_env as _ce_mod

    monkeypatch.setattr(_ce_mod.CartPoleEnv, "__init__", patched_init)

    # moviepy must appear importable so we get past the import check
    fake_moviepy = types.ModuleType("moviepy")

    with patch.dict(sys.modules, {"moviepy": fake_moviepy}), pytest.raises(_StopAfterCaptureError):
        run_main(
            ["--record-video", "--video-dir", video_dir, "--config", config_path],
            monkeypatch,
        )

    assert captured.get("SDL_VIDEODRIVER") == "offscreen", (
        "SDL_VIDEODRIVER was not set to 'offscreen' before CartPoleEnv creation"
    )
    assert captured.get("SDL_AUDIODRIVER") == "dummy", (
        "SDL_AUDIODRIVER was not set to 'dummy' before CartPoleEnv creation"
    )
    assert os.environ.get("SDL_VIDEODRIVER") is None
    assert os.environ.get("SDL_AUDIODRIVER") is None


def test_sdl_env_vars_restored_for_headless_when_preexisting(tmp_path, monkeypatch):
    """--record-video must restore pre-existing SDL env vars after temporary override."""
    config_path = minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    original_video_driver = "already-set-video"
    original_audio_driver = "already-set-audio"
    monkeypatch.setenv("SDL_VIDEODRIVER", original_video_driver)
    monkeypatch.setenv("SDL_AUDIODRIVER", original_audio_driver)

    captured = {}

    class _StopAfterCaptureError(Exception):
        pass

    def patched_init(self, *a, **kw):
        # Capture env vars at CartPoleEnv initialization time
        captured["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER")
        captured["SDL_AUDIODRIVER"] = os.environ.get("SDL_AUDIODRIVER")
        raise _StopAfterCaptureError()

    import rl_cartpole.environments.cartpole_env as _ce_mod

    monkeypatch.setattr(_ce_mod.CartPoleEnv, "__init__", patched_init)

    fake_moviepy = types.ModuleType("moviepy")

    with patch.dict(sys.modules, {"moviepy": fake_moviepy}), pytest.raises(_StopAfterCaptureError):
        run_main(
            ["--record-video", "--video-dir", video_dir, "--config", config_path],
            monkeypatch,
        )

    assert captured.get("SDL_VIDEODRIVER") == original_video_driver
    assert captured.get("SDL_AUDIODRIVER") == original_audio_driver
    assert os.environ.get("SDL_VIDEODRIVER") == original_video_driver
    assert os.environ.get("SDL_AUDIODRIVER") == original_audio_driver


def test_record_video_creates_videos(tmp_path, monkeypatch):
    """--record-video should produce MP4 files in the specified directory."""
    pytest.importorskip("moviepy")

    config_path = minimal_config(tmp_path)
    video_dir = str(tmp_path / "videos")

    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    run_main(
        ["--record-video", "--video-dir", video_dir, "--config", config_path],
        monkeypatch,
    )

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    assert len(mp4_files) > 0, "Expected at least one MP4 file to be recorded"


def _record_with_args(extra_args, tmp_path, monkeypatch):
    """Run train.py with --record-video plus extra_args; return the sorted MP4 names produced."""
    pytest.importorskip("moviepy")

    config_path = minimal_config(tmp_path)
    video_dir = tmp_path / "videos"

    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    run_main(
        ["--record-video", "--video-dir", str(video_dir), "--config", config_path] + extra_args,
        monkeypatch,
    )
    return sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))


def test_record_video_default_interval_matches_training_episodes(tmp_path, monkeypatch):
    """Default interval is eval_frequency (2): only training episodes 2 and 4 are recorded.

    Evaluation runs 10 episodes on the same env after episodes 2 and 4; they must neither be
    recorded nor shift the clip numbering.
    """
    assert _record_with_args([], tmp_path, monkeypatch) == [
        "cartpole-training-episode-2.mp4",
        "cartpole-training-episode-4.mp4",
    ]


def test_video_every_sets_recording_interval(tmp_path, monkeypatch):
    """--video-every 1 records each of the 4 training episodes."""
    assert _record_with_args(["--video-every", "1"], tmp_path, monkeypatch) == [
        f"cartpole-training-episode-{n}.mp4" for n in (1, 2, 3, 4)
    ]


def test_video_every_requires_record_video(tmp_path, monkeypatch, capsys):
    """--video-every without --record-video is a usage error."""
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--video-every", "5", "--config", config_path], monkeypatch)
    assert "--video-every requires --record-video" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3"])
def test_video_every_must_be_positive(tmp_path, monkeypatch, capsys, value):
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--record-video", "--video-every", value, "--config", config_path], monkeypatch)
    assert "--video-every must be a positive integer" in capsys.readouterr().err


def test_environment_config_is_forwarded_to_cartpole_env(tmp_path, monkeypatch):
    """train.py should forward environment config fields into CartPoleEnv."""
    import yaml

    config = {
        "environment": {
            "name": "CartPole-v1",
            "render_mode": None,
            "max_episode_steps": 20,
            "seed": 123,
            "obs_noise_std": 0.05,
            "action_noise_prob": 0.1,
            "domain_randomization": {"gravity": [8.0, 12.0]},
        },
        "agent": {
            "type": "random",
            "config": {"seed": 42},
        },
        "training": {
            "num_episodes": 1,
            "max_steps_per_episode": 10,
            "eval_frequency": 10,
            "save_frequency": 10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
        "logging": {"level": "INFO", "log_metrics": True},
    }
    config_path = str(tmp_path / "config_env_forwarding.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    captured = {}

    class _StopAfterCaptureError(Exception):
        pass

    def patched_init(self, *args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterCaptureError()

    import rl_cartpole.environments.cartpole_env as _ce_mod

    monkeypatch.setattr(_ce_mod.CartPoleEnv, "__init__", patched_init)

    with pytest.raises(_StopAfterCaptureError):
        run_main(["--config", config_path], monkeypatch)

    assert captured["env_name"] == "CartPole-v1"
    assert captured["max_episode_steps"] == 20
    assert captured["seed"] == 123
    assert captured["obs_noise_std"] == pytest.approx(0.05)
    assert captured["action_noise_prob"] == pytest.approx(0.1)
    assert captured["domain_randomization"] == {"gravity": [8.0, 12.0]}


def test_environment_null_values_fall_back_to_defaults(tmp_path, monkeypatch):
    """Explicit YAML nulls for selected environment fields should use defaults."""
    import yaml

    config = {
        "environment": {
            "name": None,
            "render_mode": None,
            "max_episode_steps": 20,
            "seed": 123,
            "obs_noise_std": None,
            "action_noise_prob": None,
        },
        "agent": {
            "type": "random",
            "config": {"seed": 42},
        },
        "training": {
            "num_episodes": 1,
            "max_steps_per_episode": 10,
            "eval_frequency": 10,
            "save_frequency": 10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
        "logging": {"level": "INFO", "log_metrics": True},
    }
    config_path = str(tmp_path / "config_env_nulls.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    captured = {}

    class _StopAfterCaptureError(Exception):
        pass

    def patched_init(self, *args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterCaptureError()

    import rl_cartpole.environments.cartpole_env as _ce_mod

    monkeypatch.setattr(_ce_mod.CartPoleEnv, "__init__", patched_init)

    with pytest.raises(_StopAfterCaptureError):
        run_main(["--config", config_path], monkeypatch)

    assert captured["env_name"] == "CartPole-v1"
    assert captured["obs_noise_std"] == pytest.approx(0.0)
    assert captured["action_noise_prob"] == pytest.approx(0.0)
