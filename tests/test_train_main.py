"""train.py's main: how a run ends (finished, interrupted, failed), how the CLI is read and
overrides the config, and what it forwards to the environment."""

import argparse
import os
import sys

import numpy as np
import pytest

import rl_core.training
from rl_cartpole.environments import CartPoleEnv
from rl_core.agents.base_agent import OBSERVATION_KEY_PREFIX
from rl_core.environments import OBSERVATION_DEFAULTS
from tests.helpers import minimal_config, run_main


class _StubTrainer:
    """Stands in for Trainer; ``outcome`` decides how train() ends."""

    outcome: BaseException | None = None

    def __init__(self, env, agent, config, logger=None, **_eval):
        self.eval_frequency = config.get("eval_frequency", 1)
        self.episode_rewards = [1.0, 2.0, 3.0]
        self.episode_lengths = [1, 2, 3]

    def train(self):
        if self.outcome is not None:
            raise self.outcome
        return {"episodes": 3}


@pytest.fixture(autouse=True)
def _fresh_train_module():
    """Drop the train module each test imported, so none keeps a stubbed Trainer for the next."""
    yield
    sys.modules.pop("train", None)


@pytest.fixture
def closed_envs(monkeypatch):
    """The ids of every CartPoleEnv closed during the test."""
    closed: list[int] = []
    original_close = CartPoleEnv.close
    monkeypatch.setattr(
        CartPoleEnv, "close", lambda self: closed.append(id(self)) or original_close(self)
    )
    return closed


def _stub_trainer(monkeypatch, outcome=None):
    trainer_cls = type("Stub", (_StubTrainer,), {"outcome": outcome})
    monkeypatch.setattr(rl_core.training, "Trainer", trainer_cls)


def test_plot_flag_writes_the_training_plot(tmp_path, monkeypatch, closed_envs):
    _stub_trainer(monkeypatch)
    plot_dir = tmp_path / "plots"
    run_main(
        ["--config", minimal_config(tmp_path), "--plot", "--plot-dir", str(plot_dir)], monkeypatch
    )
    assert [p.suffix for p in plot_dir.iterdir()] == [".png"]
    assert len(set(closed_envs)) == 2


def test_an_interrupted_run_returns_closes_both_environments_and_writes_no_plot(
    tmp_path, monkeypatch, closed_envs
):
    _stub_trainer(monkeypatch, KeyboardInterrupt())
    plot_dir = tmp_path / "plots"
    run_main(
        ["--config", minimal_config(tmp_path), "--plot", "--plot-dir", str(plot_dir)], monkeypatch
    )
    assert not plot_dir.exists()
    assert len(set(closed_envs)) == 2


def test_a_failed_run_raises_and_still_closes_both_environments(tmp_path, monkeypatch, closed_envs):
    _stub_trainer(monkeypatch, RuntimeError("diverged"))
    with pytest.raises(RuntimeError, match="diverged"):
        run_main(["--config", minimal_config(tmp_path)], monkeypatch)
    assert len(set(closed_envs)) == 2


# ---------------------------------------------------------------------------
# The helpers main is built from
# ---------------------------------------------------------------------------


def _train_module():
    # Imported afresh, like run_main does, so no earlier test's stubbed Trainer is bound in it.
    sys.modules.pop("train", None)
    import train  # noqa: PLC0415

    return train


def _parse_error(argv, capsys) -> str:
    with pytest.raises(SystemExit):
        _train_module().parse_args(argv)
    return capsys.readouterr().err


def test_parse_args_reads_the_defaults_without_sys_argv():
    args = _train_module().parse_args([])
    assert args.config == "configs/cartpole_default.yaml"
    assert (args.render, args.record_video, args.plot) == (False, False, False)
    assert (args.num_episodes, args.total_steps, args.video_every) == (None, None, None)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--render", "--record-video", "--video-every", "0"], "mutually exclusive"),
        (["--num-episodes", "0", "--video-every", "3"], "--num-episodes must be a positive"),
        (["--total-steps", "-1"], "--total-steps must be a positive"),
        (["--video-every", "2"], "--video-every requires --record-video"),
        (["--video-every", "0"], "--video-every requires --record-video"),
        (["--record-video", "--video-every", "0"], "--video-every must be a positive"),
    ],
)
def test_parse_args_reports_the_first_problem(argv, message, capsys):
    assert message in _parse_error(argv, capsys)


@pytest.mark.parametrize(
    ("flags", "configured", "expected"),
    [
        ({"record_video": True, "render": False}, "human", "rgb_array"),
        ({"record_video": False, "render": True}, None, "human"),
        ({"record_video": False, "render": False}, "rgb_array", "rgb_array"),
        ({"record_video": False, "render": False}, None, None),
    ],
)
def test_render_mode_follows_the_flags_then_the_config(flags, configured, expected):
    config = {"environment": {"render_mode": configured}}
    assert _train_module().resolve_render_mode(argparse.Namespace(**flags), config) == expected


def test_output_dirs_nest_under_challenge_and_agent_without_touching_the_config():
    train = _train_module()
    args = train.parse_args(["--plot-dir", "my_plots"])
    config = {"training": {"checkpoint_dir": "ck", "log_dir": "lg"}}
    dirs = train.output_dirs(config, "dqn", "cartpole", args)
    assert dirs == (
        os.path.join("ck", "cartpole", "dqn"),
        os.path.join("lg", "cartpole", "dqn"),
        os.path.join("./videos", "cartpole", "dqn"),
        "my_plots",
    )
    assert config == {"training": {"checkpoint_dir": "ck", "log_dir": "lg"}}


def test_output_dirs_default_to_the_working_directory():
    train = _train_module()
    dirs = train.output_dirs({"training": {}}, "ppo", "pendulum", train.parse_args([]))
    assert dirs == tuple(
        os.path.join(base, "pendulum", "ppo")
        for base in ("./checkpoints", "./logs", "./videos", "./plots")
    )


def test_train_py_saves_a_final_checkpoint_at_the_last_episode(tmp_path, monkeypatch):
    """4 episodes with save_frequency 10: no periodic save, yet the run ends with a checkpoint."""
    run_main(["--config", minimal_config(tmp_path)], monkeypatch)
    (folder,) = (tmp_path / "checkpoints").glob("*/*")
    assert sorted(p.name for p in folder.iterdir()) == [
        "agent_episode_4.pt",
        "trainer_episode_4.npz",
    ]


# ---------------------------------------------------------------------------
# CLI overrides and the environment config train.py forwards
# ---------------------------------------------------------------------------


def test_num_episodes_override_requires_positive_int(tmp_path, monkeypatch):
    """--num-episodes must be a positive integer."""
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--num-episodes", "0", "--config", config_path], monkeypatch)


def test_agent_type_and_num_episodes_overrides_apply(tmp_path, monkeypatch):
    """--agent-type and --num-episodes replace the config's: a real random agent trains for one
    episode, is saved under its own name, and records the run's observation settings."""
    config_path = minimal_config(tmp_path, agent={"type": "reinforce"})
    run_main(
        ["--config", config_path, "--agent-type", "random", "--num-episodes", "1"], monkeypatch
    )
    (folder,) = (tmp_path / "checkpoints").glob("*/*")
    assert folder.relative_to(tmp_path / "checkpoints").parts == ("cartpole", "random")
    assert sorted(p.name for p in folder.iterdir()) == [
        "agent_episode_1.pt",
        "trainer_episode_1.npz",
    ]
    with np.load(folder / "agent_episode_1.pt") as saved:
        dims = (int(saved["observation_dim"]), int(saved["action_dim"]))
        recorded = {
            key.removeprefix(OBSERVATION_KEY_PREFIX): saved[key].item()
            for key in saved.files
            if key.startswith(OBSERVATION_KEY_PREFIX)
        }
    assert dims == (4, 2), "built for CartPole's 4 observations and 2 actions"
    assert recorded == OBSERVATION_DEFAULTS


def _cartpole_env_kwargs(args, monkeypatch) -> dict:
    """The keyword arguments train.py builds its first CartPoleEnv with; it stops there."""
    captured = {}

    class _StopAfterCaptureError(Exception):
        pass

    def patched_init(self, *args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterCaptureError()

    monkeypatch.setattr(CartPoleEnv, "__init__", patched_init)
    with pytest.raises(_StopAfterCaptureError):
        run_main(args, monkeypatch)
    return captured


def test_environment_config_is_forwarded_to_cartpole_env(tmp_path, monkeypatch):
    """train.py should forward environment config fields into CartPoleEnv."""
    config_path = minimal_config(
        tmp_path,
        environment={
            "seed": 123,
            "obs_noise_std": 0.05,
            "action_noise_prob": 0.1,
            "domain_randomization": {"gravity": [8.0, 12.0]},
        },
    )

    captured = _cartpole_env_kwargs(["--config", config_path], monkeypatch)

    assert captured["env_name"] == "CartPole-v1"
    assert captured["max_episode_steps"] == 20
    assert captured["seed"] == 123
    assert captured["obs_noise_std"] == pytest.approx(0.05)
    assert captured["action_noise_prob"] == pytest.approx(0.1)
    assert captured["domain_randomization"] == {"gravity": (8.0, 12.0)}


def test_environment_null_values_fall_back_to_defaults(tmp_path, monkeypatch):
    """Explicit YAML nulls for environment fields use defaults; --challenge supplies the name."""
    config_path = minimal_config(
        tmp_path,
        environment={"name": None, "seed": 123, "obs_noise_std": None, "action_noise_prob": None},
    )

    captured = _cartpole_env_kwargs(
        ["--config", config_path, "--challenge", "cartpole"], monkeypatch
    )

    assert captured["env_name"] == "CartPole-v1"
    assert captured["obs_noise_std"] == pytest.approx(0.0)
    assert captured["action_noise_prob"] == pytest.approx(0.0)


def test_total_steps_override_sets_a_step_budget(tmp_path):
    from train import load_run_config

    config, _ = load_run_config(minimal_config(tmp_path), total_steps=300)
    assert config["training"]["total_steps"] == 300


def test_num_episodes_override_clears_a_step_budget(tmp_path):
    from train import load_run_config

    path = minimal_config(tmp_path, training={"total_steps": 900})
    config, _ = load_run_config(path, num_episodes=2)
    assert config["training"]["num_episodes"] == 2 and "total_steps" not in config["training"]


def test_num_episodes_and_total_steps_together_are_rejected(tmp_path, monkeypatch):
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(
            ["--num-episodes", "2", "--total-steps", "50", "--config", config_path], monkeypatch
        )


def test_total_steps_override_requires_positive_int(tmp_path, monkeypatch):
    config_path = minimal_config(tmp_path)
    with pytest.raises(SystemExit):
        run_main(["--total-steps", "0", "--config", config_path], monkeypatch)
