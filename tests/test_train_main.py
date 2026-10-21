"""train.py's main: how a run ends (finished, interrupted, failed) and how the CLI is read."""

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rl_core.training
from rl_cartpole.environments import CartPoleEnv
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
