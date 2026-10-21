"""Tests for the Trainer class."""

import json
import logging
import os
import sys
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import BaseAgent, DQNAgent, RandomAgent
from rl_core.training import Trainer
from rl_core.utils.logger import Logger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Provide a CartPoleEnv instance for testing."""
    e = CartPoleEnv(seed=42, max_episode_steps=20)
    yield e
    e.close()


@pytest.fixture
def agent():
    """Provide a RandomAgent for testing."""
    return RandomAgent(observation_dim=4, action_dim=2, config={})


@pytest.fixture
def trainer(env, agent, tmp_path):
    """Provide a Trainer with a minimal config."""
    config = {
        "num_episodes": 3,
        "max_steps_per_episode": 20,
        "eval_frequency": 2,
        "save_frequency": 2,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    return Trainer(env=env, agent=agent, config=config)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_trainer_creation(env, agent, tmp_path):
    """Test that Trainer is created with correct attributes."""
    ckpt_dir = str(tmp_path / "ckpt")
    config = {
        "num_episodes": 10,
        "max_steps_per_episode": 50,
        "eval_frequency": 5,
        "save_frequency": 5,
        "checkpoint_dir": ckpt_dir,
    }
    trainer = Trainer(env=env, agent=agent, config=config)

    assert trainer.env is env
    assert trainer.agent is agent
    assert trainer.num_episodes == 10
    assert trainer.max_steps_per_episode == 50
    assert trainer.eval_frequency == 5
    assert trainer.save_frequency == 5
    assert trainer.checkpoint_dir == ckpt_dir
    assert trainer.episode_rewards == []
    assert trainer.episode_lengths == []


def test_trainer_default_config_values(env, agent):
    """Verify default values are used when config keys are absent."""
    trainer = Trainer(env=env, agent=agent, config={})

    assert trainer.num_episodes == 1000
    assert trainer.max_steps_per_episode == 500
    assert trainer.eval_frequency == 100
    assert trainer.save_frequency == 100
    assert trainer.checkpoint_dir == "./checkpoints"


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"num_episodes": 0}, "num_episodes must be a positive integer"),
        ({"max_steps_per_episode": 0}, "max_steps_per_episode must be a positive integer"),
        ({"eval_frequency": 0}, "eval_frequency must be a positive integer"),
        ({"save_frequency": 0}, "save_frequency must be a positive integer"),
        (
            {"num_episodes": True},
            "Configuration value for 'num_episodes' must be a positive integer",
        ),
        (
            {"max_steps_per_episode": 1.9},
            "Configuration value for 'max_steps_per_episode' must be a positive integer",
        ),
        (
            {"eval_frequency": "abc"},
            "Configuration value for 'eval_frequency' must be a positive integer",
        ),
    ],
)
def test_trainer_invalid_config_values_raise(env, agent, config, expected_error):
    """Trainer should fail fast on invalid non-positive training config values."""
    with pytest.raises(ValueError, match=expected_error):
        Trainer(env=env, agent=agent, config=config)


def test_trainer_accepts_logger(env, agent):
    """Trainer should store an optional logger."""
    logger = MagicMock()
    trainer = Trainer(env=env, agent=agent, config={}, logger=logger)
    assert trainer.logger is logger


def test_log_info_uses_logger_when_available(env, agent):
    """_log_info should delegate to logger.info when a logger is set."""
    logger = MagicMock()
    trainer = Trainer(env=env, agent=agent, config={}, logger=logger)
    trainer._log_info("test message")
    logger.info.assert_called_once_with("test message")


def test_log_info_falls_back_to_logging_without_logger(env, agent, caplog):
    """_log_info should emit via the module logger when no logger is provided."""
    trainer = Trainer(env=env, agent=agent, config={})
    with caplog.at_level(logging.INFO, logger="rl_core.training.trainer"):
        trainer._log_info("fallback message")
    assert "fallback message" in caplog.text


# ---------------------------------------------------------------------------
# _run_episode
# ---------------------------------------------------------------------------


def test_run_episode_returns_scalars(trainer):
    """_run_episode should return (float, int)."""
    episode_reward, episode_length = trainer._run_episode(training=True)

    assert isinstance(episode_reward, float)
    assert isinstance(episode_length, int)
    assert episode_length >= 1


def test_run_episode_non_training(trainer):
    """_run_episode in eval mode should still return valid scalars."""
    episode_reward, episode_length = trainer._run_episode(training=False)

    assert isinstance(episode_reward, float)
    assert isinstance(episode_length, int)


def test_run_episode_respects_max_steps(env, agent):
    """max_steps_per_episode should cap the episode length."""
    config = {"num_episodes": 1, "max_steps_per_episode": 5}
    trainer = Trainer(env=env, agent=agent, config=config)

    _, episode_length = trainer._run_episode(training=True)
    assert episode_length <= 5


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_metrics(trainer):
    """evaluate should return a dict with mean/std reward and mean length."""
    stats = trainer.evaluate(num_episodes=3)

    assert isinstance(stats, dict)
    assert "mean_reward" in stats
    assert "std_reward" in stats
    assert "mean_length" in stats
    assert isinstance(stats["mean_reward"], float)
    assert isinstance(stats["std_reward"], float)
    assert isinstance(stats["mean_length"], float)


def test_evaluate_std_for_single_episode(trainer):
    """std_reward is zero for a single-episode evaluation."""
    stats = trainer.evaluate(num_episodes=1)
    assert stats["std_reward"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_file(env, tmp_path):
    """_save_checkpoint should create the checkpoint directory and call agent.save."""
    tmpdir = str(tmp_path / "ckpt")
    config = {
        "num_episodes": 1,
        "checkpoint_dir": tmpdir,
    }
    mock_agent = MagicMock()
    trainer = Trainer(env=env, agent=mock_agent, config=config)

    trainer._save_checkpoint(episode=1)

    expected_path = os.path.join(tmpdir, "agent_episode_1.pt")
    assert os.path.isdir(tmpdir)
    mock_agent.save.assert_called_once_with(expected_path)


def test_save_checkpoint_creates_nested_directory(env, agent, tmp_path):
    """_save_checkpoint should create nested checkpoint directories."""
    nested = str(tmp_path / "deep" / "nested" / "checkpoints")
    config = {"num_episodes": 1, "checkpoint_dir": nested}
    trainer = Trainer(env=env, agent=agent, config=config)

    trainer._save_checkpoint(episode=5)

    assert os.path.isdir(nested)


# ---------------------------------------------------------------------------
# train (full loop)
# ---------------------------------------------------------------------------


def test_train_returns_stats(trainer):
    """train() should run successfully and return a statistics dict."""
    stats = trainer.train()

    assert isinstance(stats, dict)
    assert "total_episodes" in stats
    assert "avg_reward" in stats
    assert "avg_length" in stats
    assert "final_avg_reward" in stats
    assert stats["total_episodes"] == trainer.num_episodes


def test_train_accumulates_metrics(trainer):
    """After training, episode_rewards and episode_lengths should be populated."""
    trainer.train()

    assert len(trainer.episode_rewards) == trainer.num_episodes
    assert len(trainer.episode_lengths) == trainer.num_episodes
    assert all(isinstance(r, float) for r in trainer.episode_rewards)
    assert all(isinstance(ep_len, int) for ep_len in trainer.episode_lengths)


def test_train_exposes_training_progress(trainer):
    """current_episode tracks the 1-based training episode; evaluating is set only during eval."""
    assert trainer.current_episode == 0
    assert trainer.evaluating is False

    seen = []
    original_run_episode = trainer._run_episode

    def spy_run_episode(training=True):
        seen.append((training, trainer.current_episode, trainer.evaluating))
        return original_run_episode(training=training)

    with patch.object(trainer, "_run_episode", side_effect=spy_run_episode):
        trainer.train()

    training_calls = [(ep, ev) for training, ep, ev in seen if training]
    eval_calls = [(ep, ev) for training, ep, ev in seen if not training]
    assert training_calls == [(1, False), (2, False), (3, False)]
    # eval_frequency=2 -> one evaluation of 10 episodes after training episode 2
    assert eval_calls == [(2, True)] * 10
    assert trainer.evaluating is False


def test_evaluating_flag_reset_when_evaluation_fails(trainer):
    """evaluating must be cleared even if an evaluation episode raises."""
    failing_episode = patch.object(trainer, "_run_episode", side_effect=RuntimeError("boom"))
    with failing_episode, pytest.raises(RuntimeError):
        trainer.evaluate()

    assert trainer.evaluating is False


def test_train_with_logger_calls_log(env, agent, tmp_path):
    """Trainer should call logger.log when a logger is provided."""
    logger = MagicMock()
    config = {
        "num_episodes": 10,
        "max_steps_per_episode": 20,
        "eval_frequency": 10,
        "save_frequency": 10,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    trainer = Trainer(env=env, agent=agent, config=config, logger=logger)
    trainer.train()

    # logger.log should have been called at least once (every 10 episodes)
    assert logger.log.called


def test_train_triggers_checkpoint_save(env, agent, tmp_path):
    """Trainer should call _save_checkpoint when episode reaches save_frequency."""
    config = {
        "num_episodes": 2,
        "max_steps_per_episode": 20,
        "eval_frequency": 10,
        "save_frequency": 2,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    trainer = Trainer(env=env, agent=agent, config=config)

    # Patch _save_checkpoint to verify that it is called during training.
    with patch.object(trainer, "_save_checkpoint", wraps=trainer._save_checkpoint) as mock_save:
        trainer.train()
        assert mock_save.called


def test_train_triggers_evaluation(env, agent, tmp_path):
    """Trainer should call evaluate when episode reaches eval_frequency."""
    config = {
        "num_episodes": 2,
        "max_steps_per_episode": 20,
        "eval_frequency": 2,
        "save_frequency": 100,
        "checkpoint_dir": str(tmp_path / "no_save"),
    }
    trainer = Trainer(env=env, agent=agent, config=config)

    with patch.object(trainer, "evaluate", wraps=trainer.evaluate) as mock_eval:
        trainer.train()
        assert mock_eval.called


def test_train_final_avg_reward_uses_last_100(env, agent):
    """final_avg_reward should be the mean of the last min(100, total) episodes."""
    config = {
        "num_episodes": 5,
        "max_steps_per_episode": 20,
        "eval_frequency": 100,
        "save_frequency": 100,
    }
    trainer = Trainer(env=env, agent=agent, config=config)
    stats = trainer.train()

    expected = float(np.mean(trainer.episode_rewards[-100:]))
    assert stats["final_avg_reward"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Update metrics in the periodic log
# ---------------------------------------------------------------------------


class _OneStepEnv:
    """Minimal environment: every episode lasts one step and pays the next scripted reward."""

    def __init__(self, rewards=None):
        self._rewards = list(rewards) if rewards is not None else None
        self._episode = -1

    def reset(self, **_kwargs):
        self._episode += 1
        return np.zeros(4), {}

    def step(self, _action):
        reward = 1.0 if self._rewards is None else self._rewards[self._episode]
        return np.zeros(4), reward, True, False, {}

    def close(self):
        pass


class _ScriptedAgent(BaseAgent):
    """Agent whose update() returns the next entry of a script, one per training episode."""

    def __init__(self, script):
        super().__init__(observation_dim=4, action_dim=2, config={})
        self._script = list(script)
        self.update_calls = 0

    def select_action(self, observation, training=True):
        return 0

    def update(self, batch):
        metrics = self._script[self.update_calls]
        self.update_calls += 1
        return metrics

    def _checkpoint_arrays(self):
        return {}

    def _load_checkpoint_arrays(self, data):
        pass


class _RecordingLogger:
    """Logger double that keeps what the trainer sends, so tests can read real values."""

    def __init__(self):
        self.records = []
        self.infos = []
        self.warnings = []

    def log(self, metrics):
        self.records.append(metrics)

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def periodic(self):
        """The periodic training records (those carrying an episode number), in order."""
        return [r for r in self.records if "episode" in r]


def _train_scripted(script, tmp_path, logger=None, rewards=None):
    """Train a scripted agent for len(script) one-step episodes and return (trainer, logger)."""
    logger = _RecordingLogger() if logger is None else logger
    n = len(script)
    config = {
        "num_episodes": n,
        "max_steps_per_episode": 5,
        "eval_frequency": n + 1,
        "save_frequency": n + 1,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    trainer = Trainer(
        env=_OneStepEnv(rewards), agent=_ScriptedAgent(script), config=config, logger=logger
    )
    trainer.train()
    return trainer, logger


def _reject_constant(token):
    raise ValueError(f"{token} is not strict JSON")


def test_train_logs_mean_of_update_metrics_per_interval(tmp_path):
    """The periodic record carries the mean of each update metric over its ten episodes."""
    # loss 1..10 -> mean 5.5; plain ints and NumPy scalars must be accepted like Python floats.
    script = [{"loss": np.float32(i), "steps": np.int64(2 * i), "count": i} for i in range(1, 11)]

    _, logger = _train_scripted(script, tmp_path)

    (record,) = logger.periodic()
    assert record["episode"] == 10
    assert record["update"] == {
        "loss": pytest.approx(5.5),
        "steps": pytest.approx(11.0),
        "count": pytest.approx(5.5),
    }
    assert "non_finite" not in record, "All values were finite, so nothing should be flagged."


def test_train_update_metric_window_resets_each_interval(tmp_path):
    """Each periodic record averages only the episodes since the previous one."""
    script = [{"loss": 1.0}] * 10 + [{"loss": 100.0}] * 10

    _, logger = _train_scripted(script, tmp_path)

    first, second = logger.periodic()
    assert first["update"]["loss"] == pytest.approx(1.0)
    assert second["update"]["loss"] == pytest.approx(100.0), "Episodes 1-10 leaked into 11-20."


def test_train_omits_update_key_when_agent_returns_no_metrics(tmp_path):
    """An agent whose update() returns nothing produces records without update fields."""
    _, logger = _train_scripted([{}] * 10, tmp_path)

    (record,) = logger.periodic()
    assert "update" not in record
    assert "non_finite" not in record


def test_train_averages_metric_over_episodes_that_reported_it(tmp_path):
    """A metric reported in only some episodes is averaged over those episodes alone."""
    # 'loss' appears in 4 of 10 episodes with values 2, 4, 6, 8 -> mean 5, not 20 / 10.
    script = [{"loss": 2.0}, {}, {"loss": 4.0}, {}, {"loss": 6.0}, {}, {"loss": 8.0}, {}, {}, {}]

    _, logger = _train_scripted(script, tmp_path)

    (record,) = logger.periodic()
    assert record["update"] == {"loss": pytest.approx(5.0)}


def test_train_ignores_non_numeric_update_metric_values(tmp_path):
    """Strings, None and booleans in the update dictionary are skipped, not averaged."""
    script = [{"loss": 3.0, "note": "warmup", "missing": None, "flag": True}] * 10

    _, logger = _train_scripted(script, tmp_path)

    (record,) = logger.periodic()
    assert record["update"] == {"loss": pytest.approx(3.0)}


def test_train_ignores_update_return_that_is_not_a_mapping(tmp_path):
    """An update() that returns None (or anything that is not a mapping) is tolerated."""
    trainer, logger = _train_scripted([None] * 10, tmp_path)

    (record,) = logger.periodic()
    assert "update" not in record
    assert trainer.first_non_finite_episode is None


def test_train_warns_once_per_metric_at_first_non_finite_value(tmp_path):
    """Each metric is reported once, at the episode its first non-finite value appeared."""
    script = [{"loss": 1.0, "mean_q": 1.0} for _ in range(10)]
    for episode in range(3, 11):
        script[episode - 1]["loss"] = float("nan")
    for episode in range(7, 11):
        script[episode - 1]["mean_q"] = float("inf")

    trainer, logger = _train_scripted(script, tmp_path)

    assert len(logger.warnings) == 2, f"Expected one warning per metric, got {logger.warnings}"
    loss_warning, q_warning = logger.warnings
    assert "'loss'" in loss_warning and "nan" in loss_warning and "episode 3." in loss_warning
    assert "'mean_q'" in q_warning and "inf" in q_warning and "episode 7." in q_warning
    assert trainer.first_non_finite_episode == 3, "The first episode must not be overwritten."


def test_train_without_non_finite_metrics_reports_none(tmp_path):
    """A healthy run raises no warning and leaves first_non_finite_episode unset."""
    trainer, logger = _train_scripted([{"loss": 1.0}] * 10, tmp_path)

    assert logger.warnings == []
    assert trainer.first_non_finite_episode is None


def test_train_non_finite_metric_is_written_as_null_and_listed(tmp_path):
    """A metric with any non-finite value in the window is null and named; others keep means."""
    script = [{"loss": 1.0, "entropy": 0.5} for _ in range(10)]
    script[4]["loss"] = float("nan")

    _, logger = _train_scripted(script, tmp_path)

    (record,) = logger.periodic()
    assert record["update"] == {"loss": None, "entropy": pytest.approx(0.5)}
    assert record["non_finite"] == ["loss"]
    json.dumps(record, allow_nan=False)  # raises if any NaN or inf is left in the record


def test_train_lists_metric_whose_mean_overflows_as_non_finite(tmp_path):
    """Finite values whose mean overflows are still flagged, never a silent null."""
    script = [{"loss": 1e308, "entropy": 0.5}] * 10

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the overflow itself must not leak a RuntimeWarning
        _, logger = _train_scripted(script, tmp_path)

    (record,) = logger.periodic()
    assert record["update"] == {"loss": None, "entropy": pytest.approx(0.5)}
    assert record["non_finite"] == ["loss"]


def test_train_metric_that_recovers_is_finite_again_in_next_window(tmp_path):
    """Non-finite state belongs to one window: the next record is clean if its values are."""
    script = [{"loss": float("inf")}] * 10 + [{"loss": 2.0}] * 10

    _, logger = _train_scripted(script, tmp_path)

    first, second = logger.periodic()
    assert first["update"]["loss"] is None
    assert first["non_finite"] == ["loss"], "inf must be listed just as NaN is."
    assert second["update"]["loss"] == pytest.approx(2.0)
    assert "non_finite" not in second


def test_train_non_finite_reward_is_written_as_null(tmp_path):
    """Every record is strict JSON, including the reward average when a reward is NaN."""
    rewards = [1.0] * 10
    rewards[2] = float("nan")

    _, logger = _train_scripted([{"loss": 1.0}] * 10, tmp_path, rewards=rewards)

    (record,) = logger.periodic()
    assert record["avg_reward"] is None
    assert record["avg_length"] == pytest.approx(1.0)
    json.dumps(record, allow_nan=False)


def test_evaluation_episodes_do_not_record_update_metrics(tmp_path):
    """evaluate() never calls update(), so it adds nothing to the next periodic record."""
    logger = _RecordingLogger()
    agent = _ScriptedAgent([{"loss": 7.0}] * 10)
    config = {
        "num_episodes": 10,
        "max_steps_per_episode": 5,
        "eval_frequency": 5,
        "save_frequency": 11,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    trainer = Trainer(env=_OneStepEnv(), agent=agent, config=config, logger=logger)

    trainer.train()

    assert agent.update_calls == 10, "update() must run once per training episode only."
    (record,) = logger.periodic()
    assert record["update"] == {"loss": pytest.approx(7.0)}


def test_non_finite_warning_reaches_standard_logger(tmp_path, caplog):
    """A standard logging.Logger (as the benchmark passes) receives the warning."""
    std_logger = logging.getLogger("test_trainer_non_finite_std")
    script = [{"loss": float("nan")}] * 10

    with caplog.at_level(logging.WARNING, logger=std_logger.name):
        _train_scripted(script, tmp_path, logger=std_logger)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'loss'" in warnings[0].getMessage() and "episode 1." in warnings[0].getMessage()


def test_non_finite_warning_falls_back_to_module_logger(tmp_path, caplog):
    """With no logger configured, the warning goes to the trainer module's logger."""
    config = {
        "num_episodes": 10,
        "max_steps_per_episode": 5,
        "eval_frequency": 11,
        "save_frequency": 11,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    agent = _ScriptedAgent([{"loss": float("nan")}] * 10)
    trainer = Trainer(env=_OneStepEnv(), agent=agent, config=config)

    with caplog.at_level(logging.WARNING, logger="rl_core.training.trainer"):
        trainer.train()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'loss'" in warnings[0].getMessage()


def _read_strict_jsonl(path):
    """Parse a JSONL file, failing on the NaN and Infinity tokens strict JSON forbids."""
    with open(path) as f:
        return [json.loads(line, parse_constant=_reject_constant) for line in f if line.strip()]


def test_train_writes_null_for_non_finite_metric_through_real_logger(tmp_path):
    """Through the real Logger, the metrics file stays strict JSON when a metric goes NaN."""
    logger = Logger(name="test_trainer_nan_jsonl", log_dir=str(tmp_path / "logs"))
    try:
        _train_scripted([{"loss": float("nan"), "entropy": 0.25}] * 10, tmp_path, logger=logger)
        rows = _read_strict_jsonl(logger.metrics_file)
    finally:
        logger.close()

    (row,) = [r for r in rows if "episode" in r]
    assert row["update"] == {"loss": None, "entropy": pytest.approx(0.25)}
    assert row["non_finite"] == ["loss"]


def test_train_writes_update_metrics_to_jsonl_with_real_agent(env, tmp_path):
    """A real DQN run writes its loss, epsilon and mean Q-value to the metrics file."""
    agent_config = {"seed": 0, "hidden_dim": 8, "batch_size": 8, "min_buffer_size": 16}
    agent = DQNAgent(observation_dim=4, action_dim=2, config=agent_config)
    config = {
        "num_episodes": 20,
        "max_steps_per_episode": 20,
        "eval_frequency": 21,
        "save_frequency": 21,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    logger = Logger(name="test_trainer_dqn_jsonl", log_dir=str(tmp_path / "logs"))
    try:
        Trainer(env=env, agent=agent, config=config, logger=logger).train()
        rows = _read_strict_jsonl(logger.metrics_file)
    finally:
        logger.close()

    last = [r for r in rows if "episode" in r][-1]
    assert last["episode"] == 20
    assert set(last["update"]) == {"loss", "epsilon", "mean_q"}
    assert last["update"]["loss"] > 0.0, "A zero loss means no real update was averaged."
    assert last["update"]["epsilon"] < 1.0, "Epsilon should have decayed from its start value."
    assert "non_finite" not in last


# ---------------------------------------------------------------------------
# _log_metrics: one normalised record, sent to whichever logger the trainer has
# ---------------------------------------------------------------------------

_RAW_METRICS = {"episode": np.int64(10), "loss": float("nan")}
_SAFE_METRICS = {"episode": 10, "loss": None}


def test_log_metrics_without_a_logger_uses_the_module_logger(env, agent, caplog):
    trainer = Trainer(env=env, agent=agent, config={})
    with caplog.at_level(logging.INFO, logger="rl_core.training.trainer"):
        trainer._log_metrics(_RAW_METRICS)
    assert f"METRICS: {_SAFE_METRICS}" in caplog.messages


def test_log_metrics_with_a_standard_logger_uses_info(env, agent, caplog):
    std_logger = logging.getLogger("test_log_metrics_standard")
    trainer = Trainer(env=env, agent=agent, config={}, logger=std_logger)
    with caplog.at_level(logging.INFO, logger="test_log_metrics_standard"):
        trainer._log_metrics(_RAW_METRICS)
    assert caplog.messages == [f"Metrics: {_SAFE_METRICS}"]


def test_log_metrics_hands_a_custom_logger_the_normalised_dict(env, agent):
    custom = MagicMock()
    Trainer(env=env, agent=agent, config={}, logger=custom)._log_metrics(_RAW_METRICS)
    custom.log.assert_called_once_with(_SAFE_METRICS)
    custom.info.assert_not_called()


def test_log_metrics_falls_back_to_info_when_log_takes_other_arguments(env, agent):
    class LevelLogger:
        def __init__(self):
            self.lines = []

        def log(self, level, msg):  # logging.Logger.log's shape: a dict alone is a TypeError
            self.lines.append(("log", level, msg))

        def info(self, msg):
            self.lines.append(msg)

    custom = LevelLogger()
    Trainer(env=env, agent=agent, config={}, logger=custom)._log_metrics(_RAW_METRICS)
    assert custom.lines == [f"Metrics: {_SAFE_METRICS}"]


def test_log_metrics_with_neither_log_nor_info_uses_the_module_logger(env, agent, caplog):
    trainer = Trainer(env=env, agent=agent, config={}, logger=object())
    with caplog.at_level(logging.INFO, logger="rl_core.training.trainer"):
        trainer._log_metrics(_RAW_METRICS)
    assert f"METRICS: {_SAFE_METRICS}" in caplog.messages
