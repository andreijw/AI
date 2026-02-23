"""Tests for the Logger and setup_logger utilities."""

import json
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from rl_cartpole.utils.logger import Logger, setup_logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log_dir(tmp_path):
    """Return a temporary directory for log files."""
    return str(tmp_path)


@pytest.fixture
def logger(log_dir):
    """Provide a Logger instance backed by a temporary directory."""
    return Logger(name="test_logger", log_dir=log_dir, level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Logger creation
# ---------------------------------------------------------------------------


def test_logger_creation(log_dir):
    """Logger should be created and the log directory should exist."""
    lg = Logger(name="test_logger", log_dir=log_dir)

    assert lg.name == "test_logger"
    assert lg.log_dir == log_dir
    assert os.path.isdir(log_dir)


def test_logger_creates_log_file(log_dir):
    """A .log file should be created in the log directory on instantiation."""
    Logger(name="my_logger", log_dir=log_dir)

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    assert len(log_files) == 1


def test_logger_creates_metrics_file(log_dir):
    """A metrics .jsonl file should be created in the log directory after the first log() call."""
    lg = Logger(name="my_logger", log_dir=log_dir)
    lg.log({"value": 1})  # triggers file creation

    jsonl_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    assert len(jsonl_files) == 1


def test_logger_creates_directory_if_missing(tmp_path):
    """Logger should create the log directory if it does not exist."""
    new_dir = str(tmp_path / "new" / "nested" / "logs")
    Logger(name="test", log_dir=new_dir)
    assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# Logger.log
# ---------------------------------------------------------------------------


def test_logger_log_writes_to_metrics_file(logger, log_dir):
    """logger.log should append a JSON line to the metrics file."""
    logger.log({"reward": 42.0, "length": 10})

    jsonl_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    assert len(jsonl_files) == 1

    with open(os.path.join(log_dir, jsonl_files[0])) as f:
        line = f.readline().strip()

    entry = json.loads(line)
    assert entry["reward"] == 42.0
    assert entry["length"] == 10


def test_logger_log_with_step_includes_step(logger, log_dir):
    """When step is provided, the JSON line should include it."""
    logger.log({"loss": 0.5}, step=100)

    jsonl_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    with open(os.path.join(log_dir, jsonl_files[0])) as f:
        entry = json.loads(f.readline().strip())

    assert entry["step"] == 100
    assert entry["loss"] == 0.5


def test_logger_log_without_step_omits_step_key(logger, log_dir):
    """When step is not provided, the JSON line should not contain a 'step' key."""
    logger.log({"episode": 1})

    jsonl_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    with open(os.path.join(log_dir, jsonl_files[0])) as f:
        entry = json.loads(f.readline().strip())

    assert "step" not in entry


def test_logger_log_appends_multiple_entries(logger, log_dir):
    """Multiple calls to log should each append a separate JSON line."""
    logger.log({"reward": 1.0})
    logger.log({"reward": 2.0})
    logger.log({"reward": 3.0})

    jsonl_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    with open(os.path.join(log_dir, jsonl_files[0])) as f:
        lines = f.readlines()

    assert len(lines) == 3
    rewards = [json.loads(line)["reward"] for line in lines]
    assert rewards == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Logger convenience methods
# ---------------------------------------------------------------------------


def test_logger_info(logger):
    """Logger.info should not raise."""
    logger.info("This is an info message")


def test_logger_warning(logger):
    """Logger.warning should not raise."""
    logger.warning("This is a warning")


def test_logger_error(logger):
    """Logger.error should not raise."""
    logger.error("This is an error")


def test_logger_debug(logger):
    """Logger.debug should not raise."""
    logger.debug("This is a debug message")


# ---------------------------------------------------------------------------
# setup_logger factory
# ---------------------------------------------------------------------------


def test_setup_logger_returns_logger_instance(log_dir):
    """setup_logger should return a Logger instance."""
    lg = setup_logger(name="setup_test", log_dir=log_dir)
    assert isinstance(lg, Logger)


def test_setup_logger_passes_parameters(log_dir):
    """setup_logger should pass name and log_dir to the Logger."""
    lg = setup_logger(name="my_run", log_dir=log_dir, level=logging.WARNING)
    assert lg.name == "my_run"
    assert lg.log_dir == log_dir


def test_setup_logger_creates_log_directory(tmp_path):
    """setup_logger should create the log directory if absent."""
    new_dir = str(tmp_path / "auto_created")
    setup_logger(name="test", log_dir=new_dir)
    assert os.path.isdir(new_dir)
