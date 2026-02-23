"""Tests for the Trainer class."""

import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl_cartpole.agents import RandomAgent
from rl_cartpole.environments import CartPoleEnv
from rl_cartpole.training import Trainer


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


def test_trainer_accepts_logger(env, agent):
    """Trainer should store an optional logger."""
    logger = MagicMock()
    trainer = Trainer(env=env, agent=agent, config={}, logger=logger)
    assert trainer.logger is logger


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
# _evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_metrics(trainer):
    """_evaluate should return a dict with mean/std reward and mean length."""
    stats = trainer._evaluate(num_episodes=3)

    assert isinstance(stats, dict)
    assert "mean_reward" in stats
    assert "std_reward" in stats
    assert "mean_length" in stats
    assert isinstance(stats["mean_reward"], float)
    assert isinstance(stats["std_reward"], float)
    assert isinstance(stats["mean_length"], float)


def test_evaluate_std_for_single_episode(trainer):
    """std_reward is zero for a single-episode evaluation."""
    stats = trainer._evaluate(num_episodes=1)
    assert stats["std_reward"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_file(env):
    """_save_checkpoint should create the checkpoint directory and call agent.save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "num_episodes": 1,
            "checkpoint_dir": tmpdir,
        }
        mock_agent = MagicMock()
        trainer = Trainer(env=env, agent=mock_agent, config=config)

        trainer._save_checkpoint(episode=1)

        expected_path = os.path.join(tmpdir, "agent_episode_1.pt")
        # Ensure the directory exists and that agent.save was called with the expected path
        assert os.path.isdir(tmpdir)
        mock_agent.save.assert_called_once_with(expected_path)


def test_save_checkpoint_creates_nested_directory(env, agent):
    """_save_checkpoint should create nested checkpoint directories."""
    with tempfile.TemporaryDirectory() as base:
        nested = os.path.join(base, "deep", "nested", "checkpoints")
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
    assert all(isinstance(l, int) for l in trainer.episode_lengths)


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


def test_train_triggers_checkpoint_save(env, agent):
    """Trainer should save a checkpoint when episode reaches save_frequency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "num_episodes": 2,
            "max_steps_per_episode": 20,
            "eval_frequency": 10,
            "save_frequency": 2,
            "checkpoint_dir": tmpdir,
        }
        trainer = Trainer(env=env, agent=agent, config=config)

        # Patch _save_checkpoint to verify that it is called during training.
        with patch.object(trainer, "_save_checkpoint", wraps=trainer._save_checkpoint) as mock_save:
            trainer.train()
            assert mock_save.called


def test_train_triggers_evaluation(env, agent, tmp_path):
    """Trainer should call _evaluate when episode reaches eval_frequency."""
    config = {
        "num_episodes": 2,
        "max_steps_per_episode": 20,
        "eval_frequency": 2,
        "save_frequency": 100,
        "checkpoint_dir": str(tmp_path / "no_save"),
    }
    trainer = Trainer(env=env, agent=agent, config=config)

    with patch.object(trainer, "_evaluate", wraps=trainer._evaluate) as mock_eval:
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
