"""Tests for CartPole environment wrapper."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_cartpole.environments import CartPoleEnv


def test_env_creation():
    """Test environment creation."""
    env = CartPoleEnv(seed=42)
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None
    env.close()


def test_env_reset():
    """Test environment reset."""
    env = CartPoleEnv(seed=42)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)  # CartPole has 4 observations
    assert isinstance(info, dict)

    env.close()


def test_env_step():
    """Test environment step."""
    env = CartPoleEnv(seed=42)
    obs, _ = env.reset()

    action = 0  # Left
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (4,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_env_episode_tracking():
    """Test episode statistics tracking."""
    env = CartPoleEnv(seed=42)
    env.reset()

    assert env.episode_steps == 0
    assert env.episode_reward == 0.0

    env.step(0)
    assert env.episode_steps == 1
    assert env.episode_reward > 0

    env.close()


def test_env_episode_completion():
    """Test complete episode."""
    env = CartPoleEnv(seed=42, max_episode_steps=10)
    obs, _ = env.reset()

    done = False
    steps = 0
    while not done and steps < 100:
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        steps += 1

        if done:
            assert "episode" in info
            assert "steps" in info["episode"]
            assert "reward" in info["episode"]

    env.close()
