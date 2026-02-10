"""Tests for CartPole environment wrapper."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
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


def test_observation_noise():
    """Test observation noise injection."""
    env = CartPoleEnv(seed=42, observation_noise_std=0.1)
    obs, _ = env.reset(seed=42)
    
    # With noise, observations should be valid but different from deterministic
    assert obs.shape == (4,)
    assert not np.any(np.isnan(obs))
    
    env.close()


def test_action_noise():
    """Test action noise (probabilistic flipping)."""
    env = CartPoleEnv(seed=42, action_noise_std=0.5)
    obs, _ = env.reset(seed=42)
    
    # Action noise should not cause errors
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (4,)
        if terminated or truncated:
            break
    
    env.close()


def test_domain_randomization():
    """Test domain randomization of physics parameters."""
    domain_randomization = {
        'gravity': (9.0, 11.0),
        'pole_length': (0.45, 0.55),
        'cart_mass': (0.9, 1.1),
    }
    
    env = CartPoleEnv(seed=42, domain_randomization=domain_randomization)
    
    # Reset and check physics parameters were randomized
    obs, _ = env.reset(seed=42)
    unwrapped = env.env.unwrapped
    
    assert 9.0 <= unwrapped.gravity <= 11.0
    assert 0.45 <= unwrapped.length <= 0.55
    assert 0.9 <= unwrapped.masscart <= 1.1
    
    env.close()


def test_combined_augmentation():
    """Test environment with all augmentation features."""
    domain_randomization = {
        'gravity': (9.5, 10.5),
    }
    
    env = CartPoleEnv(
        seed=42,
        observation_noise_std=0.01,
        action_noise_std=0.1,
        domain_randomization=domain_randomization,
    )
    
    obs, _ = env.reset(seed=42)
    assert obs.shape == (4,)
    
    # Run a few steps to ensure everything works together
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
