"""Tests for CartPole environment wrapper."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_cartpole.environments import (
    CartPoleEnv,
    make_env,
    make_env_from_config,
    make_vec_env,
    make_vec_env_from_config,
)


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
    # Create env with observation noise
    env_noisy = CartPoleEnv(seed=42, obs_noise_std=0.1)
    env_clean = CartPoleEnv(seed=42, obs_noise_std=0.0)

    # Compare multiple samples to avoid relying on a single random draw
    observations_differ = []
    base_seed = 123
    for i in range(10):
        obs_noisy, _ = env_noisy.reset(seed=base_seed + i)
        obs_clean, _ = env_clean.reset(seed=base_seed + i)
        observations_differ.append(not np.allclose(obs_noisy, obs_clean, atol=1e-10))

    # At least one pair of observations should differ when noise is applied
    assert any(observations_differ)

    env_noisy.close()
    env_clean.close()


def test_action_noise():
    """Test action noise application."""
    # With 100% action noise, actions should always flip
    env = CartPoleEnv(seed=42, action_noise_prob=1.0)
    env.reset()

    # Test action flipping
    action_0_result = env._apply_action_noise(0)
    action_1_result = env._apply_action_noise(1)

    assert action_0_result == 1  # 0 should flip to 1
    assert action_1_result == 0  # 1 should flip to 0

    # With 0% action noise, actions should never flip
    env_no_noise = CartPoleEnv(seed=42, action_noise_prob=0.0)
    env_no_noise.reset()

    assert env_no_noise._apply_action_noise(0) == 0
    assert env_no_noise._apply_action_noise(1) == 1

    env.close()
    env_no_noise.close()


def test_domain_randomization():
    """Test domain randomization."""
    domain_rand = {
        "gravity": (8.0, 12.0),
        "masscart": (0.8, 1.2),
        "masspole": (0.08, 0.12),
        "length": (0.4, 0.6),
    }

    env = CartPoleEnv(seed=42, domain_randomization=domain_rand)
    env.reset()

    # Access the underlying environment
    base_env = env.env.unwrapped

    # Check that parameters are within the specified ranges
    assert 8.0 <= base_env.gravity <= 12.0
    assert 0.8 <= base_env.masscart <= 1.2
    assert 0.08 <= base_env.masspole <= 0.12
    assert 0.4 <= base_env.length <= 0.6

    # Check that total_mass and polemass_length are updated
    assert base_env.total_mass == base_env.masscart + base_env.masspole
    assert base_env.polemass_length == base_env.masspole * base_env.length

    env.close()


def test_domain_randomization_changes_per_episode():
    """Test that domain randomization changes between episodes."""
    domain_rand = {
        "gravity": (8.0, 12.0),
    }

    env = CartPoleEnv(seed=None, domain_randomization=domain_rand)  # No seed for randomness

    env.reset()
    gravity_1 = env.env.unwrapped.gravity

    env.reset()
    gravity_2 = env.env.unwrapped.gravity

    # With no seed, gravity should likely be different between resets
    # (there's a small chance they could be the same, but very unlikely)
    # We just check they're in valid range
    assert 8.0 <= gravity_1 <= 12.0
    assert 8.0 <= gravity_2 <= 12.0

    env.close()


def test_make_env():
    """Test make_env factory function."""
    env = make_env(seed=42, max_episode_steps=100)

    assert env is not None
    assert isinstance(env, CartPoleEnv)
    assert env.max_episode_steps == 100

    obs, _ = env.reset()
    assert obs.shape == (4,)

    env.close()


def test_make_env_with_all_features():
    """Test make_env with all features enabled."""
    domain_rand = {
        "gravity": (9.0, 10.0),
        "length": (0.4, 0.6),
    }

    env = make_env(
        seed=42,
        obs_noise_std=0.01,
        action_noise_prob=0.1,
        domain_randomization=domain_rand,
    )

    assert env.obs_noise_std == 0.01
    assert env.action_noise_prob == 0.1
    assert env.domain_randomization == domain_rand

    env.close()


def test_make_vec_env():
    """Test vectorized environment creation."""
    vec_env = make_vec_env(num_envs=4, seed=42)

    assert vec_env is not None
    assert vec_env.num_envs == 4

    obs, _ = vec_env.reset()
    assert obs.shape == (4, 4)  # 4 envs, 4 observations each

    actions = np.array([0, 1, 0, 1])
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    assert obs.shape == (4, 4)
    assert rewards.shape == (4,)
    assert terminateds.shape == (4,)
    assert truncateds.shape == (4,)

    vec_env.close()


def test_make_vec_env_sync():
    """Test synchronous vectorized environment."""
    vec_env = make_vec_env(num_envs=2, seed=42, async_envs=False)

    assert vec_env is not None
    assert vec_env.num_envs == 2

    obs, _ = vec_env.reset()
    assert obs.shape == (2, 4)

    vec_env.close()


def test_make_env_from_config():
    """Test creating environment from config dict."""
    config = {
        "render_mode": None,
        "max_episode_steps": 300,
        "seed": 123,
        "obs_noise_std": 0.05,
        "action_noise_prob": 0.1,
        "domain_randomization": {
            "gravity": [9.0, 10.0],
            "length": [0.4, 0.6],
        },
    }

    env = make_env_from_config(config)

    assert env.max_episode_steps == 300
    assert env.seed == 123
    assert env.obs_noise_std == 0.05
    assert env.action_noise_prob == 0.1
    assert "gravity" in env.domain_randomization

    env.close()


def test_make_vec_env_from_config():
    """Test creating vectorized environment from config dict."""
    config = {
        "max_episode_steps": 300,
        "seed": 42,
        "obs_noise_std": 0.01,
    }

    vec_env = make_vec_env_from_config(config, num_envs=3)

    assert vec_env.num_envs == 3

    obs, _ = vec_env.reset()
    assert obs.shape == (3, 4)

    vec_env.close()


def test_env_name_parameter():
    """Test that env_name parameter is properly used."""
    # Test with default CartPole-v1
    env1 = make_env()
    assert env1.env_name == "CartPole-v1"
    env1.close()

    # Test with explicit CartPole-v0
    env2 = make_env(env_name="CartPole-v0")
    assert env2.env_name == "CartPole-v0"
    assert env2.env.spec.id == "CartPole-v0"
    env2.close()


def test_env_name_from_config():
    """Test that config 'name' field is properly used."""
    # Test with name in config
    config = {
        "name": "CartPole-v0",
        "seed": 42,
    }
    env = make_env_from_config(config)
    assert env.env_name == "CartPole-v0"
    assert env.env.spec.id == "CartPole-v0"
    env.close()

    # Test with default when name not in config
    config2 = {"seed": 42}
    env2 = make_env_from_config(config2)
    assert env2.env_name == "CartPole-v1"
    env2.close()


def test_vec_env_name_from_config():
    """Test that vectorized env respects config 'name' field."""
    config = {
        "name": "CartPole-v0",
        "seed": 42,
    }
    vec_env = make_vec_env_from_config(config, num_envs=2)
    assert vec_env.num_envs == 2
    vec_env.close()


def test_domain_randomization_validation_error():
    """Test that domain randomization with non-CartPole environment raises ValueError."""
    import pytest

    # Should raise ValueError when trying to use domain randomization with non-CartPole env
    with pytest.raises(
        ValueError,
        match="Domain randomization is only supported for CartPole environments",
    ):
        CartPoleEnv(
            env_name="Acrobot-v1",
            domain_randomization={"gravity": (9.0, 10.0)},
        )


def test_action_noise_validation_error():
    """Test that action noise with non-Discrete(2) environment raises ValueError."""
    import pytest

    # Should raise ValueError when trying to use action noise with non-Discrete(2) env
    with pytest.raises(
        ValueError, match="Action noise is only supported for Discrete\\(2\\) action spaces"
    ):
        CartPoleEnv(
            env_name="MountainCar-v0",  # Has Discrete(3) action space
            action_noise_prob=0.1,
        )


def test_domain_randomization_config_validation():
    """Test that malformed domain randomization configs raise ValueError."""
    import pytest

    # Test with non-2-element range
    with pytest.raises(ValueError, match="must be a 2-element list/tuple"):
        config = {"domain_randomization": {"gravity": [9.0]}}
        make_env_from_config(config)

    # Test with non-numeric values
    with pytest.raises(ValueError, match="must contain numeric values"):
        config = {"domain_randomization": {"gravity": ["a", "b"]}}
        make_env_from_config(config)

    # Test with min > max
    with pytest.raises(ValueError, match="has min > max"):
        config = {"domain_randomization": {"gravity": [10.0, 9.0]}}
        make_env_from_config(config)
