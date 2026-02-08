"""
Tests for CartPole environment wrapper and factory functions.
"""

import pytest
import numpy as np
import gymnasium as gym
from src.env import CartPoleEnv, make_single_env, make_vec_env, make_env_from_config


class TestCartPoleEnv:
    """Tests for the CartPole environment wrapper."""
    
    def test_basic_environment_creation(self):
        """Test that we can create a basic wrapped environment."""
        base_env = gym.make("CartPole-v1")
        env = CartPoleEnv(base_env)
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_observation_noise(self):
        """Test that observation noise is applied correctly."""
        base_env = gym.make("CartPole-v1")
        env = CartPoleEnv(base_env, observation_noise_std=0.1)
        
        obs, _ = env.reset(seed=42)
        # With noise, observations should be slightly different each time
        # but still valid
        assert obs.shape == (4,)
        assert not np.any(np.isnan(obs))
        
        env.close()
    
    def test_action_noise(self):
        """Test that action noise can be applied."""
        base_env = gym.make("CartPole-v1")
        env = CartPoleEnv(base_env, action_noise_std=0.5)
        
        obs, _ = env.reset(seed=42)
        # Action noise should sometimes flip actions
        # This is probabilistic, so we just check it doesn't error
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4,)
        env.close()
    
    def test_domain_randomization(self):
        """Test that domain randomization applies correctly."""
        base_env = gym.make("CartPole-v1")
        domain_randomization = {
            'gravity': (9.0, 11.0),
            'pole_length': (0.4, 0.6),
            'cart_mass': (0.9, 1.1),
        }
        env = CartPoleEnv(base_env, domain_randomization=domain_randomization)
        
        # Reset and check that parameters were randomized
        obs, _ = env.reset(seed=42)
        unwrapped = env.unwrapped
        
        assert 9.0 <= unwrapped.gravity <= 11.0
        assert 0.4 <= unwrapped.length <= 0.6
        assert 0.9 <= unwrapped.masscart <= 1.1
        
        env.close()
    
    def test_combined_features(self):
        """Test environment with all features enabled."""
        base_env = gym.make("CartPole-v1")
        domain_randomization = {
            'gravity': (9.0, 11.0),
        }
        env = CartPoleEnv(
            base_env,
            observation_noise_std=0.01,
            action_noise_std=0.1,
            domain_randomization=domain_randomization,
        )
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4,)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()


class TestMakeEnv:
    """Tests for environment factory functions."""
    
    def test_make_single_env_basic(self):
        """Test creating a single environment."""
        env = make_single_env()
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
    
    def test_make_single_env_with_noise(self):
        """Test creating a single environment with noise."""
        env = make_single_env(
            observation_noise_std=0.01,
            action_noise_std=0.1,
        )
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        env.close()
    
    def test_make_single_env_with_randomization(self):
        """Test creating a single environment with domain randomization."""
        domain_randomization = {
            'gravity': (9.0, 11.0),
            'pole_length': (0.45, 0.55),
        }
        env = make_single_env(domain_randomization=domain_randomization)
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        
        env.close()
    
    def test_make_vec_env(self):
        """Test creating vectorized environments."""
        num_envs = 4
        vec_env = make_vec_env(num_envs=num_envs)
        
        obs, info = vec_env.reset()
        assert obs.shape == (num_envs, 4)
        
        actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        
        assert obs.shape == (num_envs, 4)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        
        vec_env.close()
    
    def test_make_vec_env_with_features(self):
        """Test creating vectorized environments with noise and randomization."""
        num_envs = 2
        domain_randomization = {
            'gravity': (9.5, 10.5),
        }
        vec_env = make_vec_env(
            num_envs=num_envs,
            observation_noise_std=0.01,
            action_noise_std=0.05,
            domain_randomization=domain_randomization,
        )
        
        obs, info = vec_env.reset()
        assert obs.shape == (num_envs, 4)
        
        vec_env.close()
    
    def test_make_env_from_config(self):
        """Test creating environment from config file."""
        config_path = 'configs/env.yaml'
        env = make_env_from_config(config_path)
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
