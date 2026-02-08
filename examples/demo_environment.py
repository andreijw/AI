"""
Example script demonstrating the CartPole environment setup.

This script shows how to:
1. Create a basic environment
2. Create an environment with noise
3. Create an environment with domain randomization
4. Create vectorized environments
5. Load an environment from configuration
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.env import make_single_env, make_vec_env, make_env_from_config


def demo_basic_env():
    """Demonstrate basic environment usage."""
    print("=" * 60)
    print("Demo 1: Basic CartPole Environment")
    print("=" * 60)
    
    env = make_single_env()
    obs, info = env.reset(seed=42)
    
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"Total reward: {total_reward}")
    env.close()
    print()


def demo_noise_env():
    """Demonstrate environment with noise."""
    print("=" * 60)
    print("Demo 2: Environment with Observation and Action Noise")
    print("=" * 60)
    
    env = make_single_env(
        observation_noise_std=0.05,
        action_noise_std=0.1,
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation (with noise): {obs}")
    
    # Run a few steps
    for step in range(5):
        action = 0  # Always choose action 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1} - Action: 0 (may be flipped by noise), Obs: {obs[:2]}...")
        
        if terminated or truncated:
            break
    
    env.close()
    print()


def demo_domain_randomization():
    """Demonstrate environment with domain randomization."""
    print("=" * 60)
    print("Demo 3: Environment with Domain Randomization")
    print("=" * 60)
    
    domain_randomization = {
        'gravity': (8.0, 12.0),
        'pole_length': (0.4, 0.6),
        'cart_mass': (0.8, 1.2),
    }
    
    env = make_single_env(domain_randomization=domain_randomization)
    
    # Reset multiple times to see different physics parameters
    for episode in range(3):
        obs, info = env.reset(seed=42 + episode)
        unwrapped = env.unwrapped
        
        print(f"Episode {episode + 1}:")
        print(f"  Gravity: {unwrapped.gravity:.2f}")
        print(f"  Pole length: {unwrapped.length:.2f}")
        print(f"  Cart mass: {unwrapped.masscart:.2f}")
    
    env.close()
    print()


def demo_vectorized_env():
    """Demonstrate vectorized environments."""
    print("=" * 60)
    print("Demo 4: Vectorized Environments")
    print("=" * 60)
    
    num_envs = 4
    vec_env = make_vec_env(num_envs=num_envs)
    
    obs, info = vec_env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Number of environments: {num_envs}")
    
    # Run a few steps in parallel
    for step in range(3):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        
        print(f"Step {step + 1} - Rewards: {rewards}")
    
    vec_env.close()
    print()


def demo_config_env():
    """Demonstrate loading environment from configuration."""
    print("=" * 60)
    print("Demo 5: Environment from Configuration File")
    print("=" * 60)
    
    env = make_env_from_config('configs/env.yaml')
    
    obs, info = env.reset(seed=42)
    print(f"Environment created from configs/env.yaml")
    print(f"Initial observation: {obs}")
    
    # Run a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CartPole Environment Setup Demonstration")
    print("=" * 60 + "\n")
    
    demo_basic_env()
    demo_noise_env()
    demo_domain_randomization()
    demo_vectorized_env()
    demo_config_env()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
