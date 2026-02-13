"""Example script demonstrating environment features."""

import numpy as np

from rl_cartpole.environments import (
    make_env,
    make_vec_env,
    make_env_from_config,
)


def demo_basic_env():
    """Demonstrate basic environment usage."""
    print("=" * 60)
    print("Demo 1: Basic Environment")
    print("=" * 60)
    
    env = make_env(seed=42)
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print()


def demo_observation_noise():
    """Demonstrate observation noise."""
    print("=" * 60)
    print("Demo 2: Observation Noise")
    print("=" * 60)
    
    env_clean = make_env(seed=42, obs_noise_std=0.0)
    env_noisy = make_env(seed=42, obs_noise_std=0.1)
    
    obs_clean, _ = env_clean.reset(seed=123)
    obs_noisy, _ = env_noisy.reset(seed=123)
    
    print(f"Clean observation:  {obs_clean}")
    print(f"Noisy observation:  {obs_noisy}")
    print(f"Difference:         {obs_noisy - obs_clean}")
    
    env_clean.close()
    env_noisy.close()
    print()


def demo_action_noise():
    """Demonstrate action noise."""
    print("=" * 60)
    print("Demo 3: Action Noise")
    print("=" * 60)
    
    # With 50% action noise probability
    env = make_env(seed=42, action_noise_prob=0.5)
    env.reset()
    
    print("Testing action noise (50% flip probability):")
    actions = [0, 1, 0, 1, 0, 1]
    
    for action in actions:
        noisy_action = env._apply_action_noise(action)
        flipped = "FLIPPED" if action != noisy_action else "same"
        print(f"  Action {action} -> {noisy_action} ({flipped})")
    
    env.close()
    print()


def demo_domain_randomization():
    """Demonstrate domain randomization."""
    print("=" * 60)
    print("Demo 4: Domain Randomization")
    print("=" * 60)
    
    domain_rand = {
        'gravity': (9.0, 10.0),
        'masscart': (0.9, 1.1),
        'masspole': (0.09, 0.11),
        'length': (0.45, 0.55),
    }
    
    env = make_env(seed=None, domain_randomization=domain_rand)
    
    print("Running 3 episodes with domain randomization:")
    for episode in range(3):
        env.reset()
        base_env = env.env.unwrapped
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Gravity:     {base_env.gravity:.4f}")
        print(f"  Cart mass:   {base_env.masscart:.4f}")
        print(f"  Pole mass:   {base_env.masspole:.4f}")
        print(f"  Pole length: {base_env.length:.4f}")
    
    env.close()
    print()


def demo_vectorized_env():
    """Demonstrate vectorized environments."""
    print("=" * 60)
    print("Demo 5: Vectorized Environments")
    print("=" * 60)
    
    # Create 4 parallel environments
    vec_env = make_vec_env(num_envs=4, seed=42)
    
    print(f"Number of parallel environments: {vec_env.num_envs}")
    
    obs, info = vec_env.reset()
    print(f"Observation shape: {obs.shape}")  # (4, 4) - 4 envs, 4 obs each
    
    # Take random actions in all environments
    actions = np.array([0, 1, 0, 1])
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    print(f"\nAfter step:")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Rewards: {rewards}")
    print(f"  Terminateds: {terminateds}")
    print(f"  Truncateds: {truncateds}")
    
    vec_env.close()
    print()


def demo_config_based():
    """Demonstrate configuration-based environment creation."""
    print("=" * 60)
    print("Demo 6: Configuration-Based Creation")
    print("=" * 60)
    
    config = {
        'max_episode_steps': 200,
        'seed': 42,
        'obs_noise_std': 0.02,
        'action_noise_prob': 0.1,
        'domain_randomization': {
            'gravity': [9.5, 10.5],
        }
    }
    
    env = make_env_from_config(config)
    
    print(f"Environment created from config:")
    print(f"  Max episode steps: {env.max_episode_steps}")
    print(f"  Seed: {env.seed}")
    print(f"  Observation noise std: {env.obs_noise_std}")
    print(f"  Action noise prob: {env.action_noise_prob}")
    print(f"  Domain randomization: {env.domain_randomization}")
    
    env.close()
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CartPole Environment Features Demonstration")
    print("=" * 60 + "\n")
    
    demo_basic_env()
    demo_observation_noise()
    demo_action_noise()
    demo_domain_randomization()
    demo_vectorized_env()
    demo_config_based()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
