"""
Example script demonstrating basic usage of the RL CartPole framework.

This script shows how to:
1. Create an environment
2. Create an agent
3. Run a simple episode
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_cartpole.agents import RandomAgent
from rl_cartpole.environments import CartPoleEnv


def main():
    """Run a simple episode with a random agent."""
    print("=" * 60)
    print("RL CartPole Example")
    print("=" * 60)

    # Create environment
    print("\n1. Creating CartPole environment...")
    env = CartPoleEnv(seed=42, max_episode_steps=500)
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Create agent
    print("\n2. Creating Random agent...")
    agent = RandomAgent(
        observation_dim=env.observation_space.shape[0], action_dim=env.action_space.n, config={}
    )

    # Run episode
    print("\n3. Running episode...")
    obs, _ = env.reset()
    print(f"   Initial observation: {obs}")

    episode_reward = 0
    episode_steps = 0
    done = False

    while not done and episode_steps < 100:
        # Select action
        action = agent.select_action(obs)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_steps += 1

        # Print every 10 steps
        if episode_steps % 10 == 0:
            print(f"   Step {episode_steps}: reward={episode_reward:.1f}")

    # Episode summary
    print("\n4. Episode complete!")
    print(f"   Total steps: {episode_steps}")
    print(f"   Total reward: {episode_reward}")
    print(f"   Average reward per step: {episode_reward / episode_steps:.2f}")

    # Cleanup
    env.close()
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
