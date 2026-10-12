"""
Example script demonstrating basic usage of the decoupled RL framework.

This script shows how to:
1. Run a discrete control episode (CartPole-v1 with RandomAgent)
2. Run a continuous control episode (Pendulum-v1 with ContinuousRandomAgent)
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import ContinuousRandomAgent, RandomAgent
from rl_pendulum.environments import PendulumEnv


def run_cartpole_demo():
    """Run a simple discrete control episode on CartPole-v1."""
    print("=" * 60)
    print("Discrete Control Demo: CartPole-v1")
    print("=" * 60)

    env = CartPoleEnv(seed=42, max_episode_steps=200)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = RandomAgent(observation_dim=obs_dim, action_dim=action_dim, config={"seed": 42})

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False

    while not done and episode_steps < 50:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1

    print(f"CartPole Episode Finished: Steps={episode_steps}, Reward={episode_reward:.1f}")
    env.close()


def run_pendulum_demo():
    """Run a simple continuous control episode on Pendulum-v1."""
    print("\n" + "=" * 60)
    print("Continuous Control Demo: Pendulum-v1")
    print("=" * 60)

    env = PendulumEnv(seed=42, max_episode_steps=200)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = ContinuousRandomAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        config={"seed": 42, "action_low": -2.0, "action_high": 2.0},
    )

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False

    while not done and episode_steps < 50:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1

    print(f"Pendulum Episode Finished: Steps={episode_steps}, Reward={episode_reward:.1f}")
    env.close()


def main():
    """Run framework demonstrations."""
    run_cartpole_demo()
    run_pendulum_demo()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
