"""CartPole environment wrapper for reinforcement learning."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """
    Wrapper around Gymnasium's CartPole environment.
    
    Provides a standardized interface for the training pipeline and
    adds logging, monitoring, and preprocessing capabilities.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
    ):
        """
        Initialize the CartPole environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode
        
        # Create the base environment
        self.env = gym.make(
            "CartPole-v1",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        
        if seed is not None:
            self.env.action_space.seed(seed)
            np.random.seed(seed)
        
        # Environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Episode tracking
        self._episode_steps = 0
        self._episode_reward = 0.0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for this episode
            
        Returns:
            Tuple of (observation, info dict)
        """
        self._episode_steps = 0
        self._episode_reward = 0.0
        
        obs, info = self.env.reset(seed=seed)
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0 or 1 for CartPole)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._episode_steps += 1
        self._episode_reward += reward
        
        # Add episode statistics to info
        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }
        
        return obs, reward, terminated, truncated, info
    
    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self.env.close()
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    @property
    def episode_steps(self) -> int:
        """Get current episode step count."""
        return self._episode_steps
    
    @property
    def episode_reward(self) -> float:
        """Get current episode cumulative reward."""
        return self._episode_reward
