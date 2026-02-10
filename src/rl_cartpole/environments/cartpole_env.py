"""CartPole environment wrapper for reinforcement learning."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """
    Wrapper around Gymnasium's CartPole environment.
    
    Provides a standardized interface for the training pipeline and
    adds logging, monitoring, preprocessing, and augmentation capabilities.
    Supports observation noise, action noise, and domain randomization.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
        observation_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the CartPole environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            observation_noise_std: Standard deviation for observation noise (Gaussian)
            action_noise_std: Probability of flipping discrete actions (0.0-1.0)
            domain_randomization: Dictionary with keys 'gravity', 'pole_length', 'cart_mass'
                                 Each value is a tuple (min, max) for uniform sampling
        """
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode
        self.observation_noise_std = observation_noise_std
        self.action_noise_std = action_noise_std
        self.domain_randomization = domain_randomization or {}
        
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
        
        # Apply domain randomization before reset
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        obs, info = self.env.reset(seed=seed)
        
        # Apply observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0 or 1 for CartPole)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action noise (for discrete actions, randomly flip)
        if self.action_noise_std > 0:
            action = self._add_action_noise(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._episode_steps += 1
        self._episode_reward += reward
        
        # Apply observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
        
        # Add episode statistics to info
        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _add_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to observation.
        
        Args:
            observation: Original observation
            
        Returns:
            Noisy observation
        """
        noise = np.random.normal(0, self.observation_noise_std, size=observation.shape)
        return observation + noise
    
    def _add_action_noise(self, action: int) -> int:
        """
        Add noise to action. For discrete actions, randomly flip with probability
        based on noise level.
        
        Args:
            action: Original action
            
        Returns:
            Potentially flipped action
        """
        # Interpret noise_std as probability of flipping the action
        flip_prob = min(self.action_noise_std, 1.0)
        if np.random.random() < flip_prob:
            # Flip action (for CartPole: 0 -> 1, 1 -> 0)
            return 1 - action
        return action
    
    def _apply_domain_randomization(self):
        """
        Apply domain randomization to environment parameters.
        Randomizes gravity, pole length, and cart mass within specified ranges.
        """
        # Access the unwrapped environment to modify physics parameters
        unwrapped = self.env.unwrapped
        
        if 'gravity' in self.domain_randomization:
            min_g, max_g = self.domain_randomization['gravity']
            unwrapped.gravity = np.random.uniform(min_g, max_g)
            
        if 'pole_length' in self.domain_randomization:
            min_len, max_len = self.domain_randomization['pole_length']
            unwrapped.length = np.random.uniform(min_len, max_len)
            unwrapped.polemass_length = unwrapped.masspole * unwrapped.length  # Update pole mass * length
            
        if 'cart_mass' in self.domain_randomization:
            min_mass, max_mass = self.domain_randomization['cart_mass']
            unwrapped.masscart = np.random.uniform(min_mass, max_mass)
            unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart  # Update total mass
    
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
