"""
CartPole environment wrapper with noise and domain randomization support.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple


class CartPoleEnv(gym.Wrapper):
    """
    Wrapper for Gymnasium CartPole environment with support for:
    - Observation noise
    - Action noise
    - Domain randomization (gravity, pole length, cart mass)
    """
    
    def __init__(
        self,
        env: gym.Env,
        observation_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the CartPole environment wrapper.
        
        Args:
            env: Base Gymnasium CartPole environment
            observation_noise_std: Standard deviation for observation noise (Gaussian)
            action_noise_std: Probability of flipping discrete actions (0.0-1.0)
            domain_randomization: Dictionary with keys 'gravity', 'pole_length', 'cart_mass'
                                 Each value is a tuple (min, max) for uniform sampling
        """
        super().__init__(env)
        self.observation_noise_std = observation_noise_std
        self.action_noise_std = action_noise_std
        self.domain_randomization = domain_randomization or {}
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and apply domain randomization if enabled.
        
        Returns:
            observation: Initial observation (potentially with noise)
            info: Additional information dictionary
        """
        # Apply domain randomization before reset
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        obs, info = self.env.reset(**kwargs)
        
        # Apply observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
            
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation (potentially with noise)
            reward: Reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        # Apply action noise (for discrete actions, we could randomly flip)
        # For CartPole (discrete action space), we add noise as probability of flipping
        if self.action_noise_std > 0:
            action = self._add_action_noise(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
            
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
