"""CartPole environment wrapper for reinforcement learning."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """
    Wrapper around Gymnasium's CartPole environment.

    Provides a standardized interface for the training pipeline and
    adds logging, monitoring, and preprocessing capabilities.

    Features:
    - Observation noise injection
    - Action noise injection
    - Domain randomization (gravity, pole length, cart mass)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
        obs_noise_std: float = 0.0,
        action_noise_prob: float = 0.0,
        domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the CartPole environment.

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            obs_noise_std: Standard deviation of Gaussian noise added to observations
            action_noise_prob: Probability of flipping the action (0.0 to 1.0)
            domain_randomization: Dictionary with parameter ranges for randomization.
                Supported keys: 'gravity', 'masscart', 'masspole', 'length'
                Values should be tuples of (min, max) for uniform sampling
                Example: {'gravity': (8.0, 12.0), 'length': (0.3, 0.7)}
        """
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode

        # Noise parameters
        self.obs_noise_std = obs_noise_std
        self.action_noise_prob = action_noise_prob

        # Domain randomization parameters
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
        self.metadata = self.env.metadata  # Required for VectorEnv compatibility

        # Episode tracking
        self._episode_steps = 0
        self._episode_reward = 0.0

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to environment parameters."""
        if not self.domain_randomization:
            return

        # Access the underlying environment (unwrap if needed)
        base_env = self.env.unwrapped

        # Randomize gravity
        if "gravity" in self.domain_randomization:
            min_g, max_g = self.domain_randomization["gravity"]
            base_env.gravity = np.random.uniform(min_g, max_g)

        # Randomize cart mass
        if "masscart" in self.domain_randomization:
            min_m, max_m = self.domain_randomization["masscart"]
            base_env.masscart = np.random.uniform(min_m, max_m)

        # Randomize pole mass
        if "masspole" in self.domain_randomization:
            min_m, max_m = self.domain_randomization["masspole"]
            base_env.masspole = np.random.uniform(min_m, max_m)

        # Randomize pole length
        if "length" in self.domain_randomization:
            min_l, max_l = self.domain_randomization["length"]
            base_env.length = np.random.uniform(min_l, max_l)

        # Update total mass (used in dynamics)
        base_env.total_mass = base_env.masspole + base_env.masscart
        base_env.polemass_length = base_env.masspole * base_env.length

    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to observations.

        Args:
            obs: Original observation

        Returns:
            Noisy observation
        """
        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, obs.shape)
            return obs + noise
        return obs

    def _apply_action_noise(self, action: int) -> int:
        """
        Apply action noise by randomly flipping the action.

        Args:
            action: Original action

        Returns:
            Potentially flipped action
        """
        if self.action_noise_prob > 0 and np.random.random() < self.action_noise_prob:
            # Flip the action (0 -> 1, 1 -> 0)
            return 1 - action
        return action

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional seed for this episode
            options: Optional dictionary of reset options (for compatibility with Gym API)

        Returns:
            Tuple of (observation, info dict)
        """
        self._episode_steps = 0
        self._episode_reward = 0.0

        obs, info = self.env.reset(seed=seed, options=options)

        # Apply domain randomization at the start of each episode
        self._apply_domain_randomization()

        # Apply observation noise
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
        # Apply action noise
        noisy_action = self._apply_action_noise(action)

        obs, reward, terminated, truncated, info = self.env.step(noisy_action)

        self._episode_steps += 1
        self._episode_reward += reward

        # Apply observation noise
        obs = self._add_observation_noise(obs)

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
