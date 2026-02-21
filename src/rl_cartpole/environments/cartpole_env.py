"""CartPole environment wrapper for reinforcement learning."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding


class CartPoleEnv:
    """
    Wrapper around Gymnasium environments with noise injection and domain randomization.

    **Primary Use Case**: This wrapper is designed for CartPole environments but can
    work with other Gymnasium environments for basic noise injection features.

    Provides a standardized interface for the training pipeline and
    adds logging, monitoring, and preprocessing capabilities.

    Feature Compatibility:
    - **Observation noise injection**: Works with any Gymnasium environment
    - **Action noise injection**: Requires Discrete(2) action space (e.g., CartPole, but not MountainCar)
    - **Domain randomization**: CartPole-specific only (requires gravity, masscart, masspole, length attributes)

    The class is named CartPoleEnv because it was designed for CartPole environments and
    contains CartPole-specific domain randomization. When using with non-CartPole environments,
    only observation noise should be used (action noise and domain randomization will raise errors).

    Raises:
        ValueError: If domain_randomization is used with non-CartPole environments
        ValueError: If action_noise_prob > 0 with non-Discrete(2) action spaces
    """

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
        obs_noise_std: float = 0.0,
        action_noise_prob: float = 0.0,
        domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
                     Note: Non-CartPole environments only support observation noise
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            obs_noise_std: Standard deviation of Gaussian noise added to observations
            action_noise_prob: Probability of flipping the action (0.0 to 1.0)
                              Note: Only works with Discrete(2) action spaces
            domain_randomization: Dictionary with parameter ranges for randomization.
                CartPole-specific. Supported keys: 'gravity', 'masscart', 'masspole', 'length'
                Values should be tuples of (min, max) for uniform sampling
                Example: {'gravity': (8.0, 12.0), 'length': (0.3, 0.7)}

        Raises:
            ValueError: If domain_randomization is enabled with non-CartPole environment
            ValueError: If action_noise_prob > 0 with non-Discrete(2) action space
        """
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode

        # Noise parameters
        self.obs_noise_std = obs_noise_std
        self.action_noise_prob = action_noise_prob

        # Domain randomization parameters
        self.domain_randomization = domain_randomization or {}

        # Validate CartPole-specific features early
        is_cartpole = "cartpole" in env_name.lower()
        if domain_randomization and not is_cartpole:
            raise ValueError(
                f"Domain randomization is only supported for CartPole environments. "
                f"Environment '{env_name}' does not appear to be a CartPole variant. "
                f"If you need domain randomization, use a CartPole environment. "
                f"Otherwise, remove domain_randomization parameter."
            )

        # Create the base environment
        self.env = gym.make(
            env_name,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        # Validate action noise compatibility early (after env creation)
        if action_noise_prob > 0:
            action_space = self.env.action_space
            if not isinstance(action_space, gym.spaces.Discrete) or action_space.n != 2:
                raise ValueError(
                    f"Action noise is only supported for Discrete(2) action spaces; "
                    f"got {type(action_space).__name__} with n={getattr(action_space, 'n', None)}. "
                    f"Environment '{env_name}' has an incompatible action space. "
                    f"Either use an environment with Discrete(2) actions or set action_noise_prob=0."
                )

        if seed is not None:
            self.env.action_space.seed(seed)

        # Environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata  # Required for VectorEnv compatibility

        # Episode tracking
        self._episode_steps = 0
        self._episode_reward = 0.0

        # Per-environment RNG for noise and domain randomization
        # Initialize with a default RNG; will be synced with env RNG on reset()
        self._np_random, _ = seeding.np_random(seed)

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to environment parameters.

        Note: This method is CartPole-specific and requires the wrapped environment
        to have CartPole attributes (gravity, masscart, masspole, length).

        Raises:
            ValueError: If domain_randomization is enabled but the environment
                       doesn't support CartPole-specific attributes.
        """
        if not self.domain_randomization:
            return

        # Access the underlying environment (unwrap if needed)
        base_env = self.env.unwrapped

        # Validate that the environment supports CartPole domain randomization
        required_attrs = [
            "gravity",
            "masscart",
            "masspole",
            "length",
            "total_mass",
            "polemass_length",
        ]
        missing_attrs = [attr for attr in required_attrs if not hasattr(base_env, attr)]
        if missing_attrs:
            raise ValueError(
                f"Domain randomization is only supported for CartPole environments. "
                f"The environment '{self.env_name}' is missing required attributes: {missing_attrs}. "
                f"Either use a CartPole environment or disable domain_randomization."
            )

        # Randomize gravity
        if "gravity" in self.domain_randomization:
            min_g, max_g = self.domain_randomization["gravity"]
            base_env.gravity = self._np_random.uniform(min_g, max_g)  # type: ignore[attr-defined]

        # Randomize cart mass
        if "masscart" in self.domain_randomization:
            min_m, max_m = self.domain_randomization["masscart"]
            base_env.masscart = self._np_random.uniform(min_m, max_m)  # type: ignore[attr-defined]

        # Randomize pole mass
        if "masspole" in self.domain_randomization:
            min_m, max_m = self.domain_randomization["masspole"]
            base_env.masspole = self._np_random.uniform(min_m, max_m)  # type: ignore[attr-defined]

        # Randomize pole length
        if "length" in self.domain_randomization:
            min_l, max_l = self.domain_randomization["length"]
            base_env.length = self._np_random.uniform(min_l, max_l)  # type: ignore[attr-defined]

        # Update total mass (used in dynamics)
        base_env.total_mass = base_env.masspole + base_env.masscart  # type: ignore[attr-defined]
        base_env.polemass_length = base_env.masspole * base_env.length  # type: ignore[attr-defined]

    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to observations.

        Args:
            obs: Original observation

        Returns:
            Noisy observation (preserving original dtype)
        """
        if self.obs_noise_std > 0:
            noise = self._np_random.normal(0, self.obs_noise_std, obs.shape)
            # Preserve the original dtype (e.g., float32)
            return (obs + noise).astype(obs.dtype)
        return obs

    def _apply_action_noise(self, action: int) -> int:
        """
        Apply action noise by randomly flipping the action.

        Note: This method only works for Discrete(2) action spaces (binary actions).
        Validation is performed in __init__ to fail fast.

        Args:
            action: Original action

        Returns:
            Potentially flipped action
        """
        if self.action_noise_prob <= 0:
            return action

        if self._np_random.random() < self.action_noise_prob:
            # Flip the action (0 -> 1, 1 -> 0) for Discrete(2) spaces
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

        # Determine the seed to use for this episode
        episode_seed = seed if seed is not None else self.seed

        obs, info = self.env.reset(seed=episode_seed, options=options)

        # Initialize or update the wrapper's RNG using a seed derived from this episode
        # This keeps the wrapper's randomness separate from the underlying env's RNG
        self._np_random, _ = seeding.np_random(episode_seed)

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
        self._episode_reward += float(reward)

        # Apply observation noise
        obs = self._add_observation_noise(obs)

        # Add episode statistics to info
        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }

        return obs, float(reward), terminated, truncated, info

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
