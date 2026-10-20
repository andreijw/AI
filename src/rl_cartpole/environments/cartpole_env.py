"""CartPole environment wrapper for reinforcement learning."""

from collections.abc import Mapping, Sequence

import gymnasium as gym

from rl_core.environments import BaseEnvWrapper, checked_range


class CartPoleEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium environments with noise injection and domain randomization.

    **Primary Use Case**: This wrapper is designed for CartPole environments but can
    work with other Gymnasium environments for basic noise injection features.

    Seeding, observation noise and episode statistics come from
    :class:`rl_core.environments.BaseEnvWrapper`; this class adds the CartPole-specific parts.

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
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        action_noise_prob: float = 0.0,
        domain_randomization: Mapping[str, Sequence[float] | tuple[float, float]] | None = None,
    ):
        """
        Initialize the environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
                     Note: Non-CartPole environments only support observation noise
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                  so runs are reproducible and episodes differ from one another
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
        super().__init__(env_name, render_mode, max_episode_steps, seed, obs_noise_std)

        if not (0.0 <= action_noise_prob <= 1.0):
            raise ValueError(
                f"action_noise_prob must be in the interval [0, 1], got {action_noise_prob!r}"
            )
        self.action_noise_prob = action_noise_prob

        # Domain randomization parameters
        self.domain_randomization = self._validate_domain_randomization(domain_randomization)

        # Validate CartPole-specific features early
        is_cartpole = "cartpole" in env_name.lower()
        if domain_randomization and not is_cartpole:
            raise ValueError(
                f"Domain randomization is only supported for CartPole environments. "
                f"Environment '{env_name}' does not appear to be a CartPole variant. "
                f"If you need domain randomization, use a CartPole environment. "
                f"Otherwise, remove domain_randomization parameter."
            )

        self._make_env()

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

    def _validate_domain_randomization(
        self,
        domain_randomization: Mapping[str, Sequence[float] | tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Validate and normalize domain randomization configuration."""
        if not domain_randomization:
            return {}
        if not isinstance(domain_randomization, dict):
            raise ValueError("domain_randomization must be a dictionary of parameter ranges.")

        supported_params = {"gravity", "masscart", "masspole", "length"}
        validated: dict[str, tuple[float, float]] = {}
        for param, value in domain_randomization.items():
            if param not in supported_params:
                raise ValueError(
                    f"Unsupported domain randomization parameter '{param}'. "
                    f"Supported parameters: {sorted(supported_params)}."
                )

            if isinstance(value, list):
                value = tuple(value)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"Domain randomization range for '{param}' must be a 2-element list/tuple "
                    f"[min, max], got: {value!r}"
                )
            validated[param] = checked_range(param, *value)

        return validated

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

        # Randomize each requested physical parameter, always in this order
        for param in ("gravity", "masscart", "masspole", "length"):
            if param in self.domain_randomization:
                min_val, max_val = self.domain_randomization[param]
                setattr(base_env, param, self._np_random.uniform(min_val, max_val))

        # Update total mass (used in dynamics)
        base_env.total_mass = base_env.masspole + base_env.masscart  # type: ignore[attr-defined]
        base_env.polemass_length = base_env.masspole * base_env.length  # type: ignore[attr-defined]

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

    def _prepare_action(self, action: int) -> int:
        """The agent's action after action noise."""
        return self._apply_action_noise(action)
