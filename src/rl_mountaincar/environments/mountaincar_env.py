"""Mountain car environment wrapper for reinforcement learning."""

from __future__ import annotations

from rl_core.environments import BaseEnvWrapper


class MountainCarEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium's MountainCar-v0 and MountainCarContinuous-v0.

    MountainCarContinuous-v0 differs only in its action, one force in [-1, 1], its reward, -0.1
    times the force squared per step and +100 at the flag, and its 999-step limit.

    MountainCar-v0 observes the car's position and velocity, takes one of three pushes (left,
    none, right), and gives -1 per step, the last included, until the car reaches the flag (a
    termination) or 200 steps pass (a truncation). The car cannot drive straight up the hill, so
    an agent has to learn to rock back and forth. Seeding, observation noise and episode
    statistics come from :class:`rl_core.environments.BaseEnvWrapper`; actions pass through
    unchanged. Action noise and domain randomization are not supported.
    """

    def __init__(
        self,
        env_name: str = "MountainCar-v0",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
    ) -> None:
        """
        Initialize the mountain car environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment (default: "MountainCar-v0").
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Maximum steps per episode (default: 200).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
        """
        super().__init__(env_name, render_mode, max_episode_steps, seed, obs_noise_std)
        self._make_env()
