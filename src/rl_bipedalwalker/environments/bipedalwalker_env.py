"""Bipedal walker environment wrapper for reinforcement learning."""

from __future__ import annotations

from typing import Unpack

from rl_core.environments import BaseEnvWrapper, ObservationOptions


class BipedalWalkerEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium's BipedalWalker-v3 (Box2D).

    It observes 24 values: the hull's angle, angular velocity and horizontal and vertical speed,
    each hip's and knee's angle and speed, each leg's ground contact, and ten lidar distances. It
    takes four torques in [-1, 1], one per hip and knee. Moving forward earns about 300 over the
    whole course and torque costs a little. The episode terminates with -100 when the hull touches
    the ground or the walker leaves the left edge, terminates on reaching the far end, and is
    truncated at the 1600-step limit.
    Seeding, observation noise and episode statistics come from
    :class:`rl_core.environments.BaseEnvWrapper`; actions pass through unchanged. Action noise,
    domain randomization, velocity masking (it lists no velocity entries) and Gymnasium's hardcore
    terrain are not supported.
    """

    def __init__(
        self,
        env_name: str = "BipedalWalker-v3",
        render_mode: str | None = None,
        max_episode_steps: int = 1600,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        **observation: Unpack[ObservationOptions],
    ) -> None:
        """
        Initialize the bipedal walker environment wrapper.

        Args:
            env_name: Gymnasium environment id ("BipedalWalker-v3").
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Maximum steps per episode (default: 1600, Gymnasium's).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
            observation: The observation settings (``ObservationOptions``); see
                :class:`rl_core.environments.BaseEnvWrapper`. ``mask_velocity`` is refused.

        Raises:
            ValueError: If a setting is invalid, or ``mask_velocity`` is set.
            ImportError: If Box2D is not installed, naming how to install it here
                (:class:`rl_core.environments.BaseEnvWrapper` checks).
        """
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )
        self._make_env()
