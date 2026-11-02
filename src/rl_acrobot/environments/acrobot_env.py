"""Acrobot swing-up environment wrapper for reinforcement learning."""

from __future__ import annotations

from types import MappingProxyType
from typing import Unpack

from rl_core.environments import BaseEnvWrapper, ObservationOptions


class AcrobotEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium Acrobot environments.

    Acrobot-v1 observes the cosine and sine of both joint angles and both joint velocities, takes
    one of three torques (-1, 0, +1) on the second joint, and gives -1 per step until the tip
    swings above the line (a termination) or 500 steps pass (a truncation). Seeding, observation
    noise and episode statistics come from :class:`rl_core.environments.BaseEnvWrapper`; actions
    pass through unchanged. Action noise and domain randomization are not supported.
    """

    VELOCITY_INDICES = MappingProxyType({"Acrobot-v1": (4, 5)})

    def __init__(
        self,
        env_name: str = "Acrobot-v1",
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        **observation: Unpack[ObservationOptions],
    ) -> None:
        """
        Initialize the Acrobot environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment (default: "Acrobot-v1").
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Maximum steps per episode (default: 500).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
            observation: The observation settings (``ObservationOptions``); see
                :class:`rl_core.environments.BaseEnvWrapper`.
        """
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )
        self._make_env()
