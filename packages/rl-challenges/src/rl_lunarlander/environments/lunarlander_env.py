"""Lunar lander environment wrapper for reinforcement learning."""

from __future__ import annotations

from types import MappingProxyType
from typing import Unpack

from rl_core.environments import BaseEnvWrapper, ObservationOptions


class LunarLanderEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium's LunarLander-v3 and LunarLanderContinuous-v3 (Box2D).

    Both observe the lander's position, velocity, angle and angular velocity and whether each leg
    touches the ground (8 values). LunarLander-v3 takes one of four actions (no engine, left,
    main, right engine); LunarLanderContinuous-v3 two throttles in [-1, 1] (main engine, then the
    side engines). The reward is shaped: moving towards the pad slowly and upright gains, each
    leg contact gains 10, firing costs fuel, and the episode ends with +100 for coming to rest or
    -100 for a crash or flying off screen (a termination), or at the step limit (a truncation).
    Seeding, observation noise and episode statistics come from
    :class:`rl_core.environments.BaseEnvWrapper`; actions pass through unchanged. Action noise,
    domain randomization and Gymnasium's wind and gravity options are not supported.
    """

    VELOCITY_INDICES = MappingProxyType(
        {"LunarLander-v3": (2, 3, 5), "LunarLanderContinuous-v3": (2, 3, 5)}
    )

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        **observation: Unpack[ObservationOptions],
    ) -> None:
        """
        Initialize the lunar lander environment wrapper.

        Args:
            env_name: "LunarLander-v3" (discrete) or "LunarLanderContinuous-v3".
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Maximum steps per episode (default: 1000).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
            observation: The observation settings (``ObservationOptions``); see
                :class:`rl_core.environments.BaseEnvWrapper`.

        Raises:
            ImportError: If Box2D is not installed, naming how to install it here
                (:class:`rl_core.environments.BaseEnvWrapper` checks).
        """
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )
        self._make_env()
