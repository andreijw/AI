"""Lunar lander environment wrapper for reinforcement learning."""

from __future__ import annotations

import importlib.util

from rl_core.environments import BaseEnvWrapper

# Gymnasium's own hint (pip install "gymnasium[box2d]") pins box2d==2.3.10, which has no build for
# the Jetson's aarch64; box2d-py builds from source with swig (README, LunarLander).
BOX2D_INSTALL_HINT = (
    "LunarLander needs Box2D: run ./setup.sh, or install it by hand into the virtual environment "
    "with `pip install swig==4.5.1 'setuptools<85' 'wheel<0.49'` and then `PATH=<venv>/bin:$PATH "
    "pip install --no-build-isolation --no-binary box2d-py box2d-py==2.3.5` (README, LunarLander)"
)


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

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
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

        Raises:
            ImportError: If Box2D is not installed, naming how to install it here.
        """
        super().__init__(env_name, render_mode, max_episode_steps, seed, obs_noise_std)
        if importlib.util.find_spec("Box2D") is None:
            raise ImportError(BOX2D_INSTALL_HINT)
        self._make_env()
