"""The behaviour every challenge's environment wrapper shares: seeding, observation noise and episode
statistics around a Gymnasium environment."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

from rl_core.seeding import (
    WRAPPER_STREAM,
    check_generator_state,
    derive_rng,
    generator_state,
    restore_generator,
)

# Checkpoint keys of an environment's generators: the wrapper's noise stream and Gymnasium's
# start-state generator.
_GENERATOR_KEYS = ("env_wrapper_rng", "env_start_rng")


@runtime_checkable
class ResumableEnv(Protocol):
    """An environment whose random state at an episode boundary can be saved and restored."""

    def rng_state(self) -> dict[str, np.ndarray]: ...

    def check_rng_state(self, state: Mapping[str, Any]) -> None: ...

    def set_rng_state(self, state: Mapping[str, Any]) -> None: ...


class BaseEnvWrapper:
    """
    Base for a challenge's environment wrapper.

    It holds a Gymnasium environment in ``self.env`` rather than being a ``gymnasium.Wrapper``:
    ``wrap_env`` swaps in a Gymnasium wrapper (such as a video recorder) around it. The base owns
    what every challenge does the same way:

    - the constructor seed, applied on the first reset only, so that a seeded run gives a
      reproducible series of different episodes; a seed passed to ``reset`` restarts the series
    - one generator for the wrapper's own noise, on its own stream so it never repeats the numbers
      the environment draws its start states from
    - Gaussian observation noise that keeps the observation's dtype
    - the episode's step count and reward, reported in ``info["episode"]`` when it ends

    A subclass validates its own settings after ``super().__init__`` and then calls
    :meth:`_make_env`. It overrides :meth:`_prepare_action` for action noise and
    :meth:`_apply_domain_randomization` for per-episode physics; both default to doing nothing.
    """

    env: gym.Env[Any, Any]
    observation_space: gym.Space[Any]
    action_space: gym.Space[Any]
    metadata: dict[str, Any]

    def __init__(
        self,
        env_name: str,
        render_mode: str | None,
        max_episode_steps: int,
        seed: int | None,
        obs_noise_std: float,
    ) -> None:
        """
        Args:
            env_name: Gymnasium environment id.
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Steps after which an episode is truncated.
            seed: Seed applied on the first reset; later resets continue the sequence.
            obs_noise_std: Standard deviation of the Gaussian noise added to observations.

        Raises:
            ValueError: If the seed is invalid, or ``obs_noise_std`` is negative or not finite.
        """
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode

        # Created first so an invalid seed is rejected before any environment is opened.
        self._np_random = derive_rng(seed, WRAPPER_STREAM)
        # The constructor seed is applied on the first reset only; see reset().
        self._has_reset = False

        if not np.isfinite(obs_noise_std) or obs_noise_std < 0.0:
            raise ValueError(
                f"obs_noise_std must be a finite, non-negative number, got {obs_noise_std!r}"
            )
        self.obs_noise_std = float(obs_noise_std)

        self._episode_steps = 0
        self._episode_reward = 0.0

    def _generators(self) -> tuple[np.random.Generator, np.random.Generator]:
        """The wrapper's noise generator and Gymnasium's start-state one, in ``_GENERATOR_KEYS``
        order."""
        start_rng: np.random.Generator = self.env.unwrapped.np_random
        return self._np_random, start_rng

    def rng_state(self) -> dict[str, np.ndarray]:
        """
        The environment's random state between episodes: both generators and whether the first
        reset (which applies the constructor seed) has happened. Physics is not state here: domain
        randomization draws it anew at every reset. Read through any wrapper ``wrap_env`` added.
        """
        state = {
            k: generator_state(g) for k, g in zip(_GENERATOR_KEYS, self._generators(), strict=True)
        }
        state["env_has_reset"] = np.array(self._has_reset)
        return state

    def check_rng_state(self, state: Mapping[str, Any]) -> None:
        """
        Raise the ValueError :meth:`set_rng_state` would, changing nothing.

        Raises:
            ValueError: If a key is missing or holds another kind of generator state.
        """
        missing = [k for k in (*_GENERATOR_KEYS, "env_has_reset") if k not in state]
        if missing:
            raise ValueError(f"Environment state is missing {', '.join(missing)}")
        for key, generator in zip(_GENERATOR_KEYS, self._generators(), strict=True):
            check_generator_state(generator, state[key], key)

    def set_rng_state(self, state: Mapping[str, Any]) -> None:
        """
        Continue from a state :meth:`rng_state` returned, in place.

        Raises:
            ValueError: If a key is missing or holds another kind of generator state; nothing is
                changed then.
        """
        self.check_rng_state(state)
        for key, generator in zip(_GENERATOR_KEYS, self._generators(), strict=True):
            restore_generator(generator, state[key])
        self._has_reset = bool(state["env_has_reset"])

    def _make_env(self) -> None:
        """Create the Gymnasium environment and take its spaces and metadata."""
        self.env = gym.make(
            self.env_name,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps,
        )
        if self.seed is not None:
            self.env.action_space.seed(self.seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # The same dict, not a copy: required for VectorEnv compatibility, which writes into it.
        self.metadata = self.env.metadata

    def _prepare_action(self, action: Any) -> Any:
        """The action to send to the environment for the agent's ``action``; unchanged by default."""
        return action

    def _apply_domain_randomization(self) -> None:
        """Randomize the environment's physics at the start of an episode; nothing by default."""

    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to an observation.

        Args:
            obs: Original observation

        Returns:
            Noisy observation (preserving original dtype)
        """
        if self.obs_noise_std > 0:
            noise = self._np_random.normal(0, self.obs_noise_std, obs.shape)
            return (obs + noise).astype(obs.dtype)
        return obs

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        The constructor seed is applied on the first reset only. Later resets without a seed
        continue the same random sequences, so a seeded environment gives a reproducible series
        of different episodes. Passing a seed restarts that series from the given seed.

        Args:
            seed: Optional seed that restarts the start-state and noise sequences
            options: Optional dictionary of reset options (for compatibility with Gym API)

        Returns:
            Tuple of (observation, info dict)
        """
        self._episode_steps = 0
        self._episode_reward = 0.0

        if seed is None and not self._has_reset:
            seed = self.seed

        obs, info = self.env.reset(seed=seed, options=options)
        self._has_reset = True

        if seed is not None:
            self._np_random = derive_rng(seed, WRAPPER_STREAM)

        # Physics first, then the noisy observation of the new start state.
        self._apply_domain_randomization()
        obs = self._add_observation_noise(obs)

        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The agent's action, passed through :meth:`_prepare_action`

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(self._prepare_action(action))

        self._episode_steps += 1
        self._episode_reward += float(reward)

        obs = self._add_observation_noise(obs)

        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }

        return obs, float(reward), terminated, truncated, info

    def wrap_env(self, wrapper: Any) -> None:
        """Inject a Gymnasium wrapper around the inner environment.

        This is the intended way to apply wrappers (e.g. RecordVideo) without breaking the
        wrapper's interface. The wrapper must accept the current ``self.env`` as its first
        constructor argument and has already been fully constructed before being passed here. It
        must wrap the current ``self.env``: the constructor seed is applied on the first reset
        only, so a different environment installed after that reset would never be seeded.

        Args:
            wrapper: A fully-constructed Gymnasium wrapper that wraps ``self.env``.

        Example::

            from gymnasium.wrappers import RecordVideo
            record_wrapper = RecordVideo(env.env, video_folder="./videos", ...)
            env.wrap_env(record_wrapper)
        """
        self.env = wrapper

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self.env.close()

    def render(self) -> Any:
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


def checked_range(param: str, min_val: Any, max_val: Any) -> tuple[float, float]:
    """
    A domain randomization range ``[min_val, max_val]`` as floats.

    Raises:
        ValueError: Naming ``param``, if a bound is not a number, is not finite, or min > max.
    """
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise ValueError(
            f"Domain randomization range for '{param}' must contain numeric values, "
            f"got: [{min_val!r}, {max_val!r}]"
        )
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        raise ValueError(
            f"Domain randomization range for '{param}' must use finite values, "
            f"got: [{min_val!r}, {max_val!r}]"
        )
    if min_val > max_val:
        raise ValueError(
            f"Domain randomization range for '{param}' has min > max: [{min_val}, {max_val}]"
        )
    return float(min_val), float(max_val)
