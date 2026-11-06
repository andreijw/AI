"""The behaviour every challenge's environment wrapper shares: seeding, observation noise and episode
statistics around a Gymnasium environment."""

import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

from rl_core.seeding import (
    WRAPPER_STREAM,
    check_generator_state,
    derive_rng,
    generator_state,
    restore_generator,
)
from rl_core.utils.config_values import check_bool, check_positive_int

from .observation import (
    OBSERVATION_DEFAULTS,
    FrameStack,
    differenced_space,
    frame_differences,
    mask_entries,
    masked_space,
    stacked_space,
)

# How to install Box2D here. Gymnasium's own hint (pip install "gymnasium[box2d]") pins
# box2d==2.3.10, which has no build for the Jetson's aarch64; box2d-py builds from source with swig
# (README, LunarLander).
BOX2D_INSTALL_HINT = (
    "{env_name} needs Box2D: run ./setup.sh, or install it by hand into the virtual environment "
    "with `pip install swig==4.5.1 'setuptools<85' 'wheel<0.49'` and then `PATH=<venv>/bin:$PATH "
    "pip install --no-build-isolation --no-binary box2d-py box2d-py==2.3.8` (README, LunarLander)"
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


@runtime_checkable
class SchedulableEnv(Protocol):
    """An environment whose noise and randomization settings can change between episodes."""

    def check_setting(self, name: str, value: Any) -> Any: ...

    def scheduled_settings(self) -> dict[str, Any]: ...

    def apply_settings(self, settings: Mapping[str, Any]) -> None: ...


def non_negative_setting(name: str, value: Any) -> float:
    """``value`` as a float, rejecting a negative or non-finite number (a noise level)."""
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return float(value)


def uses_box2d(env_name: str) -> bool:
    """Whether ``env_name`` is one of Gymnasium's Box2D environments, by its registered entry
    point. Reading the registry imports nothing, so this works where Box2D is not installed. An id
    the registry does not know (``gym.make`` also takes "module:Env-vN") is left to ``gym.make``."""
    try:
        spec = gym.spec(env_name)
    except gym.error.Error:
        return False
    return str(spec.entry_point).startswith("gymnasium.envs.box2d")


def _require_box2d_for(env_name: str) -> None:
    """Raise :data:`BOX2D_INSTALL_HINT` for ``env_name`` if it needs Box2D (:func:`uses_box2d`)
    and Box2D is not installed, before ``gym.make`` would fail with Gymnasium's own hint."""
    if uses_box2d(env_name) and importlib.util.find_spec("Box2D") is None:
        raise ImportError(BOX2D_INSTALL_HINT.format(env_name=env_name))


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
    - velocity masking and frame stacking, applied after the noise in that order, so a frame keeps
      the noise it was observed with and a masked entry is exactly zero
    - the episode's step count and reward, reported in ``info["episode"]`` when it ends

    A subclass lists the velocity entries of each environment id it wraps in
    :attr:`VELOCITY_INDICES`; ``mask_velocity`` is refused for an id it does not list.

    A subclass validates its own settings after ``super().__init__`` and then calls
    :meth:`_make_env`. It overrides :meth:`_prepare_action` for action noise and
    :meth:`_apply_domain_randomization` for per-episode physics; both default to doing nothing.

    The settings in :attr:`SCHEDULABLE_SETTINGS` can change between episodes
    (:meth:`apply_settings`), for a curriculum. A subclass that adds one lists it there and checks
    its value in :meth:`_checked_setting`, which its constructor uses too.
    """

    # Velocity entries of the observation, by environment id, in Gymnasium's layout.
    VELOCITY_INDICES: ClassVar[Mapping[str, tuple[int, ...]]] = MappingProxyType({})

    # The settings apply_settings may change, by constructor argument name.
    SCHEDULABLE_SETTINGS: ClassVar[tuple[str, ...]] = ("obs_noise_std",)

    # The physics parameters domain_randomization accepts, each name (aliases too) mapped to its
    # canonical one, which is the attribute it sets on the unwrapped Gymnasium environment.
    RANDOMIZABLE_PARAMETERS: ClassVar[Mapping[str, str]] = MappingProxyType({})

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
        *,
        frame_stack: int = 1,
        mask_velocity: bool = False,
        frame_deltas: bool = False,
    ) -> None:
        """
        Args:
            env_name: Gymnasium environment id.
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Steps after which an episode is truncated.
            seed: Seed applied on the first reset; later resets continue the sequence.
            obs_noise_std: Standard deviation of the Gaussian noise added to observations.
            frame_stack: How many observations, oldest first, make one (1, the default, is off).
            frame_deltas: Give the stack as its newest frame, then the differences between
                consecutive frames, newest first (needs ``frame_stack`` of 2 or more).
            mask_velocity: Zero the velocity entries, making the task partially observable.

        Raises:
            ValueError: If the seed is invalid, ``obs_noise_std`` is negative or not finite,
                ``frame_stack`` is not a positive integer, ``mask_velocity`` or ``frame_deltas`` is
                not a boolean, ``frame_deltas`` is set without a ``frame_stack`` of 2 or more, or
                ``mask_velocity`` is set for an environment id :attr:`VELOCITY_INDICES` does not
                list.
        """
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode

        # Created first so an invalid seed is rejected before any environment is opened.
        self._np_random = derive_rng(seed, WRAPPER_STREAM)
        # The constructor seed is applied on the first reset only; see reset().
        self._has_reset = False

        self.obs_noise_std = self.check_setting("obs_noise_std", obs_noise_std)

        self.frame_stack = check_positive_int("frame_stack", frame_stack)
        self.mask_velocity = check_bool("mask_velocity", mask_velocity)
        self._velocity_indices: tuple[int, ...] = ()
        if self.mask_velocity:
            if env_name not in self.VELOCITY_INDICES:
                known = ", ".join(sorted(self.VELOCITY_INDICES)) or "none"
                raise ValueError(
                    f"mask_velocity is not supported for {env_name}: {type(self).__name__} knows "
                    f"the velocity entries of {known}"
                )
            self._velocity_indices = self.VELOCITY_INDICES[env_name]
        self._frames = FrameStack(self.frame_stack) if self.frame_stack > 1 else None
        # A frame_stack below 2 is refused when the space is built (differenced_space).
        self.frame_deltas = check_bool("frame_deltas", frame_deltas)

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
        has_reset = np.asarray(state["env_has_reset"])
        if has_reset.dtype != np.bool_ or has_reset.shape != ():
            raise ValueError(f"env_has_reset must be one boolean, got {has_reset!r}")

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

    def _make_env(self, **make_kwargs: Any) -> None:
        """Create the Gymnasium environment and take its spaces and metadata.

        Called once every setting is checked, so it also records which schedulable settings are on:
        only those can change between episodes.

        Args:
            **make_kwargs: Extra keyword arguments for ``gym.make``, for an environment that takes
                constructor options beyond the render mode and the step limit.

        Raises:
            ImportError: If the environment is one of Gymnasium's Box2D ones and Box2D is not
                installed, naming how to install it here."""
        _require_box2d_for(self.env_name)
        self._scheduled = tuple(n for n in self.SCHEDULABLE_SETTINGS if getattr(self, n))
        self.env = gym.make(
            self.env_name,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps,
            **make_kwargs,
        )
        if self.seed is not None:
            self.env.action_space.seed(self.seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # The same dict, not a copy: required for VectorEnv compatibility, which writes into it.
        self.metadata = self.env.metadata
        try:
            if self._velocity_indices:
                self.observation_space = masked_space(
                    self.observation_space, self._velocity_indices
                )
            if self.frame_deltas:
                self.observation_space = differenced_space(self.observation_space, self.frame_stack)
            elif self._frames is not None:
                self.observation_space = stacked_space(self.observation_space, self.frame_stack)
        except ValueError:
            self.env.close()
            raise

    @classmethod
    def check_setting(cls, name: str, value: Any) -> Any:
        """
        ``value`` of the schedulable setting ``name``, checked and normalised as the constructor
        does (a randomization mapping under its canonical names, in the order given).

        Raises:
            ValueError: If ``name`` is not in :attr:`SCHEDULABLE_SETTINGS`, or ``value`` is invalid.
        """
        if name not in cls.SCHEDULABLE_SETTINGS:
            raise ValueError(
                f"{cls.__name__} cannot change {name} between episodes; it can change "
                f"{', '.join(cls.SCHEDULABLE_SETTINGS)}"
            )
        return cls._checked_setting(name, value)

    @classmethod
    def _checked_setting(cls, name: str, value: Any) -> Any:
        """The check of each schedulable setting; a subclass handles its own names first.

        The noise levels several wrappers share are checked here: ``obs_noise_std`` and
        ``action_noise_std`` are finite and non-negative, ``action_noise_prob`` lies in
        ``[0, 1]``; each is reached only by a wrapper that lists it."""
        if name in ("obs_noise_std", "action_noise_std"):
            return non_negative_setting(name, value)
        if name == "action_noise_prob":
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"action_noise_prob must be in the interval [0, 1], got {value!r}")
            return float(value)
        raise NotImplementedError(f"{cls.__name__} lists {name} but does not check it")

    def scheduled_settings(self) -> dict[str, Any]:
        """The settings :meth:`apply_settings` may change, those on when the environment was built,
        with their current values (copies)."""
        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in ((n, getattr(self, n)) for n in self._scheduled)
        }

    def apply_settings(self, settings: Mapping[str, Any]) -> None:
        """
        Change schedulable settings, all or none. Call it between episodes: noise takes effect at
        the next reset or step (the reset's observation is noised too), randomization at the next
        reset. It draws no random numbers.

        Only a setting on at construction can change, so every capability check made then (an
        action space that supports action noise, physics that can be randomized) still holds; a
        randomization keeps its parameters, in their construction order, and only their ranges
        move.

        Raises:
            ValueError: If a setting cannot be scheduled, was off at construction, or has an
                invalid value; nothing is changed.
        """
        checked = {}
        for name, value in settings.items():
            new = self.check_setting(name, value)
            if name not in self._scheduled:
                raise ValueError(
                    f"{name} was off when the environment was built, so it cannot change between "
                    f"episodes: build the environment with {name} on"
                )
            current = getattr(self, name)
            if isinstance(current, Mapping):
                if set(new) != set(current):
                    raise ValueError(
                        f"{name} must randomize the same parameters as at construction "
                        f"({', '.join(current)}), got {', '.join(new) or 'none'}"
                    )
                new = {key: new[key] for key in current}  # the construction order of draws
            checked[name] = new
        for name, value in checked.items():
            setattr(self, name, value)

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

    def _observe(self, obs: np.ndarray, first: bool) -> np.ndarray:
        """The agent's observation for the environment's ``obs``: noise, then the velocity mask,
        then the frame stack, which ``first`` (the episode's first observation) restarts, then its
        differences."""
        obs = self._add_observation_noise(obs)
        if self._velocity_indices:
            obs = mask_entries(obs, self._velocity_indices)
        if self._frames is not None:
            obs = self._frames.reset(obs) if first else self._frames.push(obs)
        if self.frame_deltas:
            obs = frame_differences(obs, self.frame_stack)
        return obs

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Domain randomization runs first, on the wrapper's generator, so the inner environment's
        reset sees the physics drawn for the new episode; the observation noise comes after.
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

        if seed is not None:
            self._np_random = derive_rng(seed, WRAPPER_STREAM)

        # Physics first, so a parameter the inner reset reads applies to the episode it was drawn
        # for; then the inner reset, then the observation of the new start state.
        self._apply_domain_randomization()
        obs, info = self.env.reset(seed=seed, options=options)
        self._has_reset = True
        obs = self._observe(obs, first=True)

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

        obs = self._observe(obs, first=False)

        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }

        return obs, float(reward), terminated, truncated, info

    def observation_settings(self) -> dict[str, bool | int]:
        """Every observation setting (``OBSERVATION_DEFAULTS``'s keys) with this wrapper's value,
        for an agent to record in its checkpoints."""
        return {name: getattr(self, name) for name in OBSERVATION_DEFAULTS}

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


def checked_range(
    param: str, min_val: Any, max_val: Any, *, positive: bool = False
) -> tuple[float, float]:
    """
    A domain randomization range ``[min_val, max_val]`` as floats.

    Args:
        positive: Also refuse a range reaching 0 or below (a physical length, mass or gravity).

    Raises:
        ValueError: Naming ``param``, if a bound is not a number, is not finite, or min > max,
            or with ``positive`` if min <= 0.
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
    if positive and min_val <= 0:
        raise ValueError(
            f"Physical parameter '{param}' must be strictly positive, got min: {min_val}"
        )
    return float(min_val), float(max_val)


@dataclass(frozen=True)
class ParameterBounds:
    """
    The values a randomized parameter may take: ``[low, high]``, and above zero with ``positive``
    (a physical length, mass or gravity).
    """

    low: float = -math.inf
    high: float = math.inf
    positive: bool = False


POSITIVE = ParameterBounds(positive=True)


def checked_randomization(
    randomization: Mapping[str, Any] | None,
    parameters: Mapping[str, str],
    bounds: Mapping[str, ParameterBounds],
) -> dict[str, tuple[float, float]]:
    """
    A domain randomization mapping as ``{canonical name: (min, max)}``, in the order given.

    Args:
        randomization: Parameter name to a 2-element ``[min, max]`` sequence; None or empty for
            none. Names are matched case-insensitively.
        parameters: Each accepted name, aliases too, mapped to its canonical one (a wrapper's
            :attr:`BaseEnvWrapper.RANDOMIZABLE_PARAMETERS`).
        bounds: Each canonical name's allowed values; a name missing here accepts any finite
            range.

    Raises:
        ValueError: If ``randomization`` is not a mapping, names an unknown parameter or one
            parameter twice, or a range is not two finite numbers with min <= max inside the
            parameter's bounds.
    """
    if not randomization:
        return {}
    if not isinstance(randomization, Mapping):
        raise ValueError("domain_randomization must be a mapping of parameter ranges.")

    validated: dict[str, tuple[float, float]] = {}
    for param, value in randomization.items():
        param_lower = param.lower()
        if param_lower not in parameters:
            raise ValueError(
                f"Unsupported domain randomization parameter '{param}'. "
                f"Supported parameters: {sorted(parameters)}."
            )
        canon_key = parameters[param_lower]
        if canon_key in validated:
            raise ValueError(
                f"Domain randomization parameter '{canon_key}' is given twice "
                f"(as '{param}' and an earlier name)"
            )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(
                f"Domain randomization range for '{param}' must be a 2-element sequence "
                f"[min, max], got: {value!r}"
            )
        allowed = bounds.get(canon_key, ParameterBounds())
        low, high = checked_range(param, *value, positive=allowed.positive)
        if low < allowed.low or high > allowed.high:
            raise ValueError(
                f"Domain randomization range for '{param}' must lie within "
                f"[{allowed.low:g}, {allowed.high:g}], got: [{low:g}, {high:g}]"
            )
        validated[canon_key] = (low, high)
    return validated
