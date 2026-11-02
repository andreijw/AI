"""Base agent interface for reinforcement learning."""

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, final

import numpy as np

from rl_core.agents.config_values import flag
from rl_core.agents.observation_normalizer import RunningMeanStd
from rl_core.seeding import check_generator_state, generator_state, restore_generator

# What to tell someone whose config chooses a backend whose library is missing.
BACKEND_INSTALL_HINTS = {
    "torch": (
        "backend: torch needs PyTorch: pip install -e '.[torch]' (on the Jetson, add "
        "--index-url https://download.pytorch.org/whl/cu132 --extra-index-url "
        "https://pypi.org/simple; see docs/pytorch_jetson_timing.md)"
    ),
}

# Checkpoint key of a generator attribute: "rng_" and its name without leading underscores.
GENERATOR_KEY_PREFIX = "rng_"
# Checkpoint key of an observation setting the agent was built for: "observation_setting_" and its
# name. ("observation_" alone would take in the random agents' observation_dim.)
OBSERVATION_KEY_PREFIX = "observation_setting_"
# Checkpoint key of a running observation statistic (normalize_observations): "obs_norm_" and its name.
OBS_NORM_KEY_PREFIX = "obs_norm_"
# What each reserved checkpoint-key prefix holds; a subclass's own entries may not use them.
_RESERVED_KEY_PREFIXES = {
    OBSERVATION_KEY_PREFIX: "the recorded observation settings",
    OBS_NORM_KEY_PREFIX: "the observation normalisation statistics",
}


def _setting_value(name: str, value: Any) -> bool | int:
    """``value`` as a Python bool or int, keeping which of the two it is; ValueError otherwise."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    raise ValueError(f"Observation setting {name} must be a boolean or an integer, got {value!r}")


def _same_setting(a: bool | int | None, b: bool | int | None) -> bool:
    """Equal and of the same kind: ``True`` and ``1`` are different settings."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _setting_text(name: str, value: bool | int | None) -> str:
    if value is None:
        return f"no {name}"
    return f"{name}={str(value).lower() if isinstance(value, bool) else value}"


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.

    Defines the interface that all agent implementations must follow.
    This allows for easy swapping of different RL algorithms.

    A checkpoint (:meth:`save`) holds what :meth:`_checkpoint_arrays` and
    :meth:`_extra_checkpoint_state` return, plus the state of every ``np.random.Generator``
    attribute, so a loaded agent continues exactly as the saved one would have. A generator is
    stored under ``rng_<attribute name without leading underscores>``: renaming the attribute
    changes the checkpoint format. A generator held inside another object (a replay buffer) is
    that object's to save.
    """

    # Checkpoint keys an older format used for a generator attribute, read when the current key is
    # absent: {attribute name: key}.
    _LEGACY_GENERATOR_KEYS: ClassVar[dict[str, str]] = {}
    # Exploration generator, created by each subclass that samples through
    # :meth:`_select_action_from_probs` (AGENT_EXPLORATION_STREAM).
    _explore_rng: np.random.Generator
    # The library this class computes with: the value of the config's ``backend`` key it serves.
    BACKEND: ClassVar[str] = "numpy"
    # This agent in other backends: {backend: "module:Class"}, imported only when a config chooses
    # it, so a backend's library is needed only then. Each class declares its own; a subclass does
    # not inherit its parent's (a TD3Agent(DDPGAgent) is not built as TorchDDPGAgent).
    BACKENDS: ClassVar[dict[str, str]] = {}
    _observation_settings: Mapping[str, bool | int] | None = None
    # Whether ``normalize_observations`` may be set. An agent that keeps transitions in a replay
    # buffer would store inputs normalised by statistics that later change, so it says False.
    SUPPORTS_OBSERVATION_NORMALIZATION: ClassVar[bool] = True
    _observation_defaults: Mapping[str, bool | int] = MappingProxyType({})

    @final
    @classmethod
    def class_for_config(cls, config: Mapping[str, Any]) -> type["BaseAgent"]:
        """
        The class to build for ``config``: this one, or the version its ``backend`` key names.

        A config without the key asks for ``numpy``, the default backend. (Building a class
        directly with no key builds that class: see ``__init__``.)

        Raises:
            ValueError: If this agent has no version in the requested backend.
            ImportError: If the backend's library is not installed, naming how to install it.
        """
        backend = config.get("backend", "numpy")
        if backend == cls.BACKEND:
            return cls
        others: dict[str, str] = vars(cls).get("BACKENDS", {})
        if backend not in others:
            allowed = [cls.BACKEND, *others]
            raise ValueError(
                f"backend must be one of {allowed} for {cls.__name__}, got {backend!r}"
            )
        module_name, _, class_name = others[backend].partition(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != backend or backend not in BACKEND_INSTALL_HINTS:
                raise
            raise ImportError(BACKEND_INSTALL_HINTS[backend]) from error
        chosen = getattr(module, class_name)
        if not (isinstance(chosen, type) and issubclass(chosen, cls)) or backend != chosen.BACKEND:
            raise TypeError(f"{others[backend]} is not {cls.__name__} in the {backend} backend")
        return chosen

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]):
        """
        Initialize the agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            config: Configuration dictionary for the agent
        """
        backend = config.get("backend", self.BACKEND)  # no key: the class built is the choice
        if backend != self.BACKEND:
            raise ValueError(
                f"{type(self).__name__} is the {self.BACKEND} backend; the config asks for "
                f"{backend!r}: build agents with class_for_config(config), as challenges do"
            )
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self._obs_normalizer: RunningMeanStd | None = None
        if flag(config, "normalize_observations", False):
            if not self.SUPPORTS_OBSERVATION_NORMALIZATION:
                raise ValueError(
                    f"normalize_observations is not supported by {type(self).__name__}: its replay "
                    "buffer would keep observations normalised by statistics that later change"
                )
            self._obs_normalizer = RunningMeanStd(observation_dim)

    @property
    def observation_normalizer(self) -> RunningMeanStd | None:
        """The running statistics ``normalize_observations`` keeps, or None with it off."""
        return self._obs_normalizer

    @final
    def normalize_observation(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        The input the agent acts and learns on for an environment's ``observation``.

        With ``normalize_observations`` off this is ``observation`` itself. With it on, a training
        observation first updates the running statistics; the result is then a new float64 array,
        ``(observation - mean) / sqrt(var + 1e-8)`` clipped to ±10. Evaluation
        (``training=False``) only reads the statistics. ``Trainer`` calls this once for every
        observation an environment returns; a caller that drives ``select_action`` itself (the
        examples, the agent tests) must do the same for a normalising agent. Callers must not
        change the returned array.
        """
        if self._obs_normalizer is None:
            return observation
        if training:
            self._obs_normalizer.update(observation)
        return self._obs_normalizer.normalize(observation)

    # ------------------------------------------------------------------
    # Shared utility helpers
    # ------------------------------------------------------------------

    def _select_action_from_probs(self, probs: np.ndarray, training: bool) -> int:
        """
        Sample or greedily select an action from a probability distribution, sampling with the
        subclass's ``_explore_rng``.

        Args:
            probs:    Action probability distribution.
            training: If True, sample stochastically; otherwise act greedily.

        Returns:
            Selected action index.
        """
        if training:
            return int(self._explore_rng.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    @property
    def observation_settings(self) -> Mapping[str, bool | int] | None:
        """The observation settings of the environment this agent was built for (``frame_stack``,
        ``mask_velocity``), read-only, as :meth:`record_observation_settings` recorded them; None
        for an agent built directly."""
        return self._observation_settings

    @final
    def record_observation_settings(
        self, settings: Mapping[str, Any], defaults: Mapping[str, Any]
    ) -> None:
        """
        Record the observation settings of the environment this agent is built for, so
        :meth:`save` stores them and :meth:`load` refuses a checkpoint saved with others.

        Args:
            settings: Each setting's value, by name (``{"frame_stack": 4, "mask_velocity": True}``).
            defaults: Each setting's default, by name. :meth:`load` reads a setting a checkpoint
                lacks (one added after it was saved) as its default.

        Raises:
            ValueError: If a name is not a string, a value or default is not a boolean or an
                integer, or a setting has no default. Nothing is recorded then.
        """
        checked: dict[str, dict[str, bool | int]] = {"settings": {}, "defaults": {}}
        for part, values in (("settings", settings), ("defaults", defaults)):
            for name, value in values.items():
                if not isinstance(name, str):
                    raise ValueError(f"Observation setting names must be strings, got {name!r}")
                checked[part][name] = _setting_value(name, value)
        unknown = sorted(checked["settings"].keys() - checked["defaults"].keys())
        if unknown:
            raise ValueError(f"Observation setting(s) without a default: {', '.join(unknown)}")
        self._observation_settings = MappingProxyType(checked["settings"])
        self._observation_defaults = MappingProxyType(checked["defaults"])

    def _observation_entries(self) -> dict[str, np.ndarray]:
        """The recorded observation settings as checkpoint entries; none without a record."""
        settings = self.observation_settings or {}
        return {f"{OBSERVATION_KEY_PREFIX}{name}": np.array(v) for name, v in settings.items()}

    def _check_observation_settings(self, data: Mapping[str, Any], path: str) -> None:
        """
        Raise ValueError if the checkpoint ``data`` was saved with other observation settings
        than this agent records. Nothing is checked when either side has no record.
        """
        saved_keys = [key for key in data if key.startswith(OBSERVATION_KEY_PREFIX)]
        if self.observation_settings is None or not saved_keys:
            return
        saved: dict[str, bool | int] = {}
        for key in saved_keys:
            value = np.asarray(data[key])
            if value.ndim != 0:
                raise ValueError(f"Checkpoint entry {key} must hold one value, got {value!r}")
            saved[key[len(OBSERVATION_KEY_PREFIX) :]] = _setting_value(key, value.item())
        current = dict(self.observation_settings)
        # A setting the checkpoint lacks was added after it was saved: it was at its default.
        was = {n: saved.get(n, self._observation_defaults.get(n)) for n in saved.keys() | current}
        differing = [n for n in sorted(was) if not _same_setting(was[n], current.get(n))]
        if differing:
            was_text = ", ".join(_setting_text(n, was[n]) for n in differing)
            now = ", ".join(_setting_text(n, current.get(n)) for n in differing)
            raise ValueError(
                f"{path} was saved with {was_text}; this environment has {now}. Play it back or "
                "resume it with the environment section it was trained with"
            )

    def _saved_observation_statistics(
        self, data: Mapping[str, Any], path: str
    ) -> dict[str, Any] | None:
        """
        The checkpoint's observation statistics, checked, or None if it holds none.

        Raises:
            ValueError: If the checkpoint holds statistics and this agent does not normalise, or the
                other way round, or the statistics are malformed (naming the entry).
        """
        names = ("mean", "var", "count")
        held = [n for n in names if f"{OBS_NORM_KEY_PREFIX}{n}" in data]
        on = self._obs_normalizer is not None
        if bool(held) != on:
            raise ValueError(
                f"{path} was saved with agent.config.normalize_observations: "
                f"{str(bool(held)).lower()}; this agent has normalize_observations: "
                f"{str(on).lower()}. Load it into an agent built with the same setting"
            )
        if not on:
            return None
        missing = [f"{OBS_NORM_KEY_PREFIX}{n}" for n in names if n not in held]
        if missing:
            raise ValueError(f"{path} lacks {', '.join(missing)}")
        statistics = {n: data[f"{OBS_NORM_KEY_PREFIX}{n}"] for n in names}
        try:
            RunningMeanStd.check_state(statistics, self.observation_dim)
        except ValueError as error:
            raise ValueError(f"{path}: {OBS_NORM_KEY_PREFIX}{error}") from None
        return statistics

    def _extra_checkpoint_state(self) -> dict[str, np.ndarray]:
        """Arrays a subclass needs saved beyond the weights and optimizer state."""
        return {}

    def _check_extra_checkpoint_state(self, data: Any) -> None:
        """Raise ValueError if :meth:`_load_extra_checkpoint_state` would; called before anything
        is loaded, so a malformed checkpoint changes nothing. Nothing to check by default."""
        return None

    def _load_extra_checkpoint_state(self, data: Any) -> None:
        """Restore what :meth:`_extra_checkpoint_state` saved; ``data`` maps names to arrays."""
        return None  # nothing extra by default

    def _generators(self) -> dict[str, np.random.Generator]:
        """
        Every generator attribute, by checkpoint key, in sorted key order.

        Raises:
            ValueError: If two attributes map to one key (``_rng`` and ``rng``).
        """
        generators: dict[str, np.random.Generator] = {}
        for name, value in vars(self).items():
            if isinstance(value, np.random.Generator):
                key = f"{GENERATOR_KEY_PREFIX}{name.lstrip('_')}"
                if key in generators:
                    raise ValueError(f"Two generator attributes would be saved as {key}")
                generators[key] = value
        return dict(sorted(generators.items()))

    def _saved_generator_states(self, data: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
        """
        The checkpoint's state for each generator it holds, by checkpoint key (the current key, or
        the legacy one): ``{key: (generator, key read)}``; one it does not hold is left out.

        Raises:
            ValueError: Naming the key, if a state is not a generator state of the same kind.
        """
        legacy = {
            f"{GENERATOR_KEY_PREFIX}{name.lstrip('_')}": old
            for name, old in self._LEGACY_GENERATOR_KEYS.items()
        }
        found: dict[str, tuple[Any, str]] = {}
        for key, generator in self._generators().items():
            source = key if key in data else legacy.get(key, key)
            if source not in data:
                continue
            check_generator_state(generator, data[source], source)
            found[key] = (generator, source)
        return found

    @staticmethod
    def _resolve_save_path(path: str) -> str:
        """
        Validate and resolve the checkpoint save path.

        Raises ``ValueError`` for unsupported extensions. Appends ``.npz``
        when *path* has no extension.

        Args:
            path: Requested destination path.

        Returns:
            Resolved path to write to.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )
        return path if extension else f"{path}.npz"

    @staticmethod
    def _resolve_load_path(path: str) -> str:
        """
        Validate and resolve the checkpoint load path.

        Raises ``ValueError`` for unsupported extensions and
        ``FileNotFoundError`` when the file cannot be located.

        Args:
            path: Requested source path.

        Returns:
            Resolved path to read from.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {"", ".npz", ".pt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{extension}'. "
                "Checkpoint files must use '.pt', '.npz', or no extension."
            )
        has_exact_path = os.path.exists(path)
        if extension:
            # Backward-compatible fallback for older checkpoints written as "<name>.pt.npz".
            legacy_npz_path = f"{path}.npz"
            if has_exact_path or extension == ".npz":
                return path
            if os.path.exists(legacy_npz_path):
                return legacy_npz_path
            raise FileNotFoundError(
                f"Checkpoint not found at '{path}' (or legacy fallback '{legacy_npz_path}')."
            )
        # No extension: prefer the bare path if it exists, then try appending ".npz".
        if has_exact_path:
            return path
        npz_path = f"{path}.npz"
        if os.path.exists(npz_path):
            return npz_path
        raise FileNotFoundError(f"Checkpoint not found at '{path}' (or fallback '{npz_path}').")

    @staticmethod
    def _validate_dims(observation_dim: int, action_dim: int) -> None:
        """
        Validate the observation and action dimensions. Hidden widths are checked where they are
        read from the config (``rl_core.agents.config_values``).

        Args:
            observation_dim: Observation space dimensionality.
            action_dim:      Number of discrete actions.

        Raises:
            ValueError: If either dimension is non-positive.
        """
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be a positive integer, got {observation_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> Any:
        """
        Select an action given an observation.

        Args:
            observation: Current observation from environment
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action (discrete int or continuous float/vector)
        """

    def end_episode(self) -> None:
        """
        Called by the Trainer after every training episode's :meth:`update`, before anything is
        saved; never for an evaluation episode. Nothing by default: an agent resets per-episode
        state here (DDPG's exploration noise).
        """
        return None  # nothing to reset by default

    def enable_mid_episode_updates(self, update_every: int) -> None:
        """
        Called by the Trainer when ``training.update_every`` is set, before training: from then on
        :meth:`update` gets ``update_every`` steps of an episode at a time (fewer for its last
        part), and :meth:`end_episode` still follows its last part. An agent that learns from a
        replay buffer accepts (see ``mid_episode``), and so does the continuous Actor-Critic with
        ``n_steps`` equal to ``update_every``. Only an agent whose updates have a fixed length, as
        that one, needs to read ``update_every``; the others ignore it.

        Raises:
            ValueError: By default: this agent trains on whole episodes or rollouts.
        """
        raise ValueError(
            f"{type(self).__name__} trains on whole episodes: update_every needs an agent that "
            "learns from a replay buffer (DQN, DDPG, TD3) or the continuous Actor-Critic with "
            "n_steps"
        )

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """
        Update the agent's policy based on collected experience.

        Args:
            batch: Dictionary or trajectory list containing experience data

        Returns:
            Dictionary of training metrics
        """

    @abstractmethod
    def _checkpoint_arrays(self) -> dict[str, Any]:
        """The arrays :meth:`save` writes beyond generators and extras: weights, optimizer state,
        counters, buffers."""

    @abstractmethod
    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """Restore what :meth:`_checkpoint_arrays` saved, validating what must match."""

    @final
    def save(self, path: str) -> None:
        """
        Save the agent to a compressed NumPy ``.npz`` checkpoint.

        Supported extensions are ``.pt`` and ``.npz``; without one, ``.npz`` is appended. Missing
        directories are created. An agent with recorded observation settings saves each as
        ``observation_setting_<name>``; one that normalises observations saves its running
        statistics as ``obs_norm_mean``, ``obs_norm_var`` and ``obs_norm_count``.

        Args:
            path: Destination file path.

        Raises:
            ValueError: If two parts of the checkpoint use one key, or a part uses a key under the
                ``observation_setting_`` or ``obs_norm_`` prefix, which hold the recorded settings
                and the normalisation statistics.
        """
        save_path = self._resolve_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        parts = (
            self._checkpoint_arrays(),
            self._extra_checkpoint_state(),
            {key: generator_state(g) for key, g in self._generators().items()},
        )
        for prefix, holds in _RESERVED_KEY_PREFIXES.items():
            reserved = sorted(k for part in parts for k in part if k.startswith(prefix))
            if reserved:
                raise ValueError(
                    f"Checkpoint keys {', '.join(reserved)} use the prefix {prefix!r}, which holds "
                    f"{holds}"
                )
        statistics = {} if self._obs_normalizer is None else self._obs_normalizer.state()
        own = {f"{OBS_NORM_KEY_PREFIX}{name}": value for name, value in statistics.items()}
        arrays: dict[str, Any] = {}
        for part in (*parts, self._observation_entries(), own):
            clashing = sorted(arrays.keys() & part.keys())
            if clashing:
                raise ValueError(f"Checkpoint keys saved twice: {', '.join(clashing)}")
            arrays.update(part)
        with open(save_path, "wb") as f:
            np.savez_compressed(f, **arrays)

    @final
    def load(self, path: str) -> None:
        """
        Load a checkpoint written by :meth:`save`, or by an older version.

        Without an extension and no such file, ``.npz`` is appended; a ``.pt`` path also finds a
        legacy ``<name>.pt.npz``. Generator states are checked before anything is assigned; the
        agent's own arrays are checked as its ``_load_checkpoint_arrays`` documents. A generator the
        checkpoint does not hold keeps its current state.

        Before anything else, the observation settings the checkpoint records are compared with the
        ones this agent records (:meth:`record_observation_settings`); a setting the checkpoint
        lacks counts as its default. A checkpoint or an agent without a record skips the check.
        The checkpoint holds normalisation statistics exactly when this agent normalises
        observations; they are checked with the rest and restored last.

        Args:
            path: Source file path.

        Raises:
            ValueError: If the checkpoint was saved with other observation settings, naming each
                one with its saved and current value, or a saved setting is not one boolean or
                integer; if it was saved with the other ``normalize_observations``, or its
                statistics are incomplete or malformed (naming the entry). Nothing is loaded then.
        """
        load_path = self._resolve_load_path(path)
        with np.load(load_path) as data:
            self._check_observation_settings(data, load_path)
            statistics = self._saved_observation_statistics(data, load_path)
            generator_states = self._saved_generator_states(data)  # checked before any change
            self._check_extra_checkpoint_state(data)
            self._load_checkpoint_arrays(data)
            self._load_extra_checkpoint_state(data)
            if self._obs_normalizer is not None and statistics is not None:
                self._obs_normalizer.load_state(statistics)
            for generator, source in generator_states.values():
                restore_generator(generator, data[source])
