"""Base agent interface for reinforcement learning."""

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, final

import numpy as np

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
        directories are created.

        Args:
            path: Destination file path.
        """
        save_path = self._resolve_save_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        parts = (
            self._checkpoint_arrays(),
            self._extra_checkpoint_state(),
            {key: generator_state(g) for key, g in self._generators().items()},
        )
        arrays: dict[str, Any] = {}
        for part in parts:
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

        Args:
            path: Source file path.
        """
        with np.load(self._resolve_load_path(path)) as data:
            generator_states = self._saved_generator_states(data)  # checked before any change
            self._check_extra_checkpoint_state(data)
            self._load_checkpoint_arrays(data)
            self._load_extra_checkpoint_state(data)
            for generator, source in generator_states.values():
                restore_generator(generator, data[source])
