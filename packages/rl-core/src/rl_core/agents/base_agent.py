"""Base agent interface for reinforcement learning."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar, final

import numpy as np

# The checkpoint key prefixes and the shape check are importable from here, where they were defined
# before the checkpoint code moved to rl_core.agents.checkpointing.
from rl_core.agents.checkpointing import GENERATOR_KEY_PREFIX as GENERATOR_KEY_PREFIX
from rl_core.agents.checkpointing import OBS_NORM_KEY_PREFIX as OBS_NORM_KEY_PREFIX
from rl_core.agents.checkpointing import OBSERVATION_KEY_PREFIX as OBSERVATION_KEY_PREFIX
from rl_core.agents.checkpointing import CheckpointMixin
from rl_core.agents.checkpointing import check_checkpoint_shapes as check_checkpoint_shapes
from rl_core.agents.inference import InferencePolicy
from rl_core.agents.observation_normalizer import RunningMeanStd
from rl_core.utils.config_values import flag

# What to tell someone whose config chooses a backend whose library is missing.
BACKEND_INSTALL_HINTS = {
    "torch": (
        "backend: torch needs PyTorch: install rl-core's torch extra (on the Jetson from PyTorch's own "
        "index, --index-url https://download.pytorch.org/whl/cu132 --extra-index-url "
        "https://pypi.org/simple; see the repository's docs/pytorch_jetson_timing.md)"
    ),
}


class BaseAgent(CheckpointMixin, ABC):
    """
    Abstract base class for RL agents.

    Defines the interface that all agent implementations must follow.
    This allows for easy swapping of different RL algorithms.

    The checkpoint (:meth:`save`, :meth:`load` and their hooks) is :class:`CheckpointMixin`'s,
    in ``rl_core.agents.checkpointing``.
    """

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
        self._obs_normalizer: RunningMeanStd | None = None
        if flag(config, "normalize_observations", False):
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

    @staticmethod
    def _validate_dims(observation_dim: int, action_dim: int) -> None:
        """
        Validate the observation and action dimensions. Hidden widths are checked where they are
        read from the config (``rl_core.utils.config_values``).

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
        ``n_steps`` equal to ``update_every``, and PPO with several training environments
        (``rollout.RolloutWindows``). Only an agent whose updates have a fixed length, as those,
        needs to read ``update_every``; the others ignore it.

        Raises:
            ValueError: By default: this agent trains on whole episodes or rollouts.
        """
        raise ValueError(
            f"{type(self).__name__} trains on whole episodes: update_every needs an agent that "
            "learns from a replay buffer (DQN, DDPG, TD3), the continuous Actor-Critic with "
            "n_steps, or PPO with several environments"
        )

    def enable_parallel_envs(self, num_envs: int, update_every: int | None = None) -> None:
        """
        Called by the Trainer before training when ``training.num_envs`` is above 1, before
        :meth:`enable_mid_episode_updates`, with both settings.

        Without ``update_every``, :meth:`update` then gets each environment's finished episode as
        its own batch, in the order the episodes finish, while the other environments' episodes
        are still being played: REINFORCE, both Actor-Critics and the random agents accept. With
        it, :meth:`update_segments` gets every environment's steps of each window of
        ``update_every`` vector steps in one call: the continuous Actor-Critic (one A2C step) and
        PPO (whole windows per rollout) accept. An agent that learns from a replay buffer refuses
        until it is designed for it.

        Raises:
            ValueError: By default.
        """
        raise ValueError(
            f"{type(self).__name__} does not train from several environments yet: num_envs above "
            "1 is for REINFORCE, the Actor-Critics, PPO (with update_every) and the random agents"
        )

    def update_segments(self, segments: Sequence[Any]) -> list[dict[str, float]]:
        """
        Train on one window of several environments' steps (``training.num_envs`` with
        ``update_every``): ``segments`` are batches as :meth:`update` takes them, each a run of one
        environment's steps that ends where its episode or the window ended, in environment
        order. By default :meth:`update` on each in turn.

        Returns:
            The metrics of each update made, in order.
        """
        return [self.update(segment) for segment in segments]

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """
        Update the agent's policy based on collected experience.

        Args:
            batch: Dictionary or trajectory list containing experience data

        Returns:
            Dictionary of training metrics
        """

    @final
    def inference_policy(self) -> InferencePolicy:
        """
        The greedy policy (``select_action(..., training=False)``) as plain data, for exporting:
        the agent's network (:meth:`_inference_network`) with its observation normalisation, if
        on, and its recorded observation settings. The arrays are copies, so later training does
        not change it.

        Raises:
            ValueError: If the agent has no network to export (the random agents).
        """
        network = self._inference_network()
        settings = dict(self.observation_settings or {})
        statistics = self._obs_normalizer
        if statistics is None:
            return replace(network, observation_settings=settings, agent_name=type(self).__name__)
        return replace(
            network,
            obs_mean=statistics.mean,
            obs_scale=statistics.scale,
            observation_settings=settings,
            agent_name=type(self).__name__,
        )

    def _inference_network(self) -> InferencePolicy:
        """
        The layers and head of the greedy policy, without normalisation or settings, which
        :meth:`inference_policy` adds.

        Raises:
            ValueError: By default: an agent without a network has nothing to export.
        """
        raise ValueError(f"{type(self).__name__} has no network to export")
