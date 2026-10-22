"""The files of a training checkpoint: the agent's and, beside it, the trainer's."""

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

AGENT_PREFIX = "agent_episode_"
TRAINER_PREFIX = "trainer_episode_"

_AGENT_NAME = re.compile(rf"^{AGENT_PREFIX}(\d+)\.(pt|npz)$")


def checkpoint_paths(directory: str, episode: int) -> tuple[str, str]:
    """The agent and trainer files of the checkpoint taken after ``episode`` training episodes."""
    return (
        os.path.join(directory, f"{AGENT_PREFIX}{episode}.pt"),
        os.path.join(directory, f"{TRAINER_PREFIX}{episode}.npz"),
    )


def trainer_state_path(agent_checkpoint: str) -> str:
    """
    The trainer file beside an agent checkpoint the trainer wrote.

    Raises:
        ValueError: If the file is not named ``agent_episode_<n>.pt`` (or ``.npz``).
    """
    directory, name = os.path.split(agent_checkpoint)
    match = _AGENT_NAME.match(name)
    if match is None:
        raise ValueError(
            f"Can only resume from a checkpoint the trainer wrote, named "
            f"{AGENT_PREFIX}<n>.pt; got {name!r}"
        )
    return os.path.join(directory, f"{TRAINER_PREFIX}{match.group(1)}.npz")


def write_atomically(path: str, write: Callable[[str], None]) -> None:
    """
    Have ``write`` write a temporary file beside ``path`` (same extension), then move it into
    place, so an interrupted save never leaves a truncated file under the real name.
    """
    directory, name = os.path.split(path)
    temporary = os.path.join(directory, f".tmp-{name}")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def file_sha256(path: str) -> str:
    """The SHA-256 of a file's bytes, in hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainerState:
    """
    What ``trainer_episode_<n>.npz`` holds: the run it belongs to (agent file and its hash, agent
    class, environment) and what the trainer needs to continue it. Any key not listed here is the
    training environment's generator state (``env_state``), saved only for an environment that
    can restore it.
    """

    agent_checkpoint: str
    agent_sha256: str
    agent_class: str
    env_id: str
    episode_rewards: list[float]
    episode_lengths: list[int]
    total_steps_trained: int
    update_metric_window: dict[str, list[float]]
    non_finite_metrics_seen: list[str]
    first_non_finite_episode: int | None = None
    eval_seed: int | None = None
    env_state: dict[str, np.ndarray] = field(default_factory=dict)

    def to_arrays(self) -> dict[str, Any]:
        """The state as ``.npz`` arrays."""
        arrays: dict[str, Any] = {
            "agent_checkpoint": np.array(self.agent_checkpoint),
            "agent_sha256": np.array(self.agent_sha256),
            "agent_class": np.array(self.agent_class),
            "env_id": np.array(self.env_id),
            "episode_rewards": np.array(self.episode_rewards, dtype=np.float64),
            "episode_lengths": np.array(self.episode_lengths, dtype=np.int64),
            "total_steps_trained": np.array(self.total_steps_trained),
            "update_metric_window": np.array(json.dumps(self.update_metric_window)),
            "non_finite_metrics_seen": np.array(json.dumps(sorted(self.non_finite_metrics_seen))),
        }
        if self.first_non_finite_episode is not None:
            arrays["first_non_finite_episode"] = np.array(self.first_non_finite_episode)
        if self.eval_seed is not None:
            arrays["eval_seed"] = np.array(self.eval_seed)
        clashing = sorted(arrays.keys() & self.env_state.keys())
        if clashing:
            raise ValueError(
                f"Environment state keys clash with the trainer's: {', '.join(clashing)}"
            )
        return {**arrays, **self.env_state}

    @classmethod
    def from_arrays(cls, data: Mapping[str, Any], agent_checkpoint: str) -> "TrainerState":
        """
        Read and check a trainer file saved beside ``agent_checkpoint``.

        Raises:
            ValueError: Naming the key, if one is missing or malformed, the histories disagree
                with each other, with the step count or with the episode in the file name, or the
                file belongs to another agent file (name or contents).
        """
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise ValueError(f"The trainer file is missing {', '.join(missing)}")
        name = os.path.basename(agent_checkpoint)
        state = cls(
            agent_checkpoint=_text(data, "agent_checkpoint"),
            agent_sha256=_text(data, "agent_sha256"),
            agent_class=_text(data, "agent_class"),
            env_id=_text(data, "env_id"),
            episode_rewards=[float(v) for v in _history(data, "episode_rewards", np.number)],
            episode_lengths=[int(v) for v in _history(data, "episode_lengths", np.integer)],
            total_steps_trained=_integer(data, "total_steps_trained"),
            update_metric_window=_metric_window(data),
            non_finite_metrics_seen=_names(data),
            first_non_finite_episode=_optional_integer(data, "first_non_finite_episode"),
            eval_seed=_optional_integer(data, "eval_seed"),
            env_state={k: np.asarray(data[k]) for k in data if k not in _KNOWN},
        )
        match = _AGENT_NAME.match(name)
        episodes = int(match.group(1)) if match else -1
        if state.agent_checkpoint != name:
            raise ValueError(f"agent_checkpoint is {state.agent_checkpoint!r}, not {name!r}")
        if not len(state.episode_rewards) == len(state.episode_lengths) == episodes:
            raise ValueError(
                f"episode_rewards and episode_lengths hold {len(state.episode_rewards)} and "
                f"{len(state.episode_lengths)} episodes; {name} was saved after {episodes}"
            )
        if state.eval_seed is not None and state.eval_seed < 0:
            raise ValueError(f"eval_seed must be non-negative, got {state.eval_seed}")
        first = state.first_non_finite_episode
        if first is not None and not 1 <= first <= episodes:
            raise ValueError(
                f"first_non_finite_episode {first} is outside episodes 1 to {episodes}"
            )
        if state.total_steps_trained != sum(state.episode_lengths):
            raise ValueError(
                f"total_steps_trained is {state.total_steps_trained}, the episode lengths sum to "
                f"{sum(state.episode_lengths)}"
            )
        if state.agent_sha256 != file_sha256(agent_checkpoint):
            raise ValueError(
                f"agent_sha256 does not match {name}: the agent file was replaced after this "
                "trainer file was written"
            )
        return state


_REQUIRED = (
    "agent_checkpoint",
    "agent_sha256",
    "agent_class",
    "env_id",
    "episode_rewards",
    "episode_lengths",
    "total_steps_trained",
    "update_metric_window",
    "non_finite_metrics_seen",
)
_KNOWN = (*_REQUIRED, "first_non_finite_episode", "eval_seed")


def _scalar(data: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must hold a single value, got shape {value.shape}")
    return value.item()


def _text(data: Mapping[str, Any], key: str) -> str:
    value = _scalar(data, key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text, got {value!r}")
    return value


def _integer(data: Mapping[str, Any], key: str) -> int:
    value = _scalar(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return value


def _optional_integer(data: Mapping[str, Any], key: str) -> int | None:
    return _integer(data, key) if key in data else None


def _history(data: Mapping[str, Any], key: str, kind: Any) -> np.ndarray:
    """A 1-D array of ``kind`` (``np.integer``, or ``np.number`` for any real number), not bool."""
    value = np.asarray(data[key])
    if value.ndim != 1:
        raise ValueError(f"{key} must be one value per episode, got shape {value.shape}")
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, kind) or value.dtype.kind == "c":
        raise ValueError(f"{key} must hold {kind.__name__} values, got dtype {value.dtype}")
    return value


def _json(data: Mapping[str, Any], key: str) -> Any:
    try:
        return json.loads(_text(data, key))
    except json.JSONDecodeError as error:
        raise ValueError(f"{key} is not JSON: {error}") from error


def _metric_window(data: Mapping[str, Any]) -> dict[str, list[float]]:
    window = _json(data, "update_metric_window")
    if not isinstance(window, dict) or not all(
        isinstance(name, str)
        and isinstance(values, list)
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
        for name, values in window.items()
    ):
        raise ValueError("update_metric_window must map metric names to lists of numbers")
    return {name: [float(v) for v in values] for name, values in window.items()}


def _names(data: Mapping[str, Any]) -> list[str]:
    names = _json(data, "non_finite_metrics_seen")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("non_finite_metrics_seen must be a list of metric names")
    return names
