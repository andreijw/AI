"""
Curriculum scheduling: the training environment's noise and randomization move from easy start
settings to the target the config's ``environment`` section sets, one difficulty from 0 to 1.

A config sets it under ``environment.curriculum``::

    environment:
      obs_noise_std: 0.05                 # the target, which evaluation always plays
      domain_randomization: {length: [0.25, 1.0]}
      curriculum:
        mode: reward_threshold            # linear | step | reward_threshold
        start: {obs_noise_std: 0.0, domain_randomization: {length: [0.5, 0.5]}}
        stages: 4                         # step and reward_threshold
        episodes: 400                     # linear and step: episodes to reach the target
        threshold: 400.0                  # reward_threshold: the stage's training average to advance
        window: 20                        # reward_threshold: episodes in that average
        min_episodes_per_stage: 20        # reward_threshold: at least window

The scheduler knows no environment: :func:`curriculum_from_config` checks the section against one
(the settings it can change, and their values by its own rules) and builds it.
"""

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from ..environments import SchedulableEnv
from ..utils.config_values import check_positive_int

MODES = ("linear", "step", "reward_threshold")

# The pacing keys each mode needs; every one is required, any other is refused.
_PACING = {
    "linear": ("episodes",),
    "step": ("episodes", "stages"),
    "reward_threshold": ("stages", "threshold", "window", "min_episodes_per_stage"),
}
_ALL_PACING = ("episodes", "stages", "threshold", "window", "min_episodes_per_stage")


def interpolate(start: Any, target: Any, difficulty: float) -> Any:
    """
    ``(1 - d) * start + d * target``, exact at d = 0 and d = 1. A randomization mapping of
    ``(low, high)`` ranges is interpolated end by end, keyed and ordered as ``target``.
    """
    if isinstance(target, Mapping):
        return {
            key: (
                interpolate(start[key][0], low, difficulty),
                interpolate(start[key][1], high, difficulty),
            )
            for key, (low, high) in target.items()
        }
    return (1.0 - difficulty) * start + difficulty * target


class CurriculumScheduler:
    """
    The difficulty of each training episode and the settings it gives the training environment.

    ``linear`` sets d = episodes done / ``episodes``, capped at 1. ``step`` climbs ``stages``
    equal stages over ``episodes``. ``reward_threshold`` climbs one stage (of ``stages``) when the
    average training reward over the last ``window`` episodes played at the current stage reaches
    ``threshold``, and at least ``min_episodes_per_stage`` have been played there. The difficulty
    never falls, and the scheduler draws no random numbers.
    """

    def __init__(self, section: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        """
        Args:
            section: The ``environment.curriculum`` section, its ``start`` values already in the
                environment's canonical form (see :func:`curriculum_from_config`).
            target: The schedulable settings the environment was built with
                (``scheduled_settings()``).

        Raises:
            ValueError: If the section is malformed, or ``start`` sets a setting or randomizes a
                parameter the target leaves off.
        """
        if not isinstance(section, Mapping):
            raise ValueError(f"environment.curriculum must be a mapping, got {section!r}")
        mode = section.get("mode")
        if mode not in MODES:
            raise ValueError(f"curriculum.mode must be one of {', '.join(MODES)}, got {mode!r}")
        self.mode: str = mode
        for key in _ALL_PACING:
            if key in section and key not in _PACING[mode]:
                raise ValueError(f"curriculum.{key} is not used by mode {mode}")
        unknown = sorted(set(section) - {"mode", "start", *_ALL_PACING})
        if unknown:
            raise ValueError(f"curriculum has unknown key(s) {', '.join(unknown)}")
        for key in _PACING[mode]:
            if section.get(key) is None:
                raise ValueError(f"curriculum.{key} is required by mode {mode}")

        self.episodes = _optional_count(section, "episodes")
        self.stages = _optional_count(section, "stages")
        self.window = _optional_count(section, "window")
        self.min_episodes_per_stage = _optional_count(section, "min_episodes_per_stage")
        self.threshold: float | None = None
        if mode == "reward_threshold":
            threshold = section["threshold"]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int | float)
                or not math.isfinite(threshold)
            ):
                raise ValueError(f"curriculum.threshold must be a finite number, got {threshold!r}")
            self.threshold = float(threshold)
            assert self.window is not None and self.min_episodes_per_stage is not None
            if self.min_episodes_per_stage < self.window:
                raise ValueError(
                    f"curriculum.min_episodes_per_stage ({self.min_episodes_per_stage}) must be "
                    f"at least curriculum.window ({self.window})"
                )

        self.target = {name: _copied(value) for name, value in target.items()}
        self.start = self._full_start(section.get("start"))
        self._restart()

    def _full_start(self, start: Any) -> dict[str, Any]:
        """``start`` for every target setting: those it leaves out, and randomization parameters
        it does not name, start at their target."""
        if not isinstance(start, Mapping) or not start:
            raise ValueError(
                f"curriculum.start must be a mapping of at least one setting, got {start!r}"
            )
        for name, value in start.items():
            if name not in self.target:
                raise ValueError(
                    f"curriculum.start sets {name}, which the environment section leaves off: the "
                    "target is the environment section, so set it there"
                )
            if isinstance(self.target[name], Mapping):
                if not isinstance(value, Mapping):
                    raise ValueError(f"curriculum.start.{name} must be a mapping, got {value!r}")
                for param, bounds in value.items():
                    if param not in self.target[name]:
                        raise ValueError(
                            f"curriculum.start randomizes {param}, which the environment section "
                            "does not"
                        )
                    if not (isinstance(bounds, tuple | list) and len(bounds) == 2):
                        raise ValueError(
                            f"Domain randomization range for '{param}' must be [min, max], "
                            f"got {bounds!r}"
                        )
        full = {}
        for name, target in self.target.items():
            value = start.get(name, target)
            if isinstance(target, Mapping):
                value = {key: tuple(value.get(key, bounds)) for key, bounds in target.items()}
            full[name] = value
        return full

    def _restart(self) -> None:
        self._played: dict[str, float | int] | None = None
        self._episodes_done = 0
        self._stage = 0
        self._stage_rewards: list[float] = []

    @property
    def difficulty(self) -> float:
        """The difficulty of the next training episode: 0 at the start, 1 at the target."""
        if self.mode == "linear":
            assert self.episodes is not None
            return min(self._episodes_done, self.episodes) / self.episodes
        assert self.stages is not None
        return self._stage / self.stages

    @property
    def stage(self) -> int | None:
        """The current stage, from 0 to ``stages``; None for ``linear``, which has none."""
        return None if self.mode == "linear" else self._stage

    def settings(self) -> dict[str, Any]:
        """The settings for the next training episode, every target setting interpolated by the
        difficulty (one at its target throughout stays there)."""
        d = self.difficulty
        return {name: interpolate(self.start[name], t, d) for name, t in self.target.items()}

    def played_metrics(self) -> dict[str, float | int] | None:
        """:meth:`metrics` as they were for the last episode recorded (None before one): what it
        trained at. :meth:`replay` rebuilds them too."""
        return None if self._played is None else dict(self._played)

    def metrics(self) -> dict[str, float | int]:
        """The difficulty, and the stage for a staged mode, as the trainer's metrics name them."""
        fields: dict[str, float | int] = {"curriculum_difficulty": self.difficulty}
        if self.stage is not None:
            fields["curriculum_stage"] = self.stage
        return fields

    def record(self, reward: float) -> bool:
        """
        Count one finished training episode and its reward.

        Returns:
            Whether the stage advanced (always False for ``linear``).
        """
        self._played = self.metrics()
        self._episodes_done += 1
        stage = self._stage
        if self.mode == "step":
            assert self.episodes is not None and self.stages is not None
            self._stage = min(self.stages, self.stages * self._episodes_done // self.episodes)
        elif self.mode == "reward_threshold":
            assert self.stages is not None and self.window is not None
            assert self.min_episodes_per_stage is not None and self.threshold is not None
            if self._stage < self.stages:
                self._stage_rewards.append(float(reward))
                recent = self._stage_rewards[-self.window :]
                if (
                    len(self._stage_rewards) >= self.min_episodes_per_stage
                    and sum(recent) / len(recent) >= self.threshold
                ):
                    self._stage += 1
                    self._stage_rewards = []
        return self._stage != stage

    def replay(self, rewards: Iterable[float]) -> None:
        """Start again and record ``rewards``: a resumed run's scheduler reaches the state the
        stopped run's had, from the training rewards its checkpoint holds."""
        self._restart()
        for reward in rewards:
            self.record(reward)

    def spec(self) -> str:
        """The curriculum as JSON (mode, pacing, start and target), saved with a checkpoint so a
        resume with a different one is refused."""
        pacing = {key: getattr(self, key) for key in _PACING[self.mode]}
        return json.dumps({"mode": self.mode, **pacing, "start": self.start, "target": self.target})


def curriculum_from_config(env_section: Mapping[str, Any], env: Any) -> CurriculumScheduler | None:
    """
    The scheduler a config's ``environment.curriculum`` asks for, for the training ``env`` built
    from the same section, or None when the section has no curriculum.

    The target is ``env.scheduled_settings()``. Each ``start`` value is checked and normalised by
    the environment's own rule (``check_setting``: a randomization under its canonical names).

    Raises:
        ValueError: If ``env`` cannot change its settings between episodes, or the section or a
            start value is invalid.
    """
    section = env_section.get("curriculum")
    if not section:
        return None
    require_schedulable(env)
    if isinstance(section, Mapping) and isinstance(section.get("start"), Mapping):
        start = {name: env.check_setting(name, value) for name, value in section["start"].items()}
        section = {**section, "start": start}
    return CurriculumScheduler(section, env.scheduled_settings())


def require_schedulable(env: Any) -> SchedulableEnv:
    """``env``, if it can change its settings between episodes (a :class:`SchedulableEnv`).

    Raises:
        ValueError: If it cannot, so it cannot follow a curriculum.
    """
    if not isinstance(env, SchedulableEnv):
        raise ValueError(
            f"{type(env).__name__} cannot change its settings between episodes, so it cannot "
            "follow a curriculum"
        )
    return env


def _optional_count(section: Mapping[str, Any], key: str) -> int | None:
    value = section.get(key)
    return None if value is None else check_positive_int(f"curriculum.{key}", value)


def _copied(value: Any) -> Any:
    return {k: tuple(v) for k, v in value.items()} if isinstance(value, Mapping) else value
