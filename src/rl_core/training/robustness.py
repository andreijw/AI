"""
The robustness benchmark: a trained agent played greedily on environments perturbed one setting at
a time, inside and beyond the range it trained on.

Each perturbation is one setting at one value: a noise level the challenge wrapper can schedule
(``obs_noise_std``, ``action_noise_prob``, ``action_noise_std``), or a physics parameter it can
randomize (:attr:`~rl_core.environments.BaseEnvWrapper.RANDOMIZABLE_PARAMETERS`), pinned to
``(value, value)`` while the others keep their configured ranges. Every perturbation gets a new
evaluation environment built from the config's ``environment`` section, so the wrapper's own
checks apply, and episode i of every row resets with the same seed. The start states are paired
across rows; noise draws are not once a pinned parameter adds a draw at each reset.

``robustness.py`` at the repository root is the command line; the Markdown report lives there.
"""

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..agents.base_agent import BaseAgent
from ..challenges.challenge import BenchmarkChallenge
from .checkpoints import read_trainer_state, trainer_state_path
from .curriculum import CurriculumScheduler, interpolate, require_schedulable
from .trainer import Trainer, known_successes, success_rate

_RANDOMIZATION = "domain_randomization"
_SWEEP_FORM = "a sweep is SETTING=V1,V2,... with finite numbers"


@dataclass(frozen=True)
class Perturbation:
    """One setting at one value, as named on the command line (an alias is allowed)."""

    setting: str
    value: float


@dataclass(frozen=True)
class RobustnessRow:
    """
    The greedy episodes of one perturbation. The nominal row, the config's own target, has no
    ``setting``, ``value`` or ``trained_range``. ``successes`` holds each episode's success, or is
    None when the environment does not report them all.
    """

    setting: str | None
    value: float | None
    trained_range: tuple[float, float] | None
    rewards: tuple[float, ...]
    lengths: tuple[int, ...]
    successes: tuple[bool, ...] | None = None

    @property
    def in_range(self) -> bool | None:
        """Whether the value lies in the range the agent trained on, ends included."""
        if self.value is None or self.trained_range is None:
            return None
        low, high = self.trained_range
        return bool(low <= self.value <= high)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards))

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths))

    @property
    def success_rate(self) -> float | None:
        """The share of episodes that succeeded; None when success is not reported."""
        return None if self.successes is None else success_rate(self.successes)

    def solved_fraction(self, threshold: float | None) -> float | None:
        """The share of episodes whose reward reached ``threshold``; None without one."""
        if threshold is None:
            return None
        return sum(reward >= threshold for reward in self.rewards) / len(self.rewards)


def parse_sweep(text: str) -> list[Perturbation]:
    """
    ``SETTING=V1,V2,...`` as one perturbation per value, in order.

    Raises:
        ValueError: If the text is not of that form or a value is not a finite number.
    """
    setting, sep, values = text.partition("=")
    setting = setting.strip()
    if not sep or not setting or not values.strip():
        raise ValueError(f"{text!r}: {_SWEEP_FORM}")
    perturbations = []
    for item in values.split(","):
        try:
            value = float(item)
        except ValueError:
            raise ValueError(f"{text!r}: {_SWEEP_FORM}") from None
        if not math.isfinite(value):
            raise ValueError(f"{text!r}: {_SWEEP_FORM}")
        perturbations.append(Perturbation(setting, value))
    return perturbations


def _sweepable(env: Any) -> tuple[list[str], Mapping[str, str]]:
    """The noise settings and the randomizable parameters (name to canonical) of ``env``."""
    noise = [name for name in env.SCHEDULABLE_SETTINGS if name != _RANDOMIZATION]
    return noise, env.RANDOMIZABLE_PARAMETERS


def canonical_setting(env: Any, name: str) -> str:
    """
    The setting ``name`` sweeps on ``env``: a noise setting as named, a physics parameter under
    the wrapper's canonical name.

    Raises:
        ValueError: Naming what can be swept, if ``env`` can sweep nothing of that name.
    """
    noise, parameters = _sweepable(env)
    if name in noise:
        return name
    try:
        # The wrapper's own reading of the name, aliases and letter case as a config may give them.
        (canonical,) = env.check_setting(_RANDOMIZATION, {name: (1.0, 1.0)})
        return str(canonical)
    except ValueError:
        pass
    raise ValueError(
        f"{type(env).__name__} cannot sweep {name!r}; it can sweep "
        f"{', '.join([*noise, *parameters])}"
    )


def perturbed_section(
    section: Mapping[str, Any], env: Any, perturbation: Perturbation
) -> tuple[str, dict[str, Any]]:
    """
    The canonical setting and a copy of the environment ``section`` with ``perturbation``
    applied and no ``curriculum`` (evaluation plays one fixed setting). ``env`` is a wrapper of
    the challenge, whose rules check the value.

    Raises:
        ValueError: If the setting cannot be swept or the wrapper refuses the value.
    """
    setting = canonical_setting(env, perturbation.setting)
    perturbed = {key: value for key, value in section.items() if key != "curriculum"}
    if setting in env.SCHEDULABLE_SETTINGS:
        perturbed[setting] = env.check_setting(setting, perturbation.value)
        return setting, perturbed
    # Under the wrapper's own names first, so an alias in the config is not given twice.
    ranges = dict(env.check_setting(_RANDOMIZATION, section.get(_RANDOMIZATION) or {}))
    ranges[setting] = (perturbation.value, perturbation.value)
    perturbed[_RANDOMIZATION] = env.check_setting(_RANDOMIZATION, ranges)
    return setting, perturbed


def physics_defaults(challenge: BenchmarkChallenge, section: Mapping[str, Any]) -> dict[str, float]:
    """
    Each randomizable parameter's value in the unrandomized environment, by canonical name, read
    from a new environment built without randomization: the wrappers set a fresh draw at each
    reset and never put the default back.
    """
    plain = {k: v for k, v in section.items() if k not in (_RANDOMIZATION, "curriculum")}
    env = challenge.make_env(plain, render_mode=None)
    try:
        physics = env.env.unwrapped
        canonical = dict.fromkeys(_sweepable(env)[1].values())
        return {name: float(getattr(physics, name)) for name in canonical}
    finally:
        env.close()


def training_range(
    setting: str,
    target: Mapping[str, Any],
    start: Mapping[str, Any] | None,
    difficulty: float,
    defaults: Mapping[str, float],
) -> tuple[float, float]:
    """
    The values of ``setting`` the agent trained on: from the curriculum's ``start`` to the
    settings at the ``difficulty`` it reached, or the ``target`` alone without a curriculum.

    Args:
        setting: A canonical setting name (see :func:`canonical_setting`).
        target: The settings the environment was built with (``scheduled_settings()``): those
            on; a noise setting missing is 0, a physics parameter missing is fixed at its default.
        start: The curriculum's full start settings (``CurriculumScheduler.start``), or None.
        difficulty: The difficulty reached, from 0 to 1 (1 without a curriculum).
        defaults: The physics parameters' unrandomized values (:func:`physics_defaults`).
    """
    if setting in defaults:
        fixed = (defaults[setting], defaults[setting])
        end = tuple(target.get(_RANDOMIZATION, {}).get(setting, fixed))
        begin = end if start is None else tuple(start.get(_RANDOMIZATION, {}).get(setting, end))
    else:
        end = (float(target.get(setting, 0.0)),) * 2
        begin = end if start is None or setting not in start else (float(start[setting]),) * 2
    reached = [interpolate(first, last, difficulty) for first, last in zip(begin, end, strict=True)]
    return (min(begin[0], reached[0]), max(begin[1], reached[1]))


def difficulty_reached(
    agent_checkpoint: str, curriculum: CurriculumScheduler | None
) -> float | None:
    """
    The curriculum difficulty the checkpoint's last training episode played, replayed from the
    training rewards in the trainer file beside it. None without a curriculum, or without a
    trainer file (a checkpoint not named ``agent_episode_<n>``, or none beside it).

    Raises:
        ValueError: If the trainer file is malformed or belongs to another agent file, or the run
            trained with another curriculum than ``curriculum`` (or with one and the config has
            none, or the other way round), as a resume refuses.
    """
    try:
        trainer_path = trainer_state_path(agent_checkpoint)
    except ValueError:
        return None
    if not os.path.exists(trainer_path):
        return None
    state = read_trainer_state(agent_checkpoint)
    if state.curriculum != (None if curriculum is None else curriculum.spec()):
        raise ValueError(
            f"{os.path.basename(agent_checkpoint)} trained with a different curriculum (mode, "
            "pacing, start or target) from the config's: give the config it trained with"
        )
    if curriculum is None:
        return None
    curriculum.replay(state.episode_rewards)
    played = curriculum.played_metrics() or curriculum.metrics()
    return float(played["curriculum_difficulty"])


def robustness_rows(
    challenge: BenchmarkChallenge,
    config: Mapping[str, Any],
    agent: BaseAgent,
    perturbations: Sequence[Perturbation],
    *,
    episodes: int,
    eval_seed: int,
    start: Mapping[str, Any] | None = None,
    difficulty: float = 1.0,
) -> list[RobustnessRow]:
    """
    Play ``agent`` greedily for ``episodes`` on the config's target (the nominal row) and on each
    perturbation in turn, episode i of every row resetting with ``eval_seed + i``.

    Args:
        challenge: The challenge the config belongs to.
        config: The run config (its ``environment`` and ``training`` sections are read).
        agent: The loaded agent.
        perturbations: The perturbations, each checked before any episode is played.
        episodes: Greedy episodes per row.
        eval_seed: The first reset seed of each row.
        start: The curriculum's full start settings, or None without a curriculum.
        difficulty: The curriculum difficulty the agent reached (1 without a curriculum).

    Raises:
        ValueError: If ``episodes`` is not positive, or a perturbation cannot be swept or its value
            is refused.
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be a positive integer, got {episodes!r}")
    section = {k: v for k, v in config["environment"].items() if k != "curriculum"}
    nominal = challenge.make_env(section, render_mode=None)
    try:
        target = require_schedulable(nominal).scheduled_settings()
        sections: list[tuple[str | None, float | None, dict[str, Any]]] = [(None, None, section)]
        for perturbation in perturbations:
            name, perturbed = perturbed_section(section, nominal, perturbation)
            sections.append((name, perturbation.value, perturbed))
    finally:
        nominal.close()
    defaults = physics_defaults(challenge, section)

    rows = []
    for setting, value, env_section in sections:
        trained = (
            None
            if setting is None
            else training_range(setting, target, start, difficulty, defaults)
        )
        env = challenge.make_env(env_section, render_mode=None)
        try:
            trainer = Trainer(
                agent=agent,
                env=env,
                config=config["training"],
                eval_env=env,
                eval_seed=eval_seed,
            )
            played = trainer.evaluate_episodes(episodes)
        finally:
            env.close()
        successes = known_successes([episode.success for episode in played])
        rows.append(
            RobustnessRow(
                setting,
                value,
                trained,
                tuple(float(episode.reward) for episode in played),
                tuple(int(episode.length) for episode in played),
                None if successes is None else tuple(successes),
            )
        )
    return rows
